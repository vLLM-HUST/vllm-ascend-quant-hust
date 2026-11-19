"""Measure the A8W4 MSD kernel time (device-side) with torch_npu profiler.

Same inputs as probe_op.py. Prints the mean/min/max duration of the
GroupedMatmulSwigluQuantV2 kernel events from the chrome trace.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch_npu
import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op
enable_custom_op()

E = int(os.environ.get("A8W4_VALIDATE_E", "256"))
K = int(os.environ.get("A8W4_VALIDATE_K", "2048"))
N = int(os.environ.get("A8W4_VALIDATE_N", "1024"))
T = int(os.environ.get("A8W4_VALIDATE_T", "24"))
ACTIVE = min(int(os.environ.get("A8W4_VALIDATE_ACTIVE", "24")), E, T) or 1
GLTYPE = int(os.environ.get("A8W4_VALIDATE_GLTYPE", "1"))
NZ = 29

def build(device):
    torch.manual_seed(0)
    w = torch.randint(0, 127, (E, K, N // 2), dtype=torch.int8, device=device).contiguous()
    wi = [torch_npu.npu_format_cast(w, NZ).view(torch.int32)]
    scale = torch.from_numpy(np.full((E, N), 0.01, dtype=np.float32).view(np.uint32).astype(np.int64)).npu()
    bias = torch.zeros(E, N, dtype=torch.float32, device=device)
    x = torch.randint(-8, 8, (T, K), dtype=torch.int8, device=device)
    pts = torch.full((T,), 0.01, device=device)
    counts = [T // ACTIVE] * ACTIVE + [0] * (E - ACTIVE)
    gl = torch.tensor(counts if GLTYPE else list(np.cumsum(counts)), dtype=torch.int64, device=device)
    return wi, scale, bias, x, pts, gl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="k")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    wi, scale, bias, x, pts, gl = build(args.device)
    def fn():
        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=x, weight=wi, weight_scale=[scale], x_scale=pts, group_list=gl,
            weight_assist_matrix=[bias], dequant_mode=0,
            group_list_type=GLTYPE, swiglu_limit=0.0)
    for _ in range(10): fn()
    torch.npu.synchronize()
    outdir = "/tmp/a8w4_kprof_" + args.tag
    os.makedirs(outdir, exist_ok=True)
    from torch_npu.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(args.iters): fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(os.path.join(outdir, "trace.json"))
    tr = json.load(open(os.path.join(outdir, "trace.json")))
    events = tr.get("traceEvents", []) if isinstance(tr, dict) else tr
    matched = [(e.get("cat"), e.get("name"), e.get("dur")) for e in events
               if "GroupedMatmul" in str(e.get("name", ""))]
    durs = [d for (c, n, d) in matched if isinstance(d, (int, float))]
    if durs:
        print(f"{args.tag}: E={E} K={K} N={N} T={T} active={ACTIVE} gltype={GLTYPE} "
              f"kernels={len(durs)} mean={sum(durs)/len(durs):.2f}us min={min(durs):.2f} max={max(durs):.2f}")
    else:
        from collections import Counter
        cats = Counter(str(e.get("cat")) for e in events)
        sample = [(e.get("cat"), str(e.get("name"))[:60]) for e in events[:8]]
        print(args.tag, "no GroupedMatmul events; total=", len(events), "cats=", cats, "sample=", sample)


if __name__ == "__main__":
    main()
