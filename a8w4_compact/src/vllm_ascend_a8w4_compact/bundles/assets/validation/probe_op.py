"""Call the A8W4 MSD op with exactly one custom OPP package on the path.

validate.py goes through vllm_ascend.utils.enable_custom_op(), which always
prepends the in-tree vendor, so both packages end up in
ASCEND_CUSTOM_OPP_PATH and the profile cannot tell which kernel ran.  This
probe only imports vllm_ascend (to register torch.ops._C_ascend) and leaves
ASCEND_CUSTOM_OPP_PATH exactly as the shell set it.

    ASCEND_CUSTOM_OPP_PATH=<pkg> python probe_op.py --tag <name>
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

enable_custom_op()  # registers torch.ops._C_ascend.*

E = int(os.environ.get("A8W4_VALIDATE_E", "256"))
K = int(os.environ.get("A8W4_VALIDATE_K", "2048"))
N = int(os.environ.get("A8W4_VALIDATE_N", "1024"))
T = int(os.environ.get("A8W4_VALIDATE_T", "24"))
ACTIVE = min(int(os.environ.get("A8W4_VALIDATE_ACTIVE", "8")), E, T) or 1
NZ = 29


def build(device: int, gltype: int):
    torch.manual_seed(0)
    w = torch.randint(0, 127, (E, K, N // 2), dtype=torch.int8, device=device).contiguous()
    wi = [torch_npu.npu_format_cast(w, NZ).view(torch.int32)]
    scale = torch.from_numpy(
        np.full((E, N), 0.01, dtype=np.float32).view(np.uint32).astype(np.int64)
    ).npu()
    bias = torch.zeros(E, N, dtype=torch.float32, device=device)
    x = torch.randint(-8, 8, (T, K), dtype=torch.int8, device=device)
    pts = torch.full((T,), 0.01, device=device)
    counts = [T // ACTIVE] * ACTIVE + [0] * (E - ACTIVE)
    gl = torch.tensor(counts if gltype else list(np.cumsum(counts)),
                      dtype=torch.int64, device=device)
    return wi, scale, bias, x, pts, gl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--device", type=int, default=int(os.environ.get("A8W4_VALIDATE_DEVICE", "0")))
    ap.add_argument("--gltype", type=int, default=int(os.environ.get("A8W4_VALIDATE_GLTYPE", "1")))
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    wi, scale, bias, x, pts, gl = build(args.device, args.gltype)

    def fn():
        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=x, weight=wi, weight_scale=[scale], x_scale=pts, group_list=gl,
            weight_assist_matrix=[bias], dequant_mode=0,
            group_list_type=args.gltype, swiglu_limit=0.0)

    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        out = fn()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / 50 * 1e6

    outdir = os.environ.get("A8W4_PROBE_DIR", "/tmp/a8w4_probe")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, f"{args.tag}.npz"),
             y=out[0].float().cpu().numpy(),
             y_scale=out[1].float().cpu().numpy())
    print(f"{args.tag}: E={E} T={T} active={ACTIVE} {us:.2f} us/call "
          f"opp={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
