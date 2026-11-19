"""Numerics + timing check for the a8w4_compact rebuilt kernel.

The switch changes which vendor package resolves the SAME op name, so the two
variants cannot be loaded in one process.  Run each variant as its own process
and compare the dumps:

    # stock kernels
    python validate.py run --tag stock
    # rebuilt kernels
    VLLM_ASCEND_A8W4_COMPACT=1 python validate.py run --tag compact
    python validate.py compare --a stock --b compact

Bit-identical output is the acceptance bar: patch 0001 only changes where the
group_list values are read from (UB instead of GM), not what is computed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch_npu

DUMP_DIR = os.environ.get("A8W4_VALIDATE_DIR", "/tmp/a8w4_validate")
E, K, N, T, ACTIVE, NZ = 256, 2048, 1024, 24, 8, 29
E = int(os.environ.get("A8W4_VALIDATE_E", E))
T = int(os.environ.get("A8W4_VALIDATE_T", T))
ACTIVE = min(int(os.environ.get("A8W4_VALIDATE_ACTIVE", ACTIVE)), E, T) or 1


def _build(device: int):
    # Deterministic operands: both variants must see byte-identical inputs.
    torch.manual_seed(int(os.environ.get("A8W4_VALIDATE_SEED", "0")))
    # Mirror process_weights_after_loading_modelslim exactly:
    #   transpose(1,2) -> (E, k, n/2) int8, maybe_trans_nz(int8), then
    #   .view(torch.int32) -> (E, k, n/8).  The view must come *after* the NZ
    #   cast: the physical tiling stays int8-NZ, which under the int32 view is
    #   the 16x64-int4 NZ block the A8W4-MSD tiling expects (see
    #   elastic_load.py, "_cast_packed_int32_via_int8").  Casting int32 first
    #   gives int32-NZ blocks and the cube core faults.
    # The weight goes in as a ONE-element list ([layer.w13_weight] in w4a8.py);
    # a per-expert list takes the ACLNN multi-tensor branch instead.
    w = torch.randint(0, 127, (E, K, N // 2), dtype=torch.int8, device=device).contiguous()
    wi = [torch_npu.npu_format_cast(w, NZ).view(torch.int32)]
    scale = torch.from_numpy(
        np.full((E, N), 0.01, dtype=np.float32).view(np.uint32).astype(np.int64)
    ).npu()
    # w13_scale_bias after transpose(1,2).sum(axis=1) is (E, n).
    bias = torch.zeros(E, N, dtype=torch.float32, device=device)
    x = torch.randint(-8, 8, (T, K), dtype=torch.int8, device=device)
    pts = torch.full((T,), 0.01, device=device)
    # group_list_type=1 (counts): ACTIVE experts with T//ACTIVE tokens each.
    # A8W4_VALIDATE_GLTYPE=0 encodes the same assignment cumulatively.
    counts = [T // ACTIVE] * ACTIVE + [0] * (E - ACTIVE)
    if os.environ.get("A8W4_VALIDATE_GLTYPE", "1") == "0":
        gl = torch.tensor(list(np.cumsum(counts)), dtype=torch.int64, device=device)
    else:
        gl = torch.tensor(counts, dtype=torch.int64, device=device)
    return wi, scale, bias, x, pts, gl


def cmd_run(args: argparse.Namespace) -> int:
    from vllm_ascend.plugins import a8w4_compact

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    import vllm_ascend.utils as utils

    assert utils.enable_custom_op(), "custom ops unavailable"

    wi, scale, bias, x, pts, gl = _build(args.device)
    gltype = int(os.environ.get("A8W4_VALIDATE_GLTYPE", "1"))

    def fn():
        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=x, weight=wi, weight_scale=[scale], x_scale=pts, group_list=gl,
            weight_assist_matrix=[bias], dequant_mode=0, group_list_type=gltype, swiglu_limit=0.0)

    for _ in range(10):
        fn()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        out = fn()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / 50 * 1e6

    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"{args.tag}.npz")
    np.savez(path,
             y=out[0].float().cpu().numpy(),
             y_scale=out[1].float().cpu().numpy(),
             us=us,
             opp=os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
             active=a8w4_compact.activate())
    print(f"{args.tag}: {us:.2f} us/call, opp={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}")
    print(f"{args.tag}: dumped to {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = np.load(os.path.join(DUMP_DIR, f"{args.a}.npz"))
    b = np.load(os.path.join(DUMP_DIR, f"{args.b}.npz"))
    ya, yb = a["y"], b["y"]
    if ya.shape != yb.shape:
        print(f"SHAPE MISMATCH {ya.shape} vs {yb.shape}")
        return 1
    same = np.array_equal(ya.view(np.uint32), yb.view(np.uint32))
    diff = float(np.max(np.abs(ya - yb))) if ya.size else 0.0
    cos = float((ya.ravel() @ yb.ravel()) / (np.linalg.norm(ya) * np.linalg.norm(yb) + 1e-12))
    print(f"bit-identical : {same}")
    print(f"max |diff|    : {diff:.6g}")
    print(f"cosine        : {cos:.8f}")
    print(f"time {args.a:>8}: {float(a['us']):.2f} us   ({a['opp']})")
    print(f"time {args.b:>8}: {float(b['us']):.2f} us   ({b['opp']})")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--device", type=int, default=int(os.environ.get("A8W4_VALIDATE_DEVICE", "4")))
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--a", default="stock")
    c.add_argument("--b", default="compact")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
