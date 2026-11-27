"""Bit-exactness check for the GDN decode fusion patch.

Run on an Ascend NPU with the patched vllm-ascend checkout on PYTHONPATH:

    python check_bit_exact.py

It compares the fused split+norm kernel against ``split`` + ``torch.cat`` +
``l2norm_fwd`` (element for element), and the stride-aware gating kernel
against a PyTorch reference (fp32 tolerance). Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys

import torch
import torch_npu  # noqa: F401

from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.gdn_decode_fused import gdn_split_qkv_norm
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu:0"
SHAPES = (1, 3, 16, 48, 256)
HEAD_LAYOUTS = ((8, 16), (16, 32))  # (num_k_heads, num_v_heads): TP2 and TP1


def reference_split_norm(x, nk, nv, dk, dv):
    t = x.shape[0]
    qd = nk * dk
    vd = nv * dv
    q, k, v = torch.split(x, [qd, qd, vd], dim=-1)
    fused = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0)
    q = fused[: t * qd].view(1, t, nk, dk)
    k = fused[t * qd : 2 * t * qd].view(1, t, nk, dk)
    v = fused[2 * t * qd :].view(1, t, nv, dv)
    return l2norm_fwd(q), l2norm_fwd(k), v


def reference_gating(a, b, a_log, dt_bias):
    x = a.float() + dt_bias.float()
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(a_log.float()) * softplus
    beta = torch.sigmoid(b.float()).to(b.dtype)
    return g, beta


def main() -> int:
    init_device_properties_triton()
    torch.manual_seed(0)
    failures = 0
    for t in SHAPES:
        for nk, nv in HEAD_LAYOUTS:
            dk = dv = 128
            total = 2 * nk * dk + nv * dv
            x = torch.randn(t, total, dtype=torch.bfloat16, device=DEVICE)
            q, k, v = gdn_split_qkv_norm(x, nk, nv, dk, dv)
            qr, kr, vr = reference_split_norm(x, nk, nv, dk, dv)
            split_ok = torch.equal(q, qr) and torch.equal(k, kr) and torch.equal(v, vr)

            ba = torch.randn(t, 2 * nv, dtype=torch.bfloat16, device=DEVICE)
            b = ba[:, :nv]
            a = ba[:, nv:]
            a_log = torch.randn(nv, dtype=torch.float32, device=DEVICE)
            dt_bias = torch.randn(nv, dtype=torch.float32, device=DEVICE)
            g, beta = fused_gdn_gating_patch(a_log, a, b, dt_bias)
            g_ref, beta_ref = reference_gating(a, b, a_log, dt_bias)
            g_ok = torch.allclose(g.squeeze(0), g_ref, atol=1e-3, rtol=1e-3)
            beta_ok = torch.allclose(beta.squeeze(0), beta_ref, atol=1e-2, rtol=1e-2)

            ok = split_ok and g_ok and beta_ok
            failures += 0 if ok else 1
            print(
                f"T={t:3d} Nk={nk:2d} Nv={nv:2d}  split_norm={'PASS' if split_ok else 'FAIL'}"
                f"  gating={'PASS' if (g_ok and beta_ok) else 'FAIL'}"
            )
    if failures:
        print(f"{failures} case(s) failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
