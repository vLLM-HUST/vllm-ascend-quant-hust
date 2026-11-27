# How the GDN decode fusion works

## Stock sequence (per GDN layer, decode)

1. `rearrange_mixed_qkv` splits the packed `q|k|v` vector into three views and
   then flattens them with three `reshape` copies followed by one `torch.cat`
   so that q/k/v are contiguous.
2. `l2norm_fwd` runs twice, once for q and once for k.
3. `RecurrentGatedDeltaRule` consumes q/k/v, the gating outputs and the SSM
   state.

Before that, the gating path calls `.contiguous()` on the two halves of the
`ba` projection (`b` and `a`), although both are ordinary row-strided views.

## Patched sequence

1. One Triton kernel (`_gdn_split_norm_kernel`) reads the packed row of one
   token, normalizes each q/k head row in fp32, writes contiguous
   `[T, Nk, Dk]` and `[T, Nv, Dv]` outputs, and copies v as-is.
2. `fused_gdn_gating_kernel` takes the row strides of `b` and `a` as runtime
   arguments, so no contiguous copy is needed.

The math is unchanged: the L2 normalization uses fp32 accumulation with
`eps = 1e-6`, exactly like `l2norm_fwd`, and the outputs are cast back to the
input dtype. The gating keeps the same `softplus`/`sigmoid` formulation.

## Fallbacks

* Prefill batches keep the reference `rearrange_mixed_qkv` + `l2norm_fwd`
  sequence, because the chunk kernel already normalizes q/k in-kernel and the
  raw q/k are expected there.
* Mixed prefill+decode batches re-run the fused split only on the decode slice.
* Shapes that are not power-of-two per head, non-contiguous inputs, or dtypes
  other than bf16/fp16 fall back to the reference implementation per call.
* Non-MTP decode and the MTP draft path are unchanged outside the two call
  sites above.

## Why it is safe

The fused kernel was checked against the reference path on the NPU for
`T = 1, 3, 16, 48, 256` with both TP1 (Nk=16, Nv=32) and TP2 (Nk=8, Nv=16)
head layouts; q/k/v match element for element. The gating kernel was checked
against a PyTorch reference for the same shapes with contiguous and strided
inputs. `validation/check_bit_exact.py` reproduces both checks.
