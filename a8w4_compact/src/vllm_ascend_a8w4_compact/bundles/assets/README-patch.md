# A8W4 MSD compact active-expert descriptor (TP2-safe)

Kernel patch for `grouped_matmul_swiglu_quant_v2` (A8W4 MSD path) in the
vLLM-Ascend `custom_transformer` vendor package. It targets Qwen3.5-35B-A3B W4A8
experts on Ascend 910B3 and keeps the kernel correct under TP2.

## Problem

With 256 experts the A8W4 MSD kernel spent a large part of its time walking the
expert list with blocking scalar GM reads:

* `InitWorkSpaceSplitConfig` on **every** AIC/AIV core to obtain the GEMM row
  count `M`,
* `GMMA8W4MidProcess` per-expert loop over all `E` experts,
* `GMMA8W4PostProcess::UpdateVecConfig` and `UpdateAuxiliaryMatrix` walking the
  list again to skip inactive experts.

At decode the list is mostly empty (top-k 8, a few dozen rows), so the work is
pure overhead. Kernel time at T=24, E=256 was ~90 us versus ~44 us at E=8.

## Changes

1. One AIV core bulk-copies `group_list` into UB once, builds a compact
   descriptor `[numActive][idx[E]][cumToken[E+1]]` in UB and writes it to the
   start of the unused `mmOut2` workspace region. `M` is published through the
   same descriptor. Two `SyncAll<false>()` barriers make it visible.
2. The AIC `mid` consumes contiguous active-expert ranges per core and computes
   `x/y` offsets from `cumToken`; no `group_list` scan.
3. The AIV `post` binary-searches `cumToken` for its start expert and advances
   to the next active expert in O(1) from the descriptor.
4. `batch1` fast path for `loopCount == 1` and inactive-expert skips are kept
   (inactive experts still reset `mnConfig.m = 0`).
5. The previous phase-1 UB bulk copy in the post is **removed**: on TP2 its
   extra 2 KB UB buffer overran the post's UB budget and raised an AIV vector
   exception. It is kept in `patches/history/` only as a rejected experiment.

The descriptor path is enabled only when `loopCount == 1`, `M <= 1024` and the
`mmOut2` region is large enough; otherwise the original scan path runs
(prefill, multi-split).

## Evidence

Measured benchmark evidence is maintained in the organization's benchmark
evidence repository rather than in this source repository. The scripts and
probes in this tree reproduce the measurement; see `docs/benchmark-method.md`
for the protocol.

## Build and deploy

See `docs/build-and-merge.md`. Summary:

```bash
# from a vllm-ascend source tree
git apply /path/to/patches/0004-a8w4-msd-tp2-safe-compact.patch
cd csrc
source <CANN 9.1 env>
bash build.sh --pkg --soc=ascend910b --vendor_name=custom \
    --ops=grouped_matmul_swiglu_quant_v2 -j16 -O3
# merge the resulting custom_transformer kernel dir into a FULL in-tree vendor
bash /path/to/scripts/merge_full_vendor.sh
```

**Deploy as a full vendor package.** A partial package that contains only this
op shadows the in-tree vendor directory; the GDN custom ops of the W4A8-GDN
model then fall back to CANN built-ins and engine init fails. The merge script
copies the complete in-tree vendor and overlays only the rebuilt GMM v2 kernel
directory; the in-tree `liboptiling.so` is kept because this patch does not
change host tiling.

## Files

```
kernel/op_kernel/            patched headers (5 files)
patches/0004-*.patch         complete diff vs the stock vendor source
patches/history/             phase-1 and earlier experiments (phase-1 is TP2-unsafe)
scripts/build_partial.sh     build the partial op package
scripts/merge_full_vendor.sh merge into a full 43-op vendor package
scripts/smoke_tp2.sh         reference TP2 smoke (adapt paths)
scripts/benchmark_tp2_c1c16.sh reference C1/C16 900 s benchmark (adapt paths)
validation/probe_kernel.py   device-side kernel timing via torch_npu profiler
validation/probe_op.py       direct op probe (stock vs patched, bit-exact check)
docs/                        results, build notes, phase-1 post-mortem
```

No build trees, caches, logs, model weights or prebuilt `.run` binaries are
included. If a prebuilt package is desired, attach the `.run` to a GitHub
Release instead of committing it.

## License / origin

The patched files derive from the `custom_transformer` vendor sources shipped
with the vLLM-Ascend HUST tree and retain their Huawei copyright and CANN Open
Software License headers. Keep those headers and confirm the license terms with
the repository owners before publishing binaries.
