# vllm-ascend-a8w4-compact

Installer for the **A8W4 MSD compact active-expert descriptor** kernel patch for
`grouped_matmul_swiglu_quant_v2` in vLLM-Ascend's `custom_transformer` vendor.
It removes the O(E) expert-list scans from the 4-bit MoE GEMM on Ascend 910B and
is safe under TP2.

Tested with Qwen3.5-35B-A3B W4A8 (GDN-W8A8 variant), 2x910B3, TP2.

## Why an installer

The compiled artifact is a *partial* vendor package that contains only
`grouped_matmul_swiglu_quant_v2`. Putting it first on `ASCEND_CUSTOM_OPP_PATH`
shadows the other custom ops (notably the GDN ops of W4A8-GDN models) and makes
engine initialization fail. This package therefore:

1. locates your full in-tree `custom_transformer` vendor,
2. copies it to `~/.cache/vllm_ascend_a8w4_compact/custom_transformer`,
3. replaces only the `grouped_matmul_swiglu_quant_v2` kernel directory (and its
   kernel config JSON) with the rebuilt one,
4. keeps the in-tree `liboptiling.so` (the patch changes no host tiling).

The resulting directory is a full 43-op vendor package and is exactly where the
`a8w4_compact` plugin looks by default.

## Requirements

* Ascend 910B3 (aarch64), CANN 9.1 runtime
* vLLM-Ascend HUST tree with the `a8w4_compact` plugin
  (`VLLM_ASCEND_A8W4_COMPACT` support)
* Qwen3.5-35B-A3B W4A8 model (the W4A8-GDN-W8A8 variant was benchmarked)

## Install

```bash
pip install vllm-ascend-a8w4-compact
a8w4-compact check          # inspect platform, plugin and installation state
a8w4-compact install        # merge into the full in-tree vendor
a8w4-compact env            # print the environment variables
```

Then start vLLM in the environment that has:

```bash
export VLLM_ASCEND_A8W4_COMPACT=1
# only needed if you installed to a non-default --opp-dir:
# export VLLM_ASCEND_A8W4_COMPACT_OPP=<opp-dir>/custom_transformer
```

The plugin prepends the package to `ASCEND_CUSTOM_OPP_PATH`; verify with:

```bash
python -m vllm_ascend.plugins.a8w4_compact
# expect: active: True, note: using rebuilt vendor package at ...
```

### Options

```bash
a8w4-compact install --vendor /path/to/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
a8w4-compact install --opp-dir /custom/opp --force
a8w4-compact install --skip-platform-check   # not recommended
```

`install` refuses a vendor with fewer than 10 ops (a partial package) because it
would shadow the other custom ops.

## Evidence

This package ships the kernel, build/merge scripts, probes and reproduction
tooling. Measured benchmark evidence is maintained in the organization's
benchmark evidence repository rather than in this source repository; see that
repository for the C1/C16 comparison and kernel-level timings. Reproduce it
with `scripts/benchmark_tp2_c1c16.sh` and `validation/probe_kernel.py` (paths
printed by `a8w4-compact assets`).

## What is inside

```bash
a8w4-compact assets
```

prints a directory with:

* `kernel/op_kernel/` - the five patched headers,
* `patches/0004-a8w4-msd-tp2-safe-compact.patch` - complete diff vs the stock
  vendor source,
* `patches/history/` - earlier experiments (phase-1 is TP2-unsafe; do not use),
* `docs/` - results, build/merge notes, phase-1 post-mortem,
* `scripts/` - build, merge, smoke and C1/C16 benchmark references,
* `validation/` - profiler probes and the bit-exact comparison.

The bundled `bundles/a8w4_compact_910b_cann9.1.tar.gz` is the prebuilt partial
operator package (aarch64, CANN 9.1, 910B).

## Uninstall

```bash
rm -rf ~/.cache/vllm_ascend_a8w4_compact
unset VLLM_ASCEND_A8W4_COMPACT VLLM_ASCEND_A8W4_COMPACT_OPP
```

## Build from source

For a different CANN version or platform, apply
`patches/0004-a8w4-msd-tp2-safe-compact.patch` to the vllm-ascend tree and build
with `scripts/build_partial.sh`; then merge with
`scripts/merge_full_vendor.sh` as described in `docs/build-and-merge.md`.

## License

The installer code is Apache-2.0. The bundled kernel package and patched
sources derive from the vLLM-Ascend `custom_transformer` vendor and retain their
Huawei copyright and CANN Open Software License headers. See `NOTICE`.
