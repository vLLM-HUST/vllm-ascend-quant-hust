# Build and merge

## Environment used

* CANN 9.1.0 at `/data/jxd/Ascend-9.1.0/cann-9.1.0` (adapt paths)
* aarch64, Ascend 910B3
* vLLM-Ascend HUST source tree with the `custom_transformer` vendor package
* system cmake/ninja; a clean build tree is required

## Apply the patch

From the vllm-ascend source root:

```bash
git apply /path/to/patches/0004-a8w4-msd-tp2-safe-compact.patch
```

The patch touches five files under
`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/grouped_matmul_swiglu_quant_v2/`.
It does not change host tiling, so `liboptiling.so` stays as shipped.

## Build the op package

```bash
cd csrc
source /path/to/cann-9.1.0/set_env.sh   # or the project env script
export PATH=/path/to/python-with-regex/bin:$PATH
bash build.sh --pkg --soc=ascend910b --vendor_name=custom \
    --ops=grouped_matmul_swiglu_quant_v2 -j16 -O3
```

The package lands in `csrc/build_out/cann-ops-transformer-custom_linux-aarch64.run`
and in `csrc/build/_CPack_Packages/.../packages/vendors/custom_transformer`.

## Merge into a full vendor package

Do **not** deploy the partial package. It only contains this op and shadows the
in-tree vendor, which breaks the GDN custom ops of the W4A8-GDN model.

```bash
bash /path/to/scripts/merge_full_vendor.sh \
  <built-partial-vendor-dir> <in-tree-vendor-dir> <output-opp-dir>
```

The script copies the full in-tree vendor and overlays the rebuilt
`grouped_matmul_swiglu_quant_v2` kernel directory plus its `config/ascend910b`
JSON. The in-tree `liboptiling.so` is kept.

## Enable

```bash
export VLLM_ASCEND_A8W4_COMPACT=1
export VLLM_ASCEND_A8W4_COMPACT_OPP=<output-opp-dir>/custom_transformer
```

The plugin prepends the package to `ASCEND_CUSTOM_OPP_PATH`; verify
`python -m vllm_ascend.plugins.a8w4_compact` reports `active: True`.

## Smoke test

Use `scripts/smoke_tp2.sh` as a reference (adapt the model path, cards, port,
cache dirs). A correct run reports `READY`, the website KV size, and the greedy
prompt `' Paris.\nThe capital of France is'`.

## Benchmark

`scripts/benchmark_tp2_c1c16.sh` is the reference arm used for the published
numbers (60 s qualification + 900 s formal, C1 and C16 in one server instance).
It expects the swe-prefix-reuse wrapper and the server metadata JSON; adapt the
paths. Use a fresh vLLM/inductor cache and a complete warm Triton cache per arm
to avoid the cross-rank Triton JIT race.

## Notes

* Clean-build only: reusing an incremental `csrc/build` tree can silently keep
  stale device objects. Move the old `build/` aside before rebuilding.
* Phase-1 (`patches/history/0001`) is TP2-unsafe; do not enable it.
