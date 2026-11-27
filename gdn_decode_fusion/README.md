# vllm-ascend-gdn-decode-fusion

Installer for the **GDN decode-path fusion** patch for vLLM-Ascend. On Ascend
910B the stock GDN decode path re-packs the convolution output into q/k/v with
three reshape copies plus a `torch.cat`, launches two `l2norm_fwd` kernels, and
then hands q/k/v to the `RecurrentGatedDeltaRule` custom operator. This patch
replaces that sequence with a single Triton kernel and removes two contiguous
copies from the gating path.

The patch is pure Python/Triton: no AscendC kernel and no custom-OPP package is
rebuilt.

Tested with Qwen3.5-35B-A3B W4A8 (GDN-W8A8 variant), 2x910B3, TP2.

## What it changes

* `vllm_ascend/ops/triton/gdn_decode_fused.py` (new) - one Triton kernel that
  splits the packed `q|k|v` row of a token, normalizes every q/k head row in
  fp32 (same math as `l2norm_fwd`, eps `1e-6`) and writes contiguous outputs.
* `vllm_ascend/ops/triton/fused_gdn_gating.py` - the gating kernel reads the
  row-strided `b`/`a` halves of the `ba` projection directly instead of
  materializing two contiguous copies.
* `vllm_ascend/ops/gdn.py` - decode-only batches take the fused path; prefill
  keeps the reference path because the chunk kernel normalizes q/k in-kernel.
  Unsupported shapes fall back to the reference implementation.

## Requirements

* Ascend 910B (aarch64) with a working `torch_npu` + Triton installation
* The vLLM-Ascend HUST tree that ships the Ascend GDN implementation
  (`vllm_ascend/ops/gdn.py` with `AscendGatedDeltaNetAttention`)
* Qwen3.5-35B-A3B W4A8 (the GDN-W8A8 variant was validated)

## Install

```bash
pip install vllm-ascend-gdn-decode-fusion
gdn-decode-fusion check      # inspect the checkout and patch state
gdn-decode-fusion install    # apply the patch (auto-detects vllm_ascend)
```

Then restart the vLLM service so the patched modules are imported.

The installer applies a context patch with `git apply` (or `patch`) and keeps a
backup under `<checkout>/.gdn_decode_fusion/`. When the checkout does not match
the patch context it aborts without modifying anything.

### Options

```bash
gdn-decode-fusion install --vllm-ascend /path/to/vllm-ascend
gdn-decode-fusion install --force      # revert first, then apply again
gdn-decode-fusion revert               # restore the backed-up files
```

## Validation

`gdn-decode-fusion assets` prints the bundled assets directory. It contains
`validation/check_bit_exact.py`, a standalone NPU script that compares:

* the fused split+norm kernel against `split` + `cat` + `l2norm_fwd`,
* the stride-aware gating kernel against a PyTorch reference,

and prints `PASS`/`FAIL` per shape. Run it with the patched tree on
`PYTHONPATH`:

```bash
python validation/check_bit_exact.py
```

## Evidence

Measured benchmark evidence is maintained in the organization's benchmark
evidence repository rather than in this source repository; see that repository
for the end-to-end comparison and the kernel-level measurements. Reproduce it
with the validation script above and the TP2 harness in the evidence
repository.

## What is inside

```bash
gdn-decode-fusion assets
```

prints a directory with:

* `patches/gdn-decode-fusion.patch` - the complete diff against the vllm-ascend
  tree,
* `validation/check_bit_exact.py` - the bit-exactness check,
* `method.md` - how the fusion works and which paths fall back.

## Uninstall

```bash
gdn-decode-fusion revert
```

or restore the files from `<checkout>/.gdn_decode_fusion/backup-*` manually.

## License

The installer code is Apache-2.0. The bundled patch edits vLLM-Ascend sources
that remain under their original Apache-2.0 license and copyright headers. See
`NOTICE`.
