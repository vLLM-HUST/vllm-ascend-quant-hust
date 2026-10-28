# vllm-ascend-quant-hust

An offline post-training quantization toolkit for Ascend NPUs, plus a separately
packaged runtime compatibility extension.

This repository is the canonical source of the `ascend-quant-toolkit` PyPI
distribution. Versions are published from here under the `toolkit-v<version>`
tag; the former personal repository keeps its already-published files but no
longer publishes new versions.

| Deliverable | Distribution | Stable? | What it is |
|---|---|---|---|
| Offline quantization toolkit | `ascend-quant-toolkit` on PyPI (this README) | 0.5.0 | Recipes, artifact inspection, precision-scope verification, contract export |
| Runtime compatibility extension | `vllm-ascend-quant-ext` | frozen at 0.4.1a4 | Inert, default-off vLLM plugin; **not** part of the 0.5.0 release |
| Legacy scripts | none | unversioned | `qwen2.5_w4a4.py`, `qwen3_w4a4.py`, `test_ppl.py` — kept for reproducibility, not packaged |
| Legacy YAMLs | none | unversioned | `qwen2.5-14b-w8a8-pdmix-a030.yaml`, `qwen_w8a8.yaml` — superseded by the packaged recipes |

## Install

```bash
pip install ascend-quant-toolkit==0.5.0
```

- Python >= 3.10, `py3-none-any` wheel, **zero runtime dependencies**.
- Nothing in the wheel imports `torch`, `vllm`, `vllm_ascend` or `msmodelslim`.
  Running a quantization needs `msmodelslim` installed separately; everything
  except `quantize`/`plan` works on a CPU-only machine.
- From source: `pip install -e . --no-deps`.

## What it does

```bash
ascend-quant-toolkit list-recipes          # what ships
ascend-quant-toolkit plan --recipe ...     # print the exact msModelSlim command, run nothing
ascend-quant-toolkit quantize --recipe ... # run msModelSlim, then verify the artifact
ascend-quant-toolkit inspect-model --model PATH
ascend-quant-toolkit verify-artifact --artifact PATH --recipe NAME [--hash]
ascend-quant-toolkit export-contract --model PATH --recipe NAME   # single-scheme artifacts only
ascend-quant-toolkit doctor --path PATH
```

| Command | Status |
|---|---|
| `list-recipes`, `plan` | implemented, CPU-only |
| `inspect-model`, `verify-artifact`, `doctor` | implemented, CPU-only, run against real artifacts |
| `quantize` | implemented; executes the external `msmodelslim` CLI on an NPU |
| `export-contract`, `prepare-runtime`, `restore-modelslim` | implemented; **single-scheme artifacts only** |
| `build-smoothing`, `inspect-smoothing`, `build-online-view` | implemented for the runtime extension's ONLINE path |

## Recipes

| Recipe | Model | Weights / activations | Status |
|---|---|---|---|
| `qwen25-w8a8` | Qwen2.5-14B-Instruct | W8A8, SmoothQuant alpha=0.30, C4 calibration | verified on NPU |
| `qwen35-35b-a3b-w4a8-gdnw8a8` | Qwen3.5-35B-A3B | int4 experts + int8 GDN/attention, no rotation, `calibImages` | artifact verified; accuracy and throughput measured on 1x910B3 |

### `qwen35-35b-a3b-w4a8-gdnw8a8`

```bash
ascend-quant-toolkit plan \
  --recipe qwen35-35b-a3b-w4a8-gdnw8a8 \
  --model  /path/to/Qwen3.5-35B-A3B \
  --output /path/to/Qwen3.5-35B-A3B-W4A8-GDNW8A8

ascend-quant-toolkit quantize \
  --recipe qwen35-35b-a3b-w4a8-gdnw8a8 \
  --model  /path/to/Qwen3.5-35B-A3B \
  --output /path/to/Qwen3.5-35B-A3B-W4A8-GDNW8A8 \
  --device npu:0

ascend-quant-toolkit verify-artifact \
  --artifact /path/to/Qwen3.5-35B-A3B-W4A8-GDNW8A8 \
  --recipe qwen35-35b-a3b-w4a8-gdnw8a8
```

The recipe quantizes:

| Scope | Layers x projections | Weight / activation |
|---|---|---|
| `mlp.experts` gate/up/down | 40 x 256 experts | int4 / int8 (`W4A8_DYNAMIC`) |
| `linear_attn` `in_proj_qkv` / `in_proj_z` / `out_proj` | 30 x 3 | int8 / int8 (`W8A8_DYNAMIC`) |
| `self_attn` q/k/v | 10 x 3 | int8 / int8 (`W8A8_DYNAMIC`) |
| `self_attn.o_proj`, `linear_attn.in_proj_a` / `in_proj_b`, `shared_expert`, `lm_head`, `visual`, `mtp` | — | FLOAT |

A verified artifact has 6 `quant_model_weights-*.safetensors` shards totalling
24,047,130,272 bytes and a `quant_model_description.json` histogram of
`W4A8_DYNAMIC` 122,880 / `W8A8_DYNAMIC` 360 / `FLOAT` 1,611 / `W8A16` 0. The
recipe pins the shard count and the byte total; `--hash` additionally records a
sha256 per shard.

## What this does not claim

- **No speed claim.** The published 2026-10-23 curve is a capacity-vs-accuracy
  operating point. At C1 the three compared recipes are within 3%, which is the
  spread of a single 900 s observation; at C16 the difference moves together
  with the KV budget and cannot be attributed to the recipe. See
  [`docs/evidence/qwen35-w4a8-gdnw8a8-tp1-20261023.md`](docs/evidence/qwen35-w4a8-gdnw8a8-tp1-20261023.md).
- **No cross-hardware claim.** The artifact was verified and benchmarked on one
  910B3, TP1, with a specific engine revision.
- **No KV-cache quantization and no speculative-decoding work.** Those belong
  to other teams; MTP appears here only as a serving-side precondition.

## Limitations

- **Mixed-precision artifacts cannot carry a runtime contract.** Artifact
  contract 1.1.0 has exactly one scheme and the runtime validator accepts one
  `runtime_quant_type` across all tensors, so `prepare-runtime`,
  `restore-modelslim` and `export-contract` refuse a checkpoint that mixes
  quantization types. `quantize` runs `verify-artifact` for such recipes
  instead. Adding a multi-scheme contract is a schema change and is not planned
  here.
- **`W8A16` is refused for this model.** vLLM-Ascend's `npugraph_ex` path hits
  a `CompilerConfig` attribute this build does not provide, so the recipe sets
  `forbid_w8a16` and verification fails if any `W8A16` entry appears.
- **GDN projections are fused at load.** vLLM merges `in_proj_qkv` and
  `in_proj_z`, so a checkpoint that quantizes only one of them cannot be
  served. `verify-artifact` enforces the sibling types match, for every recipe.
- **The artifact is pinned to one recipe file.** The packaged YAML is a verbatim
  copy of the file that produced the published checkpoint (its sha256 is
  recorded in `tests/test_qwen35_recipe.py`); editing the YAML changes the
  artifact identity and needs a new evidence record.
- **Reproducing the checkpoint needs msModelSlim and an NPU.** The toolkit
  invokes `msmodelslim quant` as an external program; it does not vendor it.
- **Serving the artifact needs engine work that is not upstream.** The measured
  curve ran vLLM `0.28.1.post1.dev143+gf18cf803c` and vLLM-Ascend
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` with three local engine
  changes (MTP contract, `npugraph_ex` attribute shim, and one recorded in the
  benchmark report). The `npugraph_ex` shim is inert for this artifact because
  it contains zero `W8A16` entries. Quantization itself does not need these
  patches.

## Accuracy

Same wikitext-2 protocol (298 chunks of 999 tokens, concurrency 8, bf16 baseline
in the same session, per-chunk paired ΔNLL):

| Checkpoint | Quantization scope | PPL | vs bf16 |
|---|---|---|---|
| bf16 | — | 7.1694 | — |
| W8A8 | experts + attn q/k/v W8A8 | 7.1750 | +0.08% |
| experts int4 | experts W4A8 + attn q/k/v W8A8, GDN bf16 | 7.8551 | +9.6% |
| **this recipe** | above + GDN three projections W8A8 | **7.9070** | +10.3% |

GDN's int8 weights cost +0.66% PPL relative to the same experts-only recipe.

## Runtime extension

`runtime-extension/` is a separate distribution with its own tag namespace
(`runtime-v<version>`), manifest, and release checklist; the 0.5.0 toolkit
release does not change it. Its canonical identity, default-off activation and
rollback are recorded in
[`runtime-extension/MOD_METADATA.json`](runtime-extension/MOD_METADATA.json).
`advisor_status: unknown` is not confirmed `none`; no performance claim is
implied.

## Legacy scripts

The repository root still carries the original research scripts:

```bash
python3 qwen3_w4a4.py --model_path /root/models/Qwen3-8B \
  --save_directory /root/models/Qwen3-8B-w4a4 \
  --calib_file common/qwen_qwen3_cot_w4a4.json --trust_remote_code True --batch_size 1

python test_ppl.py --model_path /root/models/Qwen2.5-7B-Instruct-w8a8
```

They are not packaged, not covered by the release workflow, and are kept only so
older results stay reproducible.

## Dependencies and license

- Runtime dependencies: none.
- Test dependencies: `pytest`, `jsonschema`, `numpy`, `safetensors`.
- Code: Apache-2.0 (see [`LICENSE`](LICENSE)).
- External provenance is recorded in [`PROVENANCE.md`](PROVENANCE.md).

## Authors

Hongkun Wang, Xuedong Jiang, Chaiyu Gui, Xiang Xiang (HAIV Lab, https://github.com/HAIV-Lab)
