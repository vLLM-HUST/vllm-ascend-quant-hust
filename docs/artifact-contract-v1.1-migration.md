# Artifact Contract 1.1 migration

Contract 1.1 closes an integrity gap in contract 1.0. The old contract hashed
only `quant_model_description.json`; version 1.1 binds these files by exact size
and SHA-256:

- `config.json`;
- `quant_model_description.json`;
- every `*.safetensors` weight shard;
- every `*.safetensors.index.json` index;
- every evidence result named by a non-`schema_only` claim.

The `artifact_id` is derived from the canonical model-file inventory. A changed,
missing, additional, truncated, overlapping, or out-of-bounds artifact fails
before the vLLM-Ascend implementation is imported.

## Upgrade an existing W8A8 artifact

Install the Toolkit and Runtime Extension releases that match this contract —
`ascend-quant-toolkit` 0.5.0 (or later) with `vllm-ascend-quant-ext` 0.5.0 — then
regenerate the contract:

```bash
python -m pip install -e . --no-deps --no-build-isolation
python -m pip install -e ./runtime-extension --no-build-isolation

ascend-quant-toolkit export-contract \
  --model /path/to/Qwen2.5-14B-Instruct-w8a8 \
  --recipe qwen25-w8a8 \
  --evidence-level schema_only

vllm-ascend-quant-ext check \
  --model /path/to/Qwen2.5-14B-Instruct-w8a8
```

This operation rewrites `ascend_quant_artifact.json` only. It does not rewrite
the model weights or quantization description. Hashing all shards can take
noticeable time on a large model and is expected during admission.

Do not manually copy an old `npu_e2e` or `matched_benchmark` flag. Export those
levels only with `--verified-profile` and one or more `--evidence-result` files;
the Toolkit and Runtime Extension both require the files and verify their
hashes.

Contract 1.0 artifacts fail closed under Runtime Extension 0.4.1. Runtime
Extension 0.4.0 remains the historical version used by the existing NPU E2E
record; a fresh 0.4.1 NPU run is required before marking the new checklist item
complete.

## Mixed-precision artifacts are out of scope

Contract 1.1 carries exactly one scheme: `scheme`, `producer_quant_type` and
`runtime_quant_type` are single values, and the runtime validator accepts one
`runtime_quant_type` across every quantized tensor. A checkpoint that quantizes
different scopes differently — for example int4 MLP experts with int8 GDN
projections, as in the `qwen35-35b-a3b-w4a8-gdnw8a8` recipe — therefore has no
contract 1.1 representation.

The Toolkit refuses such a checkpoint instead of stamping it with one scheme:
`prepare-runtime` and `export-contract` raise, and `quantize` runs
`verify-artifact` in place of a contract export. `verify-artifact` checks the
artifact against the recipe that declares it (quantization types, forbidden
`W8A16`, fused `linear_attn` projection pairs, FLOAT scopes, shard count and
bytes). Expressing a mixed artifact to the runtime needs a multi-scheme schema;
until then, serving such a checkpoint requires a loader that reads
`quant_model_description.json` directly.
