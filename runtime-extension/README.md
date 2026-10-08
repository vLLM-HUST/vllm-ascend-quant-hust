# Ascend Quant Runtime Extension

This directory is an independently installable pre-alpha package for the
vLLM-HUST Extension Manager. It is not the offline quantization toolkit and it
does not quantize or rewrite checkpoints.

Its current responsibilities are deliberately limited to validating the
versioned artifact contract emitted by the offline toolkit and advertising the
host protocols a future serving integration requires. The manifest marks the
implementation `import_only`; the Extension Manager discovers and inspects it
but refuses enablement, so it cannot produce vLLM runtime changes.

```bash
pip install ./runtime-extension
vllm-hust-ext extension inspect org.vllm-hust.ascend-quant-runtime
vllm-hust-ext extension check org.vllm-hust.ascend-quant-runtime
```

A runtime release is blocked until vLLM Ascend provides typed, versioned seams
for quantized-artifact loading and quantized operator selection. The current
validator fails closed on unknown schemas, unexpected or missing fields, and
malformed values, but it deliberately does not claim that owner-declared dtype,
packing, shape, or operator names are supported by a serving runtime. Those
values need an owner-approved allowlist before the manifest may move beyond
`import_only`; the implementation must not monkey-patch vLLM internals.

Public package publication is additionally blocked until the repository owners
declare the package license.

## Canonical MOD metadata

Repository identity, direct responsibility, advisor status, default-off activation,
rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not confirmed
`none`, and this descriptor makes no general online performance claim.
