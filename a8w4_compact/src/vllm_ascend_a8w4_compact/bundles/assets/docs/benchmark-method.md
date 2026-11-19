# TP2 W4A8 benchmark method

## Protocol

* Model: Qwen3.5-35B-A3B-W4A8-GDN-W8A8 (W4A8 experts, GDN projections W8A8)
* Hardware: 2x Ascend 910B3, TP2, cards 6/7, port 8302
* Server flags: `--max-model-len 262144 --gpu-memory-utilization 0.95
  --quantization ascend --max-num-seqs 16 --kv-cache-memory-bytes 26038239232
  --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp",
  "num_speculative_tokens":2}' --compilation-config
  '{"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}'`
* Client: swe-prefix-reuse 0.1.2, prepared/qwen35.json
* Per point: 60 s qualification plus a fixed formal window; C1 and C16 are
  measured in one server instance per arm, with no restart and no prefix-cache
  reset between points. Only patch-on vs patch-off W4A8 is compared.
* Use a fresh vLLM/inductor cache and a complete warm Triton cache per arm to
  avoid the cross-rank Triton JIT race.

Measured results are maintained in the organization's benchmark evidence
repository, not in this source repository.
