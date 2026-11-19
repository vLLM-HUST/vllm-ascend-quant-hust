#!/usr/bin/env bash
# vllm-hust-website TP2 W4A8-GDNW8A8 launch (same command as the published doc).
# FULL merged compact package (in-tree vendor + patched GMM v2) so the model's
# custom GDN ops stay available. Early-exit on service death or init errors.
set -u
RUN=/data/whk/measure-w4a8-tp1
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RUN/logs/tp2_website_b2k4_${TS}.log
mkdir -p "$RUN/logs" "$RUN/cache-tp2-w4a8gdn-b2k4"/{vllm,torch_ext,torchinductor,triton}
exec >>"$LOG" 2>&1
log(){ echo "[b2k4 $(date -u +%H:%M:%S)] $*"; }
log "START $TS"
source "$RUN/cann91-env.sh"
rm -rf "$HOME/.triton/cache"
shopt -s nullglob
for item in "$RUN"/cache-stock/triton/*/npu_utils.so; do
  rel=${item#"$RUN/cache-stock/triton/"}; mkdir -p "$HOME/.triton/cache/$(dirname "$rel")"
  [ -f "$HOME/.triton/cache/$rel" ] || cp -a "$item" "$HOME/.triton/cache/$rel"
done
shopt -u nullglob
(
  export ASCEND_RT_VISIBLE_DEVICES=6,7
  export VLLM_CACHE_ROOT=$RUN/cache-tp2-w4a8gdn-b2k4/vllm
  export TORCH_EXTENSIONS_DIR=$RUN/cache-tp2-w4a8gdn-b2k4/torch_ext
  export TORCHINDUCTOR_CACHE_DIR=$RUN/cache-tp2-w4a8gdn-b2k4/torchinductor
  export TRITON_CACHE_DIR=$RUN/cache-tp2-w4a8gdn-b2k4/triton
  export COMPILE_CUSTOM_KERNELS=1 PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
  export VLLM_DISABLE_COMPILE_CACHE=1
  export VLLM_ASCEND_A8W4_COMPACT=1
  export VLLM_ASCEND_A8W4_COMPACT_OPP=/data/whk/a8w4_compact/opp_full_b2k4/custom_transformer
  exec /data/jxd/envs/vllm-hust-v1-cann91/bin/vllm serve /data/whk/models/Qwen3.5-35B-A3B-W4A8-GDN-W8A8 \
    --served-model-name qwen35-a3b-w4a8gdn --tensor-parallel-size 2 --max-model-len 262144 \
    --gpu-memory-utilization 0.95 --trust-remote-code --host 127.0.0.1 --port 8302 \
    --quantization ascend --max-num-seqs 16 --kv-cache-memory-bytes 26038239232 \
    --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --compilation-config '{"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}'
) >"$RUN/logs/tp2_website_b2k4_server_${TS}.log" 2>&1 &
PID=$!
log "pid=$PID"
OK=0
for i in $(seq 1 480); do
  if ! kill -0 "$PID" 2>/dev/null; then log "PROCESS DIED at i=$i"; break; fi
  if grep -qE "Engine core initialization failed|NPUModelRunner init failed|Failed to execute operator aclnn|Kernel index [0-9]+ not found|FileNotFoundError" \
      "$RUN/logs/tp2_website_b2k4_server_${TS}.log" 2>/dev/null; then
    log "INIT FAILED"; break
  fi
  c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8302/health 2>/dev/null || true)
  [ $((i % 12)) = 0 ] && log "health=$c i=$i"
  [ "$c" = "200" ] && { OK=1; log "READY"; break; }
  sleep 5
done
if [ "$OK" = "1" ]; then
  python3 - <<'PY'
import json, urllib.request
body=json.dumps({"model":"qwen35-a3b-w4a8gdn","prompt":"The capital of France is","max_tokens":8,
                 "temperature":0.0,"top_p":1.0,"top_k":1,"seed":0,"ignore_eos":True}).encode()
req=urllib.request.Request("http://127.0.0.1:8302/v1/completions", data=body,
                           headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r: d=json.load(r)
    print("PROMPT", repr(d["choices"][0].get("text","")))
except Exception as e: print("PROMPT ERROR", repr(e))
PY
else
  log "NOT READY"; tail -30 "$RUN/logs/tp2_website_b2k4_server_${TS}.log" 2>/dev/null
fi
kill "$PID" 2>/dev/null || true; sleep 5
P=$(ps -ef | grep "[v]llm serve /data/whk/models/Qwen3.5-35B-A3B-W4A8-GDN-W8A8" | awk '{print $2}')
[ -n "$P" ] && kill $P 2>/dev/null || true
log "DONE"
echo DONE > "$RUN/logs/tp2_website_b2k4_${TS}.done"
