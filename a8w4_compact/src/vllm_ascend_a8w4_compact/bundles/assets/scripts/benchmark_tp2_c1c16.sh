#!/usr/bin/env bash
# One arm of the website-caliber TP2 W4A8-GDNW8A8 sweep.
# usage: tp2_web_sweep_arm.sh <tag> <stock|patched> <opp_path> <cache_name> [duration]
set -u
RUN=/data/whk/measure-w4a8-tp1
SWE=/data/jxd/swe-prefix-reuse
CLI=$RUN/swe-prefix-reuse-wrapper
WORKLOAD=$SWE/prepared/qwen35.json
META=$RUN/server-metadata-whk-w4a8gdn-tp2.json
PORT=8302
TAG=$1; MODE=$2; OPP=$3; CACHE=$4; DUR=${5:-300}
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RUN/logs/tp2sweep_${TAG}_${TS}.log
mkdir -p "$RUN/logs" "$RUN/results" "$RUN/$CACHE"/{vllm,torch_ext,torchinductor,triton}
exec >>"$LOG" 2>&1
log(){ echo "[arm $TAG $(date -u +%H:%M:%S)] $*"; }
log "START mode=$MODE opp=$OPP cache=$CACHE dur=$DUR"
source "$RUN/cann91-env.sh"
# Fresh Triton/global caches each arm: a stale side table makes rank1 load a
# cached Triton graph whose process-local kernel index is not registered.
rm -rf "$HOME/.triton/cache"
shopt -s nullglob
for item in "$RUN"/cache-stock/triton/*/npu_utils.so; do
  rel=${item#"$RUN/cache-stock/triton/"}; mkdir -p "$HOME/.triton/cache/$(dirname "$rel")"
  [ -f "$HOME/.triton/cache/$rel" ] || cp -a "$item" "$HOME/.triton/cache/$rel"
done
shopt -u nullglob
(
  export ASCEND_RT_VISIBLE_DEVICES=6,7
  export VLLM_CACHE_ROOT=$RUN/$CACHE/vllm
  export TORCH_EXTENSIONS_DIR=$RUN/$CACHE/torch_ext
  export TORCHINDUCTOR_CACHE_DIR=$RUN/$CACHE/torchinductor
  export TRITON_CACHE_DIR=$RUN/$CACHE/triton
  export COMPILE_CUSTOM_KERNELS=1 PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
  export VLLM_DISABLE_COMPILE_CACHE=1
  if [ "$MODE" = patched ]; then
    export VLLM_ASCEND_A8W4_COMPACT=1
    export VLLM_ASCEND_A8W4_COMPACT_OPP=$OPP
  else
    unset VLLM_ASCEND_A8W4_COMPACT VLLM_ASCEND_A8W4_COMPACT_OPP
  fi
  exec /data/jxd/envs/vllm-hust-v1-cann91/bin/vllm serve /data/whk/models/Qwen3.5-35B-A3B-W4A8-GDN-W8A8 \
    --served-model-name qwen35-a3b-w4a8gdn --tensor-parallel-size 2 --max-model-len 262144 \
    --gpu-memory-utilization 0.95 --trust-remote-code --host 127.0.0.1 --port $PORT \
    --quantization ascend --max-num-seqs 16 --kv-cache-memory-bytes 26038239232 \
    --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --compilation-config '{"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}'
) >"$RUN/logs/tp2sweep_${TAG}_${TS}_server.log" 2>&1 &
PID=$!
log "server pid=$PID"
OK=0
for i in $(seq 1 360); do
  if ! kill -0 "$PID" 2>/dev/null; then log "PROCESS DIED i=$i"; break; fi
  if grep -qE "Engine core initialization failed|NPUModelRunner init failed|Failed to execute operator aclnn|Kernel index [0-9]+ not found|FileNotFoundError|HCCL watchdog thread terminated" \
      "$RUN/logs/tp2sweep_${TAG}_${TS}_server.log" 2>/dev/null; then log "INIT FAILED"; break; fi
  c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null || true)
  [ $((i % 24)) = 0 ] && log "health=$c i=$i"
  [ "$c" = "200" ] && { OK=1; log "READY"; break; }
  sleep 5
done
if [ "$OK" != "1" ]; then
  log "ABORT NOT READY"; tail -25 "$RUN/logs/tp2sweep_${TAG}_${TS}_server.log" 2>/dev/null
  kill "$PID" 2>/dev/null || true
  echo DONE > "$RUN/logs/tp2sweep_${TAG}_${TS}.done"
  exit 1
fi
python3 - "$PORT" "$TAG" <<'PY'
import json, urllib.request, sys
port=int(sys.argv[1]); tag=sys.argv[2]
body=json.dumps({"model":"qwen35-a3b-w4a8gdn","prompt":"The capital of France is","max_tokens":8,
                 "temperature":0.0,"top_p":1.0,"top_k":1,"seed":0,"ignore_eos":True}).encode()
req=urllib.request.Request("http://127.0.0.1:%d/v1/completions"%port, data=body,
                           headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r: d=json.load(r)
    t=d["choices"][0].get("text","")
    open("/data/whk/measure-w4a8-tp1/results/prompt_%s.txt"%tag,"w").write(t)
    print("PROMPT", repr(t))
except Exception as e: print("PROMPT ERROR", repr(e))
PY
for C in ${CONCS:-1 16}; do
  for W in "check:60" "${DUR}s:$DUR"; do
    NAME=${W%%:*}; D=$(( ${W##*:} ))
    OUT=$RUN/results/${TAG}-c${C}-${NAME}
    t0=$(date +%s)
    "$CLI" run --workload "$WORKLOAD" --endpoint http://127.0.0.1:$PORT/v1/completions \
      --model qwen35-a3b-w4a8gdn --concurrency "$C" --duration "$D" --chips 2 \
      --server-max-context 262144 --server-metadata "$META" --seed 0 --timeout 3600 --output "$OUT"
    rc=$?
    log "c=$C $NAME rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
    if [ "$rc" != "0" ]; then log "CLIENT FAILED"; fi
  done
done
kill "$PID" 2>/dev/null || true
sleep 5
P=$(ps -ef | grep "[v]llm serve /data/whk/models/Qwen3.5-35B-A3B-W4A8-GDN-W8A8" | awk '{print $2}')
[ -n "$P" ] && kill $P 2>/dev/null || true
log "DONE"
echo DONE > "$RUN/logs/tp2sweep_${TAG}_${TS}.done"
