#!/usr/bin/env bash
# Build the GMM v2 op package from a patched vllm-ascend csrc tree.
# usage: build_partial.sh <csrc-dir> [jobs]
# Requires the project's CANN 9.1 environment to be sourced first.
set -euo pipefail
CSRC=${1:?usage: build_partial.sh <csrc-dir> [jobs]}
JOBS=${2:-16}
cd "$CSRC"
TS=$(date -u +%Y%m%dT%H%M%SZ)
[ -d build ] && mv build "build.old-$TS"
rm -rf build_out
bash build.sh --pkg --soc=ascend910b --vendor_name=custom \
  --ops=grouped_matmul_swiglu_quant_v2 -j"$JOBS" -O3
ls -l build_out/
