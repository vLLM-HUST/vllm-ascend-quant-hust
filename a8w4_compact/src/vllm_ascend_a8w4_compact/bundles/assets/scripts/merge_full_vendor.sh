#!/usr/bin/env bash
# Merge the rebuilt GMM v2 kernel dir into a FULL in-tree vendor package.
# A partial package shadows the in-tree vendor and breaks GDN custom ops on TP2.
# usage: merge_full_vendor.sh <partial_vendor> <in_tree_vendor> <out_opp_dir>
#   partial_vendor : .../packages/vendors/custom_transformer from build.sh
#   in_tree_vendor : vllm_ascend/_cann_ops_custom/vendors/custom_transformer
#   out_opp_dir    : destination dir that will contain ./custom_transformer
set -euo pipefail
PARTIAL=${1:?}; IN_TREE=${2:?}; OUT=${3:?}
KD=op_impl/ai_core/tbe/kernel/ascend910b/grouped_matmul_swiglu_quant_v2
KCFG=op_impl/ai_core/tbe/kernel/config/ascend910b/grouped_matmul_swiglu_quant_v2.json
[ -f "$IN_TREE/version.info" ] || { echo "not an in-tree vendor: $IN_TREE" >&2; exit 1; }
[ -d "$PARTIAL/$KD" ] || { echo "no rebuilt kernel dir in $PARTIAL" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
cp -a "$IN_TREE" "$OUT/custom_transformer"
rm -rf "$OUT/custom_transformer/$KD"
cp -a "$PARTIAL/$KD" "$OUT/custom_transformer/$KD"
[ -f "$PARTIAL/$KCFG" ] && cp -a "$PARTIAL/$KCFG" "$OUT/custom_transformer/$KCFG"
# liboptiling.so is intentionally the in-tree one: this patch changes no tiling.
echo "merged into $OUT/custom_transformer"
echo "ops: $(ls "$OUT/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b" | wc -l)"
echo "tiling sha256: $(sha256sum "$OUT/custom_transformer/op_impl/ai_core/tbe/op_tiling/liboptiling.so" | cut -d" " -f1)"
