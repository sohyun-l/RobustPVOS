#!/usr/bin/env bash
# Point-prompt evaluation on MVSeg-adv (multi-modal adverse-condition segmentation).
#
# Usage: scripts/eval_mvseg.sh {pretrained|moga}
#
# Env vars:
#   MOGA_MVSEG_ROOT       Root of MVSeg-adv (default: ./data/MVSeg_Dataset_adverse_only).
#   MOGA_CKPT_ROOT        Directory containing the checkpoint (default: ./checkpoints).
#   MOGA_RESULTS_ROOT     Where JSON results go (default: ./results).
#   MOGA_MVSEG_CKPT       Overrides the checkpoint for the chosen model.
set -euo pipefail

: "${MOGA_MVSEG_ROOT:=./data/MVSeg_Dataset_adverse_only}"
: "${MOGA_CKPT_ROOT:=./checkpoints}"
: "${MOGA_RESULTS_ROOT:=./results}"

MODEL="${1:-moga}"

case "$MODEL" in
    pretrained)
        CONFIG=configs/sam2/sam2_hiera_b+.yaml
        DEFAULT_CKPT="$MOGA_CKPT_ROOT/sam2_hiera_base_plus.pt"
        ;;
    moga)
        CONFIG=configs/sam2/sam2_hiera_b+_moga.yaml
        DEFAULT_CKPT="$MOGA_CKPT_ROOT/moga_sam2_hiera_base_plus.pt"
        ;;
    *)
        echo "usage: $0 {pretrained|moga}" >&2
        exit 2
        ;;
esac

CKPT="${MOGA_MVSEG_CKPT:-$DEFAULT_CKPT}"

cd "$(dirname "$0")/.."

OUT_DIR="$MOGA_RESULTS_ROOT/mvseg_${MODEL}"
mkdir -p "$OUT_DIR"

python eval/evaluate_mvseg_unified.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --annotated_dataset "$MOGA_MVSEG_ROOT" \
    --datasets INO KAIST RGBT234 \
    --prompts_file eval/prompts/mvseg_adv_point_prompts.json \
    --save_results "$OUT_DIR/results.json"
