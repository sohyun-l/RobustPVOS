#!/usr/bin/env bash
# Point-prompt evaluation on ACDC-Video (real-world adverse conditions).
#
# Usage: scripts/eval_acdc.sh {pretrained|moga}
#
# Env vars:
#   MOGA_ACDC_ROOT        Root of ACDC video split (default: ./data/acdc_complete_split_30_object).
#   MOGA_CKPT_ROOT        Directory containing the checkpoint (default: ./checkpoints).
#   MOGA_RESULTS_ROOT     Where JSON results go (default: ./results).
#   MOGA_ACDC_CKPT        Overrides the checkpoint for the chosen model.
set -euo pipefail

: "${MOGA_ACDC_ROOT:=./data/acdc_complete_split_30_object}"
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

CKPT="${MOGA_ACDC_CKPT:-$DEFAULT_CKPT}"

cd "$(dirname "$0")/.."

OUT_DIR="$MOGA_RESULTS_ROOT/acdc_${MODEL}"
mkdir -p "$OUT_DIR"

python eval/evaluate_acdc_unified.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --data_dir "$MOGA_ACDC_ROOT" \
    --weather_conditions fog snow night rain \
    --prompts_file eval/prompts/acdc_video_point_prompts.json \
    --save_results "$OUT_DIR/results.json"
