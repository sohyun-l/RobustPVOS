#!/usr/bin/env bash
# Point-prompt evaluation on YouTube-VOS-C (synthetic) plus the clean split.
#
# Usage: scripts/eval_youtubevos.sh {pretrained|moga} [corruption ...]
#
# The default corruption list is: clean color_jitter fog gauss_noise
# ISO_noise motion_blur rain resampling_blur snow  (i.e. the 8 training
# corruptions + clean).
#
# Env vars:
#   MOGA_YTVOS_CLEAN_ROOT        Root of clean YouTube-VOS 2019 valid split
#                                (expects `valid/{JPEGImages,Annotations,meta.json}`).
#   MOGA_YTVOS_CORRUPTED_ROOT    Root of YouTube-VOS-C valid split
#                                (expects `valid/<corruption>/{JPEGImages,meta.json}`).
#   MOGA_YTVOS_PROMPTS           Point-prompt JSON (default: the file shipped in eval/prompts/).
#   MOGA_CKPT_ROOT               Directory containing the checkpoint (default: ./checkpoints).
#   MOGA_RESULTS_ROOT            Where predictions + JSON results go (default: ./results).
#   MOGA_YTVOS_CKPT              Overrides the checkpoint for the chosen model.
set -euo pipefail

: "${MOGA_YTVOS_CLEAN_ROOT:?set MOGA_YTVOS_CLEAN_ROOT to the clean YouTube-VOS valid root}"
: "${MOGA_YTVOS_CORRUPTED_ROOT:?set MOGA_YTVOS_CORRUPTED_ROOT to the YouTube-VOS-C valid root}"
: "${MOGA_CKPT_ROOT:=./checkpoints}"
: "${MOGA_RESULTS_ROOT:=./results}"
: "${MOGA_YTVOS_PROMPTS:=eval/prompts/youtubevos_point_prompts.json}"

MODEL="${1:-moga}"; shift || true
CORRUPTIONS=("$@")
if [[ ${#CORRUPTIONS[@]} -eq 0 ]]; then
    CORRUPTIONS=(clean color_jitter fog gauss_noise ISO_noise motion_blur rain resampling_blur snow)
fi

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
        echo "usage: $0 {pretrained|moga} [corruption ...]" >&2
        exit 2
        ;;
esac

CKPT="${MOGA_YTVOS_CKPT:-$DEFAULT_CKPT}"

cd "$(dirname "$0")/.."

BASE_OUT="$MOGA_RESULTS_ROOT/ytvos_${MODEL}"
GT_ROOT="${MOGA_YTVOS_CLEAN_ROOT%/}/valid/Annotations"
CLEAN_META="${MOGA_YTVOS_CLEAN_ROOT%/}/valid/meta.json"

for CORR in "${CORRUPTIONS[@]}"; do
    PRED_DIR="$BASE_OUT/$CORR/Annotations"
    mkdir -p "$PRED_DIR"

    # 1) Generate predictions.
    python eval/generate_predictions_youtubevos.py \
        --pred_root "$PRED_DIR" \
        --corruption_type "$CORR" \
        --checkpoint_path "$CKPT" \
        --config_path "$CONFIG" \
        --max_num_objects 3 \
        --prompt_file "$MOGA_YTVOS_PROMPTS" \
        --prompt_type point \
        --youtubevos_clean_root "$MOGA_YTVOS_CLEAN_ROOT" \
        --youtubevos_corrupted_root "$MOGA_YTVOS_CORRUPTED_ROOT" \
        --wandb_name "ytvos_${MODEL}_${CORR}_pred"

    # 2) Evaluate predictions.
    if [[ "$CORR" == "clean" ]]; then
        FRAMES_ROOT="${MOGA_YTVOS_CLEAN_ROOT%/}/valid/JPEGImages"
        META_JSON="$CLEAN_META"
    else
        FRAMES_ROOT="${MOGA_YTVOS_CORRUPTED_ROOT%/}/valid/$CORR/JPEGImages"
        META_JSON="${MOGA_YTVOS_CORRUPTED_ROOT%/}/valid/$CORR/meta.json"
    fi

    python eval/evaluate_youtubevos.py \
        --pred_root "$PRED_DIR" \
        --gt_root "$GT_ROOT" \
        --frames_root "$FRAMES_ROOT" \
        --meta_json "$META_JSON" \
        --max_num_objects 3 \
        --wandb_name "ytvos_${MODEL}_${CORR}_eval"
done
