#!/usr/bin/env bash
# Train MoGA (Memory-object-conditioned Gated-rank Adaptation) + image-encoder LayerNorm
# tuning. This is the main experimental setting reported in the paper.
#
# Required env vars:
#   MOGA_CORRUPTED_ROOT   Pre-rendered {MOSE-C, DAVIS-C, YouTube-VOS-C} frames.
#                         Layout: $MOGA_CORRUPTED_ROOT/<corruption>/{JPEGImages,Annotations}/<video>/
#   MOGA_CKPT_ROOT        Directory containing sam2_hiera_base_plus.pt (Meta's release).
#
# Optional:
#   MOGA_LOG_ROOT         Where checkpoints/logs go (default: ./sam2_logs).
#   MOGA_NUM_GPUS         GPUs per node (default: from YAML, i.e. 8).
#   MOGA_WANDB_PROJECT    Enables wandb logging when set.
#   MOGA_WANDB_RUN        Wandb run name (used with MOGA_WANDB_PROJECT).
set -euo pipefail

: "${MOGA_CORRUPTED_ROOT:?set MOGA_CORRUPTED_ROOT to the pre-rendered corrupted frames root}"
: "${MOGA_CKPT_ROOT:?set MOGA_CKPT_ROOT to a directory containing sam2_hiera_base_plus.pt}"

cd "$(dirname "$0")/.."

CFG=configs/sam2_training/sam2_hiera_b+_moga.yaml

CMD=(python training/train.py -c "$CFG")
if [[ -n "${MOGA_NUM_GPUS:-}" ]]; then
    CMD+=(--num-gpus "$MOGA_NUM_GPUS")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
