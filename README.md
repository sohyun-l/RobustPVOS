# RobustPVOS: Robust Promptable Video Object Segmentation

**CVPR 2026** · **[Paper](https://arxiv.org/abs/2605.12006)** · **[Project page](https://sohyun-l.github.io/RobustPVOS_project_page/)**

[Sohyun Lee](https://sohyun-l.github.io/)<sup>1</sup>, [Yeho Gwon](https://yehogwon.github.io/)<sup>1</sup>, [Lukas Hoyer](https://lhoyer.github.io/)<sup>2</sup>, [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)<sup>3</sup>, [Christos Sakaridis](https://people.phys.ethz.ch/~csakarid/)<sup>3</sup>, [Suha Kwak](https://suhakwak.github.io/)<sup>1</sup>

<sup>1</sup>POSTECH · <sup>2</sup>Google · <sup>3</sup>ETH Zürich

This repository provides the official implementation of **MoGA (Memory-object-conditioned Gated-rank Adaptation)**, a parameter-efficient adapter that makes [SAM 2](https://github.com/facebookresearch/sam2) robust to corrupted videos, together with the training and evaluation code for the **RobustPVOS benchmark** (ACDC-Video, MVSeg-adv, YouTube-VOS-C).

## Overview

MoGA attaches a shared low-rank adapter `ΔW = BA = Σᵢ bᵢaᵢᵀ` to SAM 2's memory attention and selects its rank-1 components **per tracked object**, conditioned on the object pointers that SAM 2 keeps in its memory bank:

```
αₒ,ₜ = MLP(mₒ,ₜ)                                 # object pointer of object o, stored frame t → gating logits
z̃ₒ,ₜ,ᵢ = σ((αₒ,ₜ,ᵢ + Gᵢ) / τ)                    # Gumbel-Sigmoid, temperature τ
zₒ,ₜ,ᵢ = 𝟙[z̃ₒ,ₜ,ᵢ > 0.5]                         # straight-through discretization
ΔWₒ,ₜ  = Σᵢ zₒ,ₜ,ᵢ · bᵢ aᵢᵀ                       # object-specific adapter
hₒ     = W₀ x + (1/T) Σₜ ΔWₒ,ₜ x                  # averaged over the T stored pointers of object o
```

`{aᵢ, bᵢ}` and the gating MLP are shared across objects; only the gating mask is object-specific. MoGA is applied to the inputs of the self-attention Q/K projections (one adapter + gate shared by Q and K) and the cross-attention Q projection of every memory-attention layer. Together with the LayerNorm parameters of the image encoder, **1.1 M parameters are trained**; the rest of SAM 2 stays frozen.

## Installation

```bash
git clone https://github.com/sohyun-l/RobustPVOS.git && cd RobustPVOS
python -m venv .venv && source .venv/bin/activate     # Python 3.10 recommended
pip install -r requirements.txt
pip install -e .                                       # set MOGA_BUILD_CUDA_EXT=1 to build SAM 2's optional CUDA extension

mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

## Data preparation

All paths are passed through environment variables; each script reports which ones it needs.

| Data | Env var | Layout |
|---|---|---|
| Corruption training set (MOSE-C + YouTube-VOS-C + DAVIS-C) | `MOGA_CORRUPTED_ROOT` | `<corruption>/{JPEGImages,Annotations}/<video>/` for `color_jitter, fog, gauss_noise, ISO_noise, motion_blur, rain, resampling_blur, snow` |
| ACDC-Video | `MOGA_ACDC_ROOT` | `{fog,snow,night,rain}/<video>/split_XXX/` |
| MVSeg-adv | `MOGA_MVSEG_ROOT` | `<sequence>/{visible,label}/` (INO / KAIST / RGBT234 sources) |
| YouTube-VOS 2019 valid (clean) | `MOGA_YTVOS_CLEAN_ROOT` | `valid/{JPEGImages,Annotations,meta.json}` |
| YouTube-VOS-C valid | `MOGA_YTVOS_CORRUPTED_ROOT` | `valid/<corruption>/{JPEGImages,meta.json}` |

The 4,818 training videos (1,297 MOSE + 3,471 YouTube-VOS + 50 DAVIS) are listed in `training/assets/MERGED_ALL_train_list.txt`. The synthetic sets are rendered offline with temporally varying corruption intensities; the renderer and the exact commands are in [`data_generation/`](data_generation/README.md).

Other env vars: `MOGA_CKPT_ROOT` (checkpoints, default `./checkpoints`), `MOGA_LOG_ROOT` (training outputs, default `./sam2_logs`), `MOGA_RESULTS_ROOT` (evaluation outputs, default `./results`).

## Training

```bash
export MOGA_CORRUPTED_ROOT=/path/to/ALL_corrupted_merged_severe
export MOGA_CKPT_ROOT=./checkpoints          # contains sam2_hiera_base_plus.pt
bash scripts/train_moga.sh                   # MOGA_NUM_GPUS overrides the GPU count (paper: 4)
```

`scripts/train_moga.sh` wraps `python training/train.py -c configs/sam2_training/sam2_hiera_b+_moga.yaml`. The trainer (`training/moga_trainer.py`) freezes SAM 2, re-enables the MoGA adapters/gates and the image-encoder LayerNorms, and hands exactly those 1,094,112 parameters to the optimizer.

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 5e-6 (cosine to 5e-7), weight decay 0.1, grad-clip 0.1 |
| Batch | 1 clip / GPU × 4 GPUs, 4 frames per clip, ≤ 3 objects per clip |
| Epochs | 40 |
| Adapter rank R | 128 |
| Gumbel-sigmoid τ | 0.3 (fixed) |
| Loss | SAM 2 loss: focal 20 · dice 1 · IoU 1 · occlusion 1 |
| Peak GPU memory | 22 GB |

## Evaluation

Every benchmark is evaluated with point prompts (3 points per object on the first frame, up to 3 objects per clip) followed by `propagate_in_video()`. The prompts used for the paper are shipped in `eval/prompts/` and are passed by the wrappers, so evaluation is deterministic for a given checkpoint.

```bash
scripts/eval_acdc.sh       {pretrained|moga}
scripts/eval_mvseg.sh      {pretrained|moga}
scripts/eval_youtubevos.sh {pretrained|moga} [corruption ...]   # default: 8 corruptions + clean
```

`pretrained` evaluates Meta's `sam2_hiera_base_plus.pt`. For `moga`, point the wrapper at your checkpoint with `MOGA_ACDC_CKPT` / `MOGA_MVSEG_CKPT` / `MOGA_YTVOS_CKPT` (e.g. `$MOGA_LOG_ROOT/moga/checkpoints/checkpoint.pt`). Results are written to `$MOGA_RESULTS_ROOT/<benchmark>_<model>/results.json`.

Scores are averaged over objects and then over sequences (vos-style). ACDC-Video, whose classes are heavily imbalanced, is averaged over semantic classes instead; `evaluate_acdc_unified.py` / `evaluate_mvseg_unified.py` additionally print flat, image-level and vos-style aggregates.

## Results

Training and evaluating with this code reproduces the numbers reported in Table 2 of the paper.

## Repository structure

```
RobustPVOS/
├── sam2/                       # trimmed SAM 2 package; sam2/modeling/moga.py holds MoGA
│   └── configs/                # eval configs (sam2/) and the training config (sam2_training/)
├── training/                   # SAM 2 training loop; moga_trainer.py is the MoGA trainer
│   └── assets/MERGED_ALL_train_list.txt
├── eval/                       # point-prompt evaluators for ACDC-Video, MVSeg-adv, YouTube-VOS(-C)
│   └── prompts/                # the point prompts used in the paper
├── data_generation/            # corruption renderer + training-set merger
└── scripts/                    # train_moga.sh, eval_{acdc,mvseg,youtubevos}.sh
```

## Implementation notes

- **Object-pointer conditioning.** SAM 2 stacks `memory` as `[maskmem tokens ..., obj_ptr tokens]`; the gating MLP is conditioned on the tail slice `memory[-num_obj_ptr_tokens:]`, i.e. the object pointers of the paper. Objects live on SAM 2's batch dimension, so each object receives its own adapter output, averaged over its stored pointer tokens.
- **Trainable set.** `param_allowlist` restricts the optimizer to the MoGA adapters, gating MLPs and image-encoder LayerNorms (1,094,112 parameters); a trained checkpoint differs from `sam2_hiera_base_plus.pt` only in those tensors.
- **Synthetic data** must be rendered in a NumPy < 2 environment (`imgaug`); see `data_generation/README.md`.

## Citation

```bibtex
@inproceedings{lee2026robustpvos,
  title     = {Robust Promptable Video Object Segmentation},
  author    = {Lee, Sohyun and Gwon, Yeho and Hoyer, Lukas and Schindler, Konrad and Sakaridis, Christos and Kwak, Suha},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [SAM 2](https://github.com/facebookresearch/sam2) (Meta, Apache 2.0). The gated-rank adapter follows GaRA-SAM (Lee et al., NeurIPS 2025).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
