# Synthetic corruption data

All synthetic data in the paper — the merged training set (MOSE-C + YouTube-VOS-C
+ DAVIS-C, 4,818 videos × 8 corruptions) and the YouTube-VOS-C evaluation set
(507 validation videos × 8 corruptions) — was rendered with
`generate_corruption_dataset.py`. Both are also distributed as files from the
project page; the script is provided for transparency and for rendering new data.

## Environment

The renderer depends on `imgaug`, which does not import under NumPy 2. Use a
separate environment from the training/evaluation one:

```bash
pip install "numpy<2" imgaug==0.4.0 albumentations==2.0.8 opencv-python-headless pillow tqdm scipy
```

## What it does

Eight corruptions (`color_jitter, fog, gauss_noise, ISO_noise, motion_blur, rain,
resampling_blur, snow`) built on albumentations / imgaug primitives. Every scalar
parameter of a corruption (fog density, noise variance, blur kernel, drop size, …)
follows a **temporally varying profile**: `fourier_sampling(T, M_static, C=4, seed)`
returns a length-`T` curve that is a Dirichlet-weighted mixture of `C=4` sine
bases with random frequency, amplitude and phase (DynaAugment-style), so the
corruption intensity drifts smoothly across the video instead of being constant.

Seeding is per video: `video_seed = md5(<video_id>)`, where `<video_id>` is the
path of the video directory **relative to `--input_dir`**. This makes the
Fourier profiles reproducible, but note that particle placement / noise draws
inside albumentations and imgaug use worker-process RNGs and are *not*
bit-reproducible across runs (only `resampling_blur` is fully deterministic).
`--severity` (default 5) scales the parameter ranges; the paper uses 5 throughout.

## Commands used for the paper

```bash
G=data_generation/generate_corruption_dataset.py
ALL="color_jitter fog gauss_noise ISO_noise motion_blur rain resampling_blur snow"

# MOSE train            -> video_id = JPEGImages/<vid>
python $G --input_dir data/MOSE/train              --output_dir data/MOSE-C/train        --corruptions $ALL
# YouTube-VOS train     -> video_id = JPEGImages/<vid>
python $G --input_dir data/Youtube_vos/train       --output_dir data/YouTube-VOS-C/train --corruptions $ALL
# YouTube-VOS valid (= YouTube-VOS-C evaluation set) -> video_id = <vid>
python $G --input_dir data/Youtube_vos/valid/JPEGImages --output_dir data/YouTube-VOS-C/valid --corruptions $ALL
# DAVIS 480p            -> video_id = <vid>
python $G --input_dir data/DAVIS/JPEGImages/480p   --output_dir data/DAVIS-C            --corruptions $ALL
```

Outputs land in `<output_dir>/severe/<corruption>/...` mirroring the input tree.
Then assemble the merged training layout expected by `MOGA_CORRUPTED_ROOT`:

```bash
python data_generation/build_merged_training_set.py --out data/ALL_corrupted_merged_severe \
    --mose_frames  data/MOSE-C/train/severe   --mose_ann  data/MOSE/train/Annotations \
    --ytvos_frames data/YouTube-VOS-C/train/severe --ytvos_ann data/Youtube_vos/train/Annotations \
    --davis_frames data/DAVIS-C/severe        --davis_ann data/DAVIS/Annotations/480p
```

For YouTube-VOS-C evaluation, `scripts/eval_youtubevos.sh` expects
`$MOGA_YTVOS_CORRUPTED_ROOT/valid/<corruption>/{JPEGImages,meta.json}`; copy
`meta.json` from the clean validation split next to each corruption's frames.
