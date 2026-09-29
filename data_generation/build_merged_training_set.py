#!/usr/bin/env python3
"""Assemble the merged corruption-training tree used by the training configs.

Given per-dataset outputs of generate_corruption_dataset.py, this builds

    <out>/<corruption>/JPEGImages/<prefix>_<video>/   -> symlink to rendered frames
    <out>/<corruption>/Annotations/<prefix>_<video>/  -> symlink to the *clean* dataset masks

with prefixes mose_ / ytvos_ / davis_, which is the layout expected by
`MOGA_CORRUPTED_ROOT` and by training/assets/MERGED_ALL_train_list.txt.

Example (paths as used for the paper):
    python data_generation/build_merged_training_set.py \
        --out /path/to/ALL_corrupted_merged_severe \
        --mose_frames  /path/to/MOSE-C/train/severe        --mose_ann  /path/to/MOSE/train/Annotations \
        --ytvos_frames /path/to/YouTube-VOS-C/train         --ytvos_ann /path/to/Youtube_vos/train/Annotations \
        --davis_frames /path/to/DAVIS-C/severe              --davis_ann /path/to/DAVIS/Annotations/480p
"""
import argparse, os

CORRUPTIONS = ["color_jitter", "fog", "gauss_noise", "ISO_noise", "motion_blur", "rain", "resampling_blur", "snow"]


def link(src, dst):
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


def add_dataset(out, prefix, frames_root, ann_root, frames_have_jpegimages_dir):
    n = 0
    for corr in CORRUPTIONS:
        corr_dir = os.path.join(frames_root, corr)
        if frames_have_jpegimages_dir:
            corr_dir = os.path.join(corr_dir, "JPEGImages")
        if not os.path.isdir(corr_dir):
            print(f"[warn] missing {corr_dir}")
            continue
        for vid in sorted(os.listdir(corr_dir)):
            if not os.path.isdir(os.path.join(corr_dir, vid)):
                continue
            link(os.path.join(corr_dir, vid), os.path.join(out, corr, "JPEGImages", f"{prefix}_{vid}"))
            link(os.path.join(ann_root, vid), os.path.join(out, corr, "Annotations", f"{prefix}_{vid}"))
            n += 1
    print(f"{prefix}: linked {n} (video, corruption) pairs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    for name in ("mose", "ytvos", "davis"):
        p.add_argument(f"--{name}_frames", help=f"root containing <corruption>/... rendered by generate_corruption_dataset.py")
        p.add_argument(f"--{name}_ann", help="clean Annotations root of the source dataset")
    args = p.parse_args()
    # MOSE / YouTube-VOS were rendered with --input_dir <dataset>/train, so frames sit under <corr>/JPEGImages/<vid>;
    # DAVIS was rendered with --input_dir DAVIS/JPEGImages/480p, so frames sit directly under <corr>/<vid>.
    if args.mose_frames:
        add_dataset(args.out, "mose", args.mose_frames, args.mose_ann, True)
    if args.ytvos_frames:
        add_dataset(args.out, "ytvos", args.ytvos_frames, args.ytvos_ann, True)
    if args.davis_frames:
        add_dataset(args.out, "davis", args.davis_frames, args.davis_ann, False)


if __name__ == "__main__":
    main()
