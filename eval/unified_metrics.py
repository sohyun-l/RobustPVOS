"""
Unified metric computation for video segmentation evaluation.
Provides consistent aggregation methods across datasets (MVSeg, ACDC, etc.)

Four aggregation methods:
1. Flat: Simple mean across all (object, frame) instances
2. Image-level: Average per image (frame), then average across images
3. VOS-style: Sequence -> Object -> Frame hierarchy (DAVIS standard)
4. Class-wise: Group by semantic class, per-class mean, then average across classes
"""

import numpy as np
from collections import defaultdict


def compute_unified_metrics(results):
    """
    Compute all metric aggregation methods from a list of result dicts.

    Each result dict must have:
      - sequence_key: str (unique sequence identifier)
      - object_key: str/int (object identifier within sequence)
      - frame_key: str/int (frame identifier)
      - j: float (region similarity / IoU)
      - f: float (boundary F-measure)
      - class_key: str or None (semantic class, for class-wise aggregation)
      - group_key: str or None (dataset/weather, for per-group breakdown)

    Returns dict with: flat, image_level, vos_style, class_wise, per_group
    """
    if not results:
        return {}

    # Ensure j_and_f is computed
    for r in results:
        r['j_and_f'] = (r['j'] + r['f']) / 2.0

    metrics = {
        'flat': _compute_flat(results),
        'image_level': _compute_image_level(results),
        'vos_style': _compute_vos_style(results),
    }

    # Class-wise only if class_key is provided
    has_classes = any(r.get('class_key') is not None for r in results)
    if has_classes:
        metrics['class_wise'] = _compute_class_wise(results)

    # Per-group breakdown
    groups = sorted(set(r.get('group_key') for r in results if r.get('group_key') is not None))
    if groups:
        metrics['per_group'] = {}
        for group in groups:
            group_results = [r for r in results if r.get('group_key') == group]
            group_metrics = {
                'flat': _compute_flat(group_results),
                'image_level': _compute_image_level(group_results),
                'vos_style': _compute_vos_style(group_results),
            }
            if has_classes:
                cw = _compute_class_wise(group_results)
                if cw:
                    group_metrics['class_wise'] = cw
            metrics['per_group'][group] = group_metrics

    return metrics


def _compute_flat(results):
    js = [r['j'] for r in results]
    fs = [r['f'] for r in results]
    jfs = [r['j_and_f'] for r in results]
    return {
        'mean_j': float(np.mean(js)),
        'std_j': float(np.std(js)),
        'mean_f': float(np.mean(fs)),
        'std_f': float(np.std(fs)),
        'mean_j_and_f': float(np.mean(jfs)),
        'std_j_and_f': float(np.std(jfs)),
        'count': len(results),
    }


def _compute_image_level(results):
    frame_groups = defaultdict(list)
    for r in results:
        key = (r['sequence_key'], r['frame_key'])
        frame_groups[key].append(r)

    frame_js, frame_fs, frame_jfs = [], [], []
    for group in frame_groups.values():
        frame_js.append(np.mean([r['j'] for r in group]))
        frame_fs.append(np.mean([r['f'] for r in group]))
        frame_jfs.append(np.mean([r['j_and_f'] for r in group]))

    return {
        'mean_j': float(np.mean(frame_js)),
        'std_j': float(np.std(frame_js)),
        'mean_f': float(np.mean(frame_fs)),
        'std_f': float(np.std(frame_fs)),
        'mean_j_and_f': float(np.mean(frame_jfs)),
        'std_j_and_f': float(np.std(frame_jfs)),
        'num_images': len(frame_groups),
    }


def _compute_vos_style(results):
    # Sequence -> Object -> Frames
    seq_obj_frames = defaultdict(lambda: defaultdict(list))
    for r in results:
        seq_obj_frames[r['sequence_key']][r['object_key']].append(r)

    sequence_averages = []
    for seq_key, objects in seq_obj_frames.items():
        object_averages = []
        for obj_key, frames in objects.items():
            obj_j = np.mean([r['j'] for r in frames])
            obj_f = np.mean([r['f'] for r in frames])
            obj_jf = (obj_j + obj_f) / 2.0
            object_averages.append({'j': obj_j, 'f': obj_f, 'j_and_f': obj_jf})

        seq_j = np.mean([o['j'] for o in object_averages])
        seq_f = np.mean([o['f'] for o in object_averages])
        seq_jf = (seq_j + seq_f) / 2.0
        sequence_averages.append({'j': seq_j, 'f': seq_f, 'j_and_f': seq_jf})

    return {
        'mean_j': float(np.mean([s['j'] for s in sequence_averages])),
        'std_j': float(np.std([s['j'] for s in sequence_averages])),
        'mean_f': float(np.mean([s['f'] for s in sequence_averages])),
        'std_f': float(np.std([s['f'] for s in sequence_averages])),
        'mean_j_and_f': float(np.mean([s['j_and_f'] for s in sequence_averages])),
        'std_j_and_f': float(np.std([s['j_and_f'] for s in sequence_averages])),
        'num_sequences': len(sequence_averages),
    }


def _compute_class_wise(results):
    class_groups = defaultdict(list)
    for r in results:
        if r.get('class_key') is not None:
            class_groups[r['class_key']].append(r)

    if not class_groups:
        return {}

    class_js, class_fs, class_jfs = [], [], []
    for group in class_groups.values():
        cj = np.mean([r['j'] for r in group])
        cf = np.mean([r['f'] for r in group])
        class_js.append(cj)
        class_fs.append(cf)
        class_jfs.append((cj + cf) / 2.0)

    return {
        'mean_j': float(np.mean(class_js)),
        'std_j': float(np.std(class_js)),
        'mean_f': float(np.mean(class_fs)),
        'std_f': float(np.std(class_fs)),
        'mean_j_and_f': float(np.mean(class_jfs)),
        'std_j_and_f': float(np.std(class_jfs)),
        'num_classes': len(class_groups),
    }


def format_metrics_table(metrics, title="Evaluation Results"):
    """Print a formatted table of all metric aggregation methods."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)

    method_labels = {
        'flat': 'Flat (all instances)',
        'image_level': 'Image-Level (per-frame avg)',
        'vos_style': 'VOS-Style (seq->obj->frame)',
        'class_wise': 'Class-Wise (per-class avg)',
    }

    for method_name in ['flat', 'image_level', 'vos_style', 'class_wise']:
        if method_name not in metrics:
            continue
        m = metrics[method_name]
        lines.append(f"\n  [{method_labels[method_name]}]")
        lines.append(f"    J (IoU):  {m['mean_j']:.4f} +/- {m['std_j']:.4f}")
        lines.append(f"    F:        {m['mean_f']:.4f} +/- {m['std_f']:.4f}")
        lines.append(f"    J&F:      {m['mean_j_and_f']:.4f} +/- {m['std_j_and_f']:.4f}")
        count_key = [k for k in m if k.startswith('num_') or k == 'count']
        if count_key:
            lines.append(f"    N:        {m[count_key[0]]}")

    if 'per_group' in metrics:
        lines.append(f"\n  [Per-Group Breakdown]")
        for group, gm in sorted(metrics['per_group'].items()):
            lines.append(f"\n    --- {group} ---")
            for method_name in ['flat', 'image_level', 'vos_style', 'class_wise']:
                if method_name not in gm:
                    continue
                m = gm[method_name]
                short = method_labels[method_name].split('(')[0].strip()
                lines.append(
                    f"      {short:15s}  J&F: {m['mean_j_and_f']:.4f}  "
                    f"J: {m['mean_j']:.4f}  F: {m['mean_f']:.4f}"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def get_wandb_metrics(metrics, prefix=""):
    """Convert unified metrics to a flat dict for wandb logging."""
    wandb_dict = {}
    p = f"{prefix}/" if prefix else ""

    for method_name in ['flat', 'image_level', 'vos_style', 'class_wise']:
        if method_name not in metrics:
            continue
        m = metrics[method_name]
        for key, value in m.items():
            wandb_dict[f"{p}{method_name}/{key}"] = value

    if 'per_group' in metrics:
        for group, gm in metrics['per_group'].items():
            for method_name in ['flat', 'image_level', 'vos_style', 'class_wise']:
                if method_name not in gm:
                    continue
                m = gm[method_name]
                for key, value in m.items():
                    wandb_dict[f"{p}{group}/{method_name}/{key}"] = value

    return wandb_dict
