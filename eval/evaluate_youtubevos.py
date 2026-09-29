#!/usr/bin/env python3
"""
YouTube-VOS valid: point-prompt multi-object J&F evaluation with optional qualitative visualization dumps.
Follows the training protocol (max 3 objects per sequence).
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

OBJECT_COLORS = {
    1: (255, 0, 0),      # red
    2: (0, 255, 0),      # green
    3: (0, 0, 255),      # blue
    4: (255, 255, 0),    # yellow
    5: (255, 0, 255),    # magenta
    6: (0, 255, 255),    # cyan
    7: (255, 128, 0),    # orange
    8: (128, 0, 255),    # purple
    9: (255, 192, 203),
    10: (0, 128, 128),
}

def calculate_iou(pred_mask, gt_mask):
    """Compute IoU (Jaccard index)."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """Compute Dice coefficient."""
    intersection = np.logical_and(pred_mask, gt_mask)

    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    if pred_sum + gt_sum == 0:
        return 1.0 if intersection.sum() == 0 else 0.0

    return 2.0 * intersection.sum() / (pred_sum + gt_sum)

def calculate_f_measure(pred_mask, gt_mask, bound_th=0.008):
    """
    Compute boundary F-measure (contour accuracy).
    Following DAVIS evaluation protocol with adaptive boundary threshold

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        bound_th: Boundary thickness threshold (relative to image diagonal, default 0.008)

    Returns:
        F-measure score
    """
    from scipy import ndimage

    # Convert to bool to ensure correct XOR operation (avoid uint8 XOR bool issue)
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    # Calculate adaptive boundary distance threshold (DAVIS standard)
    bound_pix = bound_th * np.linalg.norm(gt_mask.shape)

    pred_boundary = pred_mask ^ ndimage.binary_erosion(pred_mask, structure=np.ones((3,3)))
    gt_boundary = gt_mask ^ ndimage.binary_erosion(gt_mask, structure=np.ones((3,3)))

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    # Distance transforms
    pred_dist = ndimage.distance_transform_edt(~pred_boundary)
    gt_dist = ndimage.distance_transform_edt(~gt_boundary)

    precision_dists = pred_dist[gt_boundary]
    precision = np.mean(precision_dists <= bound_pix)

    recall_dists = gt_dist[pred_boundary]
    recall = np.mean(recall_dists <= bound_pix)

    # F-measure
    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure

def save_qualitative_results(video_name, frame_num, gt_mask, pred_masks, selected_objects,
                            frames_root, save_qual_dir, video_j, video_f, video_jnf, video_dice):
    """Save a colour-coded multi-object qualitative visualization."""
    try:
        video_frames_path = Path(frames_root) / video_name / "JPEGImages"
        if not video_frames_path.exists():
            video_frames_path = Path(frames_root) / video_name

        frame_files = sorted([f for f in video_frames_path.iterdir() if f.suffix in ['.jpg', '.png']])
        if frame_num >= len(frame_files):
            return

        original_img = cv2.imread(str(frame_files[frame_num]))
        if original_img is None:
            return

        h, w = original_img.shape[:2]

        combined_gt_mask = np.zeros((h, w, 3), dtype=np.uint8)
        combined_pred_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for obj_id in selected_objects:
            color = OBJECT_COLORS.get(obj_id, (128, 128, 128))

            gt_obj_mask = (gt_mask == obj_id)
            combined_gt_mask[gt_obj_mask] = color

            if obj_id in pred_masks:
                pred_obj_mask = pred_masks[obj_id]
                combined_pred_mask[pred_obj_mask] = color

        alpha = 0.5

        gt_overlay = cv2.addWeighted(original_img, 1.0 - alpha, combined_gt_mask, alpha, 0)
        pred_overlay = cv2.addWeighted(original_img, 1.0 - alpha, combined_pred_mask, alpha, 0)

        save_dir = Path(save_qual_dir) / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_name = frame_files[frame_num].stem
        cv2.imwrite(str(save_dir / f"{frame_name}_original.jpg"), original_img)
        cv2.imwrite(str(save_dir / f"{frame_name}_gt_mask.png"), combined_gt_mask)
        cv2.imwrite(str(save_dir / f"{frame_name}_pred_mask.png"), combined_pred_mask)
        cv2.imwrite(str(save_dir / f"{frame_name}_gt_overlay.jpg"), gt_overlay)
        cv2.imwrite(str(save_dir / f"{frame_name}_pred_overlay.jpg"), pred_overlay)

        with open(save_dir / f"{frame_name}_metrics.txt", 'w') as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Frame: {frame_name}\n")
            f.write(f"Objects: {selected_objects}\n")
            f.write(f"Video J: {video_j:.4f}\n")
            f.write(f"Video F: {video_f:.4f}\n")
            f.write(f"Video J&F: {video_jnf:.4f}\n")
            f.write(f"Video Dice: {video_dice:.4f}\n")

    except Exception as e:
        print(f"Error while saving qualitative result: {video_name}, frame {frame_num}, {e}")

def evaluate_video_multiobject(video_name, pred_root, gt_root, frames_root=None,
                               max_num_objects=3, save_qual_dir=None, qual_save_interval=10):
    """Evaluate one video (multi-object J/F) and optionally save qualitatives."""

    pred_video_path = Path(pred_root) / video_name
    gt_video_path = Path(gt_root) / video_name

    if not pred_video_path.exists():
        print(f"Prediction path not found: {pred_video_path}")
        return {}

    if not gt_video_path.exists():
        print(f"GT path not found: {gt_video_path}")
        return {}

    gt_frames = sorted(gt_video_path.glob("*.png"))
    if not gt_frames:
        print(f"No GT frames found: {gt_video_path}")
        return {}

    first_gt_frame = gt_frames[0]
    first_gt_mask = np.array(Image.open(first_gt_frame).convert('L'))
    all_object_ids = [val for val in np.unique(first_gt_mask) if val > 0]

    if len(all_object_ids) > max_num_objects:
        selected_objects = sorted(all_object_ids)[:max_num_objects]
    else:
        selected_objects = all_object_ids

    video_results = {
        'video_name': video_name,
        'total_objects': len(all_object_ids),
        'selected_objects': selected_objects,
        'object_metrics': {},
        'frame_results': {}
    }

    for obj_id in selected_objects:
        pred_obj_path = pred_video_path / str(obj_id)

        if not pred_obj_path.exists():
            print(f"Prediction for object {obj_id} not found: {pred_obj_path}")
            video_results['object_metrics'][str(obj_id)] = {
                'J': 0.0, 'F': 0.0, 'J&F': 0.0, 'dice': 0.0
            }
            continue

        frame_ious = []
        frame_dices = []
        frame_f_measures = []

        for frame_idx, gt_frame_path in enumerate(gt_frames):
            frame_name = gt_frame_path.stem
            pred_frame_path = pred_obj_path / f"{frame_name}.png"

            gt_mask = np.array(Image.open(gt_frame_path).convert('L'))
            gt_obj_mask = (gt_mask == obj_id)

            if gt_obj_mask.sum() == 0:
                continue

            if pred_frame_path.exists():
                pred_mask = np.array(Image.open(pred_frame_path).convert('L'))
                pred_obj_mask = (pred_mask > 0)
            else:
                pred_obj_mask = np.zeros_like(gt_obj_mask)

            iou = calculate_iou(pred_obj_mask, gt_obj_mask)
            dice = calculate_dice(pred_obj_mask, gt_obj_mask)
            f_measure = calculate_f_measure(pred_obj_mask, gt_obj_mask)

            frame_ious.append(iou)
            frame_dices.append(dice)
            frame_f_measures.append(f_measure)

            if frame_name not in video_results['frame_results']:
                video_results['frame_results'][frame_name] = {}
            video_results['frame_results'][frame_name][str(obj_id)] = {
                'J': iou,
                'F': f_measure,
                'dice': dice
            }

        if frame_ious:
            avg_j = np.mean(frame_ious)
            avg_f = np.mean(frame_f_measures)
            avg_dice = np.mean(frame_dices)
            video_results['object_metrics'][str(obj_id)] = {
                'J': avg_j,
                'F': avg_f,
                'J&F': (avg_j + avg_f) / 2.0,
                'dice': avg_dice
            }
        else:
            video_results['object_metrics'][str(obj_id)] = {
                'J': 0.0, 'F': 0.0, 'J&F': 0.0, 'dice': 0.0
            }

    if video_results['object_metrics']:
        all_j = [m['J'] for m in video_results['object_metrics'].values()]
        all_f = [m['F'] for m in video_results['object_metrics'].values()]
        all_dice = [m['dice'] for m in video_results['object_metrics'].values()]

        video_results['video_J'] = np.mean(all_j)
        video_results['video_F'] = np.mean(all_f)
        video_results['video_J&F'] = np.mean([m['J&F'] for m in video_results['object_metrics'].values()])
        video_results['video_dice'] = np.mean(all_dice)
    else:
        video_results['video_J'] = 0.0
        video_results['video_F'] = 0.0
        video_results['video_J&F'] = 0.0
        video_results['video_dice'] = 0.0

    if save_qual_dir is not None and frames_root is not None:
        for frame_idx, gt_frame_path in enumerate(gt_frames):
            if frame_idx % qual_save_interval == 0:
                frame_name = gt_frame_path.stem
                gt_mask = np.array(Image.open(gt_frame_path).convert('L'))

                pred_masks = {}
                for obj_id in selected_objects:
                    pred_obj_path = pred_video_path / str(obj_id)
                    pred_frame_path = pred_obj_path / f"{frame_name}.png"

                    if pred_frame_path.exists():
                        pred_mask = np.array(Image.open(pred_frame_path).convert('L'))
                        pred_masks[obj_id] = (pred_mask > 0)
                    else:
                        pred_masks[obj_id] = np.zeros_like(gt_mask, dtype=bool)

                save_qualitative_results(
                    video_name, frame_idx, gt_mask, pred_masks, selected_objects,
                    frames_root, save_qual_dir,
                    video_results['video_J'], video_results['video_F'],
                    video_results['video_J&F'], video_results['video_dice']
                )

    return video_results

def main():
    parser = argparse.ArgumentParser(description='YouTube-VOS multi-object J/F evaluation (optional qualitative saving)')
    parser.add_argument('--pred_root', type=str, required=True,
                       help='root directory of the predictions')
    parser.add_argument('--gt_root', type=str, required=True,
                       help='root directory of the ground-truth annotations')
    parser.add_argument('--frames_root', type=str, default=None,
                       help='root directory of the RGB frames (for qualitative saving)')
    parser.add_argument('--meta_json', type=str, required=True,
                       help='path to meta.json')
    parser.add_argument('--max_num_objects', type=int, default=3,
                       help='max number of objects per video (training protocol)')
    parser.add_argument('--wandb_project', type=str, default="RobustSAM2",
                       help='WandB project name')
    parser.add_argument('--wandb_name', type=str, required=True,
                       help='WandB run name')
    parser.add_argument('--save_qual_dir', type=str, default=None,
                       help='directory for qualitative results')
    parser.add_argument('--qual_save_interval', type=int, default=10,
                       help='qualitative saving interval in frames (default: 10)')

    args = parser.parse_args()

    print(f"Prediction Root: {args.pred_root}")
    print(f"GT Root: {args.gt_root}")
    print(f"Frames Root: {args.frames_root}")
    print(f"Meta JSON: {args.meta_json}")
    print(f"Max Objects: {args.max_num_objects}")
    print(f"Save Qual Dir: {args.save_qual_dir}")
    print(f"Qual Save Interval: {args.qual_save_interval}")

    import wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "pred_root": args.pred_root,
                "gt_root": args.gt_root,
                "frames_root": args.frames_root,
                "max_num_objects": args.max_num_objects,
                "evaluation_type": "multi_object",
                "save_qual": args.save_qual_dir is not None,
                "qual_save_interval": args.qual_save_interval
            }
        )
        print("WandB initialized")
    except Exception as e:
        print(f"WandB init failed: {e}")

    # load meta.json
    with open(args.meta_json, 'r') as f:
        meta_data = json.load(f)

    videos_info = meta_data['videos']
    video_names = list(videos_info.keys())

    print(f"Videos to evaluate: {len(video_names)}")

    # aggregate results
    all_results = []
    all_j = []
    all_f = []
    all_jnf = []
    all_dice = []

    # evaluate each video
    for video_name in tqdm(video_names, desc="Evaluating videos"):
        video_results = evaluate_video_multiobject(
            video_name,
            args.pred_root,
            args.gt_root,
            frames_root=args.frames_root,
            max_num_objects=args.max_num_objects,
            save_qual_dir=args.save_qual_dir,
            qual_save_interval=args.qual_save_interval
        )

        if video_results:
            all_results.append(video_results)
            all_j.append(video_results['video_J'])
            all_f.append(video_results['video_F'])
            all_jnf.append(video_results['video_J&F'])
            all_dice.append(video_results['video_dice'])

    # overall means
    if all_results:
        overall_j = np.mean(all_j)
        overall_f = np.mean(all_f)
        overall_jnf = np.mean(all_jnf)
        overall_dice = np.mean(all_dice)

        print(f"\n=== Overall results ===")
        print(f"Overall J (IoU): {overall_j:.4f}")
        print(f"Overall F: {overall_f:.4f}")
        print(f"Overall J&F: {overall_jnf:.4f}")
        print(f"Overall Dice: {overall_dice:.4f}")
        print(f"Videos evaluated: {len(all_results)}")

        # log overall metrics to WandB
        try:
            wandb.log({
                "overall/J": overall_j,
                "overall/F": overall_f,
                "overall/J&F": overall_jnf,
                "overall/dice": overall_dice,
                "overall/num_videos": len(all_results)
            })

            # per-video metric distribution
            wandb.log({
                "distribution/J_mean": np.mean(all_j),
                "distribution/J_std": np.std(all_j),
                "distribution/F_mean": np.mean(all_f),
                "distribution/F_std": np.std(all_f),
                "distribution/dice_mean": np.mean(all_dice),
                "distribution/dice_std": np.std(all_dice),
            })
        except Exception as e:
            print(f"WandB logging failed: {e}")

        # save results as JSON
        output_file = os.path.join(os.path.dirname(args.pred_root), "evaluation_results.json")

        def _to_builtin(o):
            # numpy scalars / arrays inside per-video results are not JSON serializable
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(output_file, 'w') as f:
            json.dump({
                "overall_metrics": {
                    "J": float(overall_j),
                    "F": float(overall_f),
                    "J&F": float(overall_jnf),
                    "dice": float(overall_dice)
                },
                "video_results": all_results
            }, f, indent=2, default=_to_builtin)
        print(f"\nResults saved to: {output_file}")

    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()
