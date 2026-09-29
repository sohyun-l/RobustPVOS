#!/usr/bin/env python3
"""
MVSeg annotated dataset multi-object evaluation script
- Each object is tracked independently
- Shared prompt file system for consistent evaluation across models
- Based on training protocol: max 3 objects per sequence
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Add SAM2 path
sys.path.append('./sam2')
from sam2.build_sam import build_sam2_video_predictor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping logging")


# Color palette for different object IDs (BGR format for OpenCV)
# Using bright, vivid colors: Red, Blue, Green as primary colors
OBJECT_COLORS = {
    1: (0, 0, 255),      # Bright Red
    2: (255, 0, 0),      # Bright Blue
    3: (0, 255, 0),      # Bright Green
    4: (0, 255, 255),    # Yellow
    5: (255, 0, 255),    # Magenta
    6: (255, 255, 0),    # Cyan
    7: (0, 128, 255),    # Orange
    8: (255, 0, 128),    # Purple
}


class MVSegMultiObjectEvaluator:
    def __init__(self,
                 checkpoint_path: str,
                 config_path: str,
                 annotated_dataset_path: str,
                 prompts_file: Optional[str] = None,
                 max_objects: int = 3,
                 image_size: Tuple[int, int] = (1024, 1024),
                 use_wandb: bool = False,
                 wandb_name: str = "mvseg_multiobject_eval",
                 save_qual_dir: Optional[str] = None):

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.annotated_dataset_path = Path(annotated_dataset_path)
        self.prompts_file = prompts_file
        self.max_objects = max_objects
        self.image_size = image_size
        self.use_wandb = use_wandb
        self.wandb_name = wandb_name
        self.save_qual_dir = save_qual_dir

        # Create qual output directory if specified
        if self.save_qual_dir:
            os.makedirs(self.save_qual_dir, exist_ok=True)
            print(f"[OK] Qualitative results will be saved to: {self.save_qual_dir}")

        # Initialize WandB
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="mvseg-multiobject-eval",
                name=self.wandb_name,
                config={
                    "checkpoint": checkpoint_path,
                    "config": config_path,
                    "max_objects": max_objects,
                    "image_size": image_size
                }
            )

        # Load SAM2 model
        self.predictor = self._load_model()
        print("[OK] SAM2 video predictor loaded successfully")

    def _load_model(self):
        """Load SAM2 video predictor"""
        return build_sam2_video_predictor(self.config_path, self.checkpoint_path)

    def get_unique_objects(self, mask_path: str) -> List[int]:
        """Extract all object IDs from mask (excluding background)"""
        try:
            if not os.path.exists(mask_path):
                return []

            mask = np.array(Image.open(mask_path).convert('L'))
            unique_vals = np.unique(mask)
            object_ids = [int(val) for val in unique_vals if val > 0]
            return object_ids[:self.max_objects]  # Limit to max_objects

        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return []

    def generate_object_prompts(self, mask_path: str, object_ids: List[int],
                               num_points: int = 3, use_random: bool = True) -> Dict[int, Dict]:
        """Generate prompts for each object in the mask"""
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            object_prompts = {}

            # Set seed for reproducible results
            if use_random:
                np.random.seed(42)

            for obj_id in object_ids:
                # Extract object mask
                obj_mask = (mask == obj_id)

                if np.sum(obj_mask) == 0:
                    continue

                # Get valid coordinates
                y_coords, x_coords = np.where(obj_mask)

                if len(x_coords) == 0:
                    continue

                if use_random:
                    # Random points within the object mask
                    if len(x_coords) >= num_points:
                        indices = np.random.choice(len(x_coords), size=num_points, replace=False)
                        points = [[x_coords[i], y_coords[i]] for i in indices]
                    else:
                        points = [[x_coords[i], y_coords[i]] for i in range(len(x_coords))]
                else:
                    # Deterministic points (center, top-left region, bottom-right region)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)

                    points = [
                        [min_x + (max_x - min_x) * 0.5, min_y + (max_y - min_y) * 0.5],  # center
                        [min_x + (max_x - min_x) * 0.25, min_y + (max_y - min_y) * 0.25],  # top-left
                        [min_x + (max_x - min_x) * 0.75, min_y + (max_y - min_y) * 0.75],  # bottom-right
                    ]

                object_prompts[obj_id] = {
                    'point_coords': np.array(points, dtype=np.float32),
                    'point_labels': np.array([1] * len(points), dtype=np.int32)
                }

            return object_prompts

        except Exception as e:
            print(f"Error generating prompts for {mask_path}: {e}")
            return {}

    def save_prompts(self, prompts: Dict, filepath: str):
        """Save prompts to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_prompts = {}
        for seq_name, seq_prompts in prompts.items():
            serializable_prompts[seq_name] = {}
            for obj_id, obj_prompts in seq_prompts.items():
                serializable_prompts[seq_name][str(obj_id)] = {
                    'point_coords': obj_prompts['point_coords'].tolist(),
                    'point_labels': obj_prompts['point_labels'].tolist()
                }

        with open(filepath, 'w') as f:
            json.dump(serializable_prompts, f, indent=2)
        print(f"Prompts saved to: {filepath}")

    def load_prompts(self, filepath: str) -> Dict:
        """Load prompts from JSON file"""
        with open(filepath, 'r') as f:
            serializable_prompts = json.load(f)

        # Convert back to numpy arrays
        prompts = {}
        for seq_name, seq_prompts in serializable_prompts.items():
            prompts[seq_name] = {}
            for obj_id_str, obj_prompts in seq_prompts.items():
                obj_id = int(obj_id_str)
                prompts[seq_name][obj_id] = {
                    'point_coords': np.array(obj_prompts['point_coords'], dtype=np.float32),
                    'point_labels': np.array(obj_prompts['point_labels'], dtype=np.int32)
                }

        return prompts

    def generate_or_load_prompts(self, sequences: List[Dict]) -> Dict:
        """Generate prompts or load from file if exists"""
        if self.prompts_file and os.path.exists(self.prompts_file):
            print(f"Loading existing prompts from: {self.prompts_file}")
            return self.load_prompts(self.prompts_file)

        print("Generating new prompts...")
        all_prompts = {}

        for sequence_info in sequences:
            seq_name = sequence_info['full_name']
            label_dir = sequence_info['path'] / 'label'

            # Get first label file (for prompt generation)
            label_files = sorted([f for f in label_dir.iterdir()
                                if f.is_file() and f.suffix.lower() in ['.jpg', '.png']])

            if not label_files:
                continue

            first_label_path = str(label_files[0])
            object_ids = self.get_unique_objects(first_label_path)

            if object_ids:
                object_prompts = self.generate_object_prompts(first_label_path, object_ids)
                if object_prompts:
                    all_prompts[seq_name] = object_prompts
                    print(f"Generated prompts for {seq_name}: {len(object_prompts)} objects")

        # Save prompts for future use
        if self.prompts_file:
            self.save_prompts(all_prompts, self.prompts_file)

        return all_prompts

    def find_annotated_sequences(self, datasets: List[str]) -> List[Dict]:
        """Find all annotated sequences in specified datasets"""
        sequences = []

        for item in self.annotated_dataset_path.iterdir():
            if not item.is_dir():
                continue

            # Parse sequence name to extract dataset
            seq_name = item.name
            dataset = seq_name.split('_')[0]  # Extract dataset name

            if datasets and dataset not in datasets:
                continue

            # Check if it has both visible and label directories
            visible_dir = item / 'visible'
            label_dir = item / 'label'

            if visible_dir.exists() and label_dir.exists():
                sequences.append({
                    'full_name': seq_name,
                    'dataset': dataset,
                    'path': item
                })

        print(f"Found {len(sequences)} annotated sequences")
        return sequences

    def calculate_iou(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calculate IoU between GT and predicted masks"""
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        return intersection / union if union > 0 else 0.0

    def calculate_dice(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        total = gt_mask.sum() + pred_mask.sum()
        return 2 * intersection / total if total > 0 else 0.0

    def calculate_precision_recall(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float]:
        """Calculate precision and recall"""
        tp = np.logical_and(gt_mask, pred_mask).sum()
        fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
        fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall

    def calculate_boundary_f_measure(self, gt_mask: np.ndarray, pred_mask: np.ndarray, bound_th: float = 0.008) -> float:
        """
        Calculate boundary F-measure (F) between GT and predicted masks
        Following DAVIS evaluation protocol

        Args:
            gt_mask: Ground truth binary mask
            pred_mask: Predicted binary mask
            bound_th: Boundary thickness threshold (relative to image diagonal)

        Returns:
            F-measure score
        """
        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            return 1.0
        if gt_mask.sum() == 0 or pred_mask.sum() == 0:
            return 0.0

        # Convert to binary
        gt_mask = gt_mask.astype(np.uint8)
        pred_mask = pred_mask.astype(np.uint8)

        # Calculate boundary distance threshold
        bound_pix = bound_th * np.linalg.norm(gt_mask.shape)

        # Get boundaries using morphological operations
        from scipy.ndimage import binary_erosion

        # GT boundary
        gt_boundary = gt_mask - binary_erosion(gt_mask).astype(np.uint8)

        # Pred boundary
        pred_boundary = pred_mask - binary_erosion(pred_mask).astype(np.uint8)

        if gt_boundary.sum() == 0 or pred_boundary.sum() == 0:
            return 0.0

        # Calculate distance transforms
        from scipy.ndimage import distance_transform_edt

        gt_dists = distance_transform_edt(np.logical_not(gt_boundary))
        pred_dists = distance_transform_edt(np.logical_not(pred_boundary))

        # Get boundary pixels
        gt_boundary_pixels = np.where(gt_boundary)
        pred_boundary_pixels = np.where(pred_boundary)

        # Precision: for each predicted boundary pixel, check if close to GT boundary
        pred_to_gt_dists = gt_dists[pred_boundary_pixels]
        precision = (pred_to_gt_dists <= bound_pix).sum() / len(pred_to_gt_dists) if len(pred_to_gt_dists) > 0 else 0.0

        # Recall: for each GT boundary pixel, check if close to predicted boundary
        gt_to_pred_dists = pred_dists[gt_boundary_pixels]
        recall = (gt_to_pred_dists <= bound_pix).sum() / len(gt_to_pred_dists) if len(gt_to_pred_dists) > 0 else 0.0

        # F-measure
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def process_sequence_multiobject(self, sequence_info: Dict, prompts: Dict) -> List[Dict]:
        """Process a single sequence with multi-object tracking"""
        seq_name = sequence_info['full_name']
        print(f"\nProcessing {seq_name}")

        if seq_name not in prompts:
            print(f"  No prompts found for {seq_name}")
            return []

        # Clear existing qualitative results for this sequence (to avoid duplicates from multiple runs)
        if self.save_qual_dir:
            qual_seq_dir = os.path.join(self.save_qual_dir, seq_name)
            if os.path.exists(qual_seq_dir):
                print(f"  Clearing existing qualitative results for {seq_name}")
                shutil.rmtree(qual_seq_dir)

        # Get all frame pairs
        visible_dir = sequence_info['path'] / 'visible'
        label_dir = sequence_info['path'] / 'label'

        visible_files = sorted([f.name for f in visible_dir.iterdir()
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.png']])
        label_files = sorted([f.name for f in label_dir.iterdir()
                            if f.is_file() and f.suffix.lower() in ['.jpg', '.png']])

        # Match visible and label files by name pattern, but keep all visible frames for propagation
        label_for_idx = {}
        labeled_indices = []
        for idx, visible_file in enumerate(visible_files):
            base_name = visible_file.rsplit('.', 1)[0]
            if base_name.endswith('v'):
                number_part = base_name[:-1]
                expected_label = number_part + 'vl.png'
                if expected_label in label_files:
                    label_for_idx[idx] = expected_label
                    labeled_indices.append(idx)

        if not visible_files:
            print(f"  No visible frames found")
            return []

        print(f"  Found {len(visible_files)} visible frames ({len(labeled_indices)} labeled)")

        # Create temporary directory for video frames
        temp_dir = tempfile.mkdtemp(prefix='mvseg_multiobject_')
        video_dir = os.path.join(temp_dir, 'video_frames')
        os.makedirs(video_dir, exist_ok=True)

        try:
            # Copy ALL visible frames to temporary directory for propagation
            # NOTE: These are already CORRUPTED images (corruption type encoded in sequence name like ".-2", ".-3" etc.)
            frame_paths = []
            for i, visible_file in enumerate(visible_files):
                visible_path = visible_dir / visible_file
                image = cv2.imread(str(visible_path))
                frame_path = os.path.join(video_dir, f'{i:05d}.jpg')
                cv2.imwrite(frame_path, image)
                frame_paths.append(frame_path)

            # Initialize SAM2 predictor
            inference_state = self.predictor.init_state(video_path=video_dir)

            # Process each object independently
            object_results = {}
            sequence_prompts = prompts[seq_name]

            for obj_id, obj_prompts in sequence_prompts.items():
                print(f"  Processing object {obj_id}")

                # Add object to tracking
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=obj_prompts['point_coords'],
                    labels=obj_prompts['point_labels'],
                )

                # Propagate through video
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    for obj_id_out, mask_logit in zip(out_obj_ids, out_mask_logits):
                        if obj_id_out == obj_id:
                            mask = (mask_logit > 0.0).cpu().numpy()[0]
                            video_segments[out_frame_idx] = mask

                object_results[obj_id] = video_segments

            # Evaluate results only on labeled frames
            results = []

            # VOS-style object tracking: accumulate metrics per object
            from collections import defaultdict
            object_j_sums = defaultdict(float)
            object_f_sums = defaultdict(float)
            object_dice_sums = defaultdict(float)
            object_frame_counts = defaultdict(int)

            for i in sorted(label_for_idx.keys()):
                visible_file = visible_files[i]
                label_file = label_for_idx[i]

                # Load GT mask
                gt_path = label_dir / label_file
                gt_mask = np.array(Image.open(gt_path).convert('L'))

                # Load corrupted visible image for qualitative saving
                # NOTE: visible_dir already contains corrupted images (corruption type is encoded in sequence name)
                visible_path = visible_dir / visible_file
                visible_img = cv2.imread(str(visible_path)) if self.save_qual_dir else None

                # Create combined masks for all objects (for visualization)
                if self.save_qual_dir and visible_img is not None:
                    h, w = gt_mask.shape
                    combined_pred_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    combined_gt_mask = np.zeros((h, w, 3), dtype=np.uint8)

                # Calculate mean IoU for this frame (for quality determination)
                frame_ious = []

                # Process each object
                for obj_id in sequence_prompts.keys():
                    if i not in object_results[obj_id]:
                        continue

                    # Extract GT mask for this object
                    gt_obj_mask = (gt_mask == obj_id)
                    if np.sum(gt_obj_mask) == 0:
                        continue

                    # Get predicted mask
                    pred_mask = object_results[obj_id][i]

                    # Calculate metrics
                    iou = self.calculate_iou(gt_obj_mask, pred_mask)  # J (region similarity)
                    dice = self.calculate_dice(gt_obj_mask, pred_mask)
                    precision, recall = self.calculate_precision_recall(gt_obj_mask, pred_mask)
                    f_measure = self.calculate_boundary_f_measure(gt_obj_mask, pred_mask)  # F (boundary accuracy)
                    j_and_f = (iou + f_measure) / 2.0  # J&F

                    result = {
                        'sequence_name': seq_name,
                        'dataset': sequence_info['dataset'],
                        'frame_idx': i,
                        'visible_file': visible_file,
                        'label_file': label_file,
                        'object_id': obj_id,
                        'iou': iou,
                        'j': iou,  # J = IoU
                        'f': f_measure,  # F = boundary F-measure
                        'j_and_f': j_and_f,  # J&F = (J + F) / 2
                        'dice': dice,
                        'precision': precision,
                        'recall': recall,
                        'gt_area': int(np.sum(gt_obj_mask)),
                        'pred_area': int(np.sum(pred_mask))
                    }
                    results.append(result)
                    frame_ious.append(iou)

                    # VOS-style: accumulate per-object metrics
                    object_j_sums[obj_id] += iou
                    object_f_sums[obj_id] += f_measure
                    object_dice_sums[obj_id] += dice
                    object_frame_counts[obj_id] += 1

                    # Add this object to combined masks with its color
                    if self.save_qual_dir and visible_img is not None:
                        color = OBJECT_COLORS.get(obj_id, OBJECT_COLORS.get((obj_id % 8) + 1, (0, 0, 255)))  # Cycle through colors
                        combined_pred_mask[pred_mask > 0] = color
                        combined_gt_mask[gt_obj_mask] = color

                # Save qualitative results for this frame (all objects combined)
                if self.save_qual_dir and visible_img is not None and len(frame_ious) > 0:
                    try:
                        mean_frame_iou = np.mean(frame_ious)
                        status = "good" if mean_frame_iou > 0.5 else "poor"
                        qual_pred_dir = os.path.join(self.save_qual_dir, seq_name, 'predictions')
                        os.makedirs(qual_pred_dir, exist_ok=True)

                        base_name = f'{status}_frame{i:05d}_iou{mean_frame_iou:.3f}'

                        # Save original image
                        img_path = os.path.join(qual_pred_dir, f'{base_name}_image.png')
                        cv2.imwrite(img_path, visible_img)

                        # Save combined prediction mask (all objects with different colors)
                        pred_path = os.path.join(qual_pred_dir, f'{base_name}_pred.png')
                        cv2.imwrite(pred_path, combined_pred_mask)

                        # Save combined GT mask (all objects with different colors)
                        gt_path = os.path.join(qual_pred_dir, f'{base_name}_gt.png')
                        cv2.imwrite(gt_path, combined_gt_mask)

                        # Create and save overlay visualizations (image + transparent mask + prompts)
                        # For predictions - ONLY overlay object regions (exclude background/black)
                        alpha = 0.5
                        pred_overlay = visible_img.copy()
                        # Apply color ONLY where mask has actual objects (non-zero values)
                        # This ensures we don't overlay black background over the corrupted image
                        mask_region = np.any(combined_pred_mask > 0, axis=2)
                        if np.any(mask_region):  # Only blend if there are objects
                            pred_overlay[mask_region] = (
                                visible_img[mask_region] * (1 - alpha) +
                                combined_pred_mask[mask_region] * alpha
                            ).astype(np.uint8)

                        # Add prompt points on first frame
                        if i == min(label_for_idx.keys()):
                            for obj_id in sequence_prompts.keys():
                                if obj_id in sequence_prompts:
                                    color = OBJECT_COLORS.get(obj_id, OBJECT_COLORS.get((obj_id % 8) + 1, (0, 0, 255)))
                                    points = sequence_prompts[obj_id]['point_coords']
                                    for point in points:
                                        x, y = int(point[0]), int(point[1])
                                        cv2.circle(pred_overlay, (x, y), 8, color, -1)
                                        cv2.circle(pred_overlay, (x, y), 10, (255, 255, 255), 2)

                        pred_overlay_path = os.path.join(qual_pred_dir, f'{base_name}_pred_overlay.png')
                        cv2.imwrite(pred_overlay_path, pred_overlay)

                        # For GT - ONLY overlay object regions (exclude background/black)
                        gt_overlay = visible_img.copy()
                        # Apply color ONLY where mask has actual objects (non-zero values)
                        # This ensures we don't overlay black background over the corrupted image
                        mask_region = np.any(combined_gt_mask > 0, axis=2)
                        if np.any(mask_region):  # Only blend if there are objects
                            gt_overlay[mask_region] = (
                                visible_img[mask_region] * (1 - alpha) +
                                combined_gt_mask[mask_region] * alpha
                            ).astype(np.uint8)

                        # Add prompt points on first frame
                        if i == min(label_for_idx.keys()):
                            for obj_id in sequence_prompts.keys():
                                if obj_id in sequence_prompts:
                                    color = OBJECT_COLORS.get(obj_id, OBJECT_COLORS.get((obj_id % 8) + 1, (0, 0, 255)))
                                    points = sequence_prompts[obj_id]['point_coords']
                                    for point in points:
                                        x, y = int(point[0]), int(point[1])
                                        cv2.circle(gt_overlay, (x, y), 8, color, -1)
                                        cv2.circle(gt_overlay, (x, y), 10, (255, 255, 255), 2)

                        gt_overlay_path = os.path.join(qual_pred_dir, f'{base_name}_gt_overlay.png')
                        cv2.imwrite(gt_overlay_path, gt_overlay)

                        # Save additional visualizations for first frame
                        if i == min(label_for_idx.keys()):
                            # 1. Image with only prompts (no prediction)
                            img_with_prompts = visible_img.copy()
                            for obj_id in sequence_prompts.keys():
                                if obj_id in sequence_prompts:
                                    color = OBJECT_COLORS.get(obj_id, OBJECT_COLORS.get((obj_id % 8) + 1, (0, 0, 255)))
                                    points = sequence_prompts[obj_id]['point_coords']
                                    for point in points:
                                        x, y = int(point[0]), int(point[1])
                                        cv2.circle(img_with_prompts, (x, y), 8, color, -1)
                                        cv2.circle(img_with_prompts, (x, y), 10, (255, 255, 255), 2)

                            prompt_only_path = os.path.join(qual_pred_dir, f'{base_name}_prompts_only.png')
                            cv2.imwrite(prompt_only_path, img_with_prompts)

                            # 2. Combined view: Image + Prediction + Prompts (horizontal concatenation)
                            # Resize all to same height if needed
                            h, w = visible_img.shape[:2]

                            # Create combined image (Image | Pred_overlay with prompts)
                            combined_horizontal = np.hstack([img_with_prompts, pred_overlay])
                            combined_path = os.path.join(qual_pred_dir, f'{base_name}_combined_img_pred_prompt.png')
                            cv2.imwrite(combined_path, combined_horizontal)

                            # 3. Full combined view: Image | Pred | GT (all with prompts on first frame)
                            full_combined = np.hstack([img_with_prompts, pred_overlay, gt_overlay])
                            full_combined_path = os.path.join(qual_pred_dir, f'{base_name}_full_combined.png')
                            cv2.imwrite(full_combined_path, full_combined)

                    except Exception as e:
                        print(f"      Failed to save qualitative results: {e}")

            # VOS-style: Calculate per-object averages and sequence average
            sequence_object_metrics = {}
            for obj_id in sequence_prompts.keys():
                if object_frame_counts[obj_id] > 0:
                    obj_j = object_j_sums[obj_id] / object_frame_counts[obj_id]
                    obj_f = object_f_sums[obj_id] / object_frame_counts[obj_id]
                    obj_dice = object_dice_sums[obj_id] / object_frame_counts[obj_id]
                    obj_jnf = (obj_j + obj_f) / 2.0

                    sequence_object_metrics[f'obj_{obj_id}_J'] = obj_j
                    sequence_object_metrics[f'obj_{obj_id}_F'] = obj_f
                    sequence_object_metrics[f'obj_{obj_id}_J&F'] = obj_jnf
                    sequence_object_metrics[f'obj_{obj_id}_dice'] = obj_dice
                    sequence_object_metrics[f'obj_{obj_id}_frames'] = object_frame_counts[obj_id]

                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            f"sequence_objects/{seq_name}_obj{obj_id}_J": obj_j,
                            f"sequence_objects/{seq_name}_obj{obj_id}_F": obj_f,
                            f"sequence_objects/{seq_name}_obj{obj_id}_J&F": obj_jnf,
                            f"sequence_objects/{seq_name}_obj{obj_id}_dice": obj_dice,
                            f"sequence_objects/{seq_name}_obj{obj_id}_frames": object_frame_counts[obj_id],
                        })

            # Sequence average (all objects in this sequence)
            if sequence_object_metrics:
                seq_j = np.mean([v for k, v in sequence_object_metrics.items() if k.endswith('_J')])
                seq_f = np.mean([v for k, v in sequence_object_metrics.items() if k.endswith('_F')])
                seq_jnf = np.mean([v for k, v in sequence_object_metrics.items() if k.endswith('_J&F')])
                seq_dice = np.mean([v for k, v in sequence_object_metrics.items() if k.endswith('_dice')])

                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        f"sequence/{seq_name}_J": seq_j,
                        f"sequence/{seq_name}_F": seq_f,
                        f"sequence/{seq_name}_J&F": seq_jnf,
                        f"sequence/{seq_name}_dice": seq_dice,
                        f"sequence/{seq_name}_num_objects": len([k for k in sequence_object_metrics.keys() if k.endswith('_J')]),
                    })

                print(f"  Sequence {seq_name} VOS-style: J={seq_j:.3f}, F={seq_f:.3f}, J&F={seq_jnf:.3f}, Dice={seq_dice:.3f}")

            print(f"  Evaluated {len(results)} object instances")
            return results

        except Exception as e:
            print(f"  Error processing {seq_name}: {e}")
            return []

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def evaluate_dataset(self, datasets: List[str] = None, max_sequences: Optional[int] = None) -> Dict:
        """Evaluate annotated dataset"""
        print(f"Starting MVSeg multi-object annotated dataset evaluation")
        print(f"Annotated dataset: {self.annotated_dataset_path}")
        print(f"Max objects per sequence: {self.max_objects}")

        # Find sequences
        sequences = self.find_annotated_sequences(datasets or [])
        if max_sequences:
            sequences = sequences[:max_sequences]

        print(f"Testing {len(sequences)} annotated sequences")

        # Generate or load prompts
        prompts = self.generate_or_load_prompts(sequences)

        # Process all sequences
        all_results = []
        for sequence_info in sequences:
            seq_results = self.process_sequence_multiobject(sequence_info, prompts)
            all_results.extend(seq_results)

        if not all_results:
            print("No evaluation results obtained")
            return {}

        # Calculate summary statistics
        summary = self._calculate_summary_stats(all_results)

        # Log to WandB
        if self.use_wandb and WANDB_AVAILABLE:
            self._log_to_wandb(summary, all_results, prompts)

        return {
            'summary': summary,
            'detailed_results': all_results,
            'total_sequences': len(sequences),
            'total_objects': len(all_results)
        }

    def _calculate_summary_stats(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics with both object-level and image-level metrics"""
        if not results:
            return {}

        # Object-level stats (existing - per object)
        ious = [r['iou'] for r in results]
        js = [r['j'] for r in results]  # J = IoU
        fs = [r['f'] for r in results]  # F = boundary F-measure
        j_and_fs = [r['j_and_f'] for r in results]  # J&F
        dices = [r['dice'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]

        # Image-level stats (average IoU per image, then across images)
        # Group results by sequence and frame
        image_level_metrics = {}
        for r in results:
            seq_frame_key = f"{r['sequence_name']}_{r.get('frame_idx', 0)}"
            if seq_frame_key not in image_level_metrics:
                image_level_metrics[seq_frame_key] = {
                    'ious': [],
                    'js': [],
                    'fs': [],
                    'j_and_fs': [],
                    'dices': [],
                    'precisions': [],
                    'recalls': []
                }
            image_level_metrics[seq_frame_key]['ious'].append(r['iou'])
            image_level_metrics[seq_frame_key]['js'].append(r['j'])
            image_level_metrics[seq_frame_key]['fs'].append(r['f'])
            image_level_metrics[seq_frame_key]['j_and_fs'].append(r['j_and_f'])
            image_level_metrics[seq_frame_key]['dices'].append(r['dice'])
            image_level_metrics[seq_frame_key]['precisions'].append(r['precision'])
            image_level_metrics[seq_frame_key]['recalls'].append(r['recall'])

        # Calculate average metrics per image
        image_avg_ious = [np.mean(metrics['ious']) for metrics in image_level_metrics.values()]
        image_avg_js = [np.mean(metrics['js']) for metrics in image_level_metrics.values()]
        image_avg_fs = [np.mean(metrics['fs']) for metrics in image_level_metrics.values()]
        image_avg_j_and_fs = [np.mean(metrics['j_and_fs']) for metrics in image_level_metrics.values()]
        image_avg_dices = [np.mean(metrics['dices']) for metrics in image_level_metrics.values()]
        image_avg_precisions = [np.mean(metrics['precisions']) for metrics in image_level_metrics.values()]
        image_avg_recalls = [np.mean(metrics['recalls']) for metrics in image_level_metrics.values()]

        # VOS-style sequence-wise metrics (sequence -> object -> frames)
        # Group by sequence and object
        from collections import defaultdict
        sequence_object_metrics = defaultdict(lambda: defaultdict(lambda: {
            'js': [], 'fs': [], 'dices': [], 'precisions': [], 'recalls': []
        }))

        for r in results:
            seq_name = r['sequence_name']
            obj_id = r['object_id']
            sequence_object_metrics[seq_name][obj_id]['js'].append(r['j'])
            sequence_object_metrics[seq_name][obj_id]['fs'].append(r['f'])
            sequence_object_metrics[seq_name][obj_id]['dices'].append(r['dice'])
            sequence_object_metrics[seq_name][obj_id]['precisions'].append(r['precision'])
            sequence_object_metrics[seq_name][obj_id]['recalls'].append(r['recall'])

        # Calculate per-object averages, then per-sequence averages, then overall average
        sequence_averages = []
        for seq_name, objects in sequence_object_metrics.items():
            object_averages = []
            for obj_id, metrics in objects.items():
                obj_j = np.mean(metrics['js'])
                obj_f = np.mean(metrics['fs'])
                obj_jnf = (obj_j + obj_f) / 2.0
                object_averages.append({
                    'j': obj_j,
                    'f': obj_f,
                    'j_and_f': obj_jnf,
                    'dice': np.mean(metrics['dices']),
                    'precision': np.mean(metrics['precisions']),
                    'recall': np.mean(metrics['recalls'])
                })

            # Sequence average = average of all objects in this sequence
            if object_averages:
                seq_avg = {
                    'j': np.mean([o['j'] for o in object_averages]),
                    'f': np.mean([o['f'] for o in object_averages]),
                    'j_and_f': np.mean([o['j_and_f'] for o in object_averages]),
                    'dice': np.mean([o['dice'] for o in object_averages]),
                    'precision': np.mean([o['precision'] for o in object_averages]),
                    'recall': np.mean([o['recall'] for o in object_averages])
                }
                sequence_averages.append(seq_avg)

        # VOS-style overall metrics = average of all sequence averages
        vos_style_metrics = {}
        if sequence_averages:
            vos_style_metrics = {
                'mean_j': np.mean([s['j'] for s in sequence_averages]),
                'std_j': np.std([s['j'] for s in sequence_averages]),
                'mean_f': np.mean([s['f'] for s in sequence_averages]),
                'std_f': np.std([s['f'] for s in sequence_averages]),
                'mean_j_and_f': np.mean([s['j_and_f'] for s in sequence_averages]),
                'std_j_and_f': np.std([s['j_and_f'] for s in sequence_averages]),
                'mean_dice': np.mean([s['dice'] for s in sequence_averages]),
                'std_dice': np.std([s['dice'] for s in sequence_averages]),
                'mean_precision': np.mean([s['precision'] for s in sequence_averages]),
                'std_precision': np.std([s['precision'] for s in sequence_averages]),
                'mean_recall': np.mean([s['recall'] for s in sequence_averages]),
                'std_recall': np.std([s['recall'] for s in sequence_averages]),
                'total_sequences': len(sequence_averages)
            }

        summary = {
            'overall': {
                # Object-level metrics (per object)
                'object_level': {
                    'mean_iou': np.mean(ious),
                    'std_iou': np.std(ious),
                    'mean_j': np.mean(js),
                    'std_j': np.std(js),
                    'mean_f': np.mean(fs),
                    'std_f': np.std(fs),
                    'mean_j_and_f': np.mean(j_and_fs),
                    'std_j_and_f': np.std(j_and_fs),
                    'mean_dice': np.mean(dices),
                    'std_dice': np.std(dices),
                    'mean_precision': np.mean(precisions),
                    'std_precision': np.std(precisions),
                    'mean_recall': np.mean(recalls),
                    'std_recall': np.std(recalls),
                    'total_objects': len(results)
                },
                # Image-level metrics (average per image)
                'image_level': {
                    'mean_iou': np.mean(image_avg_ious),
                    'std_iou': np.std(image_avg_ious),
                    'mean_j': np.mean(image_avg_js),
                    'std_j': np.std(image_avg_js),
                    'mean_f': np.mean(image_avg_fs),
                    'std_f': np.std(image_avg_fs),
                    'mean_j_and_f': np.mean(image_avg_j_and_fs),
                    'std_j_and_f': np.std(image_avg_j_and_fs),
                    'mean_dice': np.mean(image_avg_dices),
                    'std_dice': np.std(image_avg_dices),
                    'mean_precision': np.mean(image_avg_precisions),
                    'std_precision': np.std(image_avg_precisions),
                    'mean_recall': np.mean(image_avg_recalls),
                    'std_recall': np.std(image_avg_recalls),
                    'total_images': len(image_level_metrics)
                },
                # VOS-style metrics (sequence -> object -> frames)
                'vos_style': vos_style_metrics,
                # Flat keys kept alongside the nested results
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'mean_j': np.mean(js),
                'std_j': np.std(js),
                'mean_f': np.mean(fs),
                'std_f': np.std(fs),
                'mean_j_and_f': np.mean(j_and_fs),
                'std_j_and_f': np.std(j_and_fs),
                'mean_dice': np.mean(dices),
                'std_dice': np.std(dices),
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'mean_recall': np.mean(recalls),
                'std_recall': np.std(recalls),
                'total_objects': len(results)
            }
        }

        # Per-dataset stats with both object-level and image-level
        datasets = list(set(r['dataset'] for r in results))
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            dataset_ious = [r['iou'] for r in dataset_results]
            dataset_js = [r['j'] for r in dataset_results]
            dataset_fs = [r['f'] for r in dataset_results]
            dataset_j_and_fs = [r['j_and_f'] for r in dataset_results]
            dataset_dices = [r['dice'] for r in dataset_results]
            dataset_precisions = [r['precision'] for r in dataset_results]
            dataset_recalls = [r['recall'] for r in dataset_results]

            # Dataset image-level metrics
            dataset_image_metrics = {}
            for r in dataset_results:
                seq_frame_key = f"{r['sequence_name']}_{r.get('frame_idx', 0)}"
                if seq_frame_key not in dataset_image_metrics:
                    dataset_image_metrics[seq_frame_key] = {
                        'ious': [],
                        'js': [],
                        'fs': [],
                        'j_and_fs': [],
                        'dices': [],
                        'precisions': [],
                        'recalls': []
                    }
                dataset_image_metrics[seq_frame_key]['ious'].append(r['iou'])
                dataset_image_metrics[seq_frame_key]['js'].append(r['j'])
                dataset_image_metrics[seq_frame_key]['fs'].append(r['f'])
                dataset_image_metrics[seq_frame_key]['j_and_fs'].append(r['j_and_f'])
                dataset_image_metrics[seq_frame_key]['dices'].append(r['dice'])
                dataset_image_metrics[seq_frame_key]['precisions'].append(r['precision'])
                dataset_image_metrics[seq_frame_key]['recalls'].append(r['recall'])

            dataset_image_avg_ious = [np.mean(metrics['ious']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_js = [np.mean(metrics['js']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_fs = [np.mean(metrics['fs']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_j_and_fs = [np.mean(metrics['j_and_fs']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_dices = [np.mean(metrics['dices']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_precisions = [np.mean(metrics['precisions']) for metrics in dataset_image_metrics.values()]
            dataset_image_avg_recalls = [np.mean(metrics['recalls']) for metrics in dataset_image_metrics.values()]

            summary[dataset] = {
                # Object-level metrics
                'object_level': {
                    'mean_iou': np.mean(dataset_ious),
                    'std_iou': np.std(dataset_ious),
                    'mean_j': np.mean(dataset_js),
                    'std_j': np.std(dataset_js),
                    'mean_f': np.mean(dataset_fs),
                    'std_f': np.std(dataset_fs),
                    'mean_j_and_f': np.mean(dataset_j_and_fs),
                    'std_j_and_f': np.std(dataset_j_and_fs),
                    'mean_dice': np.mean(dataset_dices),
                    'std_dice': np.std(dataset_dices),
                    'mean_precision': np.mean(dataset_precisions),
                    'std_precision': np.std(dataset_precisions),
                    'mean_recall': np.mean(dataset_recalls),
                    'std_recall': np.std(dataset_recalls),
                    'total_objects': len(dataset_results)
                },
                # Image-level metrics
                'image_level': {
                    'mean_iou': np.mean(dataset_image_avg_ious),
                    'std_iou': np.std(dataset_image_avg_ious),
                    'mean_j': np.mean(dataset_image_avg_js),
                    'std_j': np.std(dataset_image_avg_js),
                    'mean_f': np.mean(dataset_image_avg_fs),
                    'std_f': np.std(dataset_image_avg_fs),
                    'mean_j_and_f': np.mean(dataset_image_avg_j_and_fs),
                    'std_j_and_f': np.std(dataset_image_avg_j_and_fs),
                    'mean_dice': np.mean(dataset_image_avg_dices),
                    'std_dice': np.std(dataset_image_avg_dices),
                    'mean_precision': np.mean(dataset_image_avg_precisions),
                    'std_precision': np.std(dataset_image_avg_precisions),
                    'mean_recall': np.mean(dataset_image_avg_recalls),
                    'std_recall': np.std(dataset_image_avg_recalls),
                    'total_images': len(dataset_image_metrics)
                },
                # Flat keys kept alongside the nested results
                'mean_iou': np.mean(dataset_ious),
                'std_iou': np.std(dataset_ious),
                'mean_j': np.mean(dataset_js),
                'std_j': np.std(dataset_js),
                'mean_f': np.mean(dataset_fs),
                'std_f': np.std(dataset_fs),
                'mean_j_and_f': np.mean(dataset_j_and_fs),
                'std_j_and_f': np.std(dataset_j_and_fs),
                'mean_dice': np.mean(dataset_dices),
                'std_dice': np.std(dataset_dices),
                'mean_precision': np.mean(dataset_precisions),
                'std_precision': np.std(dataset_precisions),
                'mean_recall': np.mean(dataset_recalls),
                'std_recall': np.std(dataset_recalls),
                'total_objects': len(dataset_results)
            }

        return summary

    def _create_prompt_visualization(self, image_path: str, prompts: Dict, seq_name: str, object_areas: Dict = None) -> np.ndarray:
        """Create prompt visualization image with point sizes based on object areas"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Define colors for different objects (RGB format for matplotlib)
            # Using bright, vivid colors: Red, Blue, Green as primary colors
            colors = [
                (255, 0, 0),    # Bright Red
                (0, 0, 255),    # Bright Blue
                (0, 255, 0),    # Bright Green
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
            ]

            # Calculate radius based on object areas if provided
            point_radii = {}
            if object_areas and len(object_areas) > 0:
                max_area = max(object_areas.values())
                min_area = min(object_areas.values())
                area_range = max_area - min_area if max_area > min_area else 1

                for obj_id, area in object_areas.items():
                    if area_range > 0:
                        normalized_area = (area - min_area) / area_range  # 0 to 1
                        point_radii[obj_id] = int(4 + normalized_area * 8)  # radius from 4 to 12
                    else:
                        point_radii[obj_id] = 8
            else:
                # Default radius for all objects
                for obj_id in prompts.keys():
                    point_radii[obj_id] = 8

            # Draw prompts for each object
            color_idx = 0
            for obj_id, obj_prompts in prompts.items():
                color = colors[color_idx % len(colors)]
                points = obj_prompts['point_coords']
                radius = point_radii.get(obj_id, 8)

                # Draw points for this object
                for point in points:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(image, (x, y), radius, color, -1)
                    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 2)

                # Add object ID label near first point with size adjusted to point
                if len(points) > 0:
                    x, y = int(points[0][0]), int(points[0][1])
                    font_scale = max(0.5, radius * 0.075)
                    area_text = f" (area={object_areas.get(obj_id, '?')})" if object_areas else ""
                    cv2.putText(image, f"Obj {obj_id}{area_text}", (x + radius + 5, y - radius - 2),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

                color_idx += 1

            return image

        except Exception as e:
            print(f"Error creating prompt visualization: {e}")
            return None

    def _log_to_wandb(self, summary: Dict, results: List[Dict], prompts: Dict = None):
        """Log results to WandB"""
        try:
            # Log summary metrics (both object-level and image-level)
            if 'overall' in summary:
                # Object-level metrics
                if 'object_level' in summary['overall']:
                    wandb.log({
                        "overall/object_level/mean_iou": summary['overall']['object_level']['mean_iou'],
                        "overall/object_level/mean_j": summary['overall']['object_level']['mean_j'],
                        "overall/object_level/mean_f": summary['overall']['object_level']['mean_f'],
                        "overall/object_level/mean_j_and_f": summary['overall']['object_level']['mean_j_and_f'],
                        "overall/object_level/mean_dice": summary['overall']['object_level']['mean_dice'],
                        "overall/object_level/mean_precision": summary['overall']['object_level']['mean_precision'],
                        "overall/object_level/mean_recall": summary['overall']['object_level']['mean_recall'],
                        "overall/object_level/total_objects": summary['overall']['object_level']['total_objects']
                    })

                # Image-level metrics
                if 'image_level' in summary['overall']:
                    wandb.log({
                        "overall/image_level/mean_iou": summary['overall']['image_level']['mean_iou'],
                        "overall/image_level/mean_j": summary['overall']['image_level']['mean_j'],
                        "overall/image_level/mean_f": summary['overall']['image_level']['mean_f'],
                        "overall/image_level/mean_j_and_f": summary['overall']['image_level']['mean_j_and_f'],
                        "overall/image_level/mean_dice": summary['overall']['image_level']['mean_dice'],
                        "overall/image_level/mean_precision": summary['overall']['image_level']['mean_precision'],
                        "overall/image_level/mean_recall": summary['overall']['image_level']['mean_recall'],
                        "overall/image_level/total_images": summary['overall']['image_level']['total_images']
                    })

                # VOS-style metrics (sequence-wise)
                if 'vos_style' in summary['overall'] and summary['overall']['vos_style']:
                    wandb.log({
                        "overall/vos_style/mean_j": summary['overall']['vos_style']['mean_j'],
                        "overall/vos_style/mean_f": summary['overall']['vos_style']['mean_f'],
                        "overall/vos_style/mean_j_and_f": summary['overall']['vos_style']['mean_j_and_f'],
                        "overall/vos_style/mean_dice": summary['overall']['vos_style']['mean_dice'],
                        "overall/vos_style/mean_precision": summary['overall']['vos_style']['mean_precision'],
                        "overall/vos_style/mean_recall": summary['overall']['vos_style']['mean_recall'],
                        "overall/vos_style/total_sequences": summary['overall']['vos_style']['total_sequences']
                    })

                # Object-wise metrics (alias for object_level for consistency with ACDC)
                if 'object_level' in summary['overall']:
                    wandb.log({
                        "metrics/object_wise_mean_j": summary['overall']['object_level']['mean_j'],
                        "metrics/object_wise_mean_f": summary['overall']['object_level']['mean_f'],
                        "metrics/object_wise_mean_j_and_f": summary['overall']['object_level']['mean_j_and_f'],
                        "metrics/object_wise_mean_dice": summary['overall']['object_level']['mean_dice'],
                        "metrics/object_wise_std_j": summary['overall']['object_level']['std_j'],
                        "metrics/object_wise_std_f": summary['overall']['object_level']['std_f'],
                        "metrics/object_wise_std_j_and_f": summary['overall']['object_level']['std_j_and_f'],
                        "metrics/object_wise_std_dice": summary['overall']['object_level']['std_dice'],
                    })

                # Flat keys kept alongside the nested results
                wandb.log({
                    "overall/mean_iou": summary['overall']['mean_iou'],
                    "overall/mean_j": summary['overall']['mean_j'],
                    "overall/mean_f": summary['overall']['mean_f'],
                    "overall/mean_j_and_f": summary['overall']['mean_j_and_f'],
                    "overall/mean_dice": summary['overall']['mean_dice'],
                    "overall/mean_precision": summary['overall']['mean_precision'],
                    "overall/mean_recall": summary['overall']['mean_recall'],
                    "overall/total_objects": summary['overall']['total_objects']
                })

            # Log per-dataset metrics (both object-level and image-level)
            for dataset, stats in summary.items():
                if dataset != 'overall':
                    # Object-level metrics
                    if 'object_level' in stats:
                        wandb.log({
                            f"{dataset}/object_level/mean_iou": stats['object_level']['mean_iou'],
                            f"{dataset}/object_level/mean_j": stats['object_level']['mean_j'],
                            f"{dataset}/object_level/mean_f": stats['object_level']['mean_f'],
                            f"{dataset}/object_level/mean_j_and_f": stats['object_level']['mean_j_and_f'],
                            f"{dataset}/object_level/mean_dice": stats['object_level']['mean_dice'],
                            f"{dataset}/object_level/mean_precision": stats['object_level']['mean_precision'],
                            f"{dataset}/object_level/mean_recall": stats['object_level']['mean_recall'],
                            f"{dataset}/object_level/total_objects": stats['object_level']['total_objects']
                        })

                    # Image-level metrics
                    if 'image_level' in stats:
                        wandb.log({
                            f"{dataset}/image_level/mean_iou": stats['image_level']['mean_iou'],
                            f"{dataset}/image_level/mean_j": stats['image_level']['mean_j'],
                            f"{dataset}/image_level/mean_f": stats['image_level']['mean_f'],
                            f"{dataset}/image_level/mean_j_and_f": stats['image_level']['mean_j_and_f'],
                            f"{dataset}/image_level/mean_dice": stats['image_level']['mean_dice'],
                            f"{dataset}/image_level/mean_precision": stats['image_level']['mean_precision'],
                            f"{dataset}/image_level/mean_recall": stats['image_level']['mean_recall'],
                            f"{dataset}/image_level/total_images": stats['image_level']['total_images']
                        })

                    # VOS-style metrics
                    if 'vos_style' in stats and stats['vos_style']:
                        wandb.log({
                            f"{dataset}/vos_style/mean_j": stats['vos_style']['mean_j'],
                            f"{dataset}/vos_style/mean_f": stats['vos_style']['mean_f'],
                            f"{dataset}/vos_style/mean_j_and_f": stats['vos_style']['mean_j_and_f'],
                            f"{dataset}/vos_style/mean_dice": stats['vos_style']['mean_dice'],
                            f"{dataset}/vos_style/total_sequences": stats['vos_style']['total_sequences']
                        })

                    # Object-wise metrics (alias for object_level)
                    if 'object_level' in stats:
                        wandb.log({
                            f"{dataset}/object_wise_mean_j": stats['object_level']['mean_j'],
                            f"{dataset}/object_wise_mean_f": stats['object_level']['mean_f'],
                            f"{dataset}/object_wise_mean_j_and_f": stats['object_level']['mean_j_and_f'],
                            f"{dataset}/object_wise_mean_dice": stats['object_level']['mean_dice'],
                        })

                    # Flat keys kept alongside the nested results
                    wandb.log({
                        f"{dataset}/mean_iou": stats['mean_iou'],
                        f"{dataset}/mean_j": stats['mean_j'],
                        f"{dataset}/mean_f": stats['mean_f'],
                        f"{dataset}/mean_j_and_f": stats['mean_j_and_f'],
                        f"{dataset}/mean_dice": stats['mean_dice'],
                        f"{dataset}/mean_precision": stats['mean_precision'],
                        f"{dataset}/mean_recall": stats['mean_recall'],
                        f"{dataset}/total_objects": stats['total_objects']
                    })

            # Log/Save prompt visualizations (first few sequences)
            if prompts and (self.use_wandb or self.save_qual_dir):
                print("Processing prompt visualizations...")
                prompt_images = []
                count = 0
                max_vis = 10  # Process first 10 sequences

                for seq_name, seq_prompts in prompts.items():
                    if count >= max_vis:
                        break

                    # Find the sequence path - sequences are directly under the dataset path
                    seq_path = Path(self.annotated_dataset_path) / seq_name / 'visible'
                    label_path = Path(self.annotated_dataset_path) / seq_name / 'labels'

                    if seq_path.exists():
                        # Get first frame
                        visible_files = sorted([f for f in seq_path.iterdir() if f.suffix.lower() in ['.png', '.jpg']])
                        if visible_files:
                            first_frame = visible_files[0]

                            # Try to get first label frame for object areas
                            object_areas = {}
                            if label_path.exists():
                                label_files = sorted([f for f in label_path.iterdir() if f.suffix.lower() == '.png'])
                                if label_files:
                                    first_label = label_path / label_files[0]
                                    if first_label.exists():
                                        gt_mask = np.array(Image.open(first_label).convert('L'))
                                        for obj_id in seq_prompts.keys():
                                            obj_mask = (gt_mask == obj_id)
                                            object_areas[obj_id] = int(np.sum(obj_mask))

                            # Create visualization with object areas
                            vis_image = self._create_prompt_visualization(str(first_frame), seq_prompts, seq_name, object_areas)
                            if vis_image is not None:
                                # Save to disk if save_qual_dir is specified
                                if self.save_qual_dir:
                                    qual_seq_dir = os.path.join(self.save_qual_dir, seq_name, 'prompts')
                                    os.makedirs(qual_seq_dir, exist_ok=True)

                                    # Save visualization
                                    vis_path = os.path.join(qual_seq_dir, 'all_prompts_vis.png')
                                    cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                                    # Save original image
                                    img_path = os.path.join(qual_seq_dir, 'init_frame_image.png')
                                    shutil.copy(str(first_frame), img_path)

                                    # Save GT masks for each object (blue color)
                                    if label_path.exists() and label_files:
                                        first_label = label_path / label_files[0]
                                        if first_label.exists():
                                            gt_mask = np.array(Image.open(first_label).convert('L'))
                                            for obj_id in seq_prompts.keys():
                                                obj_mask = (gt_mask == obj_id).astype(np.uint8)
                                                # Create blue mask
                                                gt_blue = np.zeros((*obj_mask.shape, 3), dtype=np.uint8)
                                                gt_blue[obj_mask > 0] = [255, 0, 0]  # BGR format: blue
                                                gt_mask_path = os.path.join(qual_seq_dir, f'init_frame_obj{obj_id:02d}_gt.png')
                                                cv2.imwrite(gt_mask_path, gt_blue)

                                # Log to wandb if use_wandb is enabled
                                if self.use_wandb:
                                    prompt_images.append(wandb.Image(vis_image, caption=f"Prompts: {seq_name}"))

                                count += 1

                if prompt_images:
                    # Log each image individually with unique keys
                    for i, img in enumerate(prompt_images):
                        wandb.log({f"prompt_viz_{i}": img})
                    # Also log as a list
                    wandb.log({"prompt_visualizations": prompt_images})
                    print(f"[OK] Logged {len(prompt_images)} prompt visualizations to WandB")

            print("[OK] Summary metrics logged to WandB")

        except Exception as e:
            print(f"Failed to log to WandB: {e}")


def main():
    parser = argparse.ArgumentParser(description='MVSeg Multi-Object Annotated Dataset Evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--annotated_dataset',
                       default=os.environ.get('MOGA_MVSEG_ROOT', './data/MVSeg_Dataset_annotated'),
                       help='Path to annotated dataset')
    parser.add_argument('--datasets', nargs='+', default=['INO', 'KAIST', 'OSU', 'RGBT234'],
                       help='Datasets to evaluate')
    parser.add_argument('--prompts_file', help='Path to prompts JSON file (will be created if not exists)')
    parser.add_argument('--max_objects', type=int, default=3, help='Maximum objects per sequence')
    parser.add_argument('--max_sequences', type=int, help='Maximum sequences to evaluate')
    parser.add_argument('--save_results', help='Path to save detailed results JSON')
    parser.add_argument('--save_qual_dir', type=str, default=None,
                        help='Directory to save qualitative visualizations (prompts and predictions)')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--wandb_name', default='mvseg_multiobject_eval', help='WandB run name')

    args = parser.parse_args()

    print("Starting MVSeg multi-object annotated dataset evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Annotated dataset: {args.annotated_dataset}")
    print(f"Datasets: {args.datasets}")
    print(f"Max objects: {args.max_objects}")

    # Initialize evaluator
    evaluator = MVSegMultiObjectEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        annotated_dataset_path=args.annotated_dataset,
        prompts_file=args.prompts_file,
        max_objects=args.max_objects,
        use_wandb=args.use_wandb,
        wandb_name=args.wandb_name,
        save_qual_dir=args.save_qual_dir
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        datasets=args.datasets,
        max_sequences=args.max_sequences
    )

    if results:
        # Print summary
        summary = results['summary']
        if 'overall' in summary:
            overall = summary['overall']
            print(f"\n{'='*60}")
            print("MULTI-OBJECT EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"Mean IoU: {overall['mean_iou']:.4f} ± {overall['std_iou']:.4f}")
            print(f"Mean Dice: {overall['mean_dice']:.4f} ± {overall['std_dice']:.4f}")
            print(f"Mean Precision: {overall['mean_precision']:.4f} ± {overall['std_precision']:.4f}")
            print(f"Mean Recall: {overall['mean_recall']:.4f} ± {overall['std_recall']:.4f}")
            print(f"Total Objects: {overall['total_objects']}")
            print(f"Total Sequences: {results['total_sequences']}")

            # Per-dataset results
            print(f"\n{'='*60}")
            print("DATASET-SPECIFIC RESULTS")
            print(f"{'='*60}")
            for dataset, stats in summary.items():
                if dataset != 'overall':
                    print(f"\n{dataset}:")
                    print(f"  IoU: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f}")
                    print(f"  Dice: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")
                    print(f"  Precision: {stats['mean_precision']:.4f} ± {stats['std_precision']:.4f}")
                    print(f"  Recall: {stats['mean_recall']:.4f} ± {stats['std_recall']:.4f}")
                    print(f"  Objects: {stats['total_objects']}")

        # Save results
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {args.save_results}")

    print("Multi-object annotated dataset evaluation completed!")


if __name__ == "__main__":
    main()