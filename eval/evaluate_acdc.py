#!/usr/bin/env python3
"""
Evaluate SAM2 on ACDC Multi-Object Dataset
This script processes ACDC video sequences with multiple objects per frame
and evaluates segmentation performance following SAM2 video protocol.

Data Structure: acdc_complete_split_30_object/{weather}/{video_name}/split_XXX/
Each split folder contains a sequence of frames with corresponding GT annotations.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
from collections import defaultdict
import argparse
import json
import tempfile
import shutil
import wandb
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sam2.build_sam import build_sam2_video_predictor

def calculate_f_measure(pred_mask, gt_mask, bound_th=0.008):
    """
    Calculate F-measure (contour accuracy) between prediction and GT
    Following DAVIS evaluation protocol with adaptive boundary threshold

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        bound_th: Boundary thickness threshold (relative to image diagonal, default 0.008)

    Returns:
        F-measure score
    """
    # Convert to bool to ensure correct XOR operation (avoid uint8 XOR bool issue)
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    # Calculate adaptive boundary distance threshold (DAVIS standard)
    bound_pix = bound_th * np.linalg.norm(gt_mask.shape)

    # Extract boundaries
    pred_boundary = pred_mask ^ ndimage.binary_erosion(pred_mask, structure=np.ones((3,3)))
    gt_boundary = gt_mask ^ ndimage.binary_erosion(gt_mask, structure=np.ones((3,3)))

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    # Distance transforms
    pred_dist = ndimage.distance_transform_edt(~pred_boundary)
    gt_dist = ndimage.distance_transform_edt(~gt_boundary)

    # Precision: GT boundary pixels and prediction boundary distance
    precision_dists = pred_dist[gt_boundary]
    precision = np.mean(precision_dists <= bound_pix)

    # Recall: prediction boundary pixels and GT boundary distance
    recall_dists = gt_dist[pred_boundary]
    recall = np.mean(recall_dists <= bound_pix)

    # F-measure
    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure

# ACDC/Cityscapes class mapping
CITYSCAPES_CLASSES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic_light',
    7: 'traffic_sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'ignore'
}

# Cityscapes standard color palette (BGR format for OpenCV)
CITYSCAPES_COLORS = {
    0: (128, 64, 128),    # road - purple
    1: (244, 35, 232),    # sidewalk - pink
    2: (70, 70, 70),      # building - gray
    3: (102, 102, 156),   # wall - light purple
    4: (190, 153, 153),   # fence - light pink
    5: (153, 153, 153),   # pole - gray
    6: (250, 170, 30),    # traffic light - orange
    7: (220, 220, 0),     # traffic sign - yellow
    8: (107, 142, 35),    # vegetation - green
    9: (152, 251, 152),   # terrain - light green
    10: (70, 130, 180),   # sky - blue
    11: (220, 20, 60),    # person - red
    12: (255, 0, 0),      # rider - bright red
    13: (0, 0, 142),      # car - dark blue
    14: (0, 0, 70),       # truck - darker blue
    15: (0, 60, 100),     # bus - blue-green
    16: (0, 80, 100),     # train - blue
    17: (0, 0, 230),      # motorcycle - bright blue
    18: (119, 11, 32),    # bicycle - dark red
    255: (0, 0, 0),       # ignore - black
}

# Important classes for autonomous driving
# Note: We will dynamically read all labeled classes from GT masks instead of using hardcoded classes

class ACDCMultiObjectDataset:
    """Dataset class for ACDC multi-object data structure"""

    def __init__(self, data_dir, weather_conditions=['fog', 'snow', 'night', 'rain'], gt_data_dir=None):
        self.data_dir = data_dir
        self.gt_data_dir = gt_data_dir if gt_data_dir is not None else data_dir
        self.weather_conditions = weather_conditions
        self.sequences = self._discover_sequences()

    def _discover_sequences(self):
        """Discover all video sequences and their splits"""
        sequences = []

        for weather in self.weather_conditions:
            weather_dir = os.path.join(self.data_dir, weather)
            if not os.path.exists(weather_dir):
                print(f"Warning: Weather directory not found: {weather_dir}")
                continue

            for video_name in os.listdir(weather_dir):
                video_dir = os.path.join(weather_dir, video_name)
                if not os.path.isdir(video_dir):
                    continue

                # Load metadata if available
                metadata_path = os.path.join(video_dir, 'metadata.json')
                metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                # Find all split folders
                splits = []
                # GT directory from gt_data_dir (for restored images with separate GT)
                gt_video_dir = os.path.join(self.gt_data_dir, weather, video_name)
                for item in os.listdir(video_dir):
                    if item.startswith('split_') and os.path.isdir(os.path.join(video_dir, item)):
                        split_dir = os.path.join(video_dir, item)
                        gt_dir = os.path.join(gt_video_dir, 'gt')

                        # Check if frames exist
                        frame_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
                        if frame_files:
                            splits.append({
                                'split_id': item,
                                'split_dir': split_dir,
                                'gt_dir': gt_dir,
                                'frame_count': len(frame_files),
                                'frames': sorted(frame_files)
                            })

                if splits:
                    sequences.append({
                        'weather': weather,
                        'video_name': video_name,
                        'video_dir': video_dir,
                        'splits': splits,
                        'metadata': metadata
                    })

        print(f"Discovered {len(sequences)} sequences across {len(self.weather_conditions)} weather conditions")
        for seq in sequences[:3]:  # Show first 3 as examples
            print(f"  {seq['weather']}/{seq['video_name']}: {len(seq['splits'])} splits")

        return sequences

def create_prompt_visualization(image, prompts, target_class_name, frame_num, target_mask=None, prompt_type='points'):
    """Create visualization showing prompts on image (single object)

    Args:
        prompts: Either (N, 2) array of points or (4,) array of box [x_min, y_min, x_max, y_max]
        prompt_type: 'points' or 'box'
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if image is not None:
        ax.imshow(image)
    else:
        ax.imshow(np.zeros((1024, 1024, 3)), cmap='gray')
        ax.text(512, 512, 'Frame Not Available', ha='center', va='center',
                color='white', fontsize=16)

    # Draw prompts based on type
    if prompts is not None:
        if prompt_type == 'box':
            # Draw box prompt
            from matplotlib.patches import Rectangle
            x_min, y_min, x_max, y_max = prompts
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=3, edgecolor='red', facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, 'Box Prompt',
                   color='red', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            # Draw point prompts
            for i, (x, y) in enumerate(prompts):
                # Draw prompt point
                circle = Circle((x, y), radius=15,
                              color='red', fill=True, alpha=0.8)
                ax.add_patch(circle)

                # Add point number
                ax.text(x + 20, y - 20, f'P{i+1}',
                       color='red', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Draw ground truth mask outline if available
    if target_mask is not None:
        # Find contours
        contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 2:
                ax.plot(contour[:, 0], contour[:, 1],
                       color='yellow', linewidth=3, alpha=0.8, label='GT Mask')

    if prompt_type == 'box':
        prompt_desc = 'Box'
    else:
        prompt_desc = f'{len(prompts)} Points' if prompts is not None else '0 Points'

    ax.set_title(f'Initial Prompt - Frame {frame_num}\n'
                f'Target: {target_class_name} | Prompt: {prompt_desc}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    if target_mask is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig

def create_all_prompts_visualization(image, all_objects, frame_num):
    """Create visualization showing ALL object prompts on a single image"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    if image is not None:
        ax.imshow(image)
    else:
        ax.imshow(np.zeros((1024, 1024, 3)), cmap='gray')
        ax.text(512, 512, 'Frame Not Available', ha='center', va='center',
                color='white', fontsize=16)

    # Define distinct colors for different objects
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    # Calculate radius based on object area
    if len(all_objects) > 0:
        max_area = max(obj['area'] for obj in all_objects)
        min_area = min(obj['area'] for obj in all_objects)
        area_range = max_area - min_area if max_area > min_area else 1

    # Draw prompts and masks for all objects
    for obj_idx, obj in enumerate(all_objects):
        color = colors[obj_idx % len(colors)]
        class_name = CITYSCAPES_CLASSES.get(obj['class_id'], 'unknown')

        # Calculate point radius based on object area (smaller objects = smaller points)
        area = obj['area']
        if area_range > 0:
            normalized_area = (area - min_area) / area_range  # 0 to 1
            radius = 6 + normalized_area * 10  # radius from 6 to 16
        else:
            radius = 12

        # Draw prompts (either box or points)
        if 'box' in obj:
            # Draw box prompt
            box = obj['box']
            x_min, y_min, x_max, y_max = box
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            # Add box label
            fontsize = max(10, int(radius * 0.8))
            ax.text(x_min, y_min - 10, f'Obj{obj_idx}-Box',
                   color=color, fontsize=fontsize, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
        elif 'points' in obj:
            # Draw prompt points
            prompts = obj['points']
            for i, (x, y) in enumerate(prompts):
                # Draw prompt point
                circle = Circle((x, y), radius=radius,
                              color=color, fill=True, alpha=0.9, edgecolor='white', linewidth=2)
                ax.add_patch(circle)

                # Add point label with size adjusted to point
                fontsize = max(8, int(radius * 0.8))
                ax.text(x + radius + 6, y - radius - 2, f'Obj{obj_idx}-P{i+1}',
                       color=color, fontsize=fontsize, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color))

        # Draw ground truth mask outline
        target_mask = obj['mask']
        if target_mask is not None:
            # Find contours
            contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and contour.shape[0] > 2:
                    ax.plot(contour[:, 0], contour[:, 1],
                           color=color, linewidth=2.5, alpha=0.8,
                           label=f'Obj{obj_idx}: {class_name} (area={area})')

    # Determine prompt type
    has_boxes = any('box' in obj for obj in all_objects)
    has_points = any('points' in obj for obj in all_objects)
    if has_boxes and has_points:
        prompt_type = "Mixed (Box+Point)"
    elif has_boxes:
        prompt_type = "Box"
    else:
        prompt_type = "Point"

    ax.set_title(f'All Object Prompts - Frame {frame_num}\n'
                f'Total Objects: {len(all_objects)} | Prompt Type: {prompt_type} | Each color = different object',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    # Create legend with object info
    if len(all_objects) > 0:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    return fig

def create_prediction_visualization(image, pred_mask, gt_mask, frame_num, iou, class_name, is_good_pred=True):
    """Create visualization showing prediction vs ground truth"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    ax1.imshow(image)
    ax1.set_title(f'Original Frame {frame_num}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Ground truth
    ax2.imshow(image)
    if gt_mask is not None:
        # Overlay GT mask
        masked_image = image.copy()
        masked_image[gt_mask > 0] = [0, 255, 0]  # Green for GT
        ax2.imshow(masked_image, alpha=0.7)

        # Draw GT contours
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 2:
                ax2.plot(contour[:, 0], contour[:, 1],
                        color='yellow', linewidth=2, alpha=0.9)

    ax2.set_title(f'Ground Truth\n{class_name}', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Prediction
    ax3.imshow(image)
    if pred_mask is not None:
        # Overlay prediction mask
        masked_image = image.copy()
        color = [0, 0, 255] if is_good_pred else [255, 0, 0]  # Blue for good, Red for bad
        masked_image[pred_mask > 0] = color
        ax3.imshow(masked_image, alpha=0.7)

        # Draw prediction contours
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 2:
                ax3.plot(contour[:, 0], contour[:, 1],
                        color='cyan', linewidth=2, alpha=0.9)

    status = "Good" if is_good_pred else "Poor"
    ax3.set_title(f'Prediction ({status})\nIoU: {iou:.3f}', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Overlay comparison
    ax4.imshow(image)
    if gt_mask is not None and pred_mask is not None:
        # Create overlay image
        overlay = image.copy()

        # GT in green, Prediction in blue, Overlap in yellow
        gt_only = (gt_mask > 0) & (pred_mask == 0)
        pred_only = (pred_mask > 0) & (gt_mask == 0)
        overlap = (gt_mask > 0) & (pred_mask > 0)

        overlay[gt_only] = [0, 255, 0]      # Green: GT only
        overlay[pred_only] = [0, 0, 255]    # Blue: Pred only
        overlay[overlap] = [255, 255, 0]    # Yellow: Overlap

        ax4.imshow(overlay, alpha=0.8)

    ax4.set_title(f'Comparison\nGreen: GT only | Blue: Pred only | Yellow: Overlap',
                 fontsize=12, fontweight='bold')
    ax4.axis('off')

    plt.tight_layout()
    return fig

class ACDCMultiObjectEvaluator:
    """Evaluator for ACDC multi-object sequences using SAM2"""

    def __init__(self, predictor, dataset, device='cuda', use_wandb=False,
                 image_size=1024, max_sequences=None, prompts_file=None, save_qual_dir=None):
        self.predictor = predictor
        self.dataset = dataset
        self.device = device
        self.use_wandb = use_wandb
        self.image_size = image_size
        self.max_sequences = max_sequences
        self.prompts_file = prompts_file
        self.shared_prompts = self.load_shared_prompts() if prompts_file else None
        self.save_qual_dir = save_qual_dir

        # Create qual output directory if specified
        if self.save_qual_dir:
            os.makedirs(self.save_qual_dir, exist_ok=True)
            print(f"[OK] Qualitative results will be saved to: {self.save_qual_dir}")

    def load_shared_prompts(self):
        """Load shared prompts from JSON file"""
        if not self.prompts_file or not os.path.exists(self.prompts_file):
            print(f"Warning: Prompts file {self.prompts_file} not found. Using systematic prompts.")
            return None

        try:
            with open(self.prompts_file, 'r') as f:
                prompts_data = json.load(f)
            print(f"[OK] Loaded shared prompts from {self.prompts_file}")
            return prompts_data
        except Exception as e:
            print(f"Error loading prompts file: {e}")
            return None

    def get_largest_connected_component(self, mask):
        """Get the largest connected component from a binary mask"""
        if mask.sum() == 0:
            return mask

        # Find connected components
        labeled_mask = label(mask)
        if labeled_mask.max() == 0:
            return mask

        # Get properties of each region
        regions = regionprops(labeled_mask)
        if not regions:
            return mask

        # Find the largest region
        largest_region = max(regions, key=lambda r: r.area)

        # Create mask for only the largest connected component
        largest_component_mask = (labeled_mask == largest_region.label)
        return largest_component_mask.astype(mask.dtype)

    def generate_prompts_from_gt(self, gt_mask, weather=None, video_name=None):
        """Generate 3 point prompts for ALL objects from ground truth mask using shared prompts if available"""
        if gt_mask is None or gt_mask.sum() == 0:
            return []

        # Find all object classes (excluding road and ignore classes)
        unique_classes = np.unique(gt_mask)
        unique_classes = unique_classes[unique_classes != 255]  # Remove ignore
        unique_classes = unique_classes[unique_classes != 0]    # Remove road

        if len(unique_classes) == 0:
            return []

        all_objects = []

        # Try to use shared prompts if available
        if self.shared_prompts and weather and video_name:
            try:
                weather_prompts = self.shared_prompts['prompts'].get(weather, {})

                # Extract base video name (remove split info if present)
                base_video_name = video_name.split('/')[0] if '/' in video_name else video_name
                video_prompts = weather_prompts.get(base_video_name, {})

                if video_prompts and 'objects' in video_prompts:
                    print(f"Using shared prompts for {weather}/{video_name}")
                    print(f"  GT classes: {unique_classes}")
                    print(f"  Available shared prompts for classes: {[obj['class_id'] for obj in video_prompts['objects']]}")

                    for obj in video_prompts['objects']:
                        class_id = obj['class_id']
                        # Support both point and box prompts
                        has_box = 'box' in obj
                        has_points = 'points' in obj

                        if has_box:
                            box = obj['box']
                        elif has_points:
                            points = obj['points']
                        else:
                            print(f"  Warning: Object has neither 'box' nor 'points' key, skipping")
                            continue

                        # Check if this class exists in the GT mask
                        if class_id in unique_classes:
                            print(f"  Processing shared prompts for class {class_id}")
                            class_mask = (gt_mask == class_id)

                            # Get all connected components for this class
                            labeled_mask = label(class_mask)
                            regions = regionprops(labeled_mask)

                            if regions:
                                # Filter regions by area threshold (area >= 10)
                                valid_regions = [r for r in regions if r.area >= 10]

                                if valid_regions:
                                    # Select only the largest component per class (deterministic)
                                    # Sort by area (desc) then by label (asc) for deterministic selection when areas are equal
                                    largest_region = max(valid_regions, key=lambda r: (r.area, -r.label))
                                    connected_mask = (labeled_mask == largest_region.label).astype(np.uint8)

                                    print(f"    Selected largest component with area {largest_region.area} (out of {len(regions)} total components)")

                                    # For box prompts, directly use the box
                                    if has_box:
                                        all_objects.append({
                                            'box': np.array(box, dtype=np.float32),
                                            'class_id': class_id,
                                            'mask': connected_mask,
                                            'area': largest_region.area
                                        })
                                        continue  # Skip point validation for box prompts

                                    # Validate that shared prompts are inside the largest component mask
                                    valid_points = []
                                    for point in points:
                                        x, y = int(point[0]), int(point[1])
                                        if (0 <= y < connected_mask.shape[0] and
                                            0 <= x < connected_mask.shape[1] and
                                            connected_mask[y, x] > 0):
                                            valid_points.append(point)
                                        else:
                                            # Snap to nearest point inside this component
                                            mask_coords = np.column_stack(np.where(connected_mask))
                                            if len(mask_coords) > 0:
                                                distances = np.sqrt((mask_coords[:, 0] - y)**2 + (mask_coords[:, 1] - x)**2)
                                                closest_idx = np.argmin(distances)
                                                ny, nx = mask_coords[closest_idx]
                                                valid_points.append([float(nx), float(ny)])

                                    # If we still have fewer than 3 valid points, generate new ones inside the mask
                                    if len(valid_points) < 3:
                                        y_coords, x_coords = np.where(connected_mask)
                                        if len(y_coords) > 0:
                                            np.random.seed(42)  # For reproducibility
                                            num_needed = 3 - len(valid_points)
                                            if len(y_coords) >= num_needed:
                                                indices = np.random.choice(len(y_coords), num_needed, replace=False)
                                                for idx in indices:
                                                    valid_points.append([float(x_coords[idx]), float(y_coords[idx])])
                                            else:
                                                for i in range(num_needed):
                                                    idx = i % len(y_coords)
                                                    valid_points.append([float(x_coords[idx]), float(y_coords[idx])])

                                    if len(valid_points) >= 3:
                                        all_objects.append({
                                            'points': np.array(valid_points[:3], dtype=np.float32),
                                            'class_id': class_id,
                                            'mask': connected_mask,
                                            'area': largest_region.area
                                        })
                                    else:
                                        print(f"    Could not generate valid points for class {class_id} largest component")

                    if len(all_objects) > 0:
                        print(f"  Successfully created {len(all_objects)} shared prompt objects")
                        return all_objects
                    else:
                        print(f"  No valid shared prompt objects created (no matching classes or too small areas)")
            except Exception as e:
                print(f"Error using shared prompts: {e}, falling back to systematic prompts")

        # Fallback to systematic prompts if shared prompts not available
        print(f"Using systematic prompts for {weather}/{video_name}")
        for class_id in unique_classes:
            class_mask = (gt_mask == class_id)

            # Get all connected components for this class
            labeled_mask = label(class_mask)
            regions = regionprops(labeled_mask)

            for region in regions:
                if region.area < 10:  # Skip small objects (consistent threshold)
                    continue

                # Create mask for this connected component
                connected_mask = (labeled_mask == region.label).astype(np.uint8)

                # Generate 3 points from this connected component
                y_coords, x_coords = np.where(connected_mask)

                if len(y_coords) == 0:
                    continue

                # 1. Center point
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))

                # 2. Top-left region point
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                tl_x = int(min_x + (max_x - min_x) * 0.25)
                tl_y = int(min_y + (max_y - min_y) * 0.25)

                # 3. Bottom-right region point
                br_x = int(min_x + (max_x - min_x) * 0.75)
                br_y = int(min_y + (max_y - min_y) * 0.75)

                # Ensure all points are within the connected component
                points = []
                candidate_points = [
                    [center_x, center_y],  # Center
                    [tl_x, tl_y],         # Top-left region
                    [br_x, br_y],         # Bottom-right region
                ]

                for px, py in candidate_points:
                    # Clamp to image bounds
                    px = max(0, min(connected_mask.shape[1] - 1, px))
                    py = max(0, min(connected_mask.shape[0] - 1, py))

                    # If point is not in the mask, find the nearest point in the mask
                    if not connected_mask[py, px]:
                        # Find closest point in the mask
                        mask_coords = np.column_stack(np.where(connected_mask))
                        distances = np.sqrt((mask_coords[:, 0] - py)**2 + (mask_coords[:, 1] - px)**2)
                        closest_idx = np.argmin(distances)
                        py, px = mask_coords[closest_idx]

                    points.append([px, py])

                all_objects.append({
                    'points': np.array(points, dtype=np.float32),
                    'class_id': class_id,
                    'mask': connected_mask,
                    'area': region.area
                })

        return all_objects

    def load_and_resize_image(self, image_path):
        """Load and resize image"""
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size and self.image_size > 0:
            image = cv2.resize(image, (self.image_size, self.image_size))

        return image

    def load_and_resize_mask(self, mask_path):
        """Load and resize GT mask"""
        if not os.path.exists(mask_path):
            return None

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        if self.image_size and self.image_size > 0:
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_NEAREST)

        return mask

    def process_split_sequence(self, sequence, split_info):
        """Process a single split sequence (video clip)"""
        weather = sequence['weather']
        video_name = sequence['video_name']
        split_id = split_info['split_id']

        print(f"\nProcessing {weather}/{video_name}/{split_id} - {split_info['frame_count']} frames")

        # Create temporary directory for SAM2 video processing
        temp_dir = tempfile.mkdtemp(prefix=f'acdc_{weather}_{video_name}_{split_id}_')
        video_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(video_dir, exist_ok=True)

        try:
            # Load and prepare frames
            frame_data = []
            frame_nums = []

            for i, frame_file in enumerate(split_info['frames']):
                frame_path = os.path.join(split_info['split_dir'], frame_file)
                frame_num = int(frame_file.split('.')[0])  # Extract frame number from filename

                # Load and resize image
                image = self.load_and_resize_image(frame_path)
                if image is None:
                    continue

                # Save frame for SAM2 video processing
                sam2_frame_path = os.path.join(video_dir, f'{i:06d}.jpg')
                cv2.imwrite(sam2_frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Load GT mask if exists
                gt_mask_path = os.path.join(split_info['gt_dir'], f'{frame_num:06d}.png')
                gt_mask = self.load_and_resize_mask(gt_mask_path)

                frame_data.append({
                    'frame_idx': i,
                    'frame_num': frame_num,
                    'image': image,
                    'gt_mask': gt_mask,
                    'has_gt': gt_mask is not None,
                    'sam2_frame_path': sam2_frame_path
                })
                frame_nums.append(frame_num)

            if not frame_data:
                print(f"No valid frames found for {weather}/{video_name}/{split_id}")
                return []

            print(f"  Loaded {len(frame_data)} frames, frame range: {min(frame_nums)}-{max(frame_nums)}")

            # Find first frame with GT to initialize prompts
            init_frame_idx = None
            init_gt = None
            for frame in frame_data:
                if frame['has_gt']:
                    init_frame_idx = frame['frame_idx']
                    init_gt = frame['gt_mask']
                    break

            if init_frame_idx is None:
                print(f"No GT found for {weather}/{video_name}/{split_id}")
                return []

            # Generate prompts from first GT frame for ALL objects
            all_objects = self.generate_prompts_from_gt(init_gt, weather, video_name)

            if not all_objects:
                print(f"No valid prompts for {weather}/{video_name}/{split_id}")
                return []

            print(f"  Initial prompts on frame {frame_data[init_frame_idx]['frame_num']}")
            print(f"  Found {len(all_objects)} objects to track:")
            for i, obj in enumerate(all_objects):
                print(f"    Object {i}: {CITYSCAPES_CLASSES.get(obj['class_id'], 'unknown')} (ID: {obj['class_id']}, area: {obj['area']})")

            # Log to wandb
            if self.use_wandb:
                prompt_info = {
                    'weather': weather,
                    'video_name': video_name,
                    'split_id': split_id,
                    'init_frame_idx': init_frame_idx,
                    'init_frame_num': frame_data[init_frame_idx]['frame_num'],
                    'num_objects': len(all_objects),
                    'objects': [],
                    'total_frames': len(frame_data),
                    'frame_range': f"{min(frame_nums)}-{max(frame_nums)}"
                }

                for i, obj in enumerate(all_objects):
                    prompt_entry = {
                        'object_id': i,
                        'class_id': int(obj['class_id']),
                        'class_name': CITYSCAPES_CLASSES.get(obj['class_id'], 'unknown'),
                        'area': int(obj['area'])
                    }
                    # Add prompt coordinates based on type
                    if 'box' in obj:
                        prompt_entry['prompt_type'] = 'box'
                        prompt_entry['prompt_coordinates'] = obj['box'].tolist()
                    elif 'points' in obj:
                        prompt_entry['prompt_type'] = 'points'
                        prompt_entry['prompt_coordinates'] = obj['points'].tolist()

                    prompt_info['objects'].append(prompt_entry)

                wandb.log({
                    f"prompt_info/{weather}_{video_name}_{split_id}": prompt_info
                })

            # Create and save/log ALL prompts visualization (do this regardless of wandb)
            if self.use_wandb or self.save_qual_dir:
                try:
                    # Visualization showing ALL objects on one image
                    all_prompts_fig = create_all_prompts_visualization(
                        frame_data[init_frame_idx]['image'],
                        all_objects,
                        frame_data[init_frame_idx]['frame_num']
                    )

                    # Save to disk if save_qual_dir is specified
                    if self.save_qual_dir:
                        qual_weather_dir = os.path.join(self.save_qual_dir, weather, video_name, split_id, 'prompts')
                        os.makedirs(qual_weather_dir, exist_ok=True)

                        # Save visualization
                        save_path = os.path.join(qual_weather_dir, f'all_prompts_frame{frame_data[init_frame_idx]["frame_num"]:06d}_vis.png')
                        all_prompts_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved ALL prompts visualization to: {save_path}")

                        # Save original image
                        img_path = os.path.join(qual_weather_dir, f'init_frame{frame_data[init_frame_idx]["frame_num"]:06d}_image.png')
                        cv2.imwrite(img_path, cv2.cvtColor(frame_data[init_frame_idx]['image'], cv2.COLOR_RGB2BGR))

                        # Save GT masks for each object (blue color)
                        for obj_idx, obj in enumerate(all_objects):
                            class_name = CITYSCAPES_CLASSES.get(obj['class_id'], 'unknown').replace(' ', '_')
                            gt_mask_path = os.path.join(qual_weather_dir,
                                f'init_frame{frame_data[init_frame_idx]["frame_num"]:06d}_obj{obj_idx:02d}_{class_name}_gt.png')
                            # Create blue mask
                            gt_blue = np.zeros((*obj['mask'].shape, 3), dtype=np.uint8)
                            gt_blue[obj['mask'] > 0] = [255, 0, 0]  # BGR format: blue
                            cv2.imwrite(gt_mask_path, gt_blue)

                    # Log to wandb if use_wandb is enabled
                    if self.use_wandb:
                        all_prompts_fig.canvas.draw()
                        img_array = np.frombuffer(all_prompts_fig.canvas.buffer_rgba(), dtype=np.uint8)
                        img_array = img_array.reshape(all_prompts_fig.canvas.get_width_height()[::-1] + (4,))
                        img_array = img_array[:, :, :3]  # Remove alpha channel

                        wandb.log({
                            f"prompt_visualizations/ALL_{weather}_{video_name}_{split_id}": wandb.Image(
                                img_array,
                                caption=f"ALL Prompts - {split_id} Frame {frame_data[init_frame_idx]['frame_num']} - {len(all_objects)} objects tracked"
                            )
                        })

                    plt.close(all_prompts_fig)

                    # Also create individual prompt visualization for the largest object
                    largest_obj = max(all_objects, key=lambda x: x['area'])
                    # Determine prompt type and coordinates
                    if 'box' in largest_obj:
                        prompt_coords = largest_obj['box']
                        prompt_type = 'box'
                    else:
                        prompt_coords = largest_obj['points']
                        prompt_type = 'points'

                    prompt_fig = create_prompt_visualization(
                        frame_data[init_frame_idx]['image'], prompt_coords,
                        CITYSCAPES_CLASSES.get(largest_obj['class_id'], 'unknown'),
                        frame_data[init_frame_idx]['frame_num'],
                        largest_obj['mask'],
                        prompt_type=prompt_type
                    )

                    # Save to disk if save_qual_dir is specified
                    if self.save_qual_dir:
                        save_path = os.path.join(qual_weather_dir, f'largest_prompt_frame{frame_data[init_frame_idx]["frame_num"]:06d}_vis.png')
                        prompt_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved largest prompt visualization to: {save_path}")

                    # Log to wandb if use_wandb is enabled
                    if self.use_wandb:
                        prompt_fig.canvas.draw()
                        img_array = np.frombuffer(prompt_fig.canvas.buffer_rgba(), dtype=np.uint8)
                        img_array = img_array.reshape(prompt_fig.canvas.get_width_height()[::-1] + (4,))
                        img_array = img_array[:, :, :3]  # Remove alpha channel

                        wandb.log({
                            f"prompt_visualizations/LARGEST_{weather}_{video_name}_{split_id}": wandb.Image(
                                img_array,
                                caption=f"Largest Object Prompt - {split_id} Frame {frame_data[init_frame_idx]['frame_num']} - {CITYSCAPES_CLASSES.get(largest_obj['class_id'], 'unknown')}"
                            )
                        })

                    plt.close(prompt_fig)

                except Exception as e:
                    print(f"  Failed to create prompt visualization: {e}")

            # SAM2 video inference with multi-object tracking
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    # Initialize SAM2 video predictor
                    inference_state = self.predictor.init_state(video_path=video_dir)

                    # Add prompts for each object on first GT frame
                    for obj_idx, obj in enumerate(all_objects):
                        # Support both box and point prompts
                        if 'box' in obj:
                            # Box prompt: [x_min, y_min, x_max, y_max]
                            box = obj['box']
                            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=init_frame_idx,
                                obj_id=obj_idx,
                                box=box,
                            )
                        else:
                            # Point prompts
                            points = obj['points']
                            num_points = points.shape[0]
                            labels = np.array([1] * num_points, dtype=np.int32)  # All positive

                            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=init_frame_idx,
                                obj_id=obj_idx,  # Use unique object ID for each object
                                points=points,
                                labels=labels,
                            )

                    # Propagate through video
                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                        video_segments[out_frame_idx] = {
                            'obj_ids': out_obj_ids,
                            'mask_logits': out_mask_logits
                        }

            # Evaluate results on GT frames - evaluate each object separately
            results = []

            from collections import defaultdict
            object_j_sums = defaultdict(float)
            object_f_sums = defaultdict(float)
            object_dice_sums = defaultdict(float)
            object_frame_counts = defaultdict(int)

            for frame in frame_data:
                # Skip evaluation for frames without GT
                if not frame['has_gt']:
                    continue

                frame_idx = frame['frame_idx']
                if frame_idx not in video_segments:
                    continue

                mask_logits = video_segments[frame_idx]['mask_logits']
                obj_ids = video_segments[frame_idx]['obj_ids']
                gt_mask = frame['gt_mask']

                # Evaluate each tracked object
                for obj_idx, obj in enumerate(all_objects):
                    obj_class_id = obj['class_id']

                    # Check if this object's class exists in current frame's GT
                    if obj_class_id not in np.unique(gt_mask):
                        continue

                    # Get predicted mask for this object
                    if obj_idx < len(obj_ids):
                        sam2_obj_id = obj_ids[obj_idx]
                        obj_mask_idx = list(obj_ids).index(sam2_obj_id)

                        if obj_mask_idx < mask_logits.shape[0]:
                            pred_mask = (mask_logits[obj_mask_idx].squeeze().cpu().numpy() > 0).astype(np.uint8)
                        else:
                            continue
                    else:
                        continue

                    # Get GT mask for this class and find best matching connected component
                    class_mask = (gt_mask == obj_class_id)
                    labeled_gt = label(class_mask)
                    gt_regions = regionprops(labeled_gt)

                    if not gt_regions:
                        continue

                    # Find the GT region with highest IoU with prediction
                    best_iou = 0.0
                    best_gt_mask = None
                    best_region_area = 0
                    best_region_label = -1

                    for region in gt_regions:
                        if region.area < 10:  # Skip small regions (consistent with prompt generation)
                            continue

                        region_mask = (labeled_gt == region.label).astype(np.uint8)

                        # Calculate IoU
                        intersection = np.logical_and(pred_mask, region_mask).sum()
                        union = np.logical_or(pred_mask, region_mask).sum()
                        iou = intersection / union if union > 0 else 0.0

                        # Deterministic selection: prefer higher IoU, then larger area, then smaller label
                        if (iou > best_iou) or (iou == best_iou and region.area > best_region_area) or \
                           (iou == best_iou and region.area == best_region_area and region.label < best_region_label):
                            best_iou = iou
                            best_gt_mask = region_mask
                            best_region_area = region.area
                            best_region_label = region.label

                    if best_gt_mask is None:
                        continue

                    # Store result
                    frame_result = {
                        'weather': weather,
                        'video_name': video_name,
                        'split_id': split_id,
                        'frame_num': frame['frame_num'],
                        'frame_idx': frame_idx,
                        'object_id': obj_idx,
                        'iou': float(best_iou),
                        'target_class_id': obj_class_id,
                        'target_class_name': CITYSCAPES_CLASSES.get(obj_class_id, 'unknown'),
                        'pred_mask': pred_mask,
                        'gt_mask': best_gt_mask,
                        'image': frame['image'],
                        'pred_area': int(pred_mask.sum()),
                        'gt_area': int(best_gt_mask.sum()),
                        'intersection': int(np.logical_and(pred_mask, best_gt_mask).sum())
                    }

                    # Calculate F-measure
                    f_measure = calculate_f_measure(pred_mask, best_gt_mask)

                    # Store additional metrics in result
                    frame_result['f_measure'] = float(f_measure)
                    frame_result['dice'] = float(2 * frame_result['intersection'] / (pred_mask.sum() + best_gt_mask.sum()) if (pred_mask.sum() + best_gt_mask.sum()) > 0 else 0.0)

                    # Append to results AFTER all metrics are calculated
                    results.append(frame_result)

                    object_j_sums[obj_idx] += best_iou
                    object_f_sums[obj_idx] += f_measure
                    object_dice_sums[obj_idx] += frame_result['dice']
                    object_frame_counts[obj_idx] += 1

                    # Log to wandb
                    if self.use_wandb:
                        intersection = frame_result['intersection']
                        precision = intersection / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
                        recall = intersection / best_gt_mask.sum() if best_gt_mask.sum() > 0 else 0.0
                        dice = frame_result['dice']

                        prediction_info = {
                            'weather': weather,
                            'video_name': video_name,
                            'split_id': split_id,
                            'frame_num': frame['frame_num'],
                            'object_id': obj_idx,
                            'class_id': obj_class_id,
                            'class_name': CITYSCAPES_CLASSES.get(obj_class_id, 'unknown'),
                            'iou': float(best_iou),
                            'f_measure': float(f_measure),
                            'j_and_f': float((best_iou + f_measure) / 2.0),
                            'precision': float(precision),
                            'recall': float(recall),
                            'dice': float(dice),
                            'pred_area': int(pred_mask.sum()),
                            'gt_area': int(best_gt_mask.sum()),
                            'intersection': int(intersection)
                        }

                        wandb.log({
                            f"predictions/{weather}_{video_name}_{split_id}_obj{obj_idx}_frame_{frame['frame_num']}": prediction_info
                        })

                        # Log visualizations for selected frames (to wandb only)
                        if (best_iou > 0.8 and len(results) % 20 == 0) or (best_iou == 0.0 and len(results) % 10 == 0):
                            try:
                                pred_fig = create_prediction_visualization(
                                    frame['image'], pred_mask, best_gt_mask,
                                    frame['frame_num'], best_iou,
                                    CITYSCAPES_CLASSES.get(obj_class_id, 'unknown'),
                                    is_good_pred=(best_iou > 0.5)
                                )

                                # Convert to wandb image
                                pred_fig.canvas.draw()
                                img_array = np.frombuffer(pred_fig.canvas.buffer_rgba(), dtype=np.uint8)
                                img_array = img_array.reshape(pred_fig.canvas.get_width_height()[::-1] + (4,))
                                img_array = img_array[:, :, :3]

                                status = "Good" if best_iou > 0.5 else "Poor"
                                wandb.log({
                                    f"prediction_visualizations/{status}_{weather}_{video_name}_{split_id}_obj{obj_idx}_frame_{frame['frame_num']}": wandb.Image(
                                        img_array,
                                        caption=f"{split_id} Obj{obj_idx} Frame {frame['frame_num']} - {CITYSCAPES_CLASSES.get(obj_class_id, 'unknown')} - IoU: {best_iou:.3f}"
                                    )
                                })

                                plt.close(pred_fig)

                            except Exception as e:
                                pass

                    # Save all predictions to disk if save_qual_dir is specified
                    if self.save_qual_dir:
                        try:
                            status = "good" if best_iou > 0.5 else "poor"
                            class_name_short = CITYSCAPES_CLASSES.get(obj_class_id, 'unknown').replace(' ', '_')

                            # Create base directory
                            qual_pred_dir = os.path.join(self.save_qual_dir, weather, video_name, split_id, 'predictions')
                            os.makedirs(qual_pred_dir, exist_ok=True)

                            # Save visualization (4-panel comparison)
                            pred_fig = create_prediction_visualization(
                                frame['image'], pred_mask, best_gt_mask,
                                frame['frame_num'], best_iou,
                                CITYSCAPES_CLASSES.get(obj_class_id, 'unknown'),
                                is_good_pred=(best_iou > 0.5)
                            )
                            vis_path = os.path.join(
                                qual_pred_dir,
                                f'{status}_obj{obj_idx:02d}_{class_name_short}_frame{frame["frame_num"]:06d}_iou{best_iou:.3f}_vis.png'
                            )
                            pred_fig.savefig(vis_path, dpi=150, bbox_inches='tight')
                            plt.close(pred_fig)

                            # Save raw images separately
                            base_name = f'{status}_obj{obj_idx:02d}_{class_name_short}_frame{frame["frame_num"]:06d}_iou{best_iou:.3f}'

                            # Save original image
                            img_path = os.path.join(qual_pred_dir, f'{base_name}_image.png')
                            cv2.imwrite(img_path, cv2.cvtColor(frame['image'], cv2.COLOR_RGB2BGR))

                            # Save prediction mask (blue color)
                            pred_path = os.path.join(qual_pred_dir, f'{base_name}_pred.png')
                            pred_blue = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                            pred_blue[pred_mask > 0] = [255, 0, 0]  # BGR format: blue
                            cv2.imwrite(pred_path, pred_blue)

                            # Save GT mask (blue color)
                            gt_path = os.path.join(qual_pred_dir, f'{base_name}_gt.png')
                            gt_blue = np.zeros((*best_gt_mask.shape, 3), dtype=np.uint8)
                            gt_blue[best_gt_mask > 0] = [255, 0, 0]  # BGR format: blue
                            cv2.imwrite(gt_path, gt_blue)

                        except Exception as e:
                            print(f"      Failed to save qualitative results: {e}")

                    print(f"    Frame {frame['frame_num']} Obj{obj_idx} ({CITYSCAPES_CLASSES.get(obj_class_id, 'unknown')}): IoU={best_iou:.3f}")

            split_object_metrics = {}
            for obj_idx in range(len(all_objects)):
                if object_frame_counts[obj_idx] > 0:
                    obj_j = object_j_sums[obj_idx] / object_frame_counts[obj_idx]
                    obj_f = object_f_sums[obj_idx] / object_frame_counts[obj_idx]
                    obj_dice = object_dice_sums[obj_idx] / object_frame_counts[obj_idx]
                    obj_jnf = (obj_j + obj_f) / 2.0

                    split_object_metrics[f'obj_{obj_idx}_J'] = obj_j
                    split_object_metrics[f'obj_{obj_idx}_F'] = obj_f
                    split_object_metrics[f'obj_{obj_idx}_J&F'] = obj_jnf
                    split_object_metrics[f'obj_{obj_idx}_dice'] = obj_dice
                    split_object_metrics[f'obj_{obj_idx}_frames'] = object_frame_counts[obj_idx]
                    split_object_metrics[f'obj_{obj_idx}_class'] = CITYSCAPES_CLASSES.get(all_objects[obj_idx]['class_id'], 'unknown')

                    if self.use_wandb:
                        wandb.log({
                            f"split_objects/{weather}_{video_name}_{split_id}_obj{obj_idx}_J": obj_j,
                            f"split_objects/{weather}_{video_name}_{split_id}_obj{obj_idx}_F": obj_f,
                            f"split_objects/{weather}_{video_name}_{split_id}_obj{obj_idx}_J&F": obj_jnf,
                            f"split_objects/{weather}_{video_name}_{split_id}_obj{obj_idx}_dice": obj_dice,
                            f"split_objects/{weather}_{video_name}_{split_id}_obj{obj_idx}_frames": object_frame_counts[obj_idx],
                        })

            if split_object_metrics:
                split_j = np.mean([v for k, v in split_object_metrics.items() if k.endswith('_J')])
                split_f = np.mean([v for k, v in split_object_metrics.items() if k.endswith('_F')])
                split_jnf = np.mean([v for k, v in split_object_metrics.items() if k.endswith('_J&F')])
                split_dice = np.mean([v for k, v in split_object_metrics.items() if k.endswith('_dice')])

                if self.use_wandb:
                    wandb.log({
                        f"split/{weather}_{video_name}_{split_id}_J": split_j,
                        f"split/{weather}_{video_name}_{split_id}_F": split_f,
                        f"split/{weather}_{video_name}_{split_id}_J&F": split_jnf,
                        f"split/{weather}_{video_name}_{split_id}_dice": split_dice,
                        f"split/{weather}_{video_name}_{split_id}_num_objects": len([k for k in split_object_metrics.keys() if k.endswith('_J')]),
                    })

                print(f"  Split {split_id} Average: J={split_j:.3f}, F={split_f:.3f}, J&F={split_jnf:.3f}, Dice={split_dice:.3f}")

            print(f"  Evaluated {len(results)} object instances across frames with GT")
            return results

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def evaluate(self):
        """Evaluate all sequences"""
        all_results = []
        weather_results = defaultdict(list)

        total_sequences = sum(len(seq['splits']) for seq in self.dataset.sequences)
        if self.max_sequences:
            total_sequences = min(total_sequences, self.max_sequences)

        processed_count = 0
        pbar = tqdm(total=total_sequences, desc="Processing split sequences")

        for sequence in self.dataset.sequences:
            if self.max_sequences and processed_count >= self.max_sequences:
                break

            for split_info in sequence['splits']:
                if self.max_sequences and processed_count >= self.max_sequences:
                    break

                split_results = self.process_split_sequence(sequence, split_info)

                if split_results:
                    all_results.extend(split_results)

                    # Aggregate by weather
                    for result in split_results:
                        weather_results[result['weather']].append(result['iou'])

                processed_count += 1
                pbar.update(1)

        pbar.close()

        return all_results, weather_results

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                **vars(args),
                'dataset': 'ACDC_MultiObject',
                'data_structure': 'acdc_complete_split_30_object'
            }
        )

    # Initialize dataset
    print("Initializing ACDC Multi-Object Dataset...")
    dataset = ACDCMultiObjectDataset(
        data_dir=args.data_dir,
        weather_conditions=args.weather_conditions,
        gt_data_dir=args.gt_data_dir
    )

    # Build SAM2 video predictor
    print(f"Loading SAM2 model...")
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device
    )
    print("[OK] SAM2 video predictor loaded successfully")

    # Create evaluator
    evaluator = ACDCMultiObjectEvaluator(
        predictor=predictor,
        dataset=dataset,
        device=device,
        use_wandb=args.use_wandb,
        image_size=args.image_size,
        max_sequences=args.max_sequences,
        prompts_file=args.prompts_file,
        save_qual_dir=args.save_qual_dir
    )

    # Run evaluation
    print("\nStarting evaluation...")
    all_results, weather_results = evaluator.evaluate()

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS - ACDC Multi-Object Dataset")
    print("="*60)

    # Calculate object-wise performance per weather
    from collections import defaultdict as dd
    weather_object_wise = dd(lambda: dd(list))
    for result in all_results:
        weather = result['weather']
        class_key = f"{result['target_class_id']}_{result['target_class_name']}"
        weather_object_wise[weather][class_key].append(result['iou'])

    overall_ious = []
    for weather in args.weather_conditions:
        if weather in weather_object_wise and weather_object_wise[weather]:
            # Calculate object-wise mean IoU for this weather
            class_mean_ious = []
            for class_key, ious in weather_object_wise[weather].items():
                class_mean_ious.append(np.mean(ious))

            object_wise_mean = np.mean(class_mean_ious)
            object_wise_std = np.std(class_mean_ious)

            # Also get image-wise metrics (all object instances)
            all_object_ious = weather_results[weather] if weather in weather_results else []

            print(f"\n{weather.upper()} Condition:")
            print(f"  Object-wise Mean IoU: {object_wise_mean:.4f} ± {object_wise_std:.4f}")
            print(f"  Number of classes: {len(class_mean_ious)}")
            print(f"  Total evaluated object instances: {len(all_object_ious)}")

            overall_ious.extend(all_object_ious)

    if overall_ious:
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Mean IoU: {np.mean(overall_ious):.4f} ± {np.std(overall_ious):.4f}")
        print(f"  Median IoU: {np.median(overall_ious):.4f}")
        print(f"  Total evaluated frames: {len(overall_ious)}")

    # Collect all metrics for overall statistics
    overall_f_measures = [r['f_measure'] for r in all_results if 'f_measure' in r]
    overall_dices = [r['dice'] for r in all_results if 'dice' in r]
    overall_j_and_f = [(r['iou'] + r['f_measure']) / 2.0 for r in all_results if 'f_measure' in r]

    # Log to wandb
    if args.use_wandb and overall_ious:
        # Log overall metrics (J, F, J&F, Dice)
        wandb.log({
            'overall/J': np.mean(overall_ious),
            'overall/F': np.mean(overall_f_measures) if overall_f_measures else 0.0,
            'overall/J&F': np.mean(overall_j_and_f) if overall_j_and_f else 0.0,
            'overall/Dice': np.mean(overall_dices) if overall_dices else 0.0,
            'overall/mean_iou': np.mean(overall_ious),
            'overall/std_iou': np.std(overall_ious),
            'overall/median_iou': np.median(overall_ious),
            'overall/min_iou': np.min(overall_ious),
            'overall/max_iou': np.max(overall_ious),
            'overall/num_frames': len(overall_ious)
        })

        # Calculate and log object-wise and image-wise IoU metrics
        from collections import defaultdict

        # Object-wise metrics: Group by class and calculate mean per class
        class_ious = defaultdict(list)
        class_f_measures = defaultdict(list)
        class_dices = defaultdict(list)
        class_j_and_fs = defaultdict(list)

        for result in all_results:
            class_key = f"{result['target_class_id']}_{result['target_class_name']}"
            class_ious[class_key].append(result['iou'])
            if 'f_measure' in result:
                class_f_measures[class_key].append(result['f_measure'])
            if 'dice' in result:
                class_dices[class_key].append(result['dice'])
            if 'f_measure' in result:
                class_j_and_fs[class_key].append((result['iou'] + result['f_measure']) / 2.0)

        # Calculate mean metrics per class
        class_mean_ious = []
        class_mean_fs = []
        class_mean_dices = []
        class_mean_j_and_fs = []
        wandb_class_metrics = {}

        for class_key, ious in class_ious.items():
            class_mean_iou = np.mean(ious)
            class_mean_ious.append(class_mean_iou)

            # Log per-class J (IoU)
            wandb_class_metrics[f'class/{class_key}/mean_j'] = class_mean_iou
            wandb_class_metrics[f'class/{class_key}/std_j'] = np.std(ious)
            wandb_class_metrics[f'class/{class_key}/num_frames'] = len(ious)

            # F-measure
            if class_key in class_f_measures and class_f_measures[class_key]:
                class_mean_f = np.mean(class_f_measures[class_key])
                class_mean_fs.append(class_mean_f)
                wandb_class_metrics[f'class/{class_key}/mean_f'] = class_mean_f
                wandb_class_metrics[f'class/{class_key}/std_f'] = np.std(class_f_measures[class_key])

            # Dice
            if class_key in class_dices and class_dices[class_key]:
                class_mean_dice = np.mean(class_dices[class_key])
                class_mean_dices.append(class_mean_dice)
                wandb_class_metrics[f'class/{class_key}/mean_dice'] = class_mean_dice
                wandb_class_metrics[f'class/{class_key}/std_dice'] = np.std(class_dices[class_key])

            # J&F
            if class_key in class_j_and_fs and class_j_and_fs[class_key]:
                class_mean_jnf = np.mean(class_j_and_fs[class_key])
                class_mean_j_and_fs.append(class_mean_jnf)
                wandb_class_metrics[f'class/{class_key}/mean_j_and_f'] = class_mean_jnf
                wandb_class_metrics[f'class/{class_key}/std_j_and_f'] = np.std(class_j_and_fs[class_key])

        # Object-wise mean metrics (mean of class means)
        object_wise_mean_iou = np.mean(class_mean_ious)
        object_wise_mean_j = object_wise_mean_iou  # J = IoU
        object_wise_mean_f = np.mean(class_mean_fs) if class_mean_fs else 0.0
        object_wise_mean_j_and_f = np.mean(class_mean_j_and_fs) if class_mean_j_and_fs else 0.0
        object_wise_mean_dice = np.mean(class_mean_dices) if class_mean_dices else 0.0

        # Image-wise IoU: Group by frame and calculate mean IoU per frame
        frame_ious = defaultdict(list)
        for result in all_results:
            frame_key = f"{result['weather']}_{result['video_name']}_{result['split_id']}_frame{result['frame_num']}"
            frame_ious[frame_key].append(result['iou'])

        # Calculate mean IoU per frame
        frame_mean_ious = []
        for frame_key, ious in frame_ious.items():
            frame_mean_iou = np.mean(ious)
            frame_mean_ious.append(frame_mean_iou)

        # Image-wise mean IoU (mean of frame means)
        image_wise_mean_iou = np.mean(frame_mean_ious)

        # Weather-wise image-wise IoU: Group frames by weather
        weather_frame_ious = defaultdict(list)
        for frame_key, ious in frame_ious.items():
            weather = frame_key.split('_')[0]  # Extract weather from frame_key
            frame_mean_iou = np.mean(ious)
            weather_frame_ious[weather].append(frame_mean_iou)

        # Calculate weather-wise image-wise mean IoU
        weather_image_wise_metrics = {}
        for weather, frame_mean_ious_weather in weather_frame_ious.items():
            weather_image_wise_mean_iou = np.mean(frame_mean_ious_weather)
            weather_image_wise_metrics[f'{weather}/image_wise_mean_iou'] = weather_image_wise_mean_iou
            weather_image_wise_metrics[f'{weather}/image_wise_std_iou'] = np.std(frame_mean_ious_weather)
            weather_image_wise_metrics[f'{weather}/image_wise_num_frames'] = len(frame_mean_ious_weather)

        # Weather-wise object-wise IoU: Group classes by weather
        weather_object_wise_metrics = {}
        for weather, class_ious_dict in weather_object_wise.items():
            # Calculate mean IoU per class for this weather
            weather_class_mean_ious = []
            for class_key, ious in class_ious_dict.items():
                weather_class_mean_ious.append(np.mean(ious))

            # Object-wise mean for this weather (mean of class means)
            weather_object_wise_mean = np.mean(weather_class_mean_ious)
            weather_object_wise_metrics[f'{weather}/object_wise_mean_iou'] = weather_object_wise_mean
            weather_object_wise_metrics[f'{weather}/object_wise_std_iou'] = np.std(weather_class_mean_ious)
            weather_object_wise_metrics[f'{weather}/object_wise_num_classes'] = len(weather_class_mean_ious)

        # Log object-wise and image-wise metrics
        wandb.log({
            # Object-wise (class-wise) metrics
            'metrics/object_wise_mean_iou': object_wise_mean_iou,
            'metrics/object_wise_mean_j': object_wise_mean_j,
            'metrics/object_wise_mean_f': object_wise_mean_f,
            'metrics/object_wise_mean_j_and_f': object_wise_mean_j_and_f,
            'metrics/object_wise_mean_dice': object_wise_mean_dice,
            'metrics/object_wise_std_iou': np.std(class_mean_ious),
            'metrics/object_wise_std_j': np.std(class_mean_ious),
            'metrics/object_wise_std_f': np.std(class_mean_fs) if class_mean_fs else 0.0,
            'metrics/object_wise_std_j_and_f': np.std(class_mean_j_and_fs) if class_mean_j_and_fs else 0.0,
            'metrics/object_wise_std_dice': np.std(class_mean_dices) if class_mean_dices else 0.0,
            # Image-wise metrics
            'metrics/image_wise_mean_iou': image_wise_mean_iou,
            'metrics/image_wise_std_iou': np.std(frame_mean_ious),
            # Counts
            'metrics/num_classes': len(class_mean_ious),
            'metrics/num_unique_frames': len(frame_mean_ious),
            # Per-class and per-weather metrics
            **wandb_class_metrics,
            **weather_image_wise_metrics,
            **weather_object_wise_metrics
        })

        print(f"\nENHANCED METRICS:")
        print(f"  Object-wise Mean J (IoU): {object_wise_mean_j:.4f} ± {np.std(class_mean_ious):.4f}")
        print(f"  Object-wise Mean F:       {object_wise_mean_f:.4f} ± {np.std(class_mean_fs) if class_mean_fs else 0.0:.4f}")
        print(f"  Object-wise Mean J&F:     {object_wise_mean_j_and_f:.4f} ± {np.std(class_mean_j_and_fs) if class_mean_j_and_fs else 0.0:.4f}")
        print(f"  Object-wise Mean Dice:    {object_wise_mean_dice:.4f} ± {np.std(class_mean_dices) if class_mean_dices else 0.0:.4f}")
        print(f"  Image-wise Mean IoU:      {image_wise_mean_iou:.4f} ± {np.std(frame_mean_ious):.4f}")
        print(f"  Number of classes:        {len(class_mean_ious)}")
        print(f"  Number of frames:         {len(frame_mean_ious)}")
        print(f"\n🌤️  Weather-wise Object-wise IoU:")
        for weather, class_ious_dict in weather_object_wise.items():
            weather_class_mean_ious = [np.mean(ious) for ious in class_ious_dict.values()]
            weather_object_wise_mean = np.mean(weather_class_mean_ious)
            weather_object_wise_std = np.std(weather_class_mean_ious)
            print(f"    {weather.capitalize()}: {weather_object_wise_mean:.4f} ± {weather_object_wise_std:.4f} ({len(weather_class_mean_ious)} classes)")

        # Log per-weather metrics
        for weather in weather_results:
            if weather_results[weather]:
                wandb.log({
                    f'{weather}/mean_iou': np.mean(weather_results[weather]),
                    f'{weather}/std_iou': np.std(weather_results[weather]),
                    f'{weather}/median_iou': np.median(weather_results[weather]),
                    f'{weather}/min_iou': np.min(weather_results[weather]),
                    f'{weather}/max_iou': np.max(weather_results[weather]),
                    f'{weather}/num_frames': len(weather_results[weather])
                })

        # Create wandb table for detailed results
        table_data = []
        for result in all_results[:500]:  # Log first 500 frames
            table_data.append([
                result['weather'],
                result['video_name'],
                result['split_id'],
                result['frame_num'],
                result['target_class_name'],
                result['iou']
            ])

        table = wandb.Table(
            columns=['Weather', 'Video', 'Split', 'Frame', 'Class', 'IoU'],
            data=table_data
        )
        wandb.log({'frame_results': table})

        # Create histogram of IoU values
        wandb.log({'iou_distribution': wandb.Histogram(overall_ious)})

    # Save results
    if args.save_results:
        # Clean results for JSON serialization
        clean_results = []
        for result in all_results:
            clean_result = {
                'weather': result['weather'],
                'video_name': result['video_name'],
                'split_id': result['split_id'],
                'frame_num': result['frame_num'],
                'frame_idx': result['frame_idx'],
                'iou': result['iou'],
                'f_measure': result.get('f_measure', 0.0),
                'dice': result.get('dice', 0.0),
                'target_class_id': int(result['target_class_id']),
                'target_class_name': result['target_class_name'],
                'pred_area': result['pred_area'],
                'gt_area': result['gt_area'],
                'intersection': result['intersection']
            }
            clean_results.append(clean_result)

        output_data = {
            'args': vars(args),
            'frame_results': clean_results,
            'weather_summary': {},
            'overall_summary': {}
        }

        for weather in weather_results:
            if weather_results[weather]:
                output_data['weather_summary'][weather] = {
                    'mean_iou': float(np.mean(weather_results[weather])),
                    'std_iou': float(np.std(weather_results[weather])),
                    'min_iou': float(np.min(weather_results[weather])),
                    'max_iou': float(np.max(weather_results[weather])),
                    'median_iou': float(np.median(weather_results[weather])),
                    'num_frames': len(weather_results[weather])
                }

        if overall_ious:
            output_data['overall_summary'] = {
                'mean_iou': float(np.mean(overall_ious)),
                'std_iou': float(np.std(overall_ious)),
                'min_iou': float(np.min(overall_ious)),
                'max_iou': float(np.max(overall_ious)),
                'median_iou': float(np.median(overall_ious)),
                'num_frames': len(overall_ious)
            }

        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to: {args.save_results}")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SAM2 on ACDC Multi-Object Dataset')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                       default=os.environ.get('MOGA_ACDC_ROOT', './data/acdc_complete_split_30_object'),
                        help='Root directory containing ACDC multi-object data')
    parser.add_argument('--gt_data_dir', type=str, default=None,
                        help='Root directory containing GT data (if different from data_dir, e.g., for restored images)')
    parser.add_argument('--weather_conditions', nargs='+',
                        default=['fog', 'snow', 'night', 'rain'],
                        help='Weather conditions to evaluate')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Resize images to this size (0 for original size)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of split sequences to process (for testing)')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to SAM2 config file')

    # Prompts arguments
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='Path to JSON file with pre-generated prompts')

    # Output arguments
    parser.add_argument('--save_results', type=str,
                        default='acdc_multiobject_eval_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--save_qual_dir', type=str, default=None,
                        help='Directory to save qualitative visualizations (prompts and predictions)')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='acdc-multiobject-eval',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='sam2_acdc_multiobject_eval',
                        help='Wandb run name')

    args = parser.parse_args()
    main(args)