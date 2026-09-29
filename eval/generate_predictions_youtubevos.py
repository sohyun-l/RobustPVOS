#!/usr/bin/env python3
"""
Generate multi-object segmentation predictions on the YouTube-VOS valid split
(clean or a YouTube-VOS-C corruption), following the training protocol
(max 3 objects per sequence).
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import json

from sam2.build_sam import build_sam2_video_predictor

def get_unique_objects(mask_path):
    """Return all object ids present in a mask (background excluded)."""
    try:
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return []
            
        mask = np.array(Image.open(mask_path).convert('L'))
        
        unique_vals = np.unique(mask)
        object_ids = [val for val in unique_vals if val > 0]
        
        return object_ids
        
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return []

def get_first_frame_points_multiobject(mask_path, object_ids, num_points=3):
    """Sample evenly distributed point prompts per object from the first-frame mask."""
    try:
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return {}

        mask = np.array(Image.open(mask_path).convert('L'))

        object_points = {}

        for obj_id in object_ids:
            obj_mask = (mask == obj_id)

            if np.sum(obj_mask) == 0:
                print(f"Empty mask for object {obj_id}: {mask_path}")
                continue

            y, x = np.where(obj_mask)

            if len(x) == 0:
                print(f"No valid mask points for object {obj_id}: {mask_path}")
                continue

            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)

            points = np.array([
                [min_x + (max_x - min_x) * 0.5, min_y + (max_y - min_y) * 0.5],
                [min_x + (max_x - min_x) * 0.25, min_y + (max_y - min_y) * 0.25],
                [min_x + (max_x - min_x) * 0.75, min_y + (max_y - min_y) * 0.75],
            ], dtype=np.float32)

            object_points[obj_id] = points

        return object_points

    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return {}

def load_prompts_from_json(prompts_file):
    """Load prompt information from a JSON file."""
    with open(prompts_file, 'r') as f:
        json_dict = json.load(f)

    prompts_dict = {}
    for video_id, object_prompts in json_dict.items():
        prompts_dict[video_id] = {}
        for obj_id, points in object_prompts.items():
            prompts_dict[video_id][int(obj_id)] = np.array(points, dtype=np.float32)

    print(f"Prompts loaded from {prompts_file}")
    return prompts_dict

def process_video_multiobject(video_name, video_info, corrupted_frames_root, gt_annotations_root, pred_root, predictor,
                              max_num_objects=3, video_prompts=None, prompt_type='point'):
    """Generate multi-object predictions for one YouTube-VOS video (same protocol as training)."""
    print(f"\nProcessing video: {video_name}")
    
    video_frames_dir = os.path.join(corrupted_frames_root, video_name)
    if not os.path.exists(video_frames_dir):
        print(f"Frame directory not found: {video_frames_dir}")
        return
    
    gt_video_path = Path(gt_annotations_root) / video_name
    if not gt_video_path.exists():
        print(f"GT annotation path not found: {gt_video_path}")
        return
    
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print(f"No frames found: {video_frames_dir}")
        return
    
    first_frame_file = frame_files[0]
    first_frame_id = first_frame_file.split('.')[0]
    first_annotation_path = gt_video_path / f"{first_frame_id}.png"
    
    if not first_annotation_path.exists():
        print(f"First-frame annotation not found: {first_annotation_path}")
        return
    
    all_object_ids = get_unique_objects(str(first_annotation_path))
    if not all_object_ids:
        print(f"No objects in video {video_name}.")
        return
    
    if len(all_object_ids) > max_num_objects:
        selected_objects = sorted(all_object_ids)[:max_num_objects]
    else:
        selected_objects = all_object_ids
    
    print(f"Objects: {len(all_object_ids)}, selected: {selected_objects}")

    if video_prompts is not None and video_name in video_prompts:
        object_points = video_prompts[video_name]
        print(f"Loaded shared prompts for {len(object_points)} objects")
    else:
        object_points = get_first_frame_points_multiobject(str(first_annotation_path), selected_objects)
        if not object_points:
            print(f"No valid object points in video {video_name}.")
            return

    for obj_id, points in object_points.items():
        if prompt_type == 'point':
            print(f"Object {obj_id}: {len(points)} point(s)")
        else:
            print(f"Object {obj_id}: box {points}")
    
    inference_state = predictor.init_state(video_path=video_frames_dir)
    
    frames = []
    frame_names = []
    
    print("Loading frames...")
    for frame_file in tqdm(frame_files, desc="frame loading (JPEG)"):
        frame_path = os.path.join(video_frames_dir, frame_file)
        frame = Image.open(frame_path).convert('RGB')
        frames.append(np.array(frame))
        frame_names.append(frame_file.split('.')[0])
    
    for obj_id, points in object_points.items():
        if prompt_type == 'point':
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(obj_id),
                points=points,
                labels=np.array([1, 1, 1])
            )
        elif prompt_type == 'box':
            box_array = np.array(points, dtype=np.float32)  # [x_min, y_min, x_max, y_max]
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(obj_id),
                box=box_array
            )
    
    print("Propagating through the video...")
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    for obj_id in selected_objects:
        obj_pred_dir = os.path.join(pred_root, video_name, str(obj_id))
        os.makedirs(obj_pred_dir, exist_ok=True)
        
        for frame_idx, frame_name in enumerate(frame_names):
            if frame_idx in video_segments and int(obj_id) in video_segments[frame_idx]:
                mask = video_segments[frame_idx][int(obj_id)]
                
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                
                mask_img = (mask * 255).astype(np.uint8)
                
                mask_pil = Image.fromarray(mask_img, mode='L')
                mask_pil.save(os.path.join(obj_pred_dir, f"{frame_name}.png"))
            else:
                empty_mask = np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8)
                mask_pil = Image.fromarray(empty_mask, mode='L')
                mask_pil.save(os.path.join(obj_pred_dir, f"{frame_name}.png"))
    
    print(f"Done: {video_name}")

def main():
    parser = argparse.ArgumentParser(description='Multi-object prediction on YouTube-VOS valid')
    parser.add_argument('--corruption_type', type=str, required=True,
                       choices=['clean', 'snow', 'fog', 'rain', 'gauss_noise', 'resampling_blur',
                               'color_jitter', 'ISO_noise', 'motion_blur'],
                       help='corruption type')
    parser.add_argument('--pred_root', type=str, required=True,
                       help='root directory to write predictions to')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (.pt)')
    parser.add_argument('--config_path', type=str,
                       default='configs/sam2/sam2_hiera_b+.yaml',
                       help='model config path')
    parser.add_argument('--max_num_objects', type=int, default=3,
                       help='max number of objects per video (training protocol)')
    parser.add_argument('--wandb_project', type=str, default="RobustSAM2",
                       help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default="YouTubeVOS_multiobject_prediction",
                       help='WandB run name')
    parser.add_argument('--data_base_path', type=str, default=None,
                       help='Custom base path that already contains '
                            '<corruption_type>/{JPEGImages,Annotations,meta.json} '
                            '(used for restoration-baseline outputs).')
    parser.add_argument('--youtubevos_clean_root', type=str,
                       default=os.environ.get('MOGA_YTVOS_CLEAN_ROOT'),
                       help='Root of clean YouTube-VOS valid split '
                            '(expects valid/{JPEGImages,Annotations,meta.json}). '
                            'Falls back to $MOGA_YTVOS_CLEAN_ROOT.')
    parser.add_argument('--youtubevos_corrupted_root', type=str,
                       default=os.environ.get('MOGA_YTVOS_CORRUPTED_ROOT'),
                       help='Root of YouTube-VOS-C valid split '
                            '(expects valid/<corruption>/{JPEGImages,meta.json}). '
                            'Falls back to $MOGA_YTVOS_CORRUPTED_ROOT.')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='Path to prompt JSON file (optional)')
    parser.add_argument('--prompt_type', type=str, default='point', choices=['point', 'box'],
                       help='Type of prompt to use: point or box')

    args = parser.parse_args()

    # Input paths
    if args.data_base_path:
        # Custom base path (restoration baselines write into this layout).
        corrupted_data_root = f"{args.data_base_path}/{args.corruption_type}/JPEGImages"
        meta_json_path = f"{args.data_base_path}/{args.corruption_type}/meta.json"
        gt_annotations_root = f"{args.data_base_path}/{args.corruption_type}/Annotations"
    elif args.corruption_type == "clean":
        if not args.youtubevos_clean_root:
            print("Error: --youtubevos_clean_root (or $MOGA_YTVOS_CLEAN_ROOT) must be set for corruption_type='clean'.")
            return
        clean_root = args.youtubevos_clean_root.rstrip('/')
        corrupted_data_root = f"{clean_root}/valid/JPEGImages"
        meta_json_path = f"{clean_root}/valid/meta.json"
        gt_annotations_root = f"{clean_root}/valid/Annotations"
    else:
        if not args.youtubevos_corrupted_root or not args.youtubevos_clean_root:
            print("Error: --youtubevos_corrupted_root and --youtubevos_clean_root "
                  "(or their $MOGA_YTVOS_*_ROOT env vars) must be set for corrupted eval.")
            return
        corrupted_root = args.youtubevos_corrupted_root.rstrip('/')
        clean_root = args.youtubevos_clean_root.rstrip('/')
        corrupted_data_root = f"{corrupted_root}/valid/{args.corruption_type}/JPEGImages"
        meta_json_path = f"{corrupted_root}/valid/{args.corruption_type}/meta.json"
        # Annotations are always the clean ones (corruption only affects frames).
        gt_annotations_root = f"{clean_root}/valid/Annotations"
    
    print(f"Corruption Type: {args.corruption_type}")
    print(f"Corrupted Data: {corrupted_data_root}")
    print(f"GT Annotations: {gt_annotations_root}")
    print(f"Meta JSON: {meta_json_path}")
    print(f"Output Directory: {args.pred_root}")
    print(f"Model Checkpoint: {args.checkpoint_path}")
    print(f"Max Objects: {args.max_num_objects}")
    
    if not os.path.exists(corrupted_data_root):
        print(f"Error: input data directory not found: {corrupted_data_root}")
        return
    
    if not os.path.exists(gt_annotations_root):
        print(f"Error: GT annotation directory not found: {gt_annotations_root}")
        return
        
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: checkpoint not found: {args.checkpoint_path}")
        return
        
    if not os.path.exists(meta_json_path):
        print(f"Error: meta.json not found: {meta_json_path}")
        return
    
    os.makedirs(args.pred_root, exist_ok=True)
    
    all_prompts = None
    if args.prompt_file is not None:
        print(f"Loading prompts from {args.prompt_file}...")
        all_prompts = load_prompts_from_json(args.prompt_file)
        print(f"Loaded prompts for {len(all_prompts)} videos")

    import wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_name}_{args.corruption_type}",
            config={
                "corruption_type": args.corruption_type,
                "checkpoint": args.checkpoint_path,
                "config": args.config_path,
                "max_num_objects": args.max_num_objects,
                "evaluation_type": "multi_object",
                "prompt_file": args.prompt_file,
                "prompt_type": args.prompt_type
            }
        )
        print("WandB initialized")
    except Exception as e:
        print(f"WandB init failed: {e}")
    
    # load meta.json
    with open(meta_json_path, 'r') as f:
        meta_data = json.load(f)
    
    videos_info = meta_data['videos']
    print(f"Videos loaded: {len(videos_info)}")
    
    print("Loading SAM 2...")
    try:
        predictor = build_sam2_video_predictor(args.config_path, args.checkpoint_path)
        print("Model loaded")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    video_names = list(videos_info.keys())

    for video_name in tqdm(video_names, desc="Processing multi-object videos"):
        video_info = videos_info[video_name]
        process_video_multiobject(
            video_name,
            video_info,
            corrupted_data_root,
            gt_annotations_root,
            args.pred_root,
            predictor,
            max_num_objects=args.max_num_objects,
            video_prompts=all_prompts,
            prompt_type=args.prompt_type
        )
    
    print("=== YouTube-VOS multi-object prediction done ===")

if __name__ == "__main__":
    main()