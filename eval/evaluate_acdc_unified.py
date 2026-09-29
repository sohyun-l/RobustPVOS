#!/usr/bin/env python3
"""
Unified ACDC evaluation with consistent metrics.
Uses the same aggregation methods as eval_mvseg_unified.py for fair comparison.

Computes: Flat, Image-Level, VOS-Style, Class-Wise (per semantic class)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2_video_predictor
from evaluate_acdc import (
    ACDCMultiObjectDataset,
    ACDCMultiObjectEvaluator,
)
from unified_metrics import compute_unified_metrics, format_metrics_table, get_wandb_metrics


def normalize_acdc_results(results):
    """Convert ACDC raw results to unified format."""
    unified = []
    for r in results:
        # sequence_key = weather/video/split (each split is an independent tracking sequence)
        seq_key = f"{r['weather']}_{r['video_name']}_{r['split_id']}"
        # class_key = semantic class
        class_key = f"{r['target_class_id']}_{r['target_class_name']}"

        unified.append({
            'sequence_key': seq_key,
            'object_key': r['object_id'],
            'frame_key': r['frame_num'],
            'j': r['iou'],
            'f': r.get('f_measure', 0.0),
            'class_key': class_key,
            'group_key': r['weather'],
            # Preserve original fields for saving
            'weather': r['weather'],
            'video_name': r['video_name'],
            'split_id': r['split_id'],
            'frame_num': r['frame_num'],
            'object_id': r['object_id'],
            'target_class_id': r.get('target_class_id'),
            'target_class_name': r.get('target_class_name', ''),
            'iou': r['iou'],
            'f_measure': r.get('f_measure', 0.0),
            'dice': r.get('dice', 0.0),
            'pred_area': r.get('pred_area', 0),
            'gt_area': r.get('gt_area', 0),
            'intersection': r.get('intersection', 0),
        })
    return unified


def main():
    parser = argparse.ArgumentParser(description='Unified ACDC Evaluation')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                       default=os.environ.get('MOGA_ACDC_ROOT', './data/acdc_complete_split_30_object'))
    parser.add_argument('--gt_data_dir', type=str, default=None)
    parser.add_argument('--weather_conditions', nargs='+', default=['fog', 'snow', 'night', 'rain'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--max_sequences', type=int, default=None)

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)

    # Prompts
    parser.add_argument('--prompts_file', type=str, default=None)

    # Output
    parser.add_argument('--save_results', type=str, default=None)
    parser.add_argument('--save_qual_dir', type=str, default=None)

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='unified-segmentation-eval')
    parser.add_argument('--wandb_name', type=str, default='acdc_unified_eval')

    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                **vars(args),
                'dataset': 'ACDC',
                'eval_type': 'unified',
            }
        )

    print("=" * 60)
    print("  Unified ACDC Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Weather conditions: {args.weather_conditions}")
    print(f"Prompts file: {args.prompts_file}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset
    dataset = ACDCMultiObjectDataset(
        data_dir=args.data_dir,
        weather_conditions=args.weather_conditions,
        gt_data_dir=args.gt_data_dir,
    )

    # Build SAM2 video predictor
    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device,
    )
    print("SAM2 video predictor loaded successfully")

    # Create evaluator with internal wandb disabled
    evaluator = ACDCMultiObjectEvaluator(
        predictor=predictor,
        dataset=dataset,
        device=device,
        use_wandb=False,
        image_size=args.image_size,
        max_sequences=args.max_sequences,
        prompts_file=args.prompts_file,
        save_qual_dir=args.save_qual_dir,
    )

    # Run evaluation
    print("\nStarting evaluation...")
    all_results, weather_results = evaluator.evaluate()

    if not all_results:
        print("No results obtained.")
        return

    print(f"\nTotal raw results: {len(all_results)}")

    # Normalize and compute unified metrics
    unified_results = normalize_acdc_results(all_results)
    metrics = compute_unified_metrics(unified_results)

    # Print results
    print(format_metrics_table(metrics, title="ACDC Unified Evaluation"))

    # Log to wandb
    if args.use_wandb:
        wandb_metrics = get_wandb_metrics(metrics)
        wandb.log(wandb_metrics)

        # Log comparison-friendly aliases (same names as MVSeg unified)
        wandb.log({
            'unified/flat/J&F': metrics['flat']['mean_j_and_f'],
            'unified/flat/J': metrics['flat']['mean_j'],
            'unified/flat/F': metrics['flat']['mean_f'],
            'unified/image_level/J&F': metrics['image_level']['mean_j_and_f'],
            'unified/image_level/J': metrics['image_level']['mean_j'],
            'unified/image_level/F': metrics['image_level']['mean_f'],
            'unified/vos_style/J&F': metrics['vos_style']['mean_j_and_f'],
            'unified/vos_style/J': metrics['vos_style']['mean_j'],
            'unified/vos_style/F': metrics['vos_style']['mean_f'],
        })

        if 'class_wise' in metrics:
            wandb.log({
                'unified/class_wise/J&F': metrics['class_wise']['mean_j_and_f'],
                'unified/class_wise/J': metrics['class_wise']['mean_j'],
                'unified/class_wise/F': metrics['class_wise']['mean_f'],
            })

        wandb.finish()

    # Save results
    if args.save_results:
        save_results = []
        for r in unified_results:
            save_r = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.float64, np.float32)):
                    save_r[k] = float(v)
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    save_r[k] = int(v)
                elif not isinstance(v, np.ndarray):
                    save_r[k] = v
            save_results.append(save_r)

        output = {
            'args': vars(args),
            'unified_metrics': metrics,
            'num_results': len(save_results),
            'results': save_results,
        }

        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    print("\nDone.")


if __name__ == '__main__':
    main()
