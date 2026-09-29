#!/usr/bin/env python3
"""
Unified MVSeg evaluation with consistent metrics.
Uses the same aggregation methods as eval_acdc_unified.py for fair comparison.

Computes: Flat, Image-Level, VOS-Style, Class-Wise (per sequence-object)
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_mvseg import MVSegMultiObjectEvaluator
from unified_metrics import compute_unified_metrics, format_metrics_table, get_wandb_metrics


def normalize_mvseg_results(results):
    """Convert MVSeg raw results to unified format."""
    unified = []
    for r in results:
        unified.append({
            'sequence_key': r['sequence_name'],
            'object_key': r['object_id'],
            'frame_key': r['frame_idx'],
            'j': r['iou'],
            'f': r['f'],
            'class_key': f"{r['sequence_name']}_obj{r['object_id']}",
            'group_key': r['dataset'],
            # Preserve original fields for saving
            'sequence_name': r['sequence_name'],
            'dataset': r['dataset'],
            'frame_idx': r['frame_idx'],
            'object_id': r['object_id'],
            'iou': r['iou'],
            'dice': r.get('dice', 0.0),
            'precision': r.get('precision', 0.0),
            'recall': r.get('recall', 0.0),
            'gt_area': r.get('gt_area', 0),
            'pred_area': r.get('pred_area', 0),
        })
    return unified


def main():
    parser = argparse.ArgumentParser(description='Unified MVSeg Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annotated_dataset', type=str,
                       default=os.environ.get('MOGA_MVSEG_ROOT', './data/MVSeg_Dataset_annotated'))
    parser.add_argument('--datasets', nargs='+', default=['INO', 'KAIST', 'OSU', 'RGBT234'])
    parser.add_argument('--prompts_file', type=str, default=None)
    parser.add_argument('--max_objects', type=int, default=3)
    parser.add_argument('--max_sequences', type=int, default=None)
    parser.add_argument('--save_results', type=str, default=None)
    parser.add_argument('--save_qual_dir', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='unified-segmentation-eval')
    parser.add_argument('--wandb_name', type=str, default='mvseg_unified_eval')
    args = parser.parse_args()

    # Initialize wandb before evaluator (evaluator's internal wandb is disabled)
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                **vars(args),
                'dataset': 'MVSeg',
                'eval_type': 'unified',
            }
        )

    print("=" * 60)
    print("  Unified MVSeg Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Datasets: {args.datasets}")
    print(f"Prompts file: {args.prompts_file}")

    # Create evaluator with internal wandb disabled
    evaluator = MVSegMultiObjectEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        annotated_dataset_path=args.annotated_dataset,
        prompts_file=args.prompts_file,
        max_objects=args.max_objects,
        use_wandb=False,
        wandb_name=args.wandb_name,
        save_qual_dir=args.save_qual_dir,
    )

    # Run evaluation - get raw results
    eval_output = evaluator.evaluate_dataset(
        datasets=args.datasets,
        max_sequences=args.max_sequences,
    )

    if not eval_output or not eval_output.get('detailed_results'):
        print("No results obtained.")
        return

    raw_results = eval_output['detailed_results']
    print(f"\nTotal raw results: {len(raw_results)}")

    # Normalize and compute unified metrics
    unified_results = normalize_mvseg_results(raw_results)
    metrics = compute_unified_metrics(unified_results)

    # Print results
    print(format_metrics_table(metrics, title="MVSeg Unified Evaluation"))

    # Log to wandb
    if args.use_wandb:
        wandb_metrics = get_wandb_metrics(metrics)
        wandb.log(wandb_metrics)

        # Log comparison-friendly aliases (same names as ACDC unified)
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
