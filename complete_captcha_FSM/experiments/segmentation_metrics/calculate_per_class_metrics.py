"""
Calculate per-class metrics for bbox and mask modes from YOLO results
Analyzes F1, precision, recall, and accuracy for each target object class.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_metrics(tp, fp, fn, tn):
    """Calculate precision, recall, accuracy, F1 from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1
    }


def analyze_per_class_metrics(model_size, conf_threshold=0.25):
    """Analyze per-class metrics for a specific YOLO model with given threshold."""
    inference_dir = Path("segmentation_inference_data")

    # Collect results by class
    class_data = defaultdict(lambda: {
        'bbox': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'mask': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'count': 0
    })

    # Read all puzzle results
    threshold_str = f"{conf_threshold:.2f}".replace('.', '_')
    for puzzle_dir in inference_dir.iterdir():
        if not puzzle_dir.is_dir():
            continue

        result_file = puzzle_dir / f"yolov8{model_size}_dual_result_conf{threshold_str}.json"
        if not result_file.exists():
            continue

        with open(result_file, 'r') as f:
            result = json.load(f)

        target_obj = result['target_object']
        class_data[target_obj]['count'] += 1

        bbox_metrics = result['bbox_results']['metrics']
        class_data[target_obj]['bbox']['tp'] += bbox_metrics['tp']
        class_data[target_obj]['bbox']['fp'] += bbox_metrics['fp']
        class_data[target_obj]['bbox']['fn'] += bbox_metrics['fn']
        class_data[target_obj]['bbox']['tn'] += bbox_metrics['tn']

        mask_metrics = result['mask_results']['metrics']
        class_data[target_obj]['mask']['tp'] += mask_metrics['tp']
        class_data[target_obj]['mask']['fp'] += mask_metrics['fp']
        class_data[target_obj]['mask']['fn'] += mask_metrics['fn']
        class_data[target_obj]['mask']['tn'] += mask_metrics['tn']

    # Calculate metrics for each class
    results = {}
    for class_name, data in class_data.items():
        bbox_cm = data['bbox']
        mask_cm = data['mask']

        results[class_name] = {
            'count': data['count'],
            'bbox': calculate_metrics(bbox_cm['tp'], bbox_cm['fp'], bbox_cm['fn'], bbox_cm['tn']),
            'mask': calculate_metrics(mask_cm['tp'], mask_cm['fp'], mask_cm['fn'], mask_cm['tn'])
        }

    return results


def print_per_class_table(results, model_name, conf_threshold=0.25):
    """Print per-class metrics in a formatted table."""
    print(f"\n{'='*100}")
    print(f"PER-CLASS METRICS: {model_name} (conf={conf_threshold})")
    print(f"{'='*100}")

    # Sort classes by count (descending)
    sorted_classes = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)

    print("\n--- BOUNDING BOX MODE ---")
    print(f"{'Class':<20} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("-" * 100)
    for class_name, data in sorted_classes:
        m = data['bbox']
        print(f"{class_name:<20} {data['count']:<8} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['accuracy']:.4f}")

    print("\n--- MASK MODE ---")
    print(f"{'Class':<20} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("-" * 100)
    for class_name, data in sorted_classes:
        m = data['mask']
        print(f"{class_name:<20} {data['count']:<8} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['accuracy']:.4f}")

    print("\n--- MASK vs BBOX IMPROVEMENT ---")
    print(f"{'Class':<20} {'Count':<8} {'d_Precision':<12} {'d_Recall':<12} {'d_F1':<12} {'d_Accuracy':<12}")
    print("-" * 100)
    for class_name, data in sorted_classes:
        bbox_m = data['bbox']
        mask_m = data['mask']
        d_prec = mask_m['precision'] - bbox_m['precision']
        d_rec = mask_m['recall'] - bbox_m['recall']
        d_f1 = mask_m['f1'] - bbox_m['f1']
        d_acc = mask_m['accuracy'] - bbox_m['accuracy']
        print(f"{class_name:<20} {data['count']:<8} {d_prec:+.4f}       {d_rec:+.4f}       {d_f1:+.4f}       {d_acc:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Calculate per-class metrics from YOLO results')
    parser.add_argument('--threshold', '-t', type=float, default=0.25,
                        help='Confidence threshold used (default: 0.25)')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['n', 'm', 'x', 'all'],
                        help='Model size to analyze')
    args = parser.parse_args()

    print("="*100)
    print(f"PER-CLASS METRICS ANALYSIS (conf={args.threshold})")
    print("="*100)
    print("\nAnalyzing results from dual-mode YOLO tests...")

    models = ['n', 'm', 'x'] if args.model == 'all' else [args.model]
    all_results = {}

    for model_size in models:
        print(f"\n\nProcessing YOLOv8{model_size}-seg...")
        results = analyze_per_class_metrics(model_size, args.threshold)
        all_results[model_size] = results
        print_per_class_table(results, f"YOLOv8{model_size}-seg", args.threshold)

    # Save results to JSON
    output = {}
    for model_size, results in all_results.items():
        output[f"yolov8{model_size}"] = results

    threshold_str = f"{args.threshold:.2f}".replace('.', '_')
    output_file = f'per_class_metrics_conf{threshold_str}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*100)
    print(f"Per-class metrics saved to {output_file}")
    print("="*100)


if __name__ == "__main__":
    main()
