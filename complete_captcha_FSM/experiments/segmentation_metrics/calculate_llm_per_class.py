import json
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


def analyze_llm_per_class():
    """Analyze per-class metrics for LLM model."""
    inference_dir = Path("segmentation_inference_data")

    # Collect results by class
    class_data = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'count': 0
    })

    # Read all puzzle results
    for puzzle_dir in inference_dir.iterdir():
        if not puzzle_dir.is_dir():
            continue

        result_file = puzzle_dir / "llm_unified_result.json"
        if not result_file.exists():
            continue

        with open(result_file, 'r') as f:
            result = json.load(f)

        target_obj = result['target_object']
        class_data[target_obj]['count'] += 1

        metrics = result['metrics']
        class_data[target_obj]['tp'] += metrics['tp']
        class_data[target_obj]['fp'] += metrics['fp']
        class_data[target_obj]['fn'] += metrics['fn']
        class_data[target_obj]['tn'] += metrics['tn']

    # Calculate metrics for each class
    results = {}
    for class_name, data in class_data.items():
        results[class_name] = {
            'count': data['count'],
            **calculate_metrics(data['tp'], data['fp'], data['fn'], data['tn'])
        }

    return results


def main():
    print("="*100)
    print("LLM PER-CLASS METRICS ANALYSIS")
    print("="*100)
    print("\nAnalyzing results from LLM unified tests...")

    results = analyze_llm_per_class()

    # Sort classes by count
    sorted_classes = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"\n{'Class':<20} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("-" * 100)
    for class_name, data in sorted_classes:
        print(f"{class_name:<20} {data['count']:<8} {data['precision']:.4f}       "
              f"{data['recall']:.4f}       {data['f1']:.4f}       {data['accuracy']:.4f}")

    # Save results to JSON
    output = {'vlm': results}
    output_file = 'llm_per_class_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*100)
    print(f"Per-class metrics saved to {output_file}")
    print("="*100)


if __name__ == "__main__":
    main()
