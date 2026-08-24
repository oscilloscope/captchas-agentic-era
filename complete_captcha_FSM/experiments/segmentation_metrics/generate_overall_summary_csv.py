"""
Generate overall model summary CSV (Table 4 results)
Combines results from all model summary JSON files into a single CSV
"""
import json
import csv

def main():
    # Load all summary files
    with open('yolov8n_dual_summary.json') as f:
        nano = json.load(f)
    with open('yolov8m_dual_summary.json') as f:
        medium = json.load(f)
    with open('yolov8x_dual_summary.json') as f:
        xlarge = json.load(f)
    with open('llm_unified_summary.json') as f:
        vlm = json.load(f)
    with open('hybrid_metrics.json') as f:
        hybrid = json.load(f)

    # Compile all results
    results = [
        {
            'Model': 'YOLOv8n-BBox',
            'Accuracy': nano['bbox_metrics']['accuracy'],
            'Precision': nano['bbox_metrics']['precision'],
            'Recall': nano['bbox_metrics']['recall'],
            'F1': nano['bbox_metrics']['f1'],
            'TP': nano['bbox_metrics']['tp'],
            'FP': nano['bbox_metrics']['fp'],
            'FN': nano['bbox_metrics']['fn'],
            'TN': nano['bbox_metrics']['tn']
        },
        {
            'Model': 'YOLOv8n-Mask',
            'Accuracy': nano['mask_metrics']['accuracy'],
            'Precision': nano['mask_metrics']['precision'],
            'Recall': nano['mask_metrics']['recall'],
            'F1': nano['mask_metrics']['f1'],
            'TP': nano['mask_metrics']['tp'],
            'FP': nano['mask_metrics']['fp'],
            'FN': nano['mask_metrics']['fn'],
            'TN': nano['mask_metrics']['tn']
        },
        {
            'Model': 'YOLOv8m-BBox',
            'Accuracy': medium['bbox_metrics']['accuracy'],
            'Precision': medium['bbox_metrics']['precision'],
            'Recall': medium['bbox_metrics']['recall'],
            'F1': medium['bbox_metrics']['f1'],
            'TP': medium['bbox_metrics']['tp'],
            'FP': medium['bbox_metrics']['fp'],
            'FN': medium['bbox_metrics']['fn'],
            'TN': medium['bbox_metrics']['tn']
        },
        {
            'Model': 'YOLOv8m-Mask',
            'Accuracy': medium['mask_metrics']['accuracy'],
            'Precision': medium['mask_metrics']['precision'],
            'Recall': medium['mask_metrics']['recall'],
            'F1': medium['mask_metrics']['f1'],
            'TP': medium['mask_metrics']['tp'],
            'FP': medium['mask_metrics']['fp'],
            'FN': medium['mask_metrics']['fn'],
            'TN': medium['mask_metrics']['tn']
        },
        {
            'Model': 'YOLOv8x-BBox',
            'Accuracy': xlarge['bbox_metrics']['accuracy'],
            'Precision': xlarge['bbox_metrics']['precision'],
            'Recall': xlarge['bbox_metrics']['recall'],
            'F1': xlarge['bbox_metrics']['f1'],
            'TP': xlarge['bbox_metrics']['tp'],
            'FP': xlarge['bbox_metrics']['fp'],
            'FN': xlarge['bbox_metrics']['fn'],
            'TN': xlarge['bbox_metrics']['tn']
        },
        {
            'Model': 'YOLOv8x-Mask',
            'Accuracy': xlarge['mask_metrics']['accuracy'],
            'Precision': xlarge['mask_metrics']['precision'],
            'Recall': xlarge['mask_metrics']['recall'],
            'F1': xlarge['mask_metrics']['f1'],
            'TP': xlarge['mask_metrics']['tp'],
            'FP': xlarge['mask_metrics']['fp'],
            'FN': xlarge['mask_metrics']['fn'],
            'TN': xlarge['mask_metrics']['tn']
        },
        {
            'Model': 'VLM (BBox)',
            'Accuracy': vlm['overall_metrics']['accuracy'],
            'Precision': vlm['overall_metrics']['precision'],
            'Recall': vlm['overall_metrics']['recall'],
            'F1': vlm['overall_metrics']['f1'],
            'TP': vlm['overall_metrics']['tp'],
            'FP': vlm['overall_metrics']['fp'],
            'FN': vlm['overall_metrics']['fn'],
            'TN': vlm['overall_metrics']['tn']
        },
        {
            'Model': 'Hybrid',
            'Accuracy': hybrid['metrics']['accuracy'],
            'Precision': hybrid['metrics']['precision'],
            'Recall': hybrid['metrics']['recall'],
            'F1': hybrid['metrics']['f1'],
            'TP': hybrid['confusion_matrix']['tp'],
            'FP': hybrid['confusion_matrix']['fp'],
            'FN': hybrid['confusion_matrix']['fn'],
            'TN': hybrid['confusion_matrix']['tn']
        }
    ]

    # Write to CSV
    output_file = 'overall_model_summary.csv'
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
                      'TP', 'FP', 'FN', 'TN']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("="*80)
    print("OVERALL MODEL SUMMARY CSV GENERATION")
    print("="*80)
    print(f"\nCreated {output_file}")
    print(f"Contains {len(results)} model entries")
    print(f"Columns: {', '.join(fieldnames)}")
    print("\nMetrics included:")
    print("  - Accuracy, Precision, Recall, F1-Score")
    print("  - Confusion matrix values (TP, FP, FN, TN)")
    print("\n" + "="*80)
    print("SUMMARY TABLE:")
    print("="*80)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-"*80)

    for row in results:
        print(f"{row['Model']:<20} {row['Accuracy']:<12.3f} {row['Precision']:<12.3f} {row['Recall']:<12.3f} {row['F1']:<10.3f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
