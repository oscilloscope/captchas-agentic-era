"""
Calculate hybrid metrics combining YOLOv8x-Mask and VLM.
Uses YOLOv8x-Mask for: bicycle, bus, fire hydrant, motorcycle, traffic light
Uses VLM for: crosswalk, stairs, taxi
"""
import json

# Load per-class results
with open('per_class_metrics_conf0_10.json') as f:
    yolo_data = json.load(f)

with open('llm_per_class_metrics.json') as f:
    vlm_data = json.load(f)

yolo_x_mask = yolo_data['yolov8x']
vlm_results = vlm_data['vlm']

# Define which model to use for each class
model_selection = {
    'bicycle': 'yolo',
    'bus': 'yolo',
    'fire hydrant': 'yolo',
    'motorcycle': 'yolo',
    'traffic light': 'yolo',
    'crosswalk': 'vlm',
    'stairs': 'vlm',
    'taxi': 'vlm'
}

print("="*100)
print("HYBRID MODEL CALCULATION")
print("="*100)
print()
print("Model Selection:")
print("-"*100)
for cls, model in sorted(model_selection.items()):
    source = "YOLOv8x-Mask" if model == 'yolo' else "VLM"
    print(f"  {cls:<20} -> {source}")

print()
print("="*100)
print("AGGREGATING CONFUSION MATRICES")
print("="*100)
print()

# Aggregate confusion matrices
total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

print(f"{'Class':<20} {'Source':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'TN':<8}")
print("-"*100)

for cls in sorted(model_selection.keys()):
    if model_selection[cls] == 'yolo':
        metrics = yolo_x_mask.get(cls, {}).get('mask', {})
        source = "YOLOv8x-Mask"
    else:
        metrics = vlm_results.get(cls, {})
        source = "VLM"
    
    tp = metrics.get('tp', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    tn = metrics.get('tn', 0)
    
    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn
    
    print(f"{cls:<20} {source:<15} {tp:<8} {fp:<8} {fn:<8} {tn:<8}")

print("-"*100)
print(f"{'TOTAL':<20} {'':<15} {total_tp:<8} {total_fp:<8} {total_fn:<8} {total_tn:<8}")

# Calculate metrics
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print()
print("="*100)
print("HYBRID MODEL METRICS")
print("="*100)
print()
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print()

# Save results
hybrid_metrics = {
    'model': 'Hybrid (YOLOv8x-Mask + VLM)',
    'model_selection': model_selection,
    'confusion_matrix': {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'tn': total_tn
    },
    'metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
}

with open('hybrid_metrics.json', 'w') as f:
    json.dump(hybrid_metrics, f, indent=2)

print("Results saved to hybrid_metrics.json")
print("="*100)
