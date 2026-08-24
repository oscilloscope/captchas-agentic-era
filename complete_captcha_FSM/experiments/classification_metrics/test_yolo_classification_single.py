import os
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
import time
import numpy as np
from ultralytics import YOLO

# Configuration
DATASET_PATH = "recaptcha-dataset-master"
MODEL_PATH = "classification_model.pt"
NUM_PER_CLASS = 999999  # Use all available samples
BATCH_SIZE = 100

# ALL 16 classes (including unsupported ones for YOLO)
# YOLO only supports 13, but we'll test all 16 to see misclassifications
YOLO_CLASSES = [
    "Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk",
    "Hydrant", "Motorcycle", "Mountain", "Other", "Palm", "Stairs", "Traffic Light",
    "Boat", "Taxi", "Tractor"  # These 3 are NOT supported by YOLO model!
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(YOLO_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(YOLO_CLASSES)}

def load_stratified_samples(num_per_class=100):
    """Load stratified samples: num_per_class images from each class"""
    samples = []

    for class_name in YOLO_CLASSES:
        class_samples = []

        # Collect from both Training and Validation
        for split in ['Training', 'Validation']:
            class_dir = Path(DATASET_PATH) / split / class_name
            if not class_dir.exists():
                continue

            for img_file in class_dir.iterdir():
                if img_file.is_file() and not img_file.name.startswith('._') and img_file.suffix.lower() in ['.jpg', '.png']:
                    class_samples.append({
                        'image_path': str(img_file),
                        'ground_truth_class': class_name
                    })

        # Take first num_per_class samples (or all if less than num_per_class)
        selected = class_samples[:num_per_class]
        samples.extend(selected)
        print(f"Selected {len(selected)} samples from {class_name}")

    return samples

def classify_with_yolo(model, image_path):
    """Classify image using YOLO model"""
    try:
        # Run inference
        start_time = time.time()
        results = model(image_path, verbose=False)
        inference_time = time.time() - start_time

        # Get top prediction
        result = results[0]

        # Get probabilities
        probs = result.probs
        top_class_idx = int(probs.top1)
        top_confidence = float(probs.top1conf)

        predicted_class = YOLO_CLASSES[top_class_idx]

        return {
            'success': True,
            'predicted_idx': top_class_idx,
            'predicted_class': predicted_class,
            'confidence': top_confidence,
            'inference_time': inference_time,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'predicted_idx': None,
            'predicted_class': None,
            'confidence': 0.0,
            'inference_time': 0,
            'error': str(e)
        }

def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, F1 for each class from confusion matrix"""
    metrics = {}
    n_classes = len(YOLO_CLASSES)

    for i, class_name in enumerate(YOLO_CLASSES):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }

    return metrics

def test_yolo_classification(model, samples, batch_size=100):
    """Test YOLO with single classification per image"""

    # Initialize confusion matrix
    n_classes = len(YOLO_CLASSES)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(samples),
            'num_classes': len(YOLO_CLASSES),
            'batch_size': batch_size,
            'model_path': MODEL_PATH
        },
        'detailed_predictions': [],
        'api_errors': 0,
        'total_time': 0
    }

    print(f"\nTesting {len(samples)} images with YOLO classification")
    print(f"Batch size: {batch_size} images\n")

    # Process in batches
    start_time = time.time()

    for batch_idx, batch_start in enumerate(range(0, len(samples), batch_size)):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}: Processing images {batch_start + 1}-{batch_end}")
        print(f"{'='*80}")

        for img_idx, sample in enumerate(batch_samples):
            global_idx = batch_start + img_idx + 1
            image_path = sample['image_path']
            ground_truth = sample['ground_truth_class']
            gt_idx = CLASS_TO_IDX[ground_truth]

            # Classify with YOLO
            result = classify_with_yolo(model, image_path)

            results['total_time'] += result['inference_time']

            if not result['success']:
                results['api_errors'] += 1
                print(f"[{global_idx}/{len(samples)}] ERROR: {result['error']}")
            else:
                predicted_class = result['predicted_class']
                pred_idx = result['predicted_idx']

                # Update confusion matrix
                confusion_matrix[gt_idx, pred_idx] += 1
                is_correct = (predicted_class == ground_truth)
                status = "Correct" if is_correct else "Wrong"

                # Store result
                results['detailed_predictions'].append({
                    'image_path': image_path,
                    'ground_truth': ground_truth,
                    'predicted_class': predicted_class,
                    'confidence': result['confidence'],
                    'inference_time': result['inference_time'],
                    'correct': is_correct
                })

                if (global_idx) % 10 == 0 or global_idx == len(samples):
                    print(f"[{global_idx}/{len(samples)}] GT: {ground_truth:15s} | Pred: {predicted_class:15s} ({result['confidence']:.2f}) {status}")

        # Save after each batch
        metrics = calculate_metrics(confusion_matrix)
        results['confusion_matrix'] = confusion_matrix.tolist()
        results['metrics'] = metrics

        output_file = f"yolo_single_classification_batch_{batch_end}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate accuracy
        correct = np.trace(confusion_matrix)
        total = confusion_matrix.sum()
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\n{'='*80}")
        print(f"Saved results ({batch_end} images) to: {output_file}")
        print(f"Current Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"{'='*80}")

    return results, confusion_matrix

def print_final_results(confusion_matrix):
    """Print final results"""
    print("\n" + "="*80)
    print("FINAL RESULTS - YOLO CLASSIFICATION")
    print("="*80)

    metrics = calculate_metrics(confusion_matrix)

    print(f"\n{'Class':<20s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 80)

    for class_name in YOLO_CLASSES:
        m = metrics[class_name]
        print(f"{class_name:<20s} {m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d} "
              f"{m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%")

    # Overall accuracy
    correct = np.trace(confusion_matrix)
    total = confusion_matrix.sum()
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*80)
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("="*80)

def main():
    print("="*80)
    print("YOLO SINGLE CLASSIFICATION TESTING")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples per class: {NUM_PER_CLASS}")
    print(f"Total classes: {len(YOLO_CLASSES)}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

    # Load samples
    print("\nLoading stratified samples...")
    samples = load_stratified_samples(NUM_PER_CLASS)
    print(f"\nTotal samples loaded: {len(samples)}")

    # Test
    results, confusion_matrix = test_yolo_classification(model, samples, BATCH_SIZE)

    # Print final results
    print_final_results(confusion_matrix)

if __name__ == "__main__":
    main()
