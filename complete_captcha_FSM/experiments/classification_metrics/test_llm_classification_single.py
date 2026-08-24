"""
Test LLM with single classification question - ask which class it sees
Much more efficient: 1 question per image instead of 16
"""

import os
import json
import io
from pathlib import Path
from PIL import Image
import requests
from datetime import datetime
import time
from collections import defaultdict
import numpy as np

# Configuration
DATASET_PATH = "recaptcha-dataset-master"
NUM_PER_CLASS = 999999  # Use all available samples
BATCH_SIZE = 100

def read_api_url():
    """Read API URL from config.json"""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            return config_data.get('api_url', 'http://localhost:5000')  # Default fallback - update with your API port
    return 'http://localhost:5000'  # Default fallback - update with your API port

API_URL = read_api_url()
API_ENDPOINT = f"{API_URL}/generate"

# All 16 classes in the dataset
ALL_CLASSES = [
    "Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk",
    "Hydrant", "Motorcycle", "Mountain", "Other", "Palm", "Traffic Light",
    "Boat", "Stairs", "Taxi", "Tractor"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ALL_CLASSES)}

def load_stratified_samples(num_per_class=100):
    """Load stratified samples: num_per_class images from each class"""
    samples = []

    for class_name in ALL_CLASSES:
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

def ask_llm_classify(image_path, max_retries=3):
    """Ask LLM to classify image into one of the 16 classes"""

    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert('RGB')

            # Prepare image for API
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Create classification prompt with class indices
            prompt = f"""Look at this image carefully and identify what object you see.

Choose ONLY ONE number from the following list:

0: Bicycle
1: Bridge
2: Bus
3: Car
4: Chimney
5: Crosswalk
6: Hydrant
7: Motorcycle
8: Mountain
9: Other
10: Palm
11: Traffic Light
12: Boat
13: Stairs
14: Taxi
15: Tractor

Answer with ONLY the number (0-15) of the class you see with the highest confidence. Do not include any explanation, just the number.

Answer: """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            files = {'image': ('image.png', img_byte_arr, 'image/png')}
            data = {'messages_json': json.dumps(messages)}

            start_time = time.time()
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=90)
            inference_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                model_output = result.get('model_output', '').strip()

                # Parse the class index
                predicted_idx = parse_class_index(model_output)
                predicted_class = IDX_TO_CLASS.get(predicted_idx, None)

                return {
                    'success': True,
                    'predicted_idx': predicted_idx,
                    'predicted_class': predicted_class,
                    'raw_response': model_output,
                    'inference_time': inference_time,
                    'error': None
                }
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {
                    'success': False,
                    'predicted_idx': None,
                    'predicted_class': None,
                    'raw_response': None,
                    'inference_time': 0,
                    'error': f"HTTP {response.status_code}"
                }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                'success': False,
                'predicted_idx': None,
                'predicted_class': None,
                'raw_response': None,
                'inference_time': 0,
                'error': str(e)
            }

    return {
        'success': False,
        'predicted_idx': None,
        'predicted_class': None,
        'raw_response': None,
        'inference_time': 0,
        'error': 'Max retries exceeded'
    }

def parse_class_index(model_output):
    """Parse model output to extract class index (0-15)"""
    # Try to find a number in the output
    import re

    # Look for standalone numbers
    numbers = re.findall(r'\b(\d+)\b', model_output)

    if numbers:
        idx = int(numbers[0])
        if 0 <= idx <= 15:
            return idx

    # Fallback: check if class name is mentioned
    output_upper = model_output.upper()
    for cls_name, idx in CLASS_TO_IDX.items():
        if cls_name.upper() in output_upper:
            return idx

    # If nothing found, return None
    return None

def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, F1 for each class from confusion matrix"""
    metrics = {}
    n_classes = len(ALL_CLASSES)

    for i, class_name in enumerate(ALL_CLASSES):
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

def test_llm_classification(samples, batch_size=100):
    """Test LLM with single classification question per image"""

    # Initialize confusion matrix
    n_classes = len(ALL_CLASSES)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(samples),
            'num_classes': len(ALL_CLASSES),
            'batch_size': batch_size
        },
        'detailed_predictions': [],
        'api_errors': 0,
        'total_time': 0
    }

    print(f"\nTesting {len(samples)} images with single classification question each")
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

            # Ask LLM to classify
            result = ask_llm_classify(image_path)

            results['total_time'] += result['inference_time']

            if not result['success']:
                results['api_errors'] += 1
                print(f"[{global_idx}/{len(samples)}] ERROR: {result['error']}")
            else:
                predicted_class = result['predicted_class']
                pred_idx = result['predicted_idx']

                # Update confusion matrix
                if pred_idx is not None:
                    confusion_matrix[gt_idx, pred_idx] += 1
                    is_correct = (predicted_class == ground_truth)
                    status = "Correct" if is_correct else "Wrong"
                else:
                    is_correct = False
                    status = "Unknown"

                # Store result
                results['detailed_predictions'].append({
                    'image_path': image_path,
                    'ground_truth': ground_truth,
                    'predicted_class': predicted_class,
                    'raw_response': result['raw_response'],
                    'inference_time': result['inference_time'],
                    'correct': is_correct
                })

                if (global_idx) % 10 == 0 or global_idx == len(samples):
                    print(f"[{global_idx}/{len(samples)}] GT: {ground_truth:15s} | Pred: {str(predicted_class):15s} {status}")

            time.sleep(0.1)

        # Save after each batch
        metrics = calculate_metrics(confusion_matrix)
        results['confusion_matrix'] = confusion_matrix.tolist()
        results['metrics'] = metrics

        output_file = f"llm_single_classification_batch_{batch_end}.json"
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
    print("FINAL RESULTS")
    print("="*80)

    metrics = calculate_metrics(confusion_matrix)

    print(f"\n{'Class':<20s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 80)

    for class_name in ALL_CLASSES:
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
    print("SINGLE CLASSIFICATION TESTING")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples per class: {NUM_PER_CLASS}")
    print(f"Total classes: {len(ALL_CLASSES)}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load samples
    print("\nLoading stratified samples...")
    samples = load_stratified_samples(NUM_PER_CLASS)
    print(f"\nTotal samples loaded: {len(samples)}")

    # Test
    results, confusion_matrix = test_llm_classification(samples, BATCH_SIZE)

    # Print final results
    print_final_results(confusion_matrix)

if __name__ == "__main__":
    main()
