"""
Unified LLM segmentation test script.
Tests LLM on the segmentation dataset.

Usage:
    python test_llm_unified.py
    python test_llm_unified.py --no-save
    python test_llm_unified.py --api-url <url>
"""

import os
import json
import cv2
import argparse
import time
import requests
import io
import re
from pathlib import Path
from PIL import Image
import easyocr


def normalize_target_object(ocr_text):
    """Normalize OCR-extracted text to consistent class names."""
    text_lower = ocr_text.lower().strip()

    target_mapping = {
        # Plurals to singular
        'bicycles': 'bicycle',
        'cars': 'car',
        'motorcycles': 'motorcycle',
        'buses': 'bus',
        'traffic lights': 'traffic light',
        'fire hydrants': 'fire hydrant',
        'stop signs': 'stop sign',
        'parking meters': 'parking meter',
        'crosswalks': 'crosswalk',

        # Abbreviations/synonyms
        'bike': 'bicycle',
        'bikes': 'bicycle',
        'hydrant': 'fire hydrant',
        'hydrants': 'fire hydrant',
    }

    return target_mapping.get(text_lower, text_lower)


def get_overlapping_cells(bbox, grid_width, grid_height, grid_size=4, threshold=0.01):
    """
    Determine which grid cells overlap with a bounding box.

    Uses cell area ratio: a cell is selected if >= threshold of its area is covered.

    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        grid_width: width of the grid image
        grid_height: height of the grid image
        grid_size: grid dimensions (4 for 4x4)
        threshold: minimum overlap ratio (0.01 = 1% of cell area)

    Returns:
        list of cell indices (0-15 for 4x4 grid)
    """
    x1, y1, x2, y2 = bbox
    cell_w = grid_width / grid_size
    cell_h = grid_height / grid_size

    overlapping_cells = []

    for row in range(grid_size):
        for col in range(grid_size):
            cell_x1 = col * cell_w
            cell_y1 = row * cell_h
            cell_x2 = cell_x1 + cell_w
            cell_y2 = cell_y1 + cell_h

            # Calculate intersection
            inter_x1 = max(x1, cell_x1)
            inter_y1 = max(y1, cell_y1)
            inter_x2 = min(x2, cell_x2)
            inter_y2 = min(y2, cell_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                cell_area = cell_w * cell_h
                overlap_ratio = intersection_area / cell_area

                if overlap_ratio >= threshold:
                    cell_idx = row * grid_size + col
                    overlapping_cells.append(cell_idx)

    return overlapping_cells


def extract_target_from_captcha(captcha_path, ocr_reader):
    """
    Extract and normalize target object name from captcha image using OCR.

    Args:
        captcha_path: path to captcha_area.png
        ocr_reader: EasyOCR reader instance

    Returns:
        normalized target object name, or None if extraction failed
    """
    img = cv2.imread(str(captcha_path))
    if img is None:
        return None

    ocr_results = ocr_reader.readtext(img, detail=0, paragraph=True)
    full_text = " ".join(ocr_results).lower()

    # Keywords to search for
    keywords = [
        'bicycle', 'bicycles', 'bike', 'bikes',
        'bus', 'buses',
        'car', 'cars',
        'motorcycle', 'motorcycles',
        'traffic light', 'traffic lights',
        'fire hydrant', 'fire hydrants', 'hydrant',
        'stop sign', 'stop signs',
        'parking meter', 'parking meters',
        'taxi', 'taxis',
        'crosswalk', 'crosswalks',
        'stairs',
    ]

    for keyword in keywords:
        if keyword in full_text:
            return normalize_target_object(keyword)

    return None


def send_image_to_llm_api(image, prompt, api_url):
    """
    Send image to LLM API with a prompt.

    Args:
        image: PIL Image object
        prompt: text prompt
        api_url: API endpoint URL

    Returns:
        dict with API response or None if failed
    """
    endpoint = f"{api_url.rstrip('/')}/generate"

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    files = {
        'image': ('image.png', img_byte_arr, 'image/png')
    }
    data = {
        'messages_json': json.dumps(messages)
    }

    try:
        response = requests.post(endpoint, files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Error: API request failed: {e}")
        return None


def extract_bounding_boxes(text, object_name):
    """
    Extract bounding boxes from LLM response text.

    Args:
        text: LLM response text
        object_name: target object name

    Returns:
        list of dicts with 'bbox' and 'object_id'
    """
    objects = []

    # Create flexible patterns for different object names
    obj_patterns = [
        object_name,
        object_name.title(),
        object_name.lower(),
        object_name.replace(' ', ''),
        object_name.replace(' ', '_')
    ]

    patterns = []
    for obj_pattern in obj_patterns:
        patterns.extend([
            rf'(?:{obj_pattern})\s*(\d+)?\s*:?\s*\[(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*)\]',
            rf'\[(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*)\].*?{obj_pattern}'
        ])

    # Fallback pattern for plain coordinates
    patterns.append(r'\[(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*),\s*(\s*\d+\s*)\]')

    obj_count = 0

    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                if pattern_idx < len(patterns) - 1:  # Labeled patterns
                    if len(match) >= 5:  # Has object number + 4 coordinates
                        obj_num = match[0] if match[0] else str(obj_count + 1)
                        x1, y1, x2, y2 = int(match[1].strip()), int(match[2].strip()), int(match[3].strip()), int(match[4].strip())
                    elif len(match) == 4:  # Just coordinates
                        obj_num = str(obj_count + 1)
                        x1, y1, x2, y2 = int(match[0].strip()), int(match[1].strip()), int(match[2].strip()), int(match[3].strip())
                    else:
                        continue
                else:  # Plain coordinates (fallback)
                    if obj_count > 0:  # Skip if objects already found
                        continue
                    obj_num = str(obj_count + 1)
                    x1, y1, x2, y2 = int(match[0].strip()), int(match[1].strip()), int(match[2].strip()), int(match[3].strip())

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                objects.append({
                    'object_id': f'{object_name.replace(" ", "_")}_{obj_num}',
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y]
                })

                obj_count += 1

            except (ValueError, IndexError):
                continue

        if objects:  # Stop after finding objects
            break

    return objects


def run_llm_detection(image_path, target_object, api_url):
    """
    Run LLM detection on an image.

    Args:
        image_path: path to image
        target_object: target object name
        api_url: API endpoint URL

    Returns:
        list of detection dicts with bbox, conf, class_name
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"  Error: Could not load image: {e}")
        return []

    # Create detection prompt
    prompt = f"""Find all {target_object} in this image and create bounding boxes that encompass the entire {target_object}.

Requirements:
- Include ALL visible parts of the {target_object}
- Capture the complete {target_object} from top to bottom and side to side
- Make sure the entire {target_object} is contained within the bounding box

For each {target_object}, return coordinates in this format:
{target_object.title()} 1: [x1, y1, x2, y2]
{target_object.title()} 2: [x1, y1, x2, y2]

Where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner."""

    result = send_image_to_llm_api(image, prompt, api_url)

    if not result:
        return []

    model_output = result.get('model_output', '')

    # Extract bounding boxes
    objects = extract_bounding_boxes(model_output, target_object)

    # Convert to detection format
    detections = []
    for obj in objects:
        detections.append({
            'bbox': obj['bbox'],
            'conf': 1.0,  # LLM doesn't provide confidence scores
            'class_name': target_object
        })

    return detections


def process_puzzle(puzzle_dir, ocr_reader, api_url):
    """
    Process a single puzzle.

    Returns:
        dict with results, or None if processing failed
    """
    concat_path = puzzle_dir / "concatenated_grid.png"
    gt_path = puzzle_dir / "ground_truth.json"
    captcha_path = puzzle_dir / "captcha_area.png"

    if not all([concat_path.exists(), gt_path.exists(), captcha_path.exists()]):
        return None

    # Load ground truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    ground_truth_cells = gt_data.get('selected_cells', [])

    # Extract target
    target_object = extract_target_from_captcha(captcha_path, ocr_reader)
    if not target_object:
        return None

    # Load image and get dimensions
    concatenated = cv2.imread(str(concat_path))
    h, w = concatenated.shape[:2]

    # Run LLM detection
    start_time = time.time()
    detections = run_llm_detection(str(concat_path), target_object, api_url)
    inference_time = time.time() - start_time

    # Map detections to cells
    selected_cells = set()
    for det in detections:
        cells = get_overlapping_cells(det['bbox'], w, h)
        selected_cells.update(cells)
    selected_cells = sorted(list(selected_cells))

    # Calculate metrics
    gt_set = set(ground_truth_cells)
    sel_set = set(selected_cells)

    tp = len(gt_set & sel_set)
    fp = len(sel_set - gt_set)
    fn = len(gt_set - sel_set)
    tn = 16 - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / 16
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'puzzle_name': puzzle_dir.name,
        'target_object': target_object,
        'ground_truth': ground_truth_cells,
        'selected_cells': selected_cells,
        'detections': detections,
        'metrics': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        },
        'inference_time': inference_time
    }


def load_api_url(override_url=None):
    """
    Load API URL from config.json or use override.

    Args:
        override_url: optional URL to use instead of config.json

    Returns:
        API URL string or None
    """
    if override_url:
        return override_url

    try:
        with open('config.json', 'r') as f:
            config_data = json.load(f)
            return config_data.get('api_url', None)
    except FileNotFoundError:
        return None

    return None


def test_llm(api_url, save_per_puzzle=True):
    """
    Test LLM on all puzzles.

    Args:
        api_url: API endpoint URL
        save_per_puzzle: whether to save per-puzzle JSON results

    Returns:
        summary dict
    """
    print(f"\n{'='*80}")
    print(f"TESTING LLM")
    print(f"{'='*80}")

    # Load OCR
    print("\nLoading OCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("OCR loaded")

    # Get puzzle directories from inference data folder
    seg_dir = Path("segmentation_inference_data")
    puzzle_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(puzzle_dirs)} puzzles")

    # Process puzzles
    print("\nProcessing...")
    results_list = []
    total_start = time.time()

    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        print(f"  [{i}/{len(puzzle_dirs)}] {puzzle_dir.name}...", end=" ")

        result = process_puzzle(puzzle_dir, ocr_reader, api_url)

        if result:
            results_list.append(result)
            print(f"(TP={result['metrics']['tp']}, FP={result['metrics']['fp']}, FN={result['metrics']['fn']})")

            # Save per-puzzle result
            if save_per_puzzle:
                result_json = {
                    'puzzle_name': result['puzzle_name'],
                    'target_object': result['target_object'],
                    'ground_truth': result['ground_truth'],
                    'selected_cells': result['selected_cells'],
                    'llm_detections': result['detections'],
                    'metrics': result['metrics'],
                    'inference_time': result['inference_time']
                }
                json_path = puzzle_dir / "llm_unified_result.json"
                with open(json_path, 'w') as f:
                    json.dump(result_json, f, indent=2)
        else:
            print("Failed")

    total_elapsed = time.time() - total_start

    # Calculate summary
    if not results_list:
        print("\nNo results!")
        return None

    total_tp = sum(r['metrics']['tp'] for r in results_list)
    total_fp = sum(r['metrics']['fp'] for r in results_list)
    total_fn = sum(r['metrics']['fn'] for r in results_list)
    total_tn = sum(r['metrics']['tn'] for r in results_list)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    avg_inference = sum(r['inference_time'] for r in results_list) / len(results_list)

    # Print summary
    print(f"\n{'-'*40}")
    print(f"SUMMARY: LLM")
    print(f"{'-'*40}")
    print(f"Processed: {len(results_list)}/{len(puzzle_dirs)} puzzles")
    print(f"\nMetrics:")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
    print(f"  Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"  Recall:    {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"  Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  F1-Score:  {overall_f1:.4f}")
    print(f"\nTiming:")
    print(f"  Total: {total_elapsed:.2f}s")
    print(f"  Avg inference: {avg_inference:.4f}s")

    # Save summary
    summary = {
        'model': 'LLM',
        'total_puzzles': len(puzzle_dirs),
        'successful': len(results_list),
        'overall_metrics': {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn,
            'precision': overall_precision,
            'recall': overall_recall,
            'accuracy': overall_accuracy,
            'f1': overall_f1
        },
        'timing': {
            'total_elapsed': total_elapsed,
            'avg_inference': avg_inference
        }
    }

    summary_path = "llm_unified_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified LLM segmentation test')
    parser.add_argument('--api-url', type=str, default=None,
                        help='API endpoint URL (overrides config.json)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save per-puzzle results')
    args = parser.parse_args()

    print("="*80)
    print("UNIFIED LLM SEGMENTATION TEST")
    print("="*80)

    # Load API URL
    api_url = load_api_url(args.api_url)
    if not api_url:
        print("\nError: Could not load API URL")        
        return

    print(f"\nAPI URL: {api_url}")

    # Test LLM
    summary = test_llm(api_url, save_per_puzzle=not args.no_save)

    if summary:
        print("\n" + "="*80)
        print("DONE")
        print("="*80)


if __name__ == "__main__":
    main()
