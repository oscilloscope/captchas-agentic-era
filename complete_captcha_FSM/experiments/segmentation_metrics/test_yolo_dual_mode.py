"""
Unified YOLO segmentation test script with DUAL metrics calculation.
Tests YOLOv8n-seg, YOLOv8m-seg, or YOLOv8x-seg on the segmentation dataset.

This script runs inference ONCE and calculates metrics using BOTH:
- Bounding box-based overlap
- Mask-based overlap

Usage:
    python test_yolo_dual_mode.py --model n    # Test YOLOv8n-seg
    python test_yolo_dual_mode.py --model m    # Test YOLOv8m-seg
    python test_yolo_dual_mode.py --model x    # Test YOLOv8x-seg
    python test_yolo_dual_mode.py --model all  # Test all models
"""

import os
import json
import cv2
import argparse
import time
import numpy as np
from pathlib import Path
import easyocr
from ultralytics import YOLO
from pycocotools import mask as mask_utils


def normalize_target_object(ocr_text):
    """Normalize OCR-extracted text to COCO-compatible class names."""
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

        # Abbreviations/synonyms for SUPPORTED classes only
        'bike': 'bicycle',
        'bikes': 'bicycle',
        'hydrant': 'fire hydrant',
        'hydrants': 'fire hydrant',
    }

    return target_mapping.get(text_lower, text_lower)


def does_detection_match_target(yolo_class_name, target_object, yolo_class_names):
    """
    Check if a YOLO detection matches the target object.

    Args:
        yolo_class_name: Class name from YOLO detection (e.g., 'car', 'bicycle')
        target_object: Normalized target object (e.g., 'bicycle', 'taxi')
        yolo_class_names: List of class names supported by YOLO model

    Returns:
        bool: True if detection matches target, False otherwise
    """
    target_lower = target_object.lower().strip()
    yolo_class_lower = yolo_class_name.lower()

    # Convert yolo_class_names to lowercase for comparison
    supported_classes = [name.lower() for name in yolo_class_names]

    # Check if target is even supported by YOLO
    if target_lower not in supported_classes:
        return False

    # Bus accepts both 'bus' and 'truck' (YOLO misclassification workaround)
    if target_lower == 'bus':
        return yolo_class_lower in ['bus', 'truck']

    # Default: exact match
    return yolo_class_lower == target_lower


def get_overlapping_cells_bbox(bbox, grid_width, grid_height, grid_size=4, threshold=0.01):
    """
    Determine which grid cells overlap with a bounding box (BBOX MODE).

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


def get_overlapping_cells_mask(mask, grid_size=4, threshold=0.01):
    """
    Determine which grid cells overlap with a segmentation mask (MASK MODE).

    Uses mask distribution: a cell is selected if >= threshold of the total mask is in that cell.

    Args:
        mask: Binary mask array (H x W)
        grid_size: Grid dimensions (default 4x4)
        threshold: Minimum ratio of mask pixels in cell (0.01 = 1% of total mask)

    Returns:
        list of cell indices (0-15 for 4x4 grid)
    """
    h, w = mask.shape
    cell_height = h / grid_size
    cell_width = w / grid_size

    total_mask_pixels = np.sum(mask > 0)
    if total_mask_pixels == 0:
        return []

    overlapping_cells = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)

            # Count mask pixels in this cell
            cell_mask = mask[y1:y2, x1:x2]
            cell_mask_pixels = np.sum(cell_mask > 0)

            # Calculate overlap ratio relative to total mask size
            overlap_ratio = cell_mask_pixels / total_mask_pixels

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


def encode_mask_to_rle(mask):
    """
    Encode a binary mask to RLE format (COCO format).

    Args:
        mask: Binary mask array (H x W) with values 0 or 1

    Returns:
        Dictionary with RLE encoding
    """
    # Ensure mask is binary and in the correct format
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def run_yolo_detection_dual(model, image_path, target_object, img_width, img_height, conf_threshold=0.25):
    """
    Run YOLO detection and extract BOTH bounding boxes AND masks.

    Args:
        model: loaded YOLO model
        image_path: path to concatenated_grid.png
        target_object: normalized target object name
        img_width: image width
        img_height: image height
        conf_threshold: Confidence threshold for YOLO inference

    Returns:
        list of detection dicts with bbox, mask_rle, conf, class_name
    """
    results = model(image_path, conf=conf_threshold, verbose=False)

    if not results or len(results) == 0:
        return []

    result = results[0]
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[int(cls_id)] for cls_id in class_ids]

        # Get all YOLO class names supported by this model
        yolo_class_names = list(result.names.values())

        # Check if masks are available
        has_masks = result.masks is not None and len(result.masks) > 0
        if has_masks:
            masks = result.masks.data.cpu().numpy()

        for i, (box, conf, class_name) in enumerate(zip(boxes, confidences, class_names)):
            if does_detection_match_target(class_name, target_object, yolo_class_names):
                x1, y1, x2, y2 = box

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': float(conf),
                    'class_name': class_name
                }

                # Add mask if available
                if has_masks:
                    # Resize mask to image dimensions
                    mask = masks[i]
                    mask_resized = cv2.resize(mask, (img_width, img_height),
                                            interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    # Encode to RLE
                    detection['mask_rle'] = encode_mask_to_rle(mask_binary)

                detections.append(detection)

    return detections


def calculate_metrics(ground_truth_cells, selected_cells):
    """Calculate TP, FP, FN, TN, precision, recall, accuracy, F1."""
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
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1
    }


def process_puzzle(puzzle_dir, model, ocr_reader, conf_threshold=0.25):
    """
    Process a single puzzle with DUAL metrics calculation.

    Args:
        puzzle_dir: Path to puzzle directory
        model: YOLO model
        ocr_reader: EasyOCR reader
        conf_threshold: Confidence threshold for YOLO inference

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

    # Run YOLO detection (gets both bbox and mask)
    start_time = time.time()
    detections = run_yolo_detection_dual(model, str(concat_path), target_object, w, h, conf_threshold)
    inference_time = time.time() - start_time

    # Calculate BBOX-based metrics
    bbox_selected_cells = set()
    for det in detections:
        cells = get_overlapping_cells_bbox(det['bbox'], w, h)
        bbox_selected_cells.update(cells)
    bbox_selected_cells = sorted(list(bbox_selected_cells))
    bbox_metrics = calculate_metrics(ground_truth_cells, bbox_selected_cells)

    # Calculate MASK-based metrics
    # Note: Always calculate mask metrics, even when masks are unavailable.
    # When no masks exist, selected_cells=[] means all ground truth are FN.
    mask_selected_cells = set()
    has_masks = any('mask_rle' in det for det in detections)

    if has_masks:
        for det in detections:
            if 'mask_rle' in det:
                # Decode RLE to binary mask
                rle = det['mask_rle'].copy()
                rle['counts'] = rle['counts'].encode('utf-8')
                mask = mask_utils.decode(rle)

                cells = get_overlapping_cells_mask(mask)
                mask_selected_cells.update(cells)

    mask_selected_cells = sorted(list(mask_selected_cells))
    mask_metrics = calculate_metrics(ground_truth_cells, mask_selected_cells)

    return {
        'puzzle_name': puzzle_dir.name,
        'target_object': target_object,
        'ground_truth': ground_truth_cells,
        'detections': detections,
        'bbox_results': {
            'selected_cells': bbox_selected_cells,
            'metrics': bbox_metrics
        },
        'mask_results': {
            'selected_cells': mask_selected_cells,
            'metrics': mask_metrics,
            'has_masks': has_masks
        },
        'inference_time': inference_time
    }


def test_model(model_size, save_per_puzzle=True, conf_threshold=0.25):
    """
    Test a specific YOLO model size with dual metrics calculation.

    Args:
        model_size: 'n', 'm', or 'x'
        save_per_puzzle: whether to save per-puzzle JSON results
        conf_threshold: confidence threshold for YOLO inference

    Returns:
        summary dict
    """
    model_name = f"yolov8{model_size}-seg"
    model_file = f"yolov8{model_size}-seg.pt"

    print(f"\n{'='*80}")
    print(f"TESTING {model_name.upper()} - DUAL MODE (BBox + Mask) - conf={conf_threshold}")
    print(f"{'='*80}")

    # Load model (auto-downloads if missing)
    print(f"\nLoading {model_name}...")
    model = YOLO(model_file)
    print(f"Model loaded")

    # Load OCR
    print("Loading OCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("OCR loaded")

    # Get puzzle directories from inference data folder
    inference_dir = Path("segmentation_inference_data")
    puzzle_dirs = sorted([d for d in inference_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(puzzle_dirs)} puzzles")

    # Process puzzles
    print("\nProcessing...")
    results_list = []
    total_start = time.time()

    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        result = process_puzzle(puzzle_dir, model, ocr_reader, conf_threshold)

        if result:
            results_list.append(result)

            # Save per-puzzle result to same directory
            if save_per_puzzle:
                result_json = {
                    'puzzle_name': result['puzzle_name'],
                    'target_object': result['target_object'],
                    'ground_truth': result['ground_truth'],
                    'detections': result['detections'],
                    'bbox_results': result['bbox_results'],
                    'mask_results': result['mask_results'],
                    'inference_time': result['inference_time']
                }
                threshold_str = f"{conf_threshold:.2f}".replace('.', '_')
                json_path = puzzle_dir / f"yolov8{model_size}_dual_result_conf{threshold_str}.json"
                with open(json_path, 'w') as f:
                    json.dump(result_json, f, indent=2)

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(puzzle_dirs)}")

    total_elapsed = time.time() - total_start

    # Calculate aggregate metrics
    if not results_list:
        print("No results!")
        return None

    # Aggregate BBOX metrics
    bbox_tp = sum(r['bbox_results']['metrics']['tp'] for r in results_list)
    bbox_fp = sum(r['bbox_results']['metrics']['fp'] for r in results_list)
    bbox_fn = sum(r['bbox_results']['metrics']['fn'] for r in results_list)
    bbox_tn = sum(r['bbox_results']['metrics']['tn'] for r in results_list)

    bbox_precision = bbox_tp / (bbox_tp + bbox_fp) if (bbox_tp + bbox_fp) > 0 else 0
    bbox_recall = bbox_tp / (bbox_tp + bbox_fn) if (bbox_tp + bbox_fn) > 0 else 0
    bbox_accuracy = (bbox_tp + bbox_tn) / (bbox_tp + bbox_fp + bbox_fn + bbox_tn)
    bbox_f1 = 2 * (bbox_precision * bbox_recall) / (bbox_precision + bbox_recall) if (bbox_precision + bbox_recall) > 0 else 0

    # Aggregate MASK metrics (now all puzzles have mask_results)
    mask_tp = sum(r['mask_results']['metrics']['tp'] for r in results_list)
    mask_fp = sum(r['mask_results']['metrics']['fp'] for r in results_list)
    mask_fn = sum(r['mask_results']['metrics']['fn'] for r in results_list)
    mask_tn = sum(r['mask_results']['metrics']['tn'] for r in results_list)

    mask_precision = mask_tp / (mask_tp + mask_fp) if (mask_tp + mask_fp) > 0 else 0
    mask_recall = mask_tp / (mask_tp + mask_fn) if (mask_tp + mask_fn) > 0 else 0
    mask_accuracy = (mask_tp + mask_tn) / (mask_tp + mask_fp + mask_fn + mask_tn)
    mask_f1 = 2 * (mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0

    # Count puzzles that actually had masks available
    puzzles_with_masks = sum(1 for r in results_list if r['mask_results']['has_masks'])

    avg_inference = sum(r['inference_time'] for r in results_list) / len(results_list)

    # Print summary
    print(f"\n{'-'*80}")
    print(f"SUMMARY: {model_name}")
    print(f"{'-'*80}")
    print(f"Processed: {len(results_list)}/{len(puzzle_dirs)} puzzles")
    print(f"Puzzles with masks: {puzzles_with_masks}/{len(results_list)}")

    print(f"\n{'Mode':<12} {'Precision':<12} {'Recall':<12} {'Accuracy':<12} {'F1':<12}")
    print(f"{'-'*60}")
    print(f"{'BBox':<12} {bbox_precision:.4f}       {bbox_recall:.4f}       {bbox_accuracy:.4f}       {bbox_f1:.4f}")
    print(f"{'Mask':<12} {mask_precision:.4f}       {mask_recall:.4f}       {mask_accuracy:.4f}       {mask_f1:.4f}")
    print(f"{'-'*60}")
    print(f"{'Improvement':<12} {mask_precision-bbox_precision:+.4f}       {mask_recall-bbox_recall:+.4f}       {mask_accuracy-bbox_accuracy:+.4f}       {mask_f1-bbox_f1:+.4f}")

    print(f"\nTiming:")
    print(f"  Total: {total_elapsed:.2f}s")
    print(f"  Avg inference: {avg_inference:.4f}s")

    # Save summary
    summary = {
        'model': model_name,
        'total_puzzles': len(puzzle_dirs),
        'successful': len(results_list),
        'puzzles_with_masks': puzzles_with_masks,
        'bbox_metrics': {
            'tp': bbox_tp,
            'fp': bbox_fp,
            'fn': bbox_fn,
            'tn': bbox_tn,
            'precision': bbox_precision,
            'recall': bbox_recall,
            'accuracy': bbox_accuracy,
            'f1': bbox_f1
        },
        'mask_metrics': {
            'tp': mask_tp,
            'fp': mask_fp,
            'fn': mask_fn,
            'tn': mask_tn,
            'precision': mask_precision,
            'recall': mask_recall,
            'accuracy': mask_accuracy,
            'f1': mask_f1
        },
        'timing': {
            'total_elapsed': total_elapsed,
            'avg_inference': avg_inference
        }
    }

    summary_path = f"yolov8{model_size}_dual_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified YOLO dual-mode test (bbox + mask)')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['n', 'm', 'x', 'all'],
                        help='Model size: n (nano), m (medium), x (xlarge), or all')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save per-puzzle results')
    parser.add_argument('--threshold', '-t', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    args = parser.parse_args()

    print("="*80)
    print("UNIFIED YOLO DUAL-MODE TEST (BBox + Mask)")
    print("="*80)

    models_to_test = ['n', 'm', 'x'] if args.model == 'all' else [args.model]

    summaries = {}
    for model_size in models_to_test:
        summary = test_model(model_size, save_per_puzzle=not args.no_save, conf_threshold=args.threshold)
        if summary:
            summaries[model_size] = summary

    # Print comparison if multiple models tested
    if len(summaries) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON - BBOX MODE")
        print("="*80)
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
        print("-"*60)
        for size, summary in summaries.items():
            m = summary['bbox_metrics']
            print(f"YOLOv8{size}-seg    {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['accuracy']:.4f}")

        print("\n" + "="*80)
        print("MODEL COMPARISON - MASK MODE")
        print("="*80)
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
        print("-"*60)
        for size, summary in summaries.items():
            m = summary['mask_metrics']
            print(f"YOLOv8{size}-seg    {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['accuracy']:.4f}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
