================================================================================
SEGMENTATION RESULTS
================================================================================

This package contains scripts and pre-generated results for reproducing
Table 4 (Overall Metrics) and Table 5 (Per-Class F1-Scores) from the paper.

================================================================================
OVERVIEW
================================================================================

Segmentation-based CAPTCHA solving identifies which cells in a 4x4 grid
contain the target object.

Models tested:
- YOLOv8n-seg, YOLOv8m-seg, YOLOv8x-seg (BBox and Mask modes)
- VLM (Vision-Language Model - Qwen-7B-VL)
- Hybrid (YOLOv8x-Mask + VLM)

Configuration: confidence=0.10, overlap=0.01
Test set: 204 puzzles, 8 object classes

================================================================================
QUICK START
================================================================================

To view pre-generated results (Tables 4 & 5):

1. Overall metrics (Table 4):
   overall_model_summary.csv - Accuracy, Precision, Recall, F1 for all models

2. Per-class F1-scores (Table 5):
   per_class_metrics_conf0_10.json (YOLO models)
   llm_per_class_metrics.json (VLM)
         - Shows F1-score for each class and model for VLM and YOLO

================================================================================
FILES
================================================================================

INFERENCE SCRIPTS:
------------------
test_yolo_dual_mode.py
    Runs YOLO segmentation inference (BBox + Mask modes)
    Usage: python test_yolo_dual_mode.py --model all --threshold 0.10
    Generates: yolov8{n/m/x}_dual_summary.json + per-puzzle results

test_llm_unified.py
    Runs VLM segmentation inference
    Usage: python test_llm_unified.py
    NOTE: Requires VLM server running (set API URL in config.json)
    Generates: llm_unified_summary.json + per-puzzle results

METRICS CALCULATION:
--------------------
calculate_per_class_metrics.py
    Calculates per-class metrics from YOLO results
    Usage: python calculate_per_class_metrics.py
    Generates: per_class_metrics_conf0_10.json

calculate_llm_per_class.py
    Calculates per-class metrics from VLM results
    Usage: python calculate_llm_per_class.py
    Generates: llm_per_class_metrics.json

calculate_hybrid_metrics.py
    Calculates hybrid model metrics (YOLOv8x-Mask + VLM)
    Usage: python calculate_hybrid_metrics.py
    Generates: hybrid_metrics.json

generate_overall_summary_csv.py
    Generates Table 4 format CSV
    Usage: python generate_overall_summary_csv.py
    Generates: overall_model_summary.csv

PRE-GENERATED RESULTS:
----------------------
yolov8n_dual_summary.json  - YOLOv8n overall metrics
yolov8m_dual_summary.json  - YOLOv8m overall metrics
yolov8x_dual_summary.json  - YOLOv8x overall metrics
llm_unified_summary.json   - VLM overall metrics
hybrid_metrics.json        - Hybrid model metrics

per_class_metrics_conf0_10.json  - YOLO per-class F1-scores (Table 5)
llm_per_class_metrics.json       - VLM per-class F1-scores (Table 5)

overall_model_summary.csv        - Table 4 format (all metrics)
overall_segmentation_metrics.csv - Simplified summary

================================================================================
REPRODUCTION
================================================================================

View Pre-Generated Results
---------------------------------------
Open the JSON/CSV files to see the results that match Tables 4 & 5.


Regenerate YOLO Results
------------------------------------
1. Run YOLO inference:
   python test_yolo_dual_mode.py --model all --threshold 0.10

2. Calculate per-class metrics:
   python calculate_per_class_metrics.py

3. Generate summary CSV:
   python generate_overall_summary_csv.py


Full Reproduction (YOLO + VLM)
-------------------------------------------
1. Run YOLO inference:
   python test_yolo_dual_mode.py --model all --threshold 0.10

2. Run VLM inference (requires VLM server):
   python test_llm_unified.py

3. Calculate all metrics:
   python calculate_per_class_metrics.py
   python calculate_llm_per_class.py
   python calculate_hybrid_metrics.py

4. Generate summaries:
   python generate_overall_summary_csv.py

================================================================================
DATA LOCATION
================================================================================

The full dataset (204 puzzles) is in:
    segmentation_inference_data

Each puzzle contains:
- concatenated_grid.png (4x4 CAPTCHA grid)
- captcha_area.png (CAPTCHA prompt)
- ground_truth.json (correct cells)

Models required:
- yolov8n-seg.pt (6.8 MB)
- yolov8m-seg.pt (53 MB)
- yolov8x-seg.pt (138 MB)
- VLM

YOLO Models will be downloaded by ultralytics if missing.

================================================================================
