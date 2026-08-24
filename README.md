# CAPTCHAs in the Agentic Era: Solvers That Learn from Every Encounter

Vision-language models (VLMs) can solve visual CAPTCHAs without task-specific training, but the agents built on them approach every challenge from scratch. For such an agent, the hundredth instance of a familiar puzzle costs as much time and compute as the first. Specialized detectors invert the trade-off, answering in milliseconds but only for categories they were trained on. Neither improves with exposure.

We study what changes when a solver improves with use. Our system pairs a fine-tuned YOLOv8 detector with an open-weight VLM behind a confidence-based router, and runs entirely from screenshots and operating-system input events, with no browser automation or DOM access. It reaches 85.4% overall and 84.2% macro accuracy across 16 classes, exceeding either component alone. Every answer VLM produces also serves as a training label, so the detector absorbs categories it was never trained for, typically after one or two encounters and without human annotation. The same loop also repairs it. A CAPTCHA operator can perturb images against the publicly released detector and drive its accuracy to 0%, but the perturbations leave VLM untouched, and its labels let the detector recover. Under a year-long simulated arms race in which the CAPTCHA operator re-crafts its perturbations each month, the solver recovers every round, and a cheap ~70%-accurate open-weight teacher hardens it as effectively as a perfect oracle. Visual CAPTCHA defenses that assume a failing bot stays failing therefore understate how quickly an adaptive solver returns.

**State-of-the-Art Performance**

- Handles all three reCAPTCHA v2 puzzle types with rare class support (classification, segmentation, dynamic)
- 85.4% overall accuracy (+10.0pp over YOLO-only baseline)
- 84.2% macro-averaged accuracy (+24.2pp over YOLO-only baseline)
- VLM invoked for only 30% of images via confidence-based cascade routing

**Adaptive Learning**

- Teacher-student distillation: VLM labels unsupported classes, YOLO learns them as reflexes
- Taxi: 96% recall from ~2 puzzle encounters, Tractor: 88% from ~3 puzzles
- Autonomous recovery from adversarial attacks via VLM-guided retraining

**Cross-Platform Compatible**

- Works on Linux, macOS and Windows
- No platform-specific dependencies or modifications required

**Cross-Browser Compatible**

- **Works with ANY browser** (Chrome, Firefox, Edge, Safari, etc.)
- **No DOM access required** - Screenshot-based interaction
- **No browser automation frameworks** - No Selenium, Playwright, or Puppeteer needed
- **Evades WebDriver detection** - No DOM manipulation signatures

**Open-Weight Models**

- Uses Qwen-7B-VL (open-weight) instead of proprietary APIs
- No API costs for inference
- Fully reproducible on-premise

## System Requirements

### Hardware Requirements

- **Minimum**: 16GB RAM, 20GB disk space. YOLO-only solver can run on CPU without GPU, though inference will be significantly slower
- **Recommended (Hybrid mode)**: 32GB RAM, 16GB+ GPU VRAM (NVIDIA RTX 3090 or equivalent), 100GB disk space

### Software Requirements

- **Operating Systems**: Linux, macOS or Windows (tested on all three)
- **Python**: 3.11
- **Browsers**: Any modern browser with screenshot capability
- **Key Libraries**: PyTorch, Transformers, Ultralytics (YOLOv8), PyAutoGUI, MSS, OpenCV
- **Display Resolution**: The system performs optimally at 1920x1080 resolution. For other display resolutions, browser zoom adjustment may be necessary to ensure reliable UI element localization

### Platform-Specific Requirements

**Linux**:

```bash
sudo apt install gnome-screenshot
```

**macOS**:

- No additional requirements (uses built-in `screencapture`)

**Windows**:

- No additional requirements (uses Windows API)

## Installation

1. Ensure Python 3.11 is installed on your system
2. Create a virtual environment:

```bash
# Linux/Mac
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Required model files:
   - `detection_model.pt` (39 MB) - UI element detection model
   - `classification_model.pt` (2.9 MB) - Classification model
   - `yolov8x-seg.pt` (140 MB) - Segmentation model (auto-download on first run)

### Command-line Arguments

- `-v, --verbose`: Enable verbose output for debugging
- `--no-mouse`: Disable mouse movement animations
- `--backend {yolo,llm,hybrid}`: Choose the solver backend
  - `yolo`: Full YOLO-based detection and classification (default, fastest)
  - `llm`: YOLO UI detection with VLM-based classification (requires VLM backend)
  - `hybrid`: Confidence-based routing between YOLO and VLM (requires VLM backend)

### Examples

```bash
# Run with YOLO only backend (default)
python main.py

# Run with VLM backend
python main.py --backend llm

# Run with hybrid mode
python main.py --backend hybrid
```

## Configuration

- `config.json`: Contains the API URL for VLM backend communication (if using llm/hybrid modes)
- `config.py`: Global configuration for mouse movement behavior

## Architecture Overview

Our system implements a human-inspired dual-process approach:

- **YOLO (Fast Reflexes)**: Handles frequent, routine patterns with millisecond inference
- **VLM (Slow Reasoning)**: Intervenes for novel or semantically complex challenges

This hybrid architecture achieves superior performance by combining:

1. Fine-tuned YOLOv8 for real-time UI element detection
2. Specialized YOLO classifiers for common object classes
3. Open-weight Qwen-7B-VL for zero-shot reasoning on rare categories
4. Confidence-based cascade routing (threshold = 0.70)
5. Finite-state machine controller for robust puzzle solving

## Project Structure

```
.
├── complete_captcha_FSM/                # CAPTCHA solver application
│   ├── main.py                          # Main entry point
│   ├── captcha_fsm.py                   # State machine for CAPTCHA solving
│   ├── unified_captcha_processor.py     # Core processor with multiple backends
│   ├── config.py                        # Global configuration
│   ├── config.json                      # API configuration
│   ├── detection_model.pt               # UI element detection model
│   ├── classification_model.pt          # Classification model
│   └── yolov8x-seg.pt                   # Segmentation model (auto-download)
├── experiments/                         # Evaluation and experiment notebooks
│   ├── classification_metrics/          # Classification evaluation
│   │   ├── classification_metrics.ipynb # Per-class metrics and confusion matrices
│   │   └── cascade_routing_analysis.ipynb # Cascade threshold analysis
│   ├── segmentation_metrics/            # Segmentation evaluation
│   ├── distillation/                    # Experience Replay fine-tuning
│   │   ├── taxi_finetune_er.ipynb
│   │   ├── tractor_finetune_er.ipynb
│   │   └── boat_finetune_er.ipynb
│   └── adversarial/                     # Adversarial robustness experiments
│       ├── 01_generate_pgd_attacks.ipynb
│       ├── 02_vlm_inference_on_adversarial.ipynb
│       ├── 03_sample_efficiency.ipynb
│       └── 04_greybox_escalation.ipynb
├── llm_backend/                         # VLM server backend
├── captcha_detector_trainer/            # UI detection dataset generation
└── requirements.txt                     # Python dependencies
```

## How It Works

1. **Detection**: Continuously monitors the screen for reCAPTCHA checkboxes using YOLOv8
2. **UI Localization**: Detects CAPTCHA area, grid cells, verify/reload buttons via screenshot analysis
3. **Puzzle Analysis**: OCR extracts target object; determines puzzle type (classification vs segmentation)
4. **Routing**: For unsupported classes, routes directly to VLM. For supported classes, YOLO classifies first; if confidence < 0.70, falls back to VLM
5. **Verification**: Checks success and handles dynamic puzzles (tiles refresh after clicks)

## Experiments

### Teacher-Student Distillation

VLM predictions serve as labeled training data to fine-tune YOLO on unsupported classes via Experience Replay. Results (10 seeds per N value):

- **Taxi**: 96% recall at N=10 (~2 puzzle encounters)
- **Tractor**: 88% recall at N=15 (~3 puzzle encounters)
- **Boat**: 48% recall at N=6 (~1 puzzle encounter)

### Adversarial Robustness

PGD attacks (epsilon=4/255, 10 steps) reduce YOLO accuracy from 83.1% to 0%, but VLM accuracy drops only 3.7pp (78.7% to 75.0%). The solver autonomously recovers by retraining on VLM-labeled adversarial samples. Most of the gain arrives within the first ~1,800 samples (roughly 200 puzzle encounters), after which returns diminish; at the full ~11,000-sample budget VLM-labeled retraining reaches 65.0% adversarial accuracy versus 71.0% for ground-truth labels, a 6-10pp penalty for label noise.

### Grey-Box Escalation

The defender re-crafts attacks against an updated surrogate each round, simulated over 12 rounds (one per month for a year). Two factors are varied: the teacher the solver learns from, and whether the defender can reproduce its training (the batch order it shuffles with).

| Teacher | Batch order | Pre-recovery | Post-recovery |
|---|---|---|---|
| Perfect (GT) | coupled | 28.4% | 60.1% |
| Perfect (GT) | independent | 52.9% | 64.2% |
| VLM (~70% acc.) | coupled | 58.2% | 62.3% |
| VLM (~70% acc.) | independent | 58.9% | 62.3% |

Recovery is universal: every variant returns to 60-64% after retraining, so each attack is only a temporary setback. What resists a transferred attack is a training trajectory decorrelated from the surrogate's, and either an independent batch order or VLM's noisy labels achieves this — the two are redundant rather than additive (58.9% combined vs 58.2% from noisy labels alone). A cheap ~70%-accurate teacher therefore hardens the solver at least as well as a perfect oracle; what it costs is ~2-6pp of clean accuracy, not robustness.

## Ethical Considerations

This research is conducted for academic purposes to advance understanding of vision-language model capabilities and GUI automation systems. As with any security research, our techniques could potentially be misused for unauthorized automation. We do not endorse such applications and emphasize that circumventing security measures without authorization is unethical and often illegal. However, advancing scientific understanding of AI capabilities requires studying both strengths and limitations of deployed systems.
