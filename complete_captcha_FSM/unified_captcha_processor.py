"""
Unified CAPTCHA Processor

This unifies the two workflows:
1. YOLO workflow: YOLO detection + YOLO classification + YOLO segmentation
2. LLM workflow: YOLO detection + LLM classification + LLM segmentation

Common functionality is shared, with pluggable backends for classification/segmentation.
"""

import cv2
import numpy as np
import pyautogui
import easyocr
import platform
from ultralytics import YOLO
import os
import time
import tempfile
import json
from datetime import datetime
from logger import logger
from config import config
# === METRICS START ===
from metrics import metrics
# === METRICS END ===
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from universal_object_analyzer import VisionLLMClient


def read_api_url_from_config():
    """Read API URL from config.json file"""
    config_file = 'config.json'
    default_url = ''

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            return config_data.get('api_url', default_url)
    except Exception:
        pass

    return default_url


class ClassificationBackend(ABC):
    """Abstract base class for classification backends"""
    
    @abstractmethod
    def classify(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Abstract method: Classify a single CAPTCHA cell image.

        Implementations should analyze the image and return classification results
        indicating which CAPTCHA object is present (e.g., traffic light, crosswalk, etc.)

        Returns:
            Dict with classification results including predicted class and confidence
        """
        pass
    
    @property
    @abstractmethod
    def names(self) -> Dict[int, str]:
        """
        Abstract property: Return mapping of class IDs to class names.
        """
        pass


class SegmentationBackend(ABC):
    """Abstract base class for segmentation backends"""

    @abstractmethod
    def segment(self, image_path: str, object_name: str, **kwargs) -> Dict[str, Any]:
        """
        Abstract method: Segment and locate objects in a full CAPTCHA grid image.

        Implementations should detect all instances of the target object across the
        entire grid and return bounding boxes or masks for each instance.

        Args:
            image_path: Path to the full CAPTCHA grid image
            object_name: Target object to find (e.g., 'traffic light', 'crosswalk')

        Returns:
            Dict with segmentation results including bounding boxes and masks
        """
        pass

    @property
    @abstractmethod
    def names(self) -> Dict[int, str]:
        """
        Abstract property: Return mapping of class IDs to class names.
        """
        pass


class YOLOClassificationBackend(ClassificationBackend):
    """YOLO-based classification backend"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def classify(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Run YOLO classification"""
        results = self.model(image_path, **kwargs)
        # Convert YOLO results to standardized format
        return {
            'success': True,
            'results': results,
            'predictions': [r.probs for r in results] if results else []
        }
    
    @property
    def names(self) -> Dict[int, str]:
        return self.model.names


class YOLOSegmentationBackend(SegmentationBackend):
    """YOLO-based segmentation backend"""
    # Use large yolo model as default
    def __init__(self, model_path: str = 'yolov8x-seg.pt'):
        self.model = YOLO(model_path)
        
    def segment(self, image_path: str, object_name: str, **kwargs) -> Dict[str, Any]:
        """Run YOLO segmentation"""
        results = self.model(image_path, **kwargs)
        # Convert YOLO results to standardized format
        return {
            'success': True,
            'results': results,
            'detections': [r.boxes for r in results] if results else [],
            'masks': [r.masks for r in results] if results else []
        }
    
    @property
    def names(self) -> Dict[int, str]:
        return self.model.names


class UnifiedCaptchaProcessor:
    """
    Unified CAPTCHA Processor supporting both YOLO and LLM backends.
    
    Usage:
        # YOLO workflow
        processor = UnifiedCaptchaProcessor(backend='yolo')
        
        # LLM workflow  
        processor = UnifiedCaptchaProcessor(backend='llm')
    """
    
    # Hybrid cascade threshold per paper Eq. 3 (tau = 0.70).
    #
    # Per cell:
    #   max(softmax) >= HYBRID_TAU_SELECT  -> trust YOLO (select if argmax == target,
    #                                        else reject)
    #   max(softmax) <  HYBRID_TAU_SELECT  -> VLM fallback with paper Appendix H
    #                                        0-15 numbered-class prompt
    # Targets unsupported by YOLO (e.g. Boat, Taxi, Tractor) route the entire
    # puzzle to VLM (paper Eq. 3 branch 1).
    HYBRID_TAU_SELECT = 0.70

    # Paper class set (16 classes, indices match Appendix H prompt).
    PAPER_CLASS_NAMES = (
        'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
        'Crosswalk', 'Hydrant', 'Motorcycle', 'Mountain',
        'Other', 'Palm', 'Stairs', 'Traffic Light',
        'Boat', 'Taxi', 'Tractor',
    )
    
    def __init__(self, backend: str = 'yolo'):
        """
        Initialize with specified backend

        Args:
            backend: 'yolo', 'llm', or 'hybrid' (intelligent per-object backend selection)
        """
        if backend not in ['yolo', 'llm', 'hybrid']:
            raise ValueError(f"Invalid backend: {backend}. Must be 'yolo', 'llm', or 'hybrid'.")

        self.backend_type = backend.lower()
        
        # Initialize display scaling
        self._init_display_scaling()
        
        # Initialize detection model (SHARED - always YOLO)
        self._init_detection_model()
        
        # Initialize classification and segmentation backends (PLUGGABLE)
        self._init_backends()
        
        # Initialize OCR (SHARED)
        self._init_ocr()
        
        # Backend selection mapping: which backend performs best for each object type
        # This allows hybrid selection - using YOLO for some objects and LLM for others
        # Not used for cascade method
        self._backend_preference = {
            # YOLO-friendly objects (better performance with YOLO for both types)
            # Including both singular and plural forms for reliable matching
            'bridge': {'classification': 'yolo', 'segmentation': 'yolo'},
            'bridges': {'classification': 'yolo', 'segmentation': 'yolo'},
            'bus': {'classification': 'yolo', 'segmentation': 'yolo'},
            'buses': {'classification': 'yolo', 'segmentation': 'yolo'},
            'car': {'classification': 'yolo', 'segmentation': 'yolo'},
            'cars': {'classification': 'yolo', 'segmentation': 'yolo'},
            'crosswalk': {'classification': 'yolo', 'segmentation': 'yolo'},
            'crosswalks': {'classification': 'yolo', 'segmentation': 'yolo'},
            'hydrant': {'classification': 'yolo', 'segmentation': 'yolo'},
            'hydrants': {'classification': 'yolo', 'segmentation': 'yolo'},
            'fire hydrant': {'classification': 'yolo', 'segmentation': 'yolo'},
            'fire hydrants': {'classification': 'yolo', 'segmentation': 'yolo'},
            'palm': {'classification': 'yolo', 'segmentation': 'yolo'},
            'palms': {'classification': 'yolo', 'segmentation': 'yolo'},
            'palm tree': {'classification': 'yolo', 'segmentation': 'yolo'},
            'palm trees': {'classification': 'yolo', 'segmentation': 'yolo'},
            'bicycle': {'classification': 'yolo', 'segmentation': 'yolo'},
            'bicycles': {'classification': 'yolo', 'segmentation': 'yolo'},
            'boat': {'classification': 'llm', 'segmentation': 'llm'},
            'boats': {'classification': 'llm', 'segmentation': 'llm'},
            'chimney': {'classification': 'llm', 'segmentation': 'llm'},
            'chimneys': {'classification': 'llm', 'segmentation': 'llm'},
            'motorcycle': {'classification': 'yolo', 'segmentation': 'yolo'},
            'motorcycles': {'classification': 'yolo', 'segmentation': 'yolo'},
            'mountain': {'classification': 'llm', 'segmentation': 'llm'},
            'mountains': {'classification': 'llm', 'segmentation': 'llm'},
            'stair': {'classification': 'llm', 'segmentation': 'llm'},
            'stairs': {'classification': 'llm', 'segmentation': 'llm'},
            'taxi': {'classification': 'llm', 'segmentation': 'llm'},
            'taxis': {'classification': 'llm', 'segmentation': 'llm'},
            'tractor': {'classification': 'llm', 'segmentation': 'llm'},
            'tractors': {'classification': 'llm', 'segmentation': 'llm'},
            'traffic light': {'classification': 'yolo', 'segmentation': 'yolo'},
            'traffic lights': {'classification': 'yolo', 'segmentation': 'yolo'},

            # Default: Any object not listed here will use LLM (safer fallback)
        }

    def _init_display_scaling(self):
        """Initialize display scaling factor"""
        try:
            logical_width, logical_height = pyautogui.size()
            screenshot = pyautogui.screenshot()
            actual_width, actual_height = screenshot.size
            
            scale_x = actual_width / logical_width
            scale_y = actual_height / logical_height
            self.display_scaling = (scale_x + scale_y) / 2

        except Exception as e:
            self.display_scaling = 1.0
    
    def _init_detection_model(self):
        """Initialize YOLO detection model (always YOLO)"""
        self.detection_model_path = "detection_model.pt"
        
        if not os.path.exists(self.detection_model_path):
            if os.path.exists("runs/detect/yolo12_multi_scale_5class3/weights/best.pt"):
                self.detection_model_path = "runs/detect/yolo12_multi_scale_5class3/weights/best.pt"
            else:
                raise FileNotFoundError(f"Detection model not found at: {self.detection_model_path}")
        
        self.detection_model = YOLO(self.detection_model_path)
        
        self.confidence_thresholds = {
            0: 0.25,  # captcha_area
            1: 0.25,  # cell
            2: 0.15,  # reload_button
            3: 0.20,  # submit_button or verify
            4: 0.30   # robot_checkbox - Higher threshold to reduce false positives
        }
    
    def _init_backends(self):
        """Initialize classification and segmentation backends (LLM or YOLO)"""
        if self.backend_type == 'yolo':
            # YOLO backends
            classification_model_path = "classification_model.pt"
            if not os.path.exists(classification_model_path):
                if os.path.exists("best.pt"):
                    classification_model_path = "best.pt"
                else:
                    raise FileNotFoundError(f"Classification model not found")

            self.classification_backend = YOLOClassificationBackend(classification_model_path)
            self.segmentation_backend = YOLOSegmentationBackend('yolov8x-seg.pt')

        elif self.backend_type == 'llm':
            # LLM backend (hybrid: YOLO detection + LLM solving)
            api_url = read_api_url_from_config()

            # Store api_url for direct LLM methods
            self.api_url = api_url
            self.analyzer = VisionLLMClient(api_url)

            # TODO: LLM mode doesn't use backend classes but it should have - classification and segmentation are 
            # handled by _solve_classification_captcha_llm_streaming() and _solve_segmentation_captcha_llm() directly

        elif self.backend_type == 'hybrid':
            # HYBRID mode: Initialize BOTH backends for dynamic selection
            # Initialize YOLO backends
            classification_model_path = "classification_model.pt"
            if not os.path.exists(classification_model_path):
                if os.path.exists("best.pt"):
                    classification_model_path = "best.pt"
                else:
                    raise FileNotFoundError(f"Classification model not found")

            self.yolo_classification_backend = YOLOClassificationBackend(classification_model_path)
            self.yolo_segmentation_backend = YOLOSegmentationBackend('yolov8x-seg.pt')

            # Initialize LLM components (no backend classes needed - LLM uses direct method calls)
            api_url = read_api_url_from_config()
            self.api_url = api_url
            self.analyzer = VisionLLMClient(api_url)

            # LLM classification uses _solve_classification_captcha_llm_streaming() directly
            # LLM segmentation uses _solve_segmentation_captcha_llm() directly
            # Set default to None - will be assigned dynamically per puzzle
            self.classification_backend = None
            self.segmentation_backend = None

        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")
    
    def _init_ocr(self):
        """Initialize OCR"""
        use_gpu = True if platform.system() != 'Darwin' else False
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    
    # For compatibility
    # TBD
    @property
    def classification_model(self):
        if self.backend_type == 'yolo':
            return self.classification_backend.model  # Return actual YOLO model
        elif self.backend_type == 'hybrid':
            # Hybrid mode: Return YOLO classification model (used when hybrid selects YOLO for current object)
            # When hybrid selects LLM, it uses _solve_classification_captcha_llm_streaming() which doesn't access this property
            return self.yolo_classification_backend.model
        else:
            # LLM mode: This property is not accessed (LLM classification uses _solve_classification_captcha_llm_streaming directly)
            raise AttributeError("classification_model should not be accessed in LLM-only mode")

    @property
    def segmentation_model(self):
        """Segmentation model compatibility layer - returns actual YOLO model for YOLO backend"""
        if self.backend_type == 'yolo':
            return self.segmentation_backend.model  # Return actual YOLO model
        elif self.backend_type == 'hybrid':
            # Hybrid mode: Return YOLO segmentation model (used when hybrid selects YOLO for current object)
            # When hybrid selects LLM, it uses _solve_segmentation_captcha_llm() which doesn't access this property
            return self.yolo_segmentation_backend.model
        else:
            # LLM mode: This property is not accessed (LLM segmentation uses _solve_segmentation_captcha_llm directly)
            raise AttributeError("segmentation_model should not be accessed in LLM-only mode")
    
    # All methods below are identical between workflows
    def _detect_captcha_type(self, ocr_text):
        """Detect if CAPTCHA is classification or segmentation based on OCR text """
        text = ocr_text.lower()
        
        segmentation_phrases = [
            "select all squares", "click on all squares", 
            "select squares", "click squares"
        ]
        
        classification_phrases = [
            "select all images", "click on all images",
            "select all pictures", "click on the images",
            "select images", "click images"
        ]
        
        for phrase in segmentation_phrases:
            if phrase in text:
                return "segmentation"
        
        for phrase in classification_phrases:
            if phrase in text:
                return "classification"
        
        return "classification"
    
    def find_object_in_captcha_area(self, class_id, captcha_area_coords=None):
        """Find object within the CAPTCHA area"""
        if captcha_area_coords is None:
            captcha_area_coords = self.find_object_on_screen(class_id=0)
            if not captcha_area_coords:
                return None
        
        x1, y1, x2, y2 = captcha_area_coords
        width, height = x2 - x1, y2 - y1
        
        # Apply display scaling
        logical_x1 = int(x1 / self.display_scaling)
        logical_y1 = int(y1 / self.display_scaling)
        logical_width = int(width / self.display_scaling)
        logical_height = int(height / self.display_scaling)
        
        # Take screenshot of CAPTCHA area
        captcha_screenshot = pyautogui.screenshot(region=(logical_x1, logical_y1, logical_width, logical_height))
        
        # Convert to numpy array
        captcha_np = np.array(captcha_screenshot)
        captcha_bgr = cv2.cvtColor(captcha_np, cv2.COLOR_RGB2BGR)
        
        # Run detection
        confidence_threshold = self.confidence_thresholds.get(class_id, 0.25)
        results = self.detection_model(captcha_bgr, verbose=False, conf=confidence_threshold)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls) == class_id:
                        x1_rel, y1_rel, x2_rel, y2_rel = box.xyxy[0].cpu().numpy()
                        
                        # Convert back to fullscreen coordinates
                        abs_x1 = int(x1 + x1_rel * self.display_scaling)
                        abs_y1 = int(y1 + y1_rel * self.display_scaling)
                        abs_x2 = int(x1 + x2_rel * self.display_scaling)
                        abs_y2 = int(y1 + y2_rel * self.display_scaling)
                        
                        return (abs_x1, abs_y1, abs_x2, abs_y2)
        
        return None
    
    def find_object_on_screen(self, class_id):
        """Find object on fullscreen using YOLO detection

        IMPORTANT: Detection may fail on large displays at default zoom levels.
        The checkbox appears too small for the model to detect reliably. Browser zoom may be required.
        TODO: Retrain with more diverse resolution samples to handle large displays better.
        """
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Use YOLO ui detection for both 'yolo' and 'llm' backends
        confidence_threshold = self.confidence_thresholds.get(class_id, 0.25)
        results = self.detection_model(screenshot_bgr, verbose=False, conf=confidence_threshold)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls) == class_id:
                        # Validate detection with OCR for critical UI elements
                        if class_id in [0, 4]:  # captcha_area or robot_checkbox
                            if not self._validate_with_ocr(screenshot_bgr, box.xyxy[0].cpu().numpy(), class_id):
                                logger.log(f"Skipping false positive for class_id={class_id}")
                                continue  # Skip this detection, try next one

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        return (int(x1), int(y1), int(x2), int(y2))

        return None
    def _click_center(self, coords):
        """Click the center of given coordinates with display scaling correction"""
        if not coords:
            return False

        x1, y1, x2, y2 = coords
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Apply display scaling correction for high-DPI displays
        if self.display_scaling and self.display_scaling > 0 and self.display_scaling != 1.0:
            scaled_x = int(center_x / self.display_scaling)
            scaled_y = int(center_y / self.display_scaling)
            logger.always_log(f"Scaling click: ({center_x}, {center_y}) -> ({scaled_x}, {scaled_y}) [display_scaling={self.display_scaling}]")
            center_x, center_y = scaled_x, scaled_y
        
        if config.get_mouse_movement():
            pyautogui.moveTo(center_x, center_y, duration=0.25)
            pyautogui.click()
        else:
            pyautogui.click(center_x, center_y)
        return True
    
    def solve_captcha(self, image_input):
        """Main CAPTCHA solving method - UNIFIED IMPLEMENTATION"""
        # Initialize coordinate tracking flag
        using_global_coords = False

        # Handle input
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                return {'success': False, 'error': f'Could not read image from path: {image_input}'}
        elif isinstance(image_input, np.ndarray):
            frame = image_input
            if frame is None:
                logger.log("Error: image_input is None")
                return {'success': False, 'error': 'Frame is None'}
        else:
            return {'success': False, 'error': 'Invalid input type for solve_captcha.'}


        # Run OCR
        ocr_results = self.ocr_reader.readtext(frame, detail=0, paragraph=True)
        full_text = " ".join(ocr_results).lower()

        # Detect CAPTCHA type
        captcha_type = self._detect_captcha_type(full_text)


        # Extract target object
        target_object = "unknown"
        original_target = "unknown"
        import re

        # For LLM and HYBRID backends, use improved extraction (supports all objects)
        if self.backend_type in ['llm', 'hybrid']:
            target_object = self._extract_clean_object_name(full_text) or "unknown"
            original_target = target_object
        else:
            # For YOLO-only backend, check supported classes
            for class_name in self.classification_model.names.values():
                lower_class_name = class_name.lower()
                if lower_class_name in full_text or (lower_class_name + 's') in full_text:
                    target_object = class_name
                    original_target = class_name
                    break

            # If not found, try pattern extraction
            if target_object == "unknown":
                patterns = [
                    r"select all (?:images|squares) with ([a-z]+)",
                    r"select all ([a-z]+)",
                    r"click on all ([a-z]+)",
                    r"click on the ([a-z]+)"
                ]
                for pattern in patterns:
                    match = re.search(pattern, full_text)
                    if match:
                        original_target = match.group(1)
                        if original_target.endswith('s') and len(original_target) > 3:
                            singular = original_target[:-1]
                            if singular in ['taxi', 'bus', 'boat']:
                                original_target = singular
                        # Give up and mark as unknown to trigger reload
                        target_object = 'unknown'
                        break


        # Grid cell detection
        logger.always_log("=== GRID CELL DETECTION STARTING ===")
        cell_result = self.get_cell_data_from_image(frame)
        all_cells_data = cell_result.get('cells', [])

        # Fallback to fullscreen if needed
        if not cell_result.get('success', False):
            logger.always_log("Grid detection failed on CAPTCHA area")
            logger.always_log("Trying fallback: Fullscreen detection...")
            try:
                full_screenshot = pyautogui.screenshot()
                full_screenshot_np = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
                fallback_result = self.get_cell_data_from_image(full_screenshot_np)
                if fallback_result.get('success', False):
                    logger.always_log(f"Fallback SUCCESS! Found {len(fallback_result.get('cells', []))} cells")
                    cell_result = fallback_result
                    all_cells_data = fallback_result.get('cells', [])
                    frame = full_screenshot_np
                    using_global_coords = True
            except Exception as e:
                logger.always_log(f"Fallback error: {e}")


        # Check if we can solve
        if target_object == "unknown" or original_target == "unknown":
            # For LLM/HYBRID: Use full OCR text as target (let LLM figure it out from context)
            if self.backend_type in ['llm', 'hybrid']:
                logger.always_log(f"Could not extract clean target object from OCR")
                logger.always_log(f"   Using full OCR text as target for LLM: '{full_text}'")
                logger.always_log(f"   LLM will analyze the full instruction to determine the target.")
                target_object = full_text  # Pass full OCR text to LLM
                original_target = full_text
            else:
                # For YOLO-only: Reload puzzle (YOLO needs a specific class name)
                logger.always_log(f"Cannot solve - target object '{original_target}' not recognized")
                logger.always_log(f"   This CAPTCHA target is not supported by the YOLO model.")
                logger.always_log(f"   Action: Reloading to get a solvable puzzle.")
                return {
                    'success': False,
                    'error': f'Target object "{original_target}" not supported',
                    'captcha_type': captcha_type,
                    'target_object': original_target,
                    'cells_extracted': True,
                    'all_cells_data': all_cells_data,
                    'selected_cells': [],
                    'retry_needed': True,  # Signal FSM to reload
                    'coordinates_are_global': using_global_coords
                }

        # Check cell count
        if not cell_result['success']:
            error_msg = f'Cell extraction issues - found {len(all_cells_data)} cells'
            if cell_result.get('retry_needed', False):
                error_msg = 'Invalid grid cell count detected - puzzle reload needed'
            return {
                'success': False,
                'error': error_msg,
                'captcha_type': captcha_type,
                'target_object': original_target,
                'cells_extracted': len(all_cells_data) > 0,
                'all_cells_data': all_cells_data,
                'selected_cells': [],
                'retry_needed': cell_result.get('retry_needed', False),
                'coordinates_are_global': using_global_coords
            }

        # CRITICAL VALIDATION: Check cell count before backend selection
        # This allows us to retry with fullscreen if validation fails
        if captcha_type == "segmentation" and len(all_cells_data) != 16 and not using_global_coords:
            logger.always_log(f"VALIDATION: Segmentation needs 16 cells but found {len(all_cells_data)}")
            logger.always_log(f"Trying fullscreen fallback before rejecting...")
            try:
                full_screenshot = pyautogui.screenshot()
                full_screenshot_np = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
                fallback_result = self.get_cell_data_from_image(full_screenshot_np)
                if fallback_result.get('success', False) and len(fallback_result.get('cells', [])) == 16:
                    logger.always_log(f"Fullscreen fallback SUCCESS! Found 16 cells")
                    cell_result = fallback_result
                    all_cells_data = fallback_result.get('cells', [])
                    frame = full_screenshot_np
                    using_global_coords = True
                else:
                    logger.always_log(f"Fullscreen fallback found {len(fallback_result.get('cells', []))} cells (still wrong)")
            except Exception as e:
                logger.always_log(f"Fullscreen fallback error: {e}")

        # HYBRID BACKEND SELECTION: Choose best backend for this specific object AND puzzle type
        # Only use hybrid selection if backend was set to 'hybrid'
        if self.backend_type == 'hybrid':
            
            # Classification puzzles use the per-cell confidence cascade from the paper
            # (Eq. 3, tau=0.70) instead of the static class-level _backend_preference table.
            # Segmentation still uses _backend_preference below.
            if captcha_type == 'classification':
                return self._solve_classification_captcha_hybrid_cascade(
                    frame, target_object, all_cells_data, using_global_coords
                )
            
            # Check if we have a preference for this target object
            preference_entry = self._backend_preference.get(target_object.lower(), None)

            # Determine which backend to use based on puzzle type
            if preference_entry is not None:
                if isinstance(preference_entry, dict):
                    # Separate preferences for classification vs segmentation
                    preferred_backend = preference_entry.get(captcha_type, None)
                    if preferred_backend is None:
                        preferred_backend = 'llm'
                        logger.always_log(f"HYBRID: '{target_object}' found but puzzle type '{captcha_type}' not in preference. Defaulting to LLM")
                    else:
                        logger.always_log(f"HYBRID MODE: Selected {preferred_backend.upper()} backend for '{target_object}' ({captcha_type} puzzle)")
                else:
                    preferred_backend = preference_entry
                    logger.always_log(f"HYBRID MODE: Selected {preferred_backend.upper()} backend for '{target_object}'")
            else:
                # Object not in preference list - default to LLM (safer for unknown/new objects)
                preferred_backend = 'llm'
                logger.always_log(f"HYBRID: '{target_object}' not in preference list. Defaulting to LLM backend (safer for new objects)")

            # Assign the selected backends for this puzzle
            if preferred_backend == 'yolo':
                self.classification_backend = self.yolo_classification_backend
                self.segmentation_backend = self.yolo_segmentation_backend
            else:  # llm
                # Note: LLM doesn't use backend classes - classification and segmentation use direct method calls
                # self.classification_backend and self.segmentation_backend remain None for LLM.
                # TBD
                pass

            selected_backend = preferred_backend
        else:
            # Non-hybrid mode: Use the specified backend exclusively
            selected_backend = self.backend_type

        # Solve based on type and backend
        # FINAL VALIDATION: After all retries, check if cell count is correct
        if captcha_type == "segmentation":
            if len(all_cells_data) != 16:
                # Segmentation CAPTCHAs MUST have exactly 16 cells (4x4 grid)
                logger.always_log(f"VALIDATION ERROR: Segmentation CAPTCHA detected but found {len(all_cells_data)} cells instead of 16!")
                logger.always_log(f"   This indicates grid detection failure.")
                logger.always_log(f"   Segmentation CAPTCHAs are always 4x4 (16 cells), never 3x3 (9 cells).")
                return {
                    'success': False,
                    'error': f'Segmentation CAPTCHA requires 16 cells but found {len(all_cells_data)}',
                    'captcha_type': captcha_type,
                    'target_object': original_target,
                    'cells_extracted': True,
                    'all_cells_data': all_cells_data,
                    'selected_cells': [],
                    'retry_needed': True,  # Signal FSM to reload
                    'coordinates_are_global': using_global_coords
                }
            # Valid segmentation CAPTCHA (16 cells)
            logger.log(f"Using segmentation approach ({selected_backend.upper()} backend)")
            if selected_backend == 'yolo':
                return self._solve_segmentation_captcha(frame, target_object, all_cells_data, using_global_coords)
            else:  # llm
                return self._solve_segmentation_captcha_llm(frame, target_object, all_cells_data)
        else:
            # Classification CAPTCHA - should have 9 cells (3x3 grid)
            if len(all_cells_data) != 9:
                logger.always_log(f"WARNING: Classification CAPTCHA detected but found {len(all_cells_data)} cells instead of 9.")
                logger.always_log(f"   Expected 3x3 grid (9 cells) for classification.")
                logger.always_log(f"   Proceeding anyway but this may indicate detection issues.")

            logger.log(f"Using classification approach ({selected_backend.upper()} backend)")
            if selected_backend == 'yolo':
                return self._solve_classification_captcha(frame, target_object, all_cells_data, using_global_coords)
            else:  # llm
                return self._solve_classification_captcha_llm_streaming(frame, target_object, all_cells_data)

    def get_cell_data_from_image(self, image_input, max_retries=3):
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                logger.log(f"Error: Could not read image from path: {image_input}")
                return {'success': False, 'cells': [], 'retry_needed': False}
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            raise TypeError("Invalid input for get_cell_data_from_image: must be path or numpy array.")

        # Try different confidence thresholds to get the correct number of cells
        confidence_levels = [0.25, 0.20, 0.15, 0.30, 0.35]  # Start with default, then try lower and higher

        for retry_count in range(max_retries):
            conf_threshold = confidence_levels[min(retry_count, len(confidence_levels) - 1)]

            results = self.detection_model(frame, verbose=False, conf=conf_threshold)
            cell_data = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                    if int(cls) == 1: # Class ID 1 is 'cell'
                        cell_data.append({'coords': tuple(map(int, box))})

            original_count = len(cell_data)

            # Filter cells to identify the main grid (handles non-standard layouts)
            cell_data = self._filter_main_grid_cells(cell_data)

            filtered_count = len(cell_data)

            # Properly sort cells into grid positions
            cell_data = self._sort_cells_into_grid(cell_data)

            cell_count = len(cell_data)

            # Validate cell count - should be exactly 9 (3x3) or 16 (4x4)
            if cell_count == 9 or cell_count == 16:
                indexed_cells = [{'cell_index': i, 'coords': data['coords']} for i, data in enumerate(cell_data)]
                return {'success': True, 'cells': indexed_cells, 'retry_needed': False}

        # If we get here, all retries failed to find correct cell count
        logger.always_log(f"GRID DETECTION FAILED: Could not find 9 or 16 cells after {max_retries} attempts")
        logger.always_log(f"   Last attempt found: {len(cell_data)} cells")
        logger.always_log(f"   Tried confidence levels: {confidence_levels[:max_retries]}")

        # Return the last attempt's data but mark as needing retry (reload)
        indexed_cells = [{'cell_index': i, 'coords': data['coords']} for i, data in enumerate(cell_data)]
        return {'success': False, 'cells': indexed_cells, 'retry_needed': True}

    def _filter_main_grid_cells(self, cell_data):
        """Filter cells to identify and keep only the main grid (3x3 or 4x4)"""
        if len(cell_data) <= 16:
            # If we have 16 or fewer cells, no filtering needed
            return cell_data
        
        # Calculate center points for each cell
        cells_with_centers = []
        for i, cell in enumerate(cell_data):
            x1, y1, x2, y2 = cell['coords']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            cells_with_centers.append({
                'coords': cell['coords'],
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'area': area,
                'index': i
            })
        
        # Try to find clusters that could form a 3x3 or 4x4 grid
        best_cluster = None
        best_score = 0
        
        for target_grid_size in [3, 4]:  # Try 3x3 and 4x4
            target_cell_count = target_grid_size * target_grid_size
            
            # Find the largest cluster of cells that could form this grid
            cluster = self._find_rectangular_cluster(cells_with_centers, target_grid_size)

            if len(cluster) == target_cell_count:
                # Perfect match - we found a complete grid
                best_cluster = cluster
                break
            elif len(cluster) > best_score:
                # This is our best candidate so far
                best_score = len(cluster)
                best_cluster = cluster
        
        if best_cluster and len(best_cluster) >= 9:
            # Return the original cell data format
            filtered_cells = [{'coords': cell['coords']} for cell in best_cluster]
            return filtered_cells
        else:
            # If we can't find a good cluster, return the original data
            return cell_data

    def _find_rectangular_cluster(self, cells_with_centers, grid_size):
        """Find the largest rectangular cluster of cells that could form a grid"""
        if len(cells_with_centers) < grid_size * grid_size:
            return []
        
        # Sort cells by position to help with clustering
        cells = sorted(cells_with_centers, key=lambda c: (c['center_y'], c['center_x']))
        
        # Find cells with similar sizes (main grid cells should be roughly the same size)
        median_area = sorted([c['area'] for c in cells])[len(cells) // 2]
        area_tolerance = median_area * 0.5  # Allow 50% variance
        
        similar_size_cells = [c for c in cells if abs(c['area'] - median_area) <= area_tolerance]
        
        if len(similar_size_cells) < grid_size * grid_size:
            # Not enough similar-sized cells, try with all cells
            similar_size_cells = cells
        
        # Try to form a rectangular grid from these cells
        best_cluster = self._extract_rectangular_region(similar_size_cells, grid_size)
        
        return best_cluster

    def _extract_rectangular_region(self, cells, grid_size):
        """Extract a rectangular region that could form a grid"""
        if len(cells) < grid_size * grid_size:
            return []
        
        # Group cells by approximate rows (Y coordinate)
        cells.sort(key=lambda c: c['center_y'])
        
        # Use clustering to find row groups
        rows = []
        current_row = [cells[0]]
        row_tolerance = 30  # Increased tolerance for grouping
        
        for cell in cells[1:]:
            if abs(cell['center_y'] - current_row[-1]['center_y']) <= row_tolerance:
                current_row.append(cell)
            else:
                rows.append(current_row)
                current_row = [cell]
        if current_row:
            rows.append(current_row)
        
        # Find the group of consecutive rows that could form our grid
        best_group = []

        # Try each possible starting position for a grid_size x grid_size grid
        for start_row in range(len(rows) - grid_size + 1):
            end_row = start_row + grid_size - 1
            candidate_rows = rows[start_row:end_row + 1]

            # Check if these rows could form a rectangular grid
            valid_grid = True
            grid_cells = []

            for row in candidate_rows:
                # Sort row by X coordinate
                row.sort(key=lambda c: c['center_x'])

                # We need exactly grid_size cells per row
                if len(row) >= grid_size:
                    # Take the first grid_size cells from this row
                    grid_cells.extend(row[:grid_size])
                else:
                    valid_grid = False
                    break

            if valid_grid and len(grid_cells) == grid_size * grid_size:
                # This is a valid rectangular region
                if len(grid_cells) > len(best_group):
                    best_group = grid_cells
        
        return best_group

    def _detect_dynamic_captcha(self, frame):
        """Detect if this is a dynamic CAPTCHA based on OCR text analysis"""
        try:
            # Extract text from the image using OCR
            ocr_results = self.ocr_reader.readtext(frame, detail=0, paragraph=True)
            full_text = " ".join(ocr_results).lower()

            # Dynamic text indicators (same as YOLO implementation)
            dynamic_indicators = [
                "none left",
                "no more",
                "keep selecting",
                "new images",
                "if there are none",
                "click verify once",
                "until"
            ]

            # Check for dynamic indicators in text
            has_dynamic_text = any(indicator in full_text for indicator in dynamic_indicators)

            logger.log(f"Dynamic CAPTCHA detection: {'YES' if has_dynamic_text else 'NO'}")
            logger.log(f"   Text: '{full_text[:100]}...'")
            logger.log(f"   Dynamic indicators found: {has_dynamic_text}")

            return has_dynamic_text

        except Exception as e:
            logger.log(f"Error detecting dynamic captcha: {e}")
            return False

    def _sort_cells_into_grid(self, cell_data):
        """Properly sort detected cells into correct grid positions"""
        if not cell_data:
            return cell_data

        cell_count = len(cell_data)

        # Determine grid size
        if cell_count == 9:
            grid_size = 3
        elif cell_count == 16:
            grid_size = 4
        else:
            # Fall back to simple sort
            cell_data.sort(key=lambda c: (c['coords'][1], c['coords'][0]))
            return cell_data

        # Get center points for each cell
        cells_with_centers = []
        for cell in cell_data:
            x1, y1, x2, y2 = cell['coords']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cells_with_centers.append({
                'coords': cell['coords'],
                'center_x': center_x,
                'center_y': center_y
            })

        # Group cells by rows using Y-coordinate clustering
        cells_with_centers.sort(key=lambda c: c['center_y'])

        # Determine row boundaries by grouping cells with similar Y coordinates
        rows = []
        current_row = [cells_with_centers[0]]
        row_tolerance = 20  # pixels tolerance for row grouping

        for cell in cells_with_centers[1:]:
            if abs(cell['center_y'] - current_row[-1]['center_y']) <= row_tolerance:
                current_row.append(cell)
            else:
                rows.append(current_row)
                current_row = [cell]
        rows.append(current_row)  # Add the last row

        # Ensure we have the right number of rows
        if len(rows) != grid_size:
            cell_data.sort(key=lambda c: (c['coords'][1], c['coords'][0]))
            return cell_data

        # Sort each row by X coordinate
        for row in rows:
            row.sort(key=lambda c: c['center_x'])

        # Validate each row has the correct number of cells
        for i, row in enumerate(rows):
            if len(row) != grid_size:
                cell_data.sort(key=lambda c: (c['coords'][1], c['coords'][0]))
                return cell_data

        # Flatten rows back into sorted list
        sorted_cells = []
        for row in rows:
            for cell in row:
                sorted_cells.append({'coords': cell['coords']})

        return sorted_cells

    def _concatenate_cells_4x4(self, cells):
        """Concatenate 4x4 grid cells into single image for segmentation model"""
        if len(cells) != 16:
            return None

        # Get average cell dimensions
        heights = [cells[i]['coords'][3] - cells[i]['coords'][1] for i in range(len(cells))]
        widths = [cells[i]['coords'][2] - cells[i]['coords'][0] for i in range(len(cells))]
        avg_height = int(np.mean(heights))
        avg_width = int(np.mean(widths))

        # Sort cells properly into 4x4 grid based on spatial position
        # First, get unique Y positions (rows) and X positions (columns)
        y_positions = []
        x_positions = []

        for cell in cells:
            x1, y1, x2, y2 = cell['coords']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            y_positions.append(center_y)
            x_positions.append(center_x)

        # Find the 4 unique row and column positions
        sorted_y = sorted(set(y_positions))
        sorted_x = sorted(set(x_positions))

        # We should have exactly 4 rows and 4 columns
        if len(sorted_y) != 4 or len(sorted_x) != 4:
            # Fall back to simple grid approximation
            sorted_y = sorted(y_positions)[::4][:4]
            sorted_x = sorted(x_positions)[::4][:4]

        # Create concatenated image
        concat_height = avg_height * 4
        concat_width = avg_width * 4
        concatenated = np.zeros((concat_height, concat_width, 3), dtype=np.uint8)

        # Assign each cell to its proper grid position
        for cell in cells:
            x1, y1, x2, y2 = cell['coords']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cell_image = cell.get('image')

            if cell_image is not None:
                # Find which row and column this cell belongs to
                # Find closest row
                row = min(range(len(sorted_y)), key=lambda i: abs(sorted_y[i] - center_y))
                # Find closest column
                col = min(range(len(sorted_x)), key=lambda i: abs(sorted_x[i] - center_x))

                # Resize cell to average size
                resized = cv2.resize(cell_image, (avg_width, avg_height))

                # Place in grid
                y_start = row * avg_height
                x_start = col * avg_width
                concatenated[y_start:y_start + avg_height, x_start:x_start + avg_width] = resized

        return concatenated


    def _validate_with_ocr(self, frame, box, class_id):
        """
        Validate detection using OCR to reduce false positives.
        Returns True if the detection is valid, False otherwise.
        """
        x1, y1, x2, y2 = map(int, box)
        
        try:
            # Validate robot_checkbox (class 4)
            if class_id == 4:
                # Add padding to capture surrounding text
                pad = 10
                h, w, _ = frame.shape
                checkbox_crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                ocr_results = self.ocr_reader.readtext(checkbox_crop, detail=0, paragraph=False)
                full_text = " ".join(ocr_results).lower()
                
                # Check if both "not" AND "robot" are present
                if "not" in full_text and "robot" in full_text:
                    return True
                else:
                    logger.log(f"Discarded robot_checkbox false positive. Text found: '{full_text}'")
                    return False
                    
            # Validate captcha_area (class 0)
            elif class_id == 0:
                captcha_crop = frame[y1:y2, x1:x2]
                ocr_results = self.ocr_reader.readtext(captcha_crop, detail=0, paragraph=True)
                full_text = " ".join(ocr_results).lower()
                
                # Keywords to look for in a valid CAPTCHA area
                captcha_keywords = ['select all', 'images', 'squares', 'click', 'verify', 
                                    'motorcycle', 'bicycle', 'bus', 'car', 'traffic', 
                                    'crosswalk', 'hydrant', 'bridge', 'chimney', 'mountain',
                                    'palm', 'tree', 'boat', 'plane', 'truck', 'taxi']
                
                if any(keyword in full_text for keyword in captcha_keywords):
                    return True
                else:
                    return False
                    
            # For other classes, no OCR validation needed
            else:
                return True
                
        except Exception as e:
            logger.log(f"OCR validation failed for class {class_id}: {e}")
            # If OCR fails, be conservative and reject the detection
            return False if class_id in [0, 4] else True


    def _solve_segmentation_captcha(self, frame, target_object, all_cells_data, using_global_coords=False):
        """Solve segmentation CAPTCHA using YOLOv8x-seg xlarge model"""

        logger.always_log(f"=== SEGMENTATION SOLVER STARTING ===")
        logger.always_log(f"   Target object: '{target_object}'")
        logger.always_log(f"   Number of cells: {len(all_cells_data)}")

        # Add cell images to cell data
        h, w = frame.shape[:2]
        for cell in all_cells_data:
            x1, y1, x2, y2 = cell['coords']

            # Validate coordinates are within frame bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                logger.always_log(f"Segmentation cell {cell.get('cell_index', '?')}: Invalid coordinates [{x1},{y1},{x2},{y2}] for frame size {w}x{h}")
                return {'success': False, 'error': f'Invalid cell coordinates for segmentation', 'coordinates_are_global': using_global_coords}

            cell_image = frame[y1:y2, x1:x2]

            # Validate cell image is not empty
            if cell_image is None or cell_image.size == 0:
                logger.always_log(f"Segmentation cell {cell.get('cell_index', '?')}: Extracted image is empty!")
                return {'success': False, 'error': f'Empty cell image for segmentation', 'coordinates_are_global': using_global_coords}

            cell['image'] = cell_image
        
        # Concatenate cells for segmentation model
        concatenated = self._concatenate_cells_4x4(all_cells_data)
        if concatenated is None:
            return {'success': False, 'error': 'Failed to concatenate cells for segmentation model', 'coordinates_are_global': using_global_coords}

        # Run segmentation model
        logger.always_log(f"Running YOLOv8x-seg model on concatenated grid...")
        results = self.segmentation_model(concatenated, verbose=False)

        selected_cells = []
        h, w = concatenated.shape[:2]
        cell_w, cell_h = w // 4, h // 4

        # Check if YOLO detected anything at all
        if results[0].boxes is None or len(results[0].boxes) == 0:
            logger.always_log(f"YOLO segmentation model detected no objects in the grid")
        else:
            logger.always_log(f"YOLO detected {len(results[0].boxes)} objects in total")

        # Check if we have masks (segmentation model)
        has_masks = results[0].masks is not None and len(results[0].masks) > 0

        if results[0].boxes is not None:
            target_matches = 0
            detected_classes = {}

            # Prepare mask data if available
            masks_data = None
            if has_masks:
                logger.always_log(f"Using YOLO segmentation MASKS (pixel-level accuracy)")
                # Get masks as numpy arrays (shape: [N, H, W] where N is number of detections)
                masks_data = results[0].masks.data.cpu().numpy()
            else:
                logger.always_log(f"No masks available, falling back to bounding boxes")

            for idx, (box, cls, conf) in enumerate(zip(results[0].boxes.xyxy.cpu().numpy(),
                                     results[0].boxes.cls.cpu().numpy(),
                                     results[0].boxes.conf.cpu().numpy())):
                class_name = self.segmentation_model.names[int(cls)]

                # Track all detected classes
                if class_name not in detected_classes:
                    detected_classes[class_name] = 0
                detected_classes[class_name] += 1

                # Check if this matches our target (flexible matching)
                is_target_match = (class_name.lower() == target_object.lower() or
                                  target_object.lower() in class_name.lower() or
                                  class_name.lower() in target_object.lower())

                if is_target_match:
                    target_matches += 1

                    x1, y1, x2, y2 = map(int, box)
                    logger.always_log(f"Found {class_name} at [{x1},{y1},{x2},{y2}] conf={conf:.3f}")

                    # Log concatenated grid dimensions for debugging
                    logger.log(f"Concatenated grid size: {w}x{h}, cell size: {cell_w}x{cell_h}")

                    overlapping_cells = []

                    # Use masks if available (more accurate!)
                    if has_masks and idx < len(masks_data):
                        mask = masks_data[idx]  # Get mask for this detection
                        # Resize mask to match concatenated grid size
                        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                        # Check which cells have mask pixels
                        for grid_row in range(4):
                            for grid_col in range(4):
                                cell_x1 = grid_col * cell_w
                                cell_y1 = grid_row * cell_h
                                cell_x2 = cell_x1 + cell_w
                                cell_y2 = cell_y1 + cell_h
                                grid_cell_idx = grid_row * 4 + grid_col

                                # Extract mask region for this cell
                                cell_mask = mask_resized[cell_y1:cell_y2, cell_x1:cell_x2]

                                # Count pixels that belong to the object (mask value > 0.5)
                                mask_pixels = np.sum(cell_mask > 0.5)
                                cell_total_pixels = cell_w * cell_h
                                mask_percentage = (mask_pixels / cell_total_pixels) * 100

                                # Threshold: At least 1% of cell must be covered by mask
                                MIN_MASK_THRESHOLD = 1.0

                                if mask_percentage >= MIN_MASK_THRESHOLD:
                                    selected_cells.append(grid_cell_idx)
                                    overlapping_cells.append(grid_cell_idx)
                    else:
                        # Fallback to bounding box overlap (old method)
                        for grid_row in range(4):
                            for grid_col in range(4):
                                cell_x1 = grid_col * cell_w
                                cell_y1 = grid_row * cell_h
                                cell_x2 = cell_x1 + cell_w
                                cell_y2 = cell_y1 + cell_h
                                grid_cell_idx = grid_row * 4 + grid_col

                                # Check overlap using intersection area
                                overlap_x1 = max(x1, cell_x1)
                                overlap_y1 = max(y1, cell_y1)
                                overlap_x2 = min(x2, cell_x2)
                                overlap_y2 = min(y2, cell_y2)

                                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                    cell_area = cell_w * cell_h
                                    overlap_percentage = (overlap_area / cell_area) * 100
                                    MIN_OVERLAP_THRESHOLD = 1.0

                                    if overlap_percentage >= MIN_OVERLAP_THRESHOLD:
                                        selected_cells.append(grid_cell_idx)
                                        overlapping_cells.append(grid_cell_idx)

                    if overlapping_cells:
                        pass
                    else:
                        pass

            # Print summary of what YOLO detected
            logger.always_log(f"YOLO Detection Summary:")
            logger.always_log(f"   Target object: '{target_object}'")
            logger.always_log(f"   Target matches found: {target_matches}")
            if detected_classes:
                logger.always_log(f"   All detected objects: {dict(detected_classes)}")
            else:
                logger.always_log(f"   No objects detected by YOLO")

            if target_matches == 0 and detected_classes:
                logger.always_log(f"MISMATCH: YOLO detected objects, but NONE match target '{target_object}'")
                logger.always_log(f"   Consider checking if the target name matches YOLO class names")
            elif target_matches > 0 and len(selected_cells) == 0:
                logger.always_log(f"TARGET FOUND BUT NO CELLS SELECTED!")
                logger.always_log(f"   YOLO found {target_matches} '{target_object}' object(s)")
                logger.always_log(f"   But NO cells met the 1% overlap threshold")


        # Remove duplicates and sort
        selected_cells = sorted(set(selected_cells))

        # Print summary
        logger.always_log(f"=== SEGMENTATION SOLVER COMPLETE ===")
        logger.always_log(f"   Total cells in grid: {len(all_cells_data)}")
        logger.always_log(f"   Cells selected: {len(selected_cells)}")
        logger.always_log(f"   Selected cell indices: {selected_cells}")

        # === METRICS START ===
        metrics.add_yolo_grid()
        metrics.add_segmentation()
        

        return {
            'success': True,
            'target_object': target_object,
            'selected_cells': selected_cells,
            'all_cells_data': all_cells_data,
            'captcha_type': 'segmentation',
            'coordinates_are_global': using_global_coords
        }


    def _solve_classification_captcha(self, frame, target_object, all_cells_data, using_global_coords=False):
        """Solve classification CAPTCHA using original cell-by-cell approach"""

        logger.always_log(f"=== CLASSIFICATION SOLVER STARTING ===")
        logger.always_log(f"   Target object: '{target_object}'")
        logger.always_log(f"   Number of cells: {len(all_cells_data)}")

        # Check if this is a dynamic CAPTCHA by re-doing OCR on the frame
        ocr_results = self.ocr_reader.readtext(frame, detail=0, paragraph=True)
        full_text = " ".join(ocr_results).lower()

        # Detect if this is a dynamic CAPTCHA (images appear after clicking)
        # Key indicators:
        # Instructions about waiting for none left and
        # presence of "new" and "keep selecting" phrases

        dynamic_indicators = [
            "none left",
            "no more",
            "keep selecting",
            "new images",
            "click verify once there are none left",  
            "until"
        ]

        # Some objects are more commonly dynamic in reCAPTCHA
        dynamic_objects = ["crosswalk", "traffic light", "bus", "bicycle", "motorcycle", "car"]

        # Check for dynamic indicators in text
        has_dynamic_text = any(indicator in full_text for indicator in dynamic_indicators)

        # Check if target object is commonly dynamic AND we have "select all images with"
        # But only if we also have some dynamic text indicators
        is_likely_dynamic_object = (target_object.lower() in dynamic_objects and
                                   ("select all images with" in full_text or "select all squares with" in full_text) and
                                   has_dynamic_text)  # Require BOTH object type AND text indicators

        # CRITICAL: 4x4 grids (16 cells) are always segmentation
        # Dynamic CAPTCHAs only occur with 3x3 grids (9 cells)
        num_cells = len(all_cells_data)
        if num_cells == 16:
            is_dynamic = False
            logger.always_log(f"4x4 grid detected ({num_cells} cells) - forcing static mode (segmentation)")
        else:
            # Consider it dynamic if we have clear indicators
            # Object type alone is not enough - we need explicit text indicators
            is_dynamic = has_dynamic_text

        logger.always_log(f"Dynamic CAPTCHA detection: {'YES' if is_dynamic else 'NO'}")
        logger.always_log(f"   Grid size: {num_cells} cells ({'3x3' if num_cells == 9 else '4x4' if num_cells == 16 else 'unknown'})")
        logger.always_log(f"   Text: '{full_text[:100]}...'")
        logger.always_log(f"   Dynamic indicators found: {has_dynamic_text}")
        logger.always_log(f"   Likely dynamic object ({target_object}): {is_likely_dynamic_object}")

        # Map target object to YOLO class name
        original_target = target_object  # Keep original for logging
        target_lower = target_object.lower().strip()

        # Step 1: Direct mappings for compound/special names (most specific)
        name_mappings = {
            'fire hydrant': 'hydrant', 'fire hydrants': 'hydrant',
            'traffic light': 'traffic light', 'traffic lights': 'traffic light',
            'palm tree': 'palm', 'palm trees': 'palm',
        }

        if target_lower in name_mappings:
            target_object = name_mappings[target_lower]
            logger.always_log(f"Mapped '{original_target}' to '{target_object}' (compound name)")

        # Step 2: Case-insensitive exact match
        target_class_idx = next((k for k, v in self.classification_model.names.items() if v.lower() == target_object.lower()), None)

        # Step 3: Try plural removal ONLY for common plural patterns (safe fallback)
        if target_class_idx is None and target_object.endswith('s') and target_object.lower() not in ['bus', 'class']:
            singular = target_object[:-1]
            test_idx = next((k for k, v in self.classification_model.names.items() if v.lower() == singular.lower()), None)
            if test_idx is not None:
                logger.always_log(f"Mapped '{target_object}' to '{singular}' (plural form)")
                target_object = singular
                target_class_idx = test_idx

        if target_class_idx is None:
            logger.always_log(f"CLASSIFICATION ERROR: Target '{original_target}' not found in model classes")
            logger.always_log(f"   Available classes: {list(self.classification_model.names.values())}")
            logger.always_log(f"   This CAPTCHA cannot be solved with the current YOLO model.")
            logger.always_log(f"   Action: Reloading to get a different puzzle.")
            return {
                'success': False,
                'error': f'Target "{original_target}" not found in model classes',
                'retry_needed': True,  # Signal FSM to reload
                'coordinates_are_global': using_global_coords
            }

        logger.always_log(f"Target class index: {target_class_idx} ('{self.classification_model.names[target_class_idx]}')")
        logger.always_log(f"Starting batch classification (processing all {len(all_cells_data)} cells at once)...")

        selected_cells_indices = []
        classification_results = []  # Store results for visualization

        # Extract all cell images and validate
        cell_images = []
        valid_cells = []
        h, w = frame.shape[:2]

        for cell in all_cells_data:
            x1, y1, x2, y2 = cell['coords']

            # Validate coordinates are within frame bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                logger.always_log(f"Cell {cell['cell_index']}: Invalid coordinates [{x1},{y1},{x2},{y2}] for frame size {w}x{h}")
                logger.always_log(f"   Skipping this cell to prevent crash")
                classification_results.append({
                    'cell_index': cell['cell_index'],
                    'coords': (x1, y1, x2, y2),
                    'target_probability': 0.0,
                    'is_selected': False,
                    'all_probs': None
                })
                continue

            cell_image = frame[y1:y2, x1:x2]

            # Validate cell image is not empty
            if cell_image is None or cell_image.size == 0:
                logger.always_log(f"Cell {cell['cell_index']}: Extracted image is empty!")
                logger.always_log(f"   Coordinates: [{x1},{y1},{x2},{y2}], Frame size: {w}x{h}")
                classification_results.append({
                    'cell_index': cell['cell_index'],
                    'coords': (x1, y1, x2, y2),
                    'target_probability': 0.0,
                    'is_selected': False,
                    'all_probs': None
                })
                continue

            # Add to batch
            cell_images.append(cell_image)
            valid_cells.append(cell)

        # Batch inference (single YOLO call for all valid cells!)
        if cell_images:
            logger.always_log(f"Running batch classification on {len(cell_images)} valid cells...")
            batch_results = self.classification_model(cell_images, verbose=False, conf=0.20)

            # Phase 3: Process results
            for cell, class_results in zip(valid_cells, batch_results):
                x1, y1, x2, y2 = cell['coords']

                if class_results and len(class_results.probs.data) > target_class_idx:
                    target_prob = float(class_results.probs.data[target_class_idx])

                    # NOTE: Paper methodology vs implementation difference -- Only if backend is YOLO alone 
                    # Paper used top-1 class prediction for both YOLO and VLM to enable fair benchmarking.
                    # This implementation uses top-3 for YOLO to provide better robustness in deployment
                    # by showing confidence across multiple predictions for debugging and analysis.
                    # The actual selection decision still uses only the target class probability.
                    probs_array = class_results.probs.data.cpu().numpy()
                    top3_indices = probs_array.argsort()[-3:][::-1]
                    top3_classes = [(self.classification_model.names[i], probs_array[i]) for i in top3_indices]

                    # Store classification result for visualization
                    classification_results.append({
                        'cell_index': cell['cell_index'],
                        'coords': (x1, y1, x2, y2),
                        'target_probability': target_prob,
                        'is_selected': target_prob > 0.20,
                        'all_probs': class_results.probs.data.cpu().numpy()
                    })

                    # Print detailed results for each cell
                    selection_status = "SELECTED" if target_prob > 0.20 else "REJECTED"
                    logger.always_log(f"   Cell {cell['cell_index']}: {selection_status} (target={target_prob:.3f})")
                    logger.always_log(f"      Top-3: {top3_classes[0][0]}={top3_classes[0][1]:.3f}, {top3_classes[1][0]}={top3_classes[1][1]:.3f}, {top3_classes[2][0]}={top3_classes[2][1]:.3f}")

                    if target_prob > 0.20:
                        selected_cells_indices.append(cell['cell_index'])
                else:
                    # No classification results or insufficient data
                    logger.always_log(f"   Cell {cell['cell_index']}: NO RESULTS (classification failed)")
                    classification_results.append({
                        'cell_index': cell['cell_index'],
                        'coords': (x1, y1, x2, y2),
                        'target_probability': 0.0,
                        'is_selected': False,
                        'all_probs': None
                    })

        # Determine final captcha type based on dynamic detection
        final_captcha_type = 'dynamic_classification' if is_dynamic else 'classification'

        # Print summary
        logger.always_log(f"=== CLASSIFICATION SOLVER COMPLETE ===")
        logger.always_log(f"   Total cells analyzed: {len(all_cells_data)}")
        logger.always_log(f"   Cells selected: {len(selected_cells_indices)}")
        logger.always_log(f"   Selected cell indices: {selected_cells_indices}")
        logger.always_log(f"   Final captcha type: {final_captcha_type}")

        # === METRICS START ===
        metrics.add_yolo_cells(len(cell_images))
        if is_dynamic:
            metrics.add_dynamic()
        metrics.add_classification()

        return {
            'success': True,
            'target_object': target_object,
            'selected_cells': selected_cells_indices,
            'all_cells_data': all_cells_data,
            'captcha_type': final_captcha_type,
            'coordinates_are_global': using_global_coords
        }

    def _extract_clean_object_name(self, ocr_text):
        """Extract clean object name from OCR text, filtering out UI noise"""
        import re

        # Clean the OCR text
        ocr_text = ocr_text.lower().strip()

        # Define known object types
        known_objects = [
            'bicycles', 'bicycle', 'buses', 'bus', 'cars', 'car', 'bridges', 'bridge',
            'traffic lights', 'traffic light', 'motorcycles', 'motorcycle', 'boats', 'boat',
            'crosswalks', 'crosswalk', 'fire hydrants', 'fire hydrant', 'stairs', 'stair',
            'chimneys', 'chimney', 'tractors', 'tractor', 'taxis', 'taxi', 'trucks', 'truck',
            'trains', 'train', 'planes', 'plane'
        ]

        # First try to find a known object in the text
        for obj in known_objects:
            if obj in ocr_text:
                return obj

        # If no known object found, try pattern matching
        patterns = [
            r'select all (?:images|squares) with ([a-z\s]+?)(?:\s+(?:please|try|again|c|q|w|skip|get|new|challenge|verify|[0-9])|$)',
            r'select all ([a-z\s]+?)(?:\s+(?:please|try|again|c|q|w|skip|get|new|challenge|verify|[0-9])|$)',
            r'click (?:on )?(?:all )?(?:the )?([a-z\s]+?)(?:\s+(?:please|try|again|c|q|w|skip|get|new|challenge|verify|[0-9])|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, ocr_text)
            if match:
                object_name = match.group(1).strip()

                # Remove common noisy words
                noise_endings = ['please', 'try', 'again', 'verify', 'skip', 'get', 'new',
                               'challenge', 'poimel', 'lalug', 'challen', 'target', 'geta',
                               'c', 'q', 'w', 'dr', 'pri', 'ti', 'eat', 'gel', 'il']

                # Split into words and filter
                words = object_name.split()
                clean_words = []

                for word in words:
                    # Stop if we hit a noisy word
                    if word in noise_endings or len(word) <= 1 or word.isdigit():
                        break
                    clean_words.append(word)

                if clean_words:
                    # Join back and check if it makes sense
                    clean_name = ' '.join(clean_words)

                    # Handle special cases
                    if clean_name == 'traffic':
                        clean_name = 'traffic lights'
                    elif clean_name == 'fire':
                        clean_name = 'fire hydrants'

                    return clean_name

        return None

    def _parse_detection_data(self, detection_data, image_path, grid_size=None):
        """Parse detection results from data dictionary"""
        selected_cells = []

        try:
            # Parse bounding boxes and map to grid cells
            objects_found = detection_data.get('objects_found', 0)
            if objects_found > 0:
                object_details = detection_data.get('object_details', [])

                # Get image dimensions for bbox conversion
                image_width = image_height = None

                if image_path and os.path.exists(image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is not None:
                            image_height, image_width = img.shape[:2]
                    except Exception as e:
                        pass

                # Determine grid size - use parameter if provided, otherwise default to 4
                if grid_size is None:
                    grid_size = 4  # Default for backward compatibility

                # Map bounding boxes to overlapping grid cells
                for obj in object_details:
                    bbox = obj.get('bbox', [])
                    if len(bbox) == 4:
                        # Calculate which grid cells this bbox overlaps
                        overlapping_cells = self._map_bbox_to_grid_cells(bbox, grid_size=grid_size,
                                                                        image_width=image_width,
                                                                        image_height=image_height,
                                                                        image_path=image_path)
                        selected_cells.extend(overlapping_cells)

            # Remove duplicates and then sort
            selected_cells = sorted(set(selected_cells))

            return selected_cells

        except Exception as e:
            logger.log(f"Error parsing detection results: {str(e)}")
            return []


    def _map_bbox_to_grid_cells(self, bbox, grid_size=4, image_width=None, image_height=None, image_path=None):
        """Map bounding box coordinates to overlapping grid cells with improved accuracy"""
        try:
            # Validate grid_size to prevent division by zero
            if grid_size <= 0:
                return []

            x1, y1, x2, y2 = bbox

            # Validate bbox coordinates
            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                return []

            # If we don't have image dimensions, we can't properly map pixel coordinates
            if image_width is None or image_height is None:
                # Fallback: assume standard proportions and try basic mapping
                image_width = image_height = 400  # Reasonable default

            # Clamp bbox coordinates to image boundaries
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            # Ensure bbox is valid (x2 > x1, y2 > y1)
            if x2 <= x1 or y2 <= y1:
                return []

            # Calculate cell dimensions in pixels
            # Double-check grid_size to prevent division by zero
            if grid_size <= 0:
                return []

            cell_width = image_width / grid_size
            cell_height = image_height / grid_size

            overlapping_cells = []

            for row in range(grid_size):
                for col in range(grid_size):
                    # Calculate cell boundaries in pixel coordinates
                    cell_x1 = col * cell_width
                    cell_y1 = row * cell_height
                    cell_x2 = (col + 1) * cell_width
                    cell_y2 = (row + 1) * cell_height

                    # Calculate overlap area between bbox and cell
                    overlap_x1 = max(x1, cell_x1)
                    overlap_y1 = max(y1, cell_y1)
                    overlap_x2 = min(x2, cell_x2)
                    overlap_y2 = min(y2, cell_y2)

                    # Check if there's actual overlap
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        # Calculate overlap area
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        cell_area = cell_width * cell_height

                        # Prevent division by zero
                        if cell_area <= 0:
                            continue

                        overlap_percentage = (overlap_area / cell_area) * 100

                        cell_index = row * grid_size + col

                        # Include any cell that has overlap with LLM-detected objects
                        # For CAPTCHA segmentation, we must select cells with even tiny overlaps
                        # as the puzzle requires selecting ALL cells containing any part of the object
                        if overlap_percentage > 0:
                            overlapping_cells.append(cell_index)

            return overlapping_cells

        except Exception as e:
            return []


    def _solve_classification_captcha_hybrid_cascade(self, frame, target_object, all_cells_data, using_global_coords=False):
        """
        Hybrid cascader:

            r(x,c) = f_VLM(x)   if c not in C_YOLO
                     f_YOLO(x)  if c in C_YOLO and max(f_YOLO(x)) >= tau
                     f_VLM(x)   if c in C_YOLO and max(f_YOLO(x)) <  tau

        A cell is selected iff the predicted class name == target name.
        - YOLO predicts via argmax of the softmax distribution.
        - VLM uses the 0-15 numbered-class prompt from paper Appendix H and
          returns a class index that maps back to PAPER_CLASS_NAMES.
        - For unsupported targets (Boat, Taxi, Tractor in this codebase),
          every cell is sent to VLM with the same paper prompt.
        """
        logger.always_log(f"=== PAPER CASCADE CLASSIFICATION SOLVER STARTING ===")
        logger.always_log(f"   Target object: '{target_object}'")
        logger.always_log(f"   Number of cells: {len(all_cells_data)}")
        logger.always_log(f"   Threshold: tau = {self.HYBRID_TAU_SELECT}")

        original_target = target_object
        target_canonical = self._canonical_class_name(target_object)
        logger.always_log(f"   Target (canonical): '{target_canonical}'")

        yolo_model = self.yolo_classification_backend.model
        yolo_class_names_canonical = {
            self._canonical_class_name(v) for v in yolo_model.names.values()
        }
        target_in_yolo = target_canonical in yolo_class_names_canonical

        # Dynamic-puzzle detection (FSM downstream uses this).
        ocr_results = self.ocr_reader.readtext(frame, detail=0, paragraph=True)
        full_text = " ".join(ocr_results).lower()
        dynamic_indicators = [
            "none left", "no more", "keep selecting", "new images",
            "click verify once there are none left", "until"
        ]
        has_dynamic_text = any(indicator in full_text for indicator in dynamic_indicators)
        # 4x4 grids are always segmentation (never reach this method, but guard anyway)
        is_dynamic = has_dynamic_text if len(all_cells_data) != 16 else False

        # Extract cell images and validate coordinates.
        h, w = frame.shape[:2]
        cell_images = []
        valid_cells = []
        for cell in all_cells_data:
            x1, y1, x2, y2 = cell['coords']
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                logger.always_log(f"Cell {cell['cell_index']}: Invalid coords [{x1},{y1},{x2},{y2}] for {w}x{h}, skipping")
                continue
            cell_image = frame[y1:y2, x1:x2]
            if cell_image is None or cell_image.size == 0:
                logger.always_log(f"Cell {cell['cell_index']}: Empty image, skipping")
                continue
            cell_images.append(cell_image)
            valid_cells.append(cell)

        if not valid_cells:
            return {
                'success': False,
                'error': 'No valid cells to classify in cascade',
                'retry_needed': True,
                'coordinates_are_global': using_global_coords,
            }

        selected_cells_indices = []
        yolo_handled = 0
        vlm_handled = 0

        if not target_in_yolo:
            # Branch 1: target unsupported by YOLO -> every cell to VLM.
            logger.always_log(
                f"PAPER CASCADE: '{original_target}' (canonical='{target_canonical}') unsupported by YOLO "
                f"-> routing all {len(valid_cells)} cells to VLM"
            )
            for cell in valid_cells:
                cell_idx = cell['cell_index']
                x1, y1, x2, y2 = cell['coords']
                vlm_result = self._vlm_classify_paper_prompt(frame[y1:y2, x1:x2])
                vlm_handled += 1
                pred = vlm_result.get('class_name')
                raw = (vlm_result.get('raw') or '')[:30]
                if vlm_result.get('success') and pred == target_canonical:
                    logger.always_log(f"   Cell {cell_idx}: VLM SELECTED (predicted='{pred}'=target, raw='{raw}')")
                    selected_cells_indices.append(cell_idx)
                else:
                    logger.always_log(f"   Cell {cell_idx}: VLM REJECTED (predicted='{pred}', raw='{raw}')")
        else:
            # Branches 2 and 3: YOLO batch, then per-cell route based on max(softmax).
            logger.always_log(f"Running batch YOLO classification on {len(cell_images)} cells...")
            batch_results = yolo_model(cell_images, verbose=False, conf=0.20)

            for cell, class_results in zip(valid_cells, batch_results):
                cell_idx = cell['cell_index']
                x1, y1, x2, y2 = cell['coords']

                # YOLO produced no usable result -> treat as max=0, escalate to VLM.
                if (not class_results or class_results.probs is None
                        or class_results.probs.data is None):
                    vlm_handled += 1
                    logger.always_log(f"   Cell {cell_idx}: YOLO returned no probs -> VLM fallback")
                    vlm_result = self._vlm_classify_paper_prompt(frame[y1:y2, x1:x2])
                    pred = vlm_result.get('class_name')
                    raw = (vlm_result.get('raw') or '')[:30]
                    if vlm_result.get('success') and pred == target_canonical:
                        logger.always_log(f"   Cell {cell_idx}: VLM SELECTED (predicted='{pred}'=target, raw='{raw}')")
                        selected_cells_indices.append(cell_idx)
                    else:
                        logger.always_log(f"   Cell {cell_idx}: VLM REJECTED (predicted='{pred}', raw='{raw}')")
                    continue

                probs = class_results.probs.data.cpu().numpy()
                max_prob = float(probs.max())
                argmax_idx = int(probs.argmax())
                argmax_class = self._canonical_class_name(yolo_model.names[argmax_idx])

                if max_prob >= self.HYBRID_TAU_SELECT:
                    # Branch 2: trust YOLO -> select iff argmax == target, else reject.
                    yolo_handled += 1
                    if argmax_class == target_canonical:
                        logger.always_log(
                            f"   Cell {cell_idx}: YOLO SELECTED "
                            f"(max={max_prob:.3f} >= tau, argmax='{argmax_class}'=target)"
                        )
                        selected_cells_indices.append(cell_idx)
                    else:
                        logger.always_log(
                            f"   Cell {cell_idx}: YOLO REJECTED "
                            f"(max={max_prob:.3f} >= tau, argmax='{argmax_class}'!=target)"
                        )
                else:
                    # Branch 3: max < tau -> VLM fallback with paper prompt.
                    vlm_handled += 1
                    logger.always_log(
                        f"   Cell {cell_idx}: YOLO uncertain "
                        f"(max={max_prob:.3f} < tau, argmax='{argmax_class}') -> VLM"
                    )
                    vlm_result = self._vlm_classify_paper_prompt(frame[y1:y2, x1:x2])
                    pred = vlm_result.get('class_name')
                    raw = (vlm_result.get('raw') or '')[:30]
                    if vlm_result.get('success') and pred == target_canonical:
                        logger.always_log(f"   Cell {cell_idx}: VLM SELECTED (predicted='{pred}'=target, raw='{raw}')")
                        selected_cells_indices.append(cell_idx)
                    else:
                        logger.always_log(f"   Cell {cell_idx}: VLM REJECTED (predicted='{pred}', raw='{raw}')")

        total_handled = yolo_handled + vlm_handled
        vlm_rate = (vlm_handled / total_handled * 100) if total_handled > 0 else 0.0
        final_captcha_type = 'dynamic_classification' if is_dynamic else 'classification'

        logger.always_log(f"=== PAPER CASCADE COMPLETE ===")
        logger.always_log(f"   YOLO-handled cells: {yolo_handled}/{total_handled}")
        logger.always_log(f"   VLM-handled cells:  {vlm_handled}/{total_handled} ({vlm_rate:.1f}%)")
        logger.always_log(f"   Total selected:     {len(selected_cells_indices)}")
        logger.always_log(f"   Selected indices:   {selected_cells_indices}")
        logger.always_log(f"   Final captcha type: {final_captcha_type}")

        # === METRICS START ===
        metrics.add_yolo_cells(yolo_handled)
        metrics.add_vlm_cells(vlm_handled)
        if is_dynamic:
            metrics.add_dynamic()
        metrics.add_classification()

        return {
            'success': True,
            'target_object': target_object,
            'selected_cells': selected_cells_indices,
            'all_cells_data': all_cells_data,
            'captcha_type': final_captcha_type,
            'coordinates_are_global': using_global_coords,
        }

    def _canonical_class_name(self, name):
        """
        Map any class string (OCR target, YOLO model class, paper-prompt class)
        to a canonical lowercase form matching one of PAPER_CLASS_NAMES.
        Returns the input lowercased+stripped if no canonical match is found.
        """
        if not name:
            return ''
        n = name.lower().strip()

        compound_mappings = {
            'fire hydrant': 'hydrant', 'fire hydrants': 'hydrant',
            'traffic lights': 'traffic light',
            'palm tree': 'palm', 'palm trees': 'palm',
        }
        if n in compound_mappings:
            return compound_mappings[n]

        canonical_set = {c.lower() for c in self.PAPER_CLASS_NAMES}
        if n in canonical_set:
            return n

        # Plural removal: 'crosswalks' -> 'crosswalk'. Skip names that are already
        # singular but happen to end in 's' (bus, stairs).
        if n.endswith('s') and n not in canonical_set:
            singular = n[:-1]
            if singular in canonical_set:
                return singular

        return n  # unrecognized; return as-is so comparisons still work

    def _vlm_classify_paper_prompt(self, cell_image_cv):
        """
        Send a single cell to VLM with the paper Appendix H 0-15 numbered-class
        prompt. Parse the response as a class index and map back to a canonical
        lowercase class name.

        Returns dict:
            'success'    : bool
            'class_name' : canonical lowercase name (one of PAPER_CLASS_NAMES) or None
            'raw'        : raw model output (for logging)
        """
        from PIL import Image
        import requests
        import io
        import re

        prompt = (
            "Choose ONLY ONE number from the following list:\n"
            "0: Bicycle, 1: Bridge, 2: Bus, 3: Car, 4: Chimney,\n"
            "5: Crosswalk, 6: Hydrant, 7: Motorcycle, 8: Mountain,\n"
            "9: Other, 10: Palm, 11: Stairs, 12: Traffic Light,\n"
            "13: Boat, 14: Taxi, 15: Tractor\n"
            "Answer with ONLY the number (0-15) of the class\n"
            "you see with the highest confidence.\n"
            "Do not include any explanation."
        )

        try:
            cell_image_rgb = cv2.cvtColor(cell_image_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cell_image_rgb)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            files = {'image': ('image.png', img_byte_arr, 'image/png')}
            data = {'messages_json': json.dumps(messages)}

            response = requests.post(f"{self.api_url}/generate", files=files, data=data, timeout=30)
            if response.status_code != 200:
                return {'success': False, 'class_name': None, 'raw': f'HTTP {response.status_code}'}

            model_output = (response.json().get('model_output') or '').strip()

            # Match the first integer 0-15 in the response (handles "5", "5.",
            # "5: Crosswalk", "Answer: 5", etc.).
            match = re.search(r'\b(1[0-5]|[0-9])\b', model_output)
            if not match:
                return {'success': False, 'class_name': None, 'raw': model_output}

            idx = int(match.group(1))
            if 0 <= idx < len(self.PAPER_CLASS_NAMES):
                return {
                    'success': True,
                    'class_name': self.PAPER_CLASS_NAMES[idx].lower(),
                    'raw': model_output,
                }
            return {'success': False, 'class_name': None, 'raw': model_output}
        except Exception as e:
            return {'success': False, 'class_name': None, 'raw': f'Exception: {e}'}


    def _solve_classification_captcha_llm_streaming(self, frame, target_object, all_cells_data):
        """Solve classification CAPTCHA using sequential VLM analysis with the
        paper Appendix H 0-15 numbered-class prompt."""
        logger.log(f"Using VLM sequential classification (paper prompt) for CAPTCHA: {target_object}")

        target_canonical = self._canonical_class_name(target_object)
        is_dynamic = self._detect_dynamic_captcha(frame)

        total_cells = len(all_cells_data)
        selected_cells = []
        start_time = time.time()

        for cell_data in all_cells_data:
            cell_index = cell_data['cell_index']
            try:
                x1, y1, x2, y2 = cell_data['coords']
                cell_image = frame[y1:y2, x1:x2]

                api_start = time.time()
                result = self._vlm_classify_paper_prompt(cell_image)
                api_time = time.time() - api_start

                pred = result.get('class_name')
                raw = (result.get('raw') or '')[:30]

                if result.get('success') and pred == target_canonical:
                    logger.log(f"Cell {cell_index}: VLM SELECTED predicted='{pred}'=target raw='{raw}' (API: {api_time:.2f}s)")
                    selected_cells.append(cell_index)
                else:
                    logger.log(f"Cell {cell_index}: VLM REJECTED predicted='{pred}' raw='{raw}' (API: {api_time:.2f}s)")
            except Exception as e:
                logger.log(f"Cell {cell_index}: Exception - {e}")

        elapsed = time.time() - start_time
        final_captcha_type = 'dynamic_classification' if is_dynamic else 'classification'

        # === METRICS START ===
        metrics.add_vlm_cells(total_cells)
        if is_dynamic:
            metrics.add_dynamic()
        metrics.add_classification()

        return {
            'success': True,
            'target_object': target_object,
            'selected_cells': selected_cells,
            'all_cells_data': all_cells_data,
            'captcha_type': final_captcha_type,
            'sequential': True,
            'total_time': elapsed,
        }
    # TBD
    # Fix retry mechanism
    def _run_llm_analysis(self, image_path, object_name, max_retries=2):
        """Run VLM bounding-box detection on a concatenated grid image with retry."""
        for attempt in range(max_retries + 1):
            try:
                result = self.analyzer.detect_objects_with_boxes(image_path, object_name)

                if result and result.get('success'):
                    return {'success': True, 'result': result, 'output_dir': os.getcwd()}
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    logger.log(f"LLM analysis failed: {error_msg}")

                    # If not the last attempt, wait and retry
                    if attempt < max_retries:
                        retry_delay = 2 + attempt  # Progressive delay: 2s, 3s...
                        logger.log(f"Retrying LLM analysis in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {'success': False, 'error': error_msg}

            except Exception as e:
                logger.log(f"Exception during LLM analysis: {str(e)}")
                if attempt < max_retries:
                    retry_delay = 2 + attempt
                    logger.log(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return {'success': False, 'error': f'Exception: {str(e)}'}

        # This should never be reached, but just in case
        return {'success': False, 'error': 'All retry attempts failed'}

    def _solve_segmentation_captcha_llm(self, frame, target_object, all_cells_data):
        """Solve segmentation CAPTCHA using LLM for object detection"""
        logger.log(f"Using LLM for segmentation CAPTCHA: {target_object}")

        # Add cell images to cell data and concatenate
        for cell in all_cells_data:
            x1, y1, x2, y2 = cell['coords']
            cell['image'] = frame[y1:y2, x1:x2]

        # Concatenate cells for LLM analysis (reCAPTCHA v2 segmentation is always 4x4)
        if len(all_cells_data) != 16:
            return {'success': False, 'error': f'Unsupported grid size: {len(all_cells_data)} cells. reCAPTCHA v2 segmentation puzzles are always 16 cells (4x4).'}

        concatenated = self._concatenate_cells_4x4(all_cells_data)

        if concatenated is None:
            return {'success': False, 'error': 'Failed to concatenate cells for LLM analysis'}

        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            concatenated_path = tmp.name
        cv2.imwrite(concatenated_path, concatenated)

        try:
            # Run LLM detection on concatenated image
            result = self._run_llm_analysis(concatenated_path, target_object)

            if result['success']:
                # Parse detection results and map to grid cells
                # Determine grid size based on number of cells
                grid_size = 3 if len(all_cells_data) == 9 else 4
                detection_data = result.get('result', {}).get('summary', {})
                selected_cells = self._parse_detection_data(detection_data, concatenated_path, grid_size=grid_size)

                # === METRICS START ===
                metrics.add_vlm_grid()
                metrics.add_segmentation()
                # === METRICS END ===

                return {
                    'success': True,
                    'target_object': target_object,
                    'selected_cells': selected_cells,
                    'all_cells_data': all_cells_data,
                    'captcha_type': 'segmentation'
                }
            else:
                return {
                    'success': False,
                    'error': f'LLM segmentation failed: {result.get("error", "Unknown error")}',
                    'selected_cells': []
                }

        finally:
            # Clean up temporary file
            if concatenated_path and os.path.exists(concatenated_path):
                os.unlink(concatenated_path)


def create_captcha_processor(backend: str = 'yolo') -> UnifiedCaptchaProcessor:
    """
    Function to create a CAPTCHA processor with specified backend
    
    Args:
        backend: 'yolo' or 'llm'
    
    Returns:
        UnifiedCaptchaProcessor instance
    """
    return UnifiedCaptchaProcessor(backend=backend)
