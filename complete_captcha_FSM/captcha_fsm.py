import pyautogui
import random
from enum import Enum
import time
import cv2
import numpy as np
from logger import logger
from config import config
from metrics import metrics

class CaptchaState(Enum):
    IDLE = "idle"
    CHECKBOX_CLICKED = "checkbox_clicked"
    PUZZLE_ANALYZING = "puzzle_analyzing"
    PUZZLE_UNSUPPORTED = "puzzle_unsupported"
    PUZZLE_SOLVING = "puzzle_solving"
    PUZZLE_REANALYZING = "puzzle_reanalyzing"
    PUZZLE_VERIFYING = "puzzle_verifying"
    PUZZLE_COMPLETED = "puzzle_completed"
    PUZZLE_RELOADING = "puzzle_reloading"
    ERROR = "error"

class CaptchaStateMachine:
    def __init__(self):
        self.current_result = None
        self.current_all_cells_data = []
        self.reload_count = 0
        self.clicked_cells = set()
        self.made_progress_in_cycle = False
        self.is_new_puzzle = False  # Will be set to True when needed
        self.verification_attempted = False
        self.reanalysis_count = 0
        self.captcha_area_offset = None
        self.last_target_object = "" # Remembers the target from the last analysis
        self.state = self._detect_initial_state()
        
    def get_state(self):
        return self.state
    
    def _detect_initial_state(self):
        from captcha_solver_bridge import check_rc_imageselect_exists

        # Detect initial state: check if puzzle already visible
        logger.always_log("Detecting initial captcha state")
        try:
            puzzle_exists = check_rc_imageselect_exists()
            if puzzle_exists:
                logger.always_log("Captcha puzzle already visible - starting from PUZZLE_ANALYZING")
                return CaptchaState.PUZZLE_ANALYZING
            else:
                logger.always_log("No puzzle visible - starting from IDLE (will click checkbox)")
                return CaptchaState.IDLE
        except Exception as e:
            logger.always_log(f"Error detecting initial state: {e}")
            logger.always_log("Defaulting to IDLE state")
            return CaptchaState.IDLE
    
    def transition_to(self, new_state, reason=""):
        logger.always_log(f"STATE: {self.state.value} -> {new_state.value}")
        if reason:
            logger.log(f"Reason: {reason}")
            # Store the reason for potential reload logging
            self._last_transition_reason = reason

        self.state = new_state

    def _compute_cell_similarity(self, image1, image2):
        """
        Compute similarity score between two cell images using histogram correlation.

        This method is robust to visual feedback (borders, highlights, checkmarks)
        while still detecting actual content changes.

        Args:
            image1: First cell image (BGR)
            image2: Second cell image (BGR)

        Returns:
            float: Similarity score between 0 and 1
                  - 1.0 = identical
                  - 0.0 = completely different
                  - Threshold: >0.85 = same puzzle
        """
        # Resize to same size if needed
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if h1 != h2 or w1 != w2:
            # Resize to smaller dimensions to ensure same size
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            image1 = cv2.resize(image1, (target_w, target_h))
            image2 = cv2.resize(image2, (target_w, target_h))

        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        # Compute histograms (8 bins per channel)
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compare using correlation (range: -1 to 1, but typically 0 to 1)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Ensure result is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))

        return similarity

    def mark_new_puzzle(self):
        """Mark that we're starting a new puzzle"""
        self.is_new_puzzle = True

        # Clear previous cell data (coordinates + hashes)
        if hasattr(self, 'previous_cell_data'):
            self.previous_cell_data = {}

        # === METRICS START ===
        metrics.add_new_puzzle()

    def run(self):
        iteration = 0
        while self.state != CaptchaState.PUZZLE_COMPLETED and self.state != CaptchaState.ERROR:
            iteration += 1
            
            if self.state == CaptchaState.IDLE:
                logger.always_log("Executing: _handle_idle()")
                self._handle_idle()
            elif self.state == CaptchaState.CHECKBOX_CLICKED:
                logger.always_log("Executing: _handle_checkbox_clicked()")
                self._handle_checkbox_clicked()
            elif self.state == CaptchaState.PUZZLE_ANALYZING:
                logger.always_log("Executing: _handle_puzzle_analyzing()")
                self._handle_puzzle_analyzing()
            elif self.state == CaptchaState.PUZZLE_UNSUPPORTED:
                logger.always_log("Executing: _handle_puzzle_unsupported()")
                self._handle_puzzle_unsupported()
            elif self.state == CaptchaState.PUZZLE_SOLVING:
                logger.always_log("Executing: _handle_puzzle_solving()")
                self._handle_puzzle_solving()
            elif self.state == CaptchaState.PUZZLE_REANALYZING:
                logger.always_log("Executing: _handle_puzzle_reanalyzing()")
                self._handle_puzzle_reanalyzing()
            elif self.state == CaptchaState.PUZZLE_VERIFYING:
                logger.always_log("Executing: _handle_puzzle_verifying()")
                self._handle_puzzle_verifying()
            elif self.state == CaptchaState.PUZZLE_RELOADING:
                logger.always_log("Executing: _handle_puzzle_reloading()")
                self._handle_puzzle_reloading()
            
            time.sleep(0.2)

        if self.state == CaptchaState.PUZZLE_COMPLETED:
            return True
        else:
            return False
    
    def _handle_idle(self):
        from captcha_solver_bridge import find_and_click_recaptcha_checkbox
        checkbox_clicked = find_and_click_recaptcha_checkbox()
        if checkbox_clicked:
            self.transition_to(CaptchaState.CHECKBOX_CLICKED, "Checkbox clicked successfully")
            time.sleep(0.5)  # Reduced wait for faster response
        else:
            self.transition_to(CaptchaState.ERROR, "Could not find or click checkbox")
    
    def _handle_checkbox_clicked(self):
        from captcha_solver_bridge import check_rc_imageselect_exists, find_and_click_recaptcha_checkbox
        logger.always_log("Checkbox clicked! Waiting for puzzle dialog to appear...")

        # Mark this as a new puzzle BEFORE any detection calls (clears cache for fresh detection)
        self.mark_new_puzzle()

        # Give dialog time to appear before first check
        initial_wait = 1.0
        time.sleep(initial_wait)

        # Wait up to 3 more seconds for puzzle to appear (4 seconds total)
        wait_time = initial_wait
        max_wait = 4.0
        check_interval = 0.5

        while wait_time < max_wait:
            puzzle_exists = check_rc_imageselect_exists()
            if puzzle_exists:
                logger.always_log(f"CAPTCHA dialog appeared after {wait_time:.1f} seconds")
                self.transition_to(CaptchaState.PUZZLE_ANALYZING, "Puzzle appeared")
                return

            time.sleep(check_interval)
            wait_time += check_interval

        # If puzzle didn't appear after 4 seconds, try clicking again
        logger.always_log("Re-clicking checkbox (Focus issue detected)")

        # Try clicking the checkbox again
        checkbox_clicked = find_and_click_recaptcha_checkbox()
        if not checkbox_clicked:
            self.transition_to(CaptchaState.ERROR, "Could not find checkbox for retry")
            return

        # Wait another 2 seconds for puzzle after second click
        time.sleep(1.5)

        puzzle_exists = check_rc_imageselect_exists()
        if puzzle_exists:
            self.transition_to(CaptchaState.PUZZLE_ANALYZING, "Puzzle appeared after retry")
        else:
            # One final check after a bit more wait
            time.sleep(1)
            if check_rc_imageselect_exists():
                self.transition_to(CaptchaState.PUZZLE_ANALYZING, "Puzzle appeared after extended wait")
            else:
                self.transition_to(CaptchaState.ERROR, "Puzzle did not appear even after retry")
    
    def _handle_puzzle_analyzing(self):
        from captcha_solver_bridge import capture_captcha_area_only, get_solver, get_supported_objects

        # Wait for CAPTCHA images to load before capturing
        if not self._wait_for_captcha_images_to_load():
            self.transition_to(CaptchaState.ERROR, "CAPTCHA images failed to load")
            return

        captured_image_data, captcha_coords = capture_captcha_area_only()
        if captured_image_data is None:
            self.transition_to(CaptchaState.ERROR, "Failed to capture puzzle")
            return

        # The captcha_coords are at full resolution, but we need them at logical resolution
        # to match the cell coordinates which are now detected in the logical-resolution screenshot
        try:
            from captcha_solver_bridge import get_solver
            processor_instance = get_solver()
            if hasattr(processor_instance, 'display_scaling') and processor_instance.display_scaling and processor_instance.display_scaling > 0 and processor_instance.display_scaling != 1.0:
                scaled_x = int(captcha_coords[0] / processor_instance.display_scaling)
                scaled_y = int(captcha_coords[1] / processor_instance.display_scaling)
                scaled_x2 = int(captcha_coords[2] / processor_instance.display_scaling)
                scaled_y2 = int(captcha_coords[3] / processor_instance.display_scaling)
                self.captcha_area_offset = (scaled_x, scaled_y)
                self.captcha_area_bounds = (scaled_x, scaled_y, scaled_x2, scaled_y2)
            else:
                self.captcha_area_offset = (captcha_coords[0], captcha_coords[1])
                self.captcha_area_bounds = captcha_coords
        except (ZeroDivisionError, AttributeError, Exception) as e:
            self.captcha_area_offset = (captcha_coords[0], captcha_coords[1])
            self.captcha_area_bounds = captcha_coords
        
        solver = get_solver()

        # Solve captcha without saving images
        result = solver.solve_captcha(captured_image_data)
        
        self.current_result = result
        
        if not result['success']:
            error_message = result.get('error', '')

            # Handle invalid cell count. This should trigger a reload
            if result.get('retry_needed', False) or 'Invalid grid cell count' in error_message:
                logger.log("Invalid grid cell count detected. Reloading puzzle...")
                self.transition_to(CaptchaState.PUZZLE_RELOADING, f"INVALID GRID: {error_message}")
                return

            if 'Could not determine target object' in error_message:
                self.transition_to(CaptchaState.PUZZLE_UNSUPPORTED, f"Unsupported target: {error_message}")
            else:
                self.transition_to(CaptchaState.ERROR, f"Puzzle analysis failed: {error_message}")
            return
        
        current_target_object = result.get('target_object', 'unknown')
        captcha_type = result.get('captcha_type', 'unknown')

        # Show detection
        logger.always_log("=" * 50)
        logger.always_log(f"PUZZLE DETECTED")
        logger.always_log(f"Type: {captcha_type.upper()}")
        logger.always_log(f"Target: '{current_target_object.upper()}'")

        # Get backend type from solver instance
        if hasattr(solver, 'backend_type'):
            backend_name = solver.backend_type.upper()
        # TODO: Legacy code. Needs further refactoring
        elif hasattr(solver, '__class__') and 'LLM' in solver.__class__.__name__:
            backend_name = "LLM"
        else:
            backend_name = "YOLO"

        logger.always_log(f"Backend: {backend_name}")
        logger.always_log("=" * 50)

        is_genuinely_new_puzzle = self.is_new_puzzle or \
                                (self.last_target_object != current_target_object and current_target_object != "unknown")

        if is_genuinely_new_puzzle:
            self.clicked_cells.clear()
            self.verification_attempted = False
            # Reset click counter for new puzzle
            self.total_clicks_this_puzzle = 0

        self.is_new_puzzle = False
        self.last_target_object = current_target_object

        # Check if we're using LLM/Hybrid mode (which supports all objects) or YOLO mode (limited objects)
        # Check backend_type attribute first, then fallback
        if hasattr(solver, 'backend_type'):
            # Hybrid mode can use LLM for unsupported objects, so treat it like LLM mode
            is_llm_mode = solver.backend_type in ['llm', 'hybrid']
        else:
            is_llm_mode = hasattr(solver, '__class__') and 'LLM' in solver.__class__.__name__

        if is_llm_mode:
            # LLM/Hybrid mode supports all objects. No need to check supported list
            pass
        else:
            # YOLO mode. Check supported objects list
            supported_objects = get_supported_objects()
            is_supported = current_target_object.lower() in [obj.lower() for obj in supported_objects]

            if not is_supported:
                self.transition_to(CaptchaState.PUZZLE_UNSUPPORTED, f"UNSUPPORTED OBJECT: '{current_target_object}' (YOLO mode only supports: {', '.join(supported_objects[:5])}...)")
                return
        
        self.current_all_cells_data = result.get('all_cells_data', [])
        selected_cells = result.get('selected_cells', [])
        captcha_type = result.get('captcha_type', 'unknown')

        # Store cell coordinates AND images for later comparison
        if captured_image_data is not None:
            self.previous_cell_data = {}
            for cell in self.current_all_cells_data:
                cell_idx = cell['cell_index']
                x1, y1, x2, y2 = cell['coords']
                # Extract cell image using these coordinates
                cell_img = captured_image_data[y1:y2, x1:x2]
                if cell_img.size > 0:
                    # Store BOTH coordinates and image for deterministic comparison
                    # Using similarity scores is more robust than hashing for visual feedback
                    self.previous_cell_data[cell_idx] = {
                        'coords': (x1, y1, x2, y2),  # CRITICAL: Store the exact coordinates
                        'image': cell_img.copy()        # Store actual image for similarity comparison
                    }

        # For classification CAPTCHAs (including dynamic), ensure we don't click the same cell multiple times
        if captcha_type in ['classification', 'dynamic_classification']:
            # Convert to sets for more robust duplicate detection
            selected_set = set(selected_cells)
            clicked_set = set(self.clicked_cells)
            new_cells_to_click = list(selected_set - clicked_set)
        else:
            # For segmentation, use original logic
            new_cells_to_click = [cell for cell in selected_cells if cell not in self.clicked_cells]

        if new_cells_to_click:
            result['selected_cells'] = new_cells_to_click
            self.current_result = result
            # Keep logging active - will be ended after clicking/verification
            self.transition_to(CaptchaState.PUZZLE_SOLVING, f"Found {len(new_cells_to_click)} new cells to click")
        else:
            if self.verification_attempted:
                logger.always_log("=" * 60)
                logger.always_log("RELOADING: All detected cells clicked, but verification failed")
                logger.always_log(f"   Selected cells: {selected_cells}")
                logger.always_log(f"   Already clicked: {list(self.clicked_cells)}")
                logger.always_log("=" * 60)
                self.transition_to(CaptchaState.PUZZLE_RELOADING, "ALL CELLS CLICKED: No new target cells found after verification attempt")
            else:
                logger.always_log("=" * 60)
                logger.always_log(f"RELOADING: No '{current_target_object}' detected in initial analysis")
                logger.always_log(f"   Captcha type: {captcha_type}")
                logger.always_log(f"   Total cells in grid: {len(self.current_all_cells_data) if self.current_all_cells_data else 0}")
                logger.always_log(f"   Cells with target: 0")
                logger.always_log("Possible reasons:")
                logger.always_log("   1. No target objects visible in this puzzle")
                logger.always_log("   2. Detection confidence too low (objects exist but below threshold)")
                logger.always_log("   3. Object name mismatch (CAPTCHA vs YOLO class names)")
                logger.always_log("=" * 60)
                self.transition_to(CaptchaState.PUZZLE_RELOADING, f"NO TARGETS FOUND: No '{current_target_object}' detected in any grid cells")

    def _handle_puzzle_unsupported(self):
        # The reason was already set when transitioning to PUZZLE_UNSUPPORTED
        self.transition_to(CaptchaState.PUZZLE_RELOADING, self._last_transition_reason if hasattr(self, '_last_transition_reason') else "UNSUPPORTED OBJECT")
    
    def _handle_puzzle_solving(self):
        result = self.current_result
        selected_cells_indices = result.get('selected_cells', [])
        target_object = result.get('target_object', 'unknown')
        captcha_type = result.get('captcha_type', 'unknown')

        all_cell_data = self.current_all_cells_data
        if not all_cell_data:
            self.transition_to(CaptchaState.ERROR, "Cell coordinate data is missing for clicking")
            return

        click_success = False
        try:
            # Get processor instance for display scaling
            from captcha_solver_bridge import get_solver
            processor_instance = get_solver()

            actually_clicked_cells = []  # Track which cells were actually clicked

            for cell_index_to_click in selected_cells_indices:
                cell_to_click = next((cell for cell in all_cell_data if cell['cell_index'] == cell_index_to_click), None)

                if cell_to_click:
                    local_x1, local_y1, local_x2, local_y2 = cell_to_click['coords']
                    local_center_x = (local_x1 + local_x2) // 2
                    local_center_y = (local_y1 + local_y2) // 2

                    offset_x, offset_y = self.captcha_area_offset

                    # Check if coordinates are already global (fallback detection)
                    coordinates_are_global = result.get('coordinates_are_global', False)

                    if coordinates_are_global:
                        # Coordinates are already global (from fullscreenshot fallback)
                        # BUT they're at full resolution, need to scale down for pyautogui
                        global_center_x = local_center_x
                        global_center_y = local_center_y

                        # Apply display scaling: fallback coords are full resoltuion, pyautogui needs logical
                        if hasattr(processor_instance, 'display_scaling') and processor_instance.display_scaling > 0:
                            scaled_x = int(global_center_x / processor_instance.display_scaling)
                            scaled_y = int(global_center_y / processor_instance.display_scaling)
                            click_x, click_y = scaled_x, scaled_y
                        else:
                            click_x, click_y = global_center_x, global_center_y
                    else:
                        # Coordinates are local to CAPTCHA area. Add offset
                        global_center_x = local_center_x + offset_x
                        global_center_y = local_center_y + offset_y
                        # Already at logical resolution for pyautogui
                        click_x, click_y = global_center_x, global_center_y

                    # Boundary validation: Ensure click is within CAPTCHA area
                    # Skip validation for global coordinates. Already scaled
                    if not coordinates_are_global and hasattr(self, 'captcha_area_bounds') and self.captcha_area_bounds:
                        bound_x1, bound_y1, bound_x2, bound_y2 = self.captcha_area_bounds
                        if not (bound_x1 <= click_x <= bound_x2 and bound_y1 <= click_y <= bound_y2):
                            logger.always_log(f"WARNING: Cell {cell_index_to_click} click position ({click_x}, {click_y}) is OUTSIDE CAPTCHA area!")
                            logger.always_log(f"   CAPTCHA bounds: ({bound_x1}, {bound_y1}) to ({bound_x2}, {bound_y2})")
                            logger.always_log(f"   Skipping this click to prevent clicking outside puzzle")
                            continue

                    if config.get_mouse_movement():
                        pyautogui.moveTo(click_x, click_y, duration=random.uniform(0.2, 0.5))
                        pyautogui.click()
                    else:
                        pyautogui.click(click_x, click_y)

                    actually_clicked_cells.append(cell_index_to_click)

            # Only consider it success if we actually clicked at least one cell
            click_success = len(actually_clicked_cells) > 0
            if not click_success and len(selected_cells_indices) > 0:
                logger.always_log(f"CRITICAL: Wanted to click {len(selected_cells_indices)} cells but all were outside")
                logger.always_log(f"   This indicates a coordinate system mismatch.")
                logger.always_log(f"   coordinates_are_global: {result.get('coordinates_are_global', False)}")
                logger.always_log(f"   CAPTCHA bounds: {self.captcha_area_bounds}")
                logger.always_log(f"   Skipped cells: {selected_cells_indices}")
        except Exception as e:
            logger.log(f"An error occurred during mouse control: {e}")
            click_success = False

        if click_success:
            self.clicked_cells.update(actually_clicked_cells)  # Update with actually clicked cells
            self.made_progress_in_cycle = True

            # Track total clicks for dynamic CAPTCHAs to prevent infinite loops
            if not hasattr(self, 'total_clicks_this_puzzle'):
                self.total_clicks_this_puzzle = 0
            self.total_clicks_this_puzzle += len(actually_clicked_cells)

            captcha_type = result.get('captcha_type', 'unknown')
            if captcha_type == 'dynamic_classification':
                self.transition_to(CaptchaState.PUZZLE_REANALYZING, "Dynamic captcha - re-analyzing after clicks")
            else:
                self.transition_to(CaptchaState.PUZZLE_VERIFYING, "Static captcha - cells clicked successfully")
        else:
            logger.always_log(f"No cells were successfully clicked - going to ERROR state")
            self.transition_to(CaptchaState.ERROR, "Failed to click cells")
    
    def _handle_puzzle_reanalyzing(self):
        # Wait for new images to load (dynamic CAPTCHAs take about 4 seconds)
        wait_time = 4.5  # Wait a bit longer to ensure images are fully loaded
        time.sleep(wait_time)

        # Re-capture and analyze the CAPTCHA area
        from captcha_solver_bridge import capture_captcha_area_only, get_solver
        captured_image_data, captcha_coords = capture_captcha_area_only()

        if captured_image_data is None:
            self.transition_to(CaptchaState.PUZZLE_VERIFYING, "Failed to re-capture, proceeding to verify")
            return

        solver = get_solver()
        result = solver.solve_captcha(captured_image_data)

        if not result['success']:
            self.transition_to(CaptchaState.PUZZLE_VERIFYING, "Reanalysis failed, proceeding to verify")
            return

        # Check if we found new cells to click
        new_selected_cells = result.get('selected_cells', [])
        reanalysis_target_object = result.get('target_object', 'unknown')

        # CRITICAL CHECK: Has the target object changed? This means a NEW puzzle appeared!
        target_changed = reanalysis_target_object != self.last_target_object and reanalysis_target_object != 'unknown'

        # Even if target is the same, check if this could be a new puzzle
        grid_changed = False
        if reanalysis_target_object == self.last_target_object and reanalysis_target_object != 'unknown':
            # Check if the grid content changed significantly
            new_selected_set = set(new_selected_cells)
            old_clicked_set = set(self.clicked_cells)

            if old_clicked_set:
                overlap = len(new_selected_set & old_clicked_set)
                overlap_percentage = (overlap / max(len(old_clicked_set), 1)) * 100

                # Multiple signals for detecting a new puzzle during reanalysis:
                very_low_overlap = overlap_percentage < 20
                no_targets_now = len(new_selected_set) == 0
                mostly_different = overlap_percentage < 40 and len(new_selected_set) > 0

                if very_low_overlap and new_selected_cells:
                    grid_changed = True
                elif no_targets_now:
                    # No targets could mean puzzle completed, but don't treat as "new puzzle"
                    pass
                elif mostly_different:
                    grid_changed = True

        if target_changed or grid_changed:
            reason = "target changed" if target_changed else "grid content changed (same target)"
            logger.always_log("=" * 70)
            logger.always_log("NEW PUZZLE DETECTED")
            logger.always_log(f"   Previous target: '{self.last_target_object}'")
            logger.always_log(f"   New target: '{reanalysis_target_object}'")
            logger.always_log(f"   Detection reason: {reason}")
            logger.always_log("   Resetting state and analyzing the new puzzle...")
            logger.always_log("=" * 70)

            # Reset state for the new puzzle
            self.last_target_object = reanalysis_target_object
            self.clicked_cells.clear()
            self.verification_attempted = False
            self.total_clicks_this_puzzle = 0
            # NOTE: Don't call mark_new_puzzle() here because we're already analyzing this puzzle
            self.is_new_puzzle = True

            # Store the new puzzle data
            self.current_result = result
            self.current_all_cells_data = result.get('all_cells_data', [])


            # If the new puzzle has cells to click, go to solving. Otherwise, go to analyzing
            if new_selected_cells:
                self.transition_to(CaptchaState.PUZZLE_SOLVING, f"New puzzle detected with {len(new_selected_cells)} cells to click ({reason})")
            else:
                # New puzzle but no targets found - analyze it again
                self.transition_to(CaptchaState.PUZZLE_ANALYZING, f"New puzzle detected but no targets found ({reason})")
            return

        # For dynamic CAPTCHAs, the same cell positions can have new target content after image replacement
        # So we should click any cells that contain the target, regardless of previous clicks
        # But we still track what we clicked to avoid infinite loops

        # In dynamic CAPTCHAs, allow re-clicking cells since content changes
        # But track total clicks to prevent infinite loops
        if not hasattr(self, 'total_clicks_this_puzzle'):
            self.total_clicks_this_puzzle = 0

        MAX_CLICKS_PER_PUZZLE = 20  # Reasonable limit for dynamic CAPTCHAs

        if new_selected_cells and self.total_clicks_this_puzzle < MAX_CLICKS_PER_PUZZLE:
            # Found new images to click
            result['selected_cells'] = new_selected_cells
            self.current_result = result
            self.current_all_cells_data = result.get('all_cells_data', [])
            self.transition_to(CaptchaState.PUZZLE_SOLVING, f"Found {len(new_selected_cells)} cells after dynamic reanalysis")
        elif self.total_clicks_this_puzzle >= MAX_CLICKS_PER_PUZZLE:
            self.transition_to(CaptchaState.PUZZLE_VERIFYING, "Maximum clicks reached")
        else:
            # No new cells found: verify or reload?
            # If we've clicked at least some cells, proceed to verification
            # If we've never clicked anything, reload the puzzle instead
            if self.total_clicks_this_puzzle > 0:
                # Check if this might be the "there are none left" case
                captured_text = self._check_for_completion_text(captured_image_data)
                if "none left" in captured_text or "no more" in captured_text:
                    self.transition_to(CaptchaState.PUZZLE_VERIFYING, "Dynamic CAPTCHA completed - no more targets")
                else:
                    # No new cells found, proceed to verification
                    self.transition_to(CaptchaState.PUZZLE_VERIFYING, "No new cells found after reanalysis")
            else:
                # No cells were ever clicked - this puzzle is unsolvable, reload it
                logger.always_log("=" * 60)
                logger.always_log(f"RELOADING: No '{reanalysis_target_object}' found in reanalysis")
                logger.always_log(f"   Total clicks made: 0")
                logger.always_log(f"   Cannot verify without clicking any cells")
                logger.always_log("=" * 60)
                self.transition_to(CaptchaState.PUZZLE_RELOADING, f"NO CLICKS MADE: No '{reanalysis_target_object}' found in any cells")

    def _handle_puzzle_verifying(self):
        from captcha_solver_bridge import find_and_click_verify_button, check_rc_imageselect_exists
        verify_clicked = find_and_click_verify_button()
        if not verify_clicked:
            self.transition_to(CaptchaState.ERROR, "Failed to click verify button")
            return

        self.verification_attempted = True

        # Wait and check multiple times for more reliable detection
        verification_result = self._check_verification_result()

        if verification_result == "success":
            
            metrics.add_success()
            self.transition_to(CaptchaState.PUZZLE_COMPLETED, "CAPTCHA completed successfully")
        elif verification_result == "failed":
            logger.always_log("=" * 70)
            logger.always_log("VERIFICATION RESULT: CAPTCHA area still exists")
            logger.always_log("Checking if this is a failed verification or a new puzzle...")
            logger.always_log("=" * 70)

            # Check if a new puzzle appeared by doing a quick OCR check
            from captcha_solver_bridge import capture_captcha_area_only
            captured_image_data, _ = capture_captcha_area_only()

            if captured_image_data is not None:
                logger.always_log("Captured current CAPTCHA area for analysis")
                # Quick OCR to detect target object
                from captcha_solver_bridge import get_solver
                solver = get_solver()

                try:
                    ocr_results = solver.ocr_reader.readtext(captured_image_data, detail=0, paragraph=True)
                    current_text = " ".join(ocr_results).lower() if ocr_results else ""
                    logger.always_log(f"OCR detected text: '{current_text[:100]}{'...' if len(current_text) > 100 else ''}'")
                except Exception as e:
                    logger.always_log(f"OCR check failed: {e}")
                    current_text = ""

                # Extract target object from OCR text using the solver's extraction method
                detected_new_target = None

                if hasattr(solver, '_extract_clean_object_name'):
                    try:
                        detected_new_target = solver._extract_clean_object_name(current_text)
                        if detected_new_target and detected_new_target.lower() != 'unknown':
                            logger.always_log(f"Extracted target from OCR: '{detected_new_target}'")
                        else:
                            logger.always_log(f"Could not extract target from OCR text: '{current_text[:100]}'")
                            detected_new_target = None
                    except Exception as e:
                        logger.always_log(f"Target extraction failed: {e}")
                        detected_new_target = None
                else:
                    logger.always_log(f"Solver doesn't have _extract_clean_object_name method")

                logger.always_log(f"Target Detection:")
                logger.always_log(f"   Previous target: '{self.last_target_object}'")
                logger.always_log(f"   Detected target: '{detected_new_target if detected_new_target else 'None (OCR extraction failed)'}")

                # Helper function to normalize target names for comparison (handle singular/plural)
                def normalize_target(target):
                    if not target:
                        return ""
                    normalized = target.lower().strip()

                    # Handle compound object names
                    # Extract the last significant word as the core object
                    compound_mappings = {
                        'fire hydrant': 'hydrant',
                        'traffic light': 'light',
                        'fire hydrants': 'hydrant',
                        'traffic lights': 'light',
                    }

                    if normalized in compound_mappings:
                        normalized = compound_mappings[normalized]

                    # Remove trailing 's' for plural forms
                    if normalized.endswith('s') and len(normalized) > 3:
                        singular = normalized[:-1]
                        # Only remove 's' if it makes sense
                        # Common plural patterns: add more as needed
                        if not (singular.endswith('s') or singular.endswith('us')):  # avoid "buses" -> "buse"
                            normalized = singular
                    return normalized

                # Check if target changed or if we should treat this as a new puzzle anyway
                normalized_detected = normalize_target(detected_new_target)
                normalized_previous = normalize_target(self.last_target_object)
                target_changed = detected_new_target and normalized_detected != normalized_previous

                logger.always_log(f"Normalized comparison: '{normalized_detected}' vs '{normalized_previous}'")

                if target_changed:
                    # Target definitely changed. This is a NEW puzzle!
                    logger.always_log("=" * 70)
                    logger.always_log("NEW PUZZLE DETECTED")
                    logger.always_log(f"   Previous target: '{self.last_target_object}'")
                    logger.always_log(f"   New target detected: '{detected_new_target}'")
                    logger.always_log(f"   Detection method: Target changed")
                    logger.always_log("   Transitioning to analyze the new puzzle...")
                    logger.always_log("=" * 70)

                    # Reset for new puzzle
                    self.clicked_cells.clear()
                    self.verification_attempted = False
                    self.mark_new_puzzle()
                    self.last_target_object = detected_new_target

                    self.transition_to(CaptchaState.PUZZLE_ANALYZING, "New puzzle detected after verification (target changed)")
                    return

                # If target is same or couldn't detect target, check grid changes
                # This handles cases where OCR fails or consecutive puzzles with same object
                grid_changed = False
                # Use normalized comparison here too!
                targets_match = detected_new_target and normalized_detected == normalized_previous
                should_check_grid = targets_match or (not detected_new_target)

                if should_check_grid:
                    reason = f"Same target detected ('{detected_new_target}')" if detected_new_target else "Could not detect target (OCR failed)"
                    logger.always_log(f"{reason}, checking if grid changed...")

                    # Initialize variables that will be used later
                    new_selected = set()
                    old_clicked = set(self.clicked_cells)

                    # Get a quick result to see what cells are detected
                    logger.always_log("Running detection to analyze current grid...")
                    quick_result = solver.solve_captcha(captured_image_data)
                    if quick_result.get('success'):
                        logger.always_log("Detection completed successfully")
                        new_selected = set(quick_result.get('selected_cells', []))

                        # Check if verification used global coordinates (fallback detection)
                        verification_used_global = quick_result.get('coordinates_are_global', False)

                        # Check if actual cell IMAGES changed using consistent coordinates
                        # CRITICAL: Only compare if both detections used the same coordinate system!
                        if hasattr(self, 'previous_cell_data') and self.previous_cell_data and not verification_used_global:
                            # We need to compare using the SAME coordinates from previous puzzle
                            # to ensure we're extracting the exact same region
                            common_cells = old_clicked & new_selected
                            if common_cells and len(common_cells) >= 2:  # Need at least 2 cells to compare
                                logger.always_log(f"Comparing cell images using CONSISTENT coordinates...")
                                logger.always_log(f"   Common cells to check: {sorted(list(common_cells))}")
                                
                                different_cells = []
                                same_cells = []
                                
                                for cell_idx in common_cells:
                                    if cell_idx in self.previous_cell_data:
                                        # Use the stored coordinates and image from previous puzzle
                                        prev_coords = self.previous_cell_data[cell_idx]['coords']
                                        prev_image = self.previous_cell_data[cell_idx]['image']

                                        # Extract from current image using SAME coordinates
                                        px1, py1, px2, py2 = prev_coords

                                        # Validate coordinates are within bounds
                                        h, w = captured_image_data.shape[:2]
                                        if px1 < 0 or py1 < 0 or px2 > w or py2 > h:
                                            logger.log(f"   Cell {cell_idx}: Coords {prev_coords} out of bounds for {w}x{h} image - skipping")
                                            continue

                                        # Extract current cell using previous coordinates
                                        current_cell_img = captured_image_data[py1:py2, px1:px2]

                                        if current_cell_img.size > 0:
                                            # Compute similarity score (robust to visual feedback!)
                                            similarity = self._compute_cell_similarity(prev_image, current_cell_img)

                                            # Threshold: similarity < 0.85 means significantly different
                                            # This is more robust than hash distance for visual feedback
                                            if similarity < 0.85:
                                                different_cells.append((cell_idx, similarity))
                                                pass
                                            else:
                                                same_cells.append((cell_idx, similarity))
                                
                                total_compared = len(different_cells) + len(same_cells)
                                
                                if total_compared > 0:
                                    different_ratio = len(different_cells) / total_compared
                                    logger.always_log(f"Image Similarity Analysis (using consistent coordinates):")
                                    logger.always_log(f"   Cells compared: {total_compared}")
                                    logger.always_log(f"   Cells with SAME images: {len(same_cells)} ({(1-different_ratio)*100:.1f}%)")
                                    logger.always_log(f"   Cells with DIFFERENT images: {len(different_cells)} ({different_ratio*100:.1f}%)")
                                    
                                    # If >50% of cells have different images, it's a new puzzle
                                    if different_ratio > 0.5:
                                        grid_changed = True
                                        logger.always_log(f"DECISION: Grid CHANGED ({different_ratio*100:.1f}% of cells have different images)")
                                        logger.always_log(f"   Reason: Majority of cells contain new images")
                                    else:
                                        logger.always_log(f"DECISION: Grid UNCHANGED ({(1-different_ratio)*100:.1f}% of cells have same images)")
                                        logger.always_log(f"   Reason: Majority of cells contain identical images")
                                else:
                                    logger.always_log(f"Could not compare any cells (coordinates out of bounds or no common cells)")
                            
                            # If grid changed via image comparison, handle new puzzle detection
                            if grid_changed:
                                # Early return for new puzzle detection
                                logger.always_log("=" * 70)
                                logger.always_log("NEW PUZZLE DETECTED VIA IMAGE COMPARISON!")
                                logger.always_log(f"   Previous target: '{self.last_target_object}'")
                                logger.always_log(f"   Detected target: '{detected_new_target if detected_new_target else 'Unknown'}'")
                                logger.always_log(f"   Detection reason: Grid images changed (>50% different)")
                                logger.always_log("   This means the previous puzzle was SOLVED!")
                                logger.always_log("   Transitioning to analyze the new puzzle...")
                                logger.always_log("=" * 70)

                                # Reset for new puzzle
                                self.clicked_cells.clear()
                                self.verification_attempted = False
                                self.mark_new_puzzle()
                                self.last_target_object = detected_new_target if detected_new_target else self.last_target_object

                                self.transition_to(CaptchaState.PUZZLE_ANALYZING, "New puzzle detected via image comparison")
                                return

                        elif verification_used_global and hasattr(self, 'previous_cell_data') and self.previous_cell_data:
                            # Coordinate systems don't match - can't reliably compare images
                            logger.always_log("=" * 70)
                            logger.always_log("COORDINATE SYSTEM MISMATCH DETECTED")
                            logger.always_log("   Initial detection: Local coordinates (CAPTCHA area)")
                            logger.always_log("   Verification detection: Global coordinates (fullscreenshot fallback)")
                            logger.always_log("   Cannot reliably compare cell images across different coordinate systems")
                            logger.always_log("   Skipping image comparison - relying on cell index comparison only")
                            logger.always_log("=" * 70)

                        # Check if grid content changed by comparing clicked vs detected cells
                        if old_clicked:
                            overlap = len(new_selected & old_clicked)

                            # For SAME target: Use strict matching
                            # For UNKNOWN target: Use Jaccard similarity 
                            if detected_new_target:
                                # Same target detected - be strict: cells must match exactly
                                cells_match = new_selected == old_clicked
                                cells_are_subset = new_selected.issubset(old_clicked) or old_clicked.issubset(new_selected)
                                exact_match = (overlap / max(len(old_clicked), 1)) * 100

                                logger.always_log(f"Grid Comparison (SAME TARGET):")
                                logger.always_log(f"   Previously clicked: {sorted(list(old_clicked))}")
                                logger.always_log(f"   Newly detected: {sorted(list(new_selected))}")
                                logger.always_log(f"   Exact match: {exact_match:.1f}% ({overlap}/{len(old_clicked)} cells)")
                                logger.always_log(f"   Sets equal: {cells_match}")

                                # For same target with different cells - NEW PUZZLE
                                # For same target with same cells - verification failed (same puzzle)
                                if not cells_match:
                                    grid_changed = True 
                                    logger.always_log(f"DECISION: Same target but different cells - NEW PUZZLE (grid changed)")
                                else:
                                    grid_changed = False 
                                    logger.always_log(f"DECISION: Same target, same cells - grid unchanged (verification failed)")
                            else:
                                # Unknown target - use Jaccard similarity
                                union = len(new_selected | old_clicked)
                                jaccard_overlap = (overlap / max(union, 1)) * 100 if union > 0 else 0
                                cell_count_ratio = len(new_selected) / max(len(old_clicked), 1)
                                very_different_count = cell_count_ratio > 2.0 or cell_count_ratio < 0.5

                                logger.always_log(f"Grid Comparison (UNKNOWN TARGET - OCR failed):")
                                logger.always_log(f"   Previously clicked: {sorted(list(old_clicked))}")
                                logger.always_log(f"   Newly detected: {sorted(list(new_selected))}")
                                logger.always_log(f"   Jaccard overlap: {jaccard_overlap:.1f}%")
                                logger.always_log(f"   Cell count ratio: {cell_count_ratio:.1f}x")

                                logger.always_log(f"Analysis:")
                                logger.always_log(f"   Very different count (>2x or <0.5x): {very_different_count}")
                                logger.always_log(f"   Low Jaccard overlap (<50%): {jaccard_overlap < 50}")

                                # Signals for new puzzle when target unknown:
                                if jaccard_overlap < 20:
                                    grid_changed = True
                                    logger.always_log(f"DECISION: Grid changed (very low overlap: {jaccard_overlap:.1f}% < 20%)")
                                elif very_different_count:
                                    grid_changed = True
                                    logger.always_log(f"DECISION: Grid changed (very different count: {len(new_selected)} vs {len(old_clicked)}, ratio={cell_count_ratio:.1f}x)")
                                elif jaccard_overlap < 50 and len(new_selected) > 0:
                                    grid_changed = True
                                    logger.always_log(f"DECISION: Grid changed (mostly different: {jaccard_overlap:.1f}% < 50%)")
                                elif len(new_selected) == 0:
                                    grid_changed = True
                                    logger.always_log(f"DECISION: Grid changed (no targets detected)")
                                else:
                                    grid_changed = False
                                    logger.always_log(f"DECISION: Grid appears unchanged (overlap: {jaccard_overlap:.1f}%)")
                        elif new_selected:
                            # We clicked nothing before, but now there are targets - new puzzle
                            grid_changed = True
                            logger.always_log(f"DECISION: New targets found where none existed before!")
                        else:
                            logger.always_log(f"No cells clicked before and no cells detected now")
                    else:
                        logger.always_log(f"Detection failed: {quick_result.get('error', 'Unknown error')}")

                # TODO 
                # If grid changed, it's a new puzzle?
                if grid_changed:
                    reason = "grid content changed (OCR could not detect target)" if not detected_new_target else "grid content changed (same target)"

                    logger.always_log("=" * 70)
                    logger.always_log("NEW PUZZLE DETECTED AFTER VERIFICATION!")
                    logger.always_log(f"   Previous target: '{self.last_target_object}'")
                    logger.always_log(f"   New target detected: '{detected_new_target if detected_new_target else 'Unknown (will be detected in ANALYZING state)'}'")
                    logger.always_log(f"   Detection reason: {reason}")
                    logger.always_log("   This means the previous puzzle was SOLVED!")
                    logger.always_log("   Transitioning to analyze the new puzzle...")
                    logger.always_log("=" * 70)

                    # Reset for new puzzle
                    self.clicked_cells.clear()
                    self.verification_attempted = False
                    self.mark_new_puzzle()
                    self.last_target_object = detected_new_target if detected_new_target else self.last_target_object

                    self.transition_to(CaptchaState.PUZZLE_ANALYZING, f"New puzzle detected after verification ({reason})")
                    return

                # Check if we detected ADDITIONAL cells
                # This happens in segmentation when we missed some cells initially
                if not grid_changed and detected_new_target == self.last_target_object and new_selected:
                    # Find cells that were NOT clicked before
                    additional_cells = list(new_selected - old_clicked)

                    if additional_cells:
                        logger.always_log("=" * 70)
                        logger.always_log("ADDITIONAL CELLS DETECTED!")
                        logger.always_log(f"   Target: '{self.last_target_object}' (unchanged)")
                        logger.always_log(f"   Previously clicked: {sorted(list(old_clicked))} ({len(old_clicked)} cells)")
                        logger.always_log(f"   Now detected: {sorted(list(new_selected))} ({len(new_selected)} cells)")
                        logger.always_log(f"   Additional cells to click: {sorted(additional_cells)} ({len(additional_cells)} cells)")
                        logger.always_log(f"   This likely means we missed cells in the initial detection")
                        logger.always_log(f"   Action: Re-analyzing to click the missed cells")
                        logger.always_log("=" * 70)

                        # Go back to analyzing to click the additional cells
                        self.mark_new_puzzle()  # Mark as new detection cycle
                        self.transition_to(CaptchaState.PUZZLE_ANALYZING, "Additional cells detected - clicking missed cells")
                        return

            # If we get here, it's a genuine verification failure with no new information
            logger.always_log("=" * 70)
            logger.always_log("FINAL DECISION: Verification failed (same puzzle)")
            logger.always_log("   Reason: Target unchanged AND grid unchanged")
            logger.always_log("   Action: Going to RELOAD state")
            logger.always_log("=" * 70)
            self.transition_to(CaptchaState.PUZZLE_RELOADING, "VERIFICATION FAILED: CAPTCHA rejected our solution")
        else:
            self.transition_to(CaptchaState.PUZZLE_RELOADING, "UNCLEAR RESULT: Could not determine if verification succeeded or failed")
    
    def _check_verification_result(self):
        """
        Fast and responsive verification result checking.
        Returns: "success" or "failed"
        """
        from captcha_solver_bridge import check_rc_imageselect_exists

        # Reduced initial wait for faster response
        time.sleep(0.5)

        start_time = time.time()
        max_wait_time = 5.0  # 5 seconds max before declaring failure
        check_count = 0

        while True:
            elapsed = time.time() - start_time
            check_count += 1

            # Use fresh detection to check current screen state
            puzzle_exists = check_rc_imageselect_exists(use_cache=False)

            if not puzzle_exists:
                # CAPTCHA disappeared - likely success
                # Double check to confirm
                time.sleep(0.3)
                puzzle_exists_recheck = check_rc_imageselect_exists(use_cache=False)
                if not puzzle_exists_recheck:
                    return "success"
                else:
                    # Probably just loading, continue checking
                    continue
            else:
                # CAPTCHA still exists. Check if we've waited long enough
                if elapsed >= max_wait_time:
                    return "failed"

                time.sleep(0.5)
    
    def _wait_for_captcha_images_to_load(self, max_wait_time=10, check_interval=0.5):
        """
        Wait for CAPTCHA images to fully load before proceeding.

        This checks if the CAPTCHA grid area has actual content (not just white/empty).
        Returns True if images loaded, False if timeout.
        """
        from captcha_solver_bridge import capture_captcha_area_only

        start_time = time.time()
        check_count = 0

        while time.time() - start_time < max_wait_time:
            check_count += 1

            # Capture current CAPTCHA area
            captured_image_data, captcha_coords = capture_captcha_area_only()

            if captured_image_data is not None:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(captured_image_data, cv2.COLOR_BGR2GRAY)

                # Check if the image has sufficient variation (not just empty/white)
                # Calculate standard deviation - empty images will have very low std dev
                std_dev = np.std(gray)

                # Also check mean brightness - pure white areas will have high mean
                mean_brightness = np.mean(gray)

                # Count non-white pixels (anything not close to 255)
                non_white_pixels = np.sum(gray < 240)  # Pixels that aren't nearly white
                total_pixels = gray.shape[0] * gray.shape[1]
                non_white_ratio = non_white_pixels / total_pixels

                # Images are considered loaded if:
                # 1. Standard deviation is high enough which indicates content variation
                # 2. There's a reasonable amount of non-white pixels
                # 3. Mean brightness isn't too high (not just white)
                if std_dev > 15 and non_white_ratio > 0.1 and mean_brightness < 220:
                    return True

            time.sleep(check_interval)

        return False
    
    def _check_for_completion_text(self, image):
        try:
            from captcha_solver_bridge import get_solver
            processor = get_solver()
            ocr_results = processor.ocr_reader.readtext(image, detail=0, paragraph=True)
            full_text = " ".join(ocr_results).lower()
            return full_text
        except Exception as e:
            return ""

    def _handle_puzzle_reloading(self):
        from captcha_solver_bridge import find_and_click_reload_button
        # === METRICS START ===
        metrics.add_reload()
        # === METRICS END ===
        self.reload_count += 1

        # Prevent infinite reloading - if we tried too many times, just give up
        MAX_RELOAD_ATTEMPTS = 10
        if self.reload_count > MAX_RELOAD_ATTEMPTS:
            self.transition_to(CaptchaState.ERROR, f"Too many reload attempts: {self.reload_count}")
            return

        # Get the reason from the transition message if available
        reload_reason = getattr(self, '_last_transition_reason', 'Unknown reason')
        
        logger.always_log("=" * 60)
        logger.always_log(f"RELOADING PUZZLE (Attempt {self.reload_count}/{MAX_RELOAD_ATTEMPTS})")
        logger.always_log(f"RELOAD REASON: {reload_reason}")
        logger.always_log("=" * 60)

        try:
            reload_clicked = find_and_click_reload_button()
            if not reload_clicked:
                logger.always_log(f"Failed to find/click reload button on attempt {self.reload_count}")
                self.transition_to(CaptchaState.ERROR, "Failed to click reload button")
                return
        except Exception as e:
            logger.always_log(f"Exception while trying to click reload button: {e}")
            import traceback
            logger.always_log(f"   Traceback: {traceback.format_exc()}")
            self.transition_to(CaptchaState.ERROR, f"Exception clicking reload: {e}")
            return

        time.sleep(0.5)  # Reduced wait for faster response
        self.mark_new_puzzle()
        # Reset any cached state for the new puzzle
        self.current_all_cells_data = None
        self.current_result = None
        self.verification_attempted = False
        self.transition_to(CaptchaState.PUZZLE_ANALYZING, "Puzzle reloaded")

def solve_captcha_with_state_machine(existing_state_machine=None):
    if existing_state_machine is not None:
        # Reuse existing state machine to preserve previous_cell_data
        state_machine = existing_state_machine
        logger.always_log("Reusing existing state machine")
    else:
        # Create new state machine
        state_machine = CaptchaStateMachine()

    success = state_machine.run()
    return success, state_machine