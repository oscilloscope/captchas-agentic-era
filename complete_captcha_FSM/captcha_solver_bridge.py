import pyautogui
import numpy as np
import cv2
from logger import logger

# Set by main.py
_processor_instance = None

def set_processor_instance(processor):
    """Set the global processor instance"""
    global _processor_instance
    _processor_instance = processor

def get_processor():
    """Get the current processor instance"""
    if _processor_instance is None:
        raise RuntimeError(
            "Processor not initialized! Call set_processor_instance() first.\n"
            "This is a bug - main.py should always initialize the processor before any bridge functions are called."
        )
    return _processor_instance


def check_rc_imageselect_exists(use_cache=True):
    """
    Check if reCAPTCHA image select puzzle exists

    Args:
        use_cache: Not currently used, kept for API compatibility.
    """
    processor = get_processor()
    return processor.find_object_on_screen(class_id=0) is not None


def _is_within_bounds(coords, bounds, margin=10):
    """
    Check if coordinates are within bounds with optional margin.

    Args:
        coords: (x1, y1, x2, y2) coordinates to check
        bounds: (x1, y1, x2, y2) boundary coordinates
        margin: pixels outside bounds that are still acceptable

    Returns:
        bool: True if coords are within bounds or close enough within margin
    """
    if not coords or not bounds:
        return False

    obj_x1, obj_y1, obj_x2, obj_y2 = coords
    bound_x1, bound_y1, bound_x2, bound_y2 = bounds

    # Calculate center point of the object
    obj_center_x = (obj_x1 + obj_x2) // 2
    obj_center_y = (obj_y1 + obj_y2) // 2

    # Check if center is within bounds (with margin)
    within_x = (bound_x1 - margin) <= obj_center_x <= (bound_x2 + margin)
    within_y = (bound_y1 - margin) <= obj_center_y <= (bound_y2 + margin)


    return within_x and within_y


def find_and_click_recaptcha_checkbox():
    """Find and click reCAPTCHA checkbox (uses detection model)"""
    from logger import logger
    processor = get_processor()
    logger.always_log("Bridge: Calling find_object_on_screen(class_id=4) for checkbox")
    coords = processor.find_object_on_screen(class_id=4)
    logger.always_log(f"Bridge: find_object_on_screen returned coords: {coords}")
    logger.always_log(f"Bridge: Calling _click_center with coords: {coords}")
    result = processor._click_center(coords)
    logger.always_log(f"Bridge: _click_center returned: {result}")
    return result


def find_and_click_verify_button():
    """Find and click verify button (uses detection model)"""
    processor = get_processor()
    # First find the CAPTCHA area
    captcha_area_coords = processor.find_object_on_screen(class_id=0)
    if not captcha_area_coords:
        logger.always_log("Could not find CAPTCHA area for verify button search")
        return False

    logger.always_log(f"VERIFY BUTTON SEARCH: CAPTCHA area = {captcha_area_coords}")

    # Search for verify button within the CAPTCHA area
    coords = processor.find_object_in_captcha_area(class_id=3, captcha_area_coords=captcha_area_coords)
    if not coords:
        logger.always_log("Could not find verify button within CAPTCHA area, trying fullscreen...")
        # Fallback to fullscreen search for verify button
        coords = processor.find_object_on_screen(class_id=3)
        if coords:
            # Validate that the found button is actually within the CAPTCHA area
            if _is_within_bounds(coords, captcha_area_coords, margin=50):
                logger.always_log(f"Found verify button on fullscreen: {coords} (validated within bounds)")
            else:
                logger.always_log("WARNING: Found verify button on fullscreen but it's OUTSIDE CAPTCHA area!")
                logger.always_log(f"   Button coords: {coords}")
                logger.always_log(f"   CAPTCHA area: {captcha_area_coords}")
                logger.always_log("   Rejecting this detection to prevent clicking outside puzzle")
                return False
        else:
            logger.always_log("Could not find verify button anywhere on screen!")
            return False
    else:
        logger.always_log(f"Found verify button within CAPTCHA area: {coords}")

    return processor._click_center(coords)


def find_and_click_reload_button():
    """Find and click reload button (uses detection model)"""
    processor = get_processor()
    # First find the CAPTCHA area
    captcha_area_coords = processor.find_object_on_screen(class_id=0)
    if not captcha_area_coords:
        logger.log("Could not find CAPTCHA area for reload button search")
        return False

    # Search for reload button within the CAPTCHA area
    coords = processor.find_object_in_captcha_area(class_id=2, captcha_area_coords=captcha_area_coords)
    if not coords:
        logger.log("Could not find reload button within CAPTCHA area")
        # Fallback to fullscreen search for reload button
        coords = processor.find_object_on_screen(class_id=2)
        if coords:
            # Validate that the found button is actually within the CAPTCHA area
            if _is_within_bounds(coords, captcha_area_coords, margin=50):
                logger.log("Found reload button on fullscreen (validated within bounds)")
            else:
                logger.always_log("WARNING: Found reload button on fullscreen but it's OUTSIDE CAPTCHA area!")
                logger.always_log(f"   Button coords: {coords}")
                logger.always_log(f"   CAPTCHA area: {captcha_area_coords}")
                logger.always_log("   Rejecting this detection to prevent clicking outside puzzle")
                return False

    return processor._click_center(coords)


def capture_captcha_area_only():
    """Capture only the CAPTCHA area (uses detection model)"""
    processor = get_processor()

    # Find the captcha area on the screen and get its global coordinates
    coords = processor.find_object_on_screen(class_id=0)

    # If coordinates are found, proceed
    if coords:
        x1, y1, x2, y2 = coords
        width, height = x2 - x1, y2 - y1

        # Apply display scaling for screenshot region (pyautogui expects logical coordinates)
        processor = get_processor()
        if hasattr(processor, 'display_scaling') and processor.display_scaling and processor.display_scaling > 0 and processor.display_scaling != 1.0:
            scaled_x1 = int(x1 / processor.display_scaling)
            scaled_y1 = int(y1 / processor.display_scaling)
            scaled_width = int(width / processor.display_scaling)
            scaled_height = int(height / processor.display_scaling)
        else:
            scaled_x1, scaled_y1 = x1, y1
            scaled_width, scaled_height = width, height

        # Take a screenshot of only that region into a memory variable (PIL Image)
        screenshot_pil = pyautogui.screenshot(region=(scaled_x1, scaled_y1, scaled_width, scaled_height))

        # Convert the PIL Image to the BGR numpy array format that OpenCV uses
        screenshot_np = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

        # Return the raw image data (numpy array) and the coordinates
        return screenshot_np, coords

    # If no captcha area was found, return two None values.
    return None, None


def get_solver():
    """Get the current solver instance (YOLO or LLM based on backend)"""
    return get_processor()


def get_supported_objects():
    """Get list of supported objects for YOLO backend (only called in YOLO mode)"""
    processor = get_processor()

    # Return objects the YOLO classification model actually supports
    try:
        # Get the actual class names from the YOLO classification model
        if hasattr(processor, 'classification_backend') and hasattr(processor.classification_backend, 'names'):
            model_classes = list(processor.classification_backend.names.values())
            # Add plural forms for common objects
            supported = []
            for obj in model_classes:
                supported.append(obj)
                # Add common plural forms
                if obj.endswith('y'):
                    supported.append(obj[:-1] + 'ies')  
                elif obj.endswith(('s', 'sh', 'ch', 'x', 'z')):
                    supported.append(obj + 'es')  
                elif obj.endswith('f'):
                    supported.append(obj[:-1] + 'ves')  
                else:
                    supported.append(obj + 's') 

            return supported
        else:
            # Fallback to actual YOLO model classes if we can't access the model directly
            return [
                "bicycle", "bicycles", "bridge", "bridges", "bus", "buses", "car", "cars",
                "chimney", "chimneys", "crosswalk", "crosswalks", "hydrant", "hydrants",
                "fire hydrant", "fire hydrants", "motorcycle", "motorcycles", "mountain", "mountains",
                "palm", "palms", "palm tree", "palm trees", "stair", "stairs", "traffic light", "traffic lights",
                "tractor", "tractors"
            ]
    except Exception as e:
        logger.log(f"Error getting YOLO model classes: {e}")
        # Return actual YOLO model classes
        return [
            "bicycle", "bicycles", "bridge", "bridges", "bus", "buses", "car", "cars",
            "chimney", "chimneys", "crosswalk", "crosswalks", "hydrant", "hydrants",
            "fire hydrant", "fire hydrants", "motorcycle", "motorcycles", "mountain", "mountains",
            "palm", "palms", "palm tree", "palm trees", "stair", "stairs", "traffic light", "traffic lights",
            "tractor", "tractors"
        ]
