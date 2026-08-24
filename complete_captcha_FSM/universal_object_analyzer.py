#!/usr/bin/env python3
import sys
import os
import argparse
import requests
import json
import io
import shutil
from PIL import Image
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import time

class VisionLLMClient:
    def __init__(self, api_url: str = ""):
        self.api_url = api_url.rstrip('/')
        self.endpoint = f"{self.api_url}/generate"
        self.last_response = None

    def send_image_to_api(self, image: Image.Image, prompt: str) -> dict:
        """Send image to the API with a prompt."""
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
            print(f"Sending request to {self.endpoint}...")
            response = requests.post(self.endpoint, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            self.last_response = result
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
            return None


    def detect_objects_with_boxes(self, image_path: str, object_name: str) -> Dict:
        print(f"\nOBJECT DETECTION MODE")
        print(f"Target: {object_name}")
        print(f"Image: {image_path}")
        print("=" * 60)

        try:
            image = Image.open(image_path)
            print(f"Image size: {image.size}")
        except Exception as e:
            return {'success': False, 'error': f"Could not load image: {e}"}

        prompt = f"""Find all {object_name} in this image and create bounding boxes that encompass the entire {object_name}.

Requirements:
- Include ALL visible parts of the {object_name}
- Capture the complete {object_name} from top to bottom and side to side
- Make sure the entire {object_name} is contained within the bounding box

For each {object_name}, return coordinates in this format:
{object_name.title()} 1: [x1, y1, x2, y2]
{object_name.title()} 2: [x1, y1, x2, y2]

Where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner."""

        result = self.send_image_to_api(image, prompt)

        if result:
            model_output = result.get('model_output', '')
            print(f"\n=== Model Response ===")
            print(model_output)

            objects = self.extract_bounding_boxes(model_output, object_name)

            if objects:
                print(f"\n[SUCCESS] Found {len(objects)} {object_name}(s)")

                summary = {
                    'mode': 'detection',
                    'object_name': object_name,
                    'image_path': image_path,
                    'objects_found': len(objects),
                    'object_details': objects,
                    'model_response': model_output,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }

                return {'success': True, 'summary': summary}
            else:
                print(f"\n[WARNING] No {object_name} detected")
                return {'success': True, 'summary': {'objects_found': 0, 'model_response': model_output}}
        else:
            return {'success': False, 'error': 'API request failed'}

    def extract_bounding_boxes(self, text: str, object_name: str) -> List[Dict]:
        """Extract bounding boxes for the specified object from API response."""
        objects = []
        # Cover different responses of LLM
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
        # Fallback
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
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })

                    print(f"Found {object_name} {obj_num}: bbox=({x1}, {y1}, {x2}, {y2})")
                    obj_count += 1

                except (ValueError, IndexError) as e:
                    continue

            if objects:  # Stop after finding objects
                break

        return objects

    # Teacher-student pipeline for efficiency
    # TBD
    def classify_objects_in_folder(self, folder_path: str, object_name: str) -> Dict:
        """
        Classify objects in individual images with YES/NO responses.

        IMPORTANT: This method is part of the LLM-to-YOLO training pipeline described in the paper.

        Purpose:
        - LLM acts as "teacher" by analyzing CAPTCHA cells and making YES/NO decisions
        - Creates labeled training data by sorting images into selected/unselected folders
        - This labeled data is used to train/fine-tune YOLO models for future use
        - Implements the "reasoning to reflex" concept: LLM decisions become YOLO training data

        Folder Creation:
        - Creates selected_{object_name}/ for positive examples
        - Creates unselected_{object_name}/ for negative examples
        - Folders persist on disk for later YOLO training

        This enables the system to improve over time: slow but accurate LLM analysis
        generates training data that makes fast YOLO-based solving possible.
        """
        print(f"\nCLASSIFICATION MODE")
        print(f"Target: {object_name}")
        print(f"Folder: {folder_path}")
        print("=" * 60)

        if not os.path.isdir(folder_path):
            return {'success': False, 'error': 'Folder does not exist'}

        safe_object_name = object_name.replace(' ', '_').replace('/', '_')
        selected_folder = f"selected_{safe_object_name}"
        unselected_folder = f"unselected_{safe_object_name}"

        os.makedirs(selected_folder, exist_ok=True)
        os.makedirs(unselected_folder, exist_ok=True)

        print(f"Created: {selected_folder}/ and {unselected_folder}/")

        image_files = []
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(file)

        image_files.sort()
        print(f"Found {len(image_files)} image files to analyze")

        if not image_files:
            return {'success': False, 'error': 'No image files found in folder'}

        results = []
        selected_count = 0
        unselected_count = 0

        for i, filename in enumerate(image_files):
            file_path = os.path.join(folder_path, filename)
            print(f"\n[{i+1}/{len(image_files)}] Processing {filename}...")

            try:
                image = Image.open(file_path)
                print(f"  Size: {image.size}")

                # NOTE: Paper methodology vs implementation difference
                # Paper used class index prompts (0-15) for VLM to enable fair benchmarking against YOLO.
                # This implementation uses YES/NO binary prompts for practical deployment because:
                # 1) Clearer binary decision making for cell classification
                # 2) Simpler prompt engineering for reliable responses
                # 3) Better suited for training data generation (LLM as teacher)
                prompt = f"""Look at this image carefully.

Question: Does this image contain {object_name}?

Please answer with YES or NO only.

If you see any type of {object_name}, answer YES.
If you see anything else that is NOT {object_name}, answer NO.

Answer: """

                result = self.send_image_to_api(image, prompt)

                if result:
                    model_output = result.get('model_output', '').strip()
                    print(f"  Response: '{model_output}'")

                    has_object = self.parse_yes_no_response(model_output)
                    decision = 'YES' if has_object else 'NO'

                    if has_object:
                        dest_folder = selected_folder
                        status = "SELECTED"
                        selected_count += 1
                    else:
                        dest_folder = unselected_folder
                        status = "UNSELECTED"
                        unselected_count += 1

                    dest_path = os.path.join(dest_folder, filename)
                    shutil.copy2(file_path, dest_path)
                    results.append({
                        'filename': filename,
                        'original_path': file_path,
                        'destination_path': dest_path,
                        'has_object': has_object,
                        'raw_response': model_output,
                        'status': status,
                        'success': True
                    })

                else:
                    print("  ERROR: API request failed")
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': 'API request failed'
                    })

                time.sleep(0.5)

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })

        summary = {
            'mode': 'classification',
            'object_name': object_name,
            'folder_path': folder_path,
            'total_images': len(image_files),
            'selected_count': selected_count,
            'unselected_count': unselected_count,
            'selected_folder': selected_folder,
            'unselected_folder': unselected_folder,
            'results': results,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        print(f"\n" + "=" * 60)
        print(f"CLASSIFICATION SUMMARY")
        print(f"=" * 60)
        print(f"Total images: {len(image_files)}")
        print(f"Selected ({object_name}): {selected_count}")
        print(f"Unselected (no {object_name}): {unselected_count}")

        return {'success': True, 'summary': summary}

    def parse_yes_no_response(self, text: str) -> bool:
        """Parse YES/NO response from model output."""
        text_lower = text.lower().strip()

        if text_lower.startswith('yes') or text_lower == 'yes':
            return True
        if text_lower.startswith('no') or text_lower == 'no':
            return False

        if 'yes' in text_lower and 'no' not in text_lower:
            return True
        if 'no' in text_lower and 'yes' not in text_lower:
            return False

        print(f"  Warning: Unclear response '{text}', defaulting to NO")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Universal Object Analyzer - Detection (images) or Classification (folders)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python universal_object_analyzer.py "bus" "captcha.png"           # Object detection
  python universal_object_analyzer.py "stairs" "individual_cells/" # Classification
  python universal_object_analyzer.py "traffic lights" "image.jpg" # Object detection
        """
    )

    config_file = 'config.json'
    default_api_url = ''

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            default_api_url = config_data.get('api_url', '')
    except Exception:
        pass

    parser.add_argument('object_name', help='Name of the object to search for')
    parser.add_argument('path', help='Path to image file (detection) or folder (classification)')
    parser.add_argument('--api-url', default=default_api_url,
                       help=f'API URL (default: {default_api_url} from config.json)')

    args = parser.parse_args()

    print("=" * 70)
    print("UNIVERSAL OBJECT ANALYZER")
    print("=" * 70)
    print(f"Object: {args.object_name}")
    print(f"Path: {args.path}")
    print(f"API: {args.api_url}")

    analyzer = VisionLLMClient(args.api_url)

    if os.path.isfile(args.path):
        result = analyzer.detect_objects_with_boxes(args.path, args.object_name)
    elif os.path.isdir(args.path):
        result = analyzer.classify_objects_in_folder(args.path, args.object_name)
    else:
        print(f"\nERROR: Path '{args.path}' does not exist or is not accessible")
        return 1

    if result['success']:
        print(f"\nAnalysis completed successfully!")
        return 0
    else:
        print(f"\nAnalysis failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())