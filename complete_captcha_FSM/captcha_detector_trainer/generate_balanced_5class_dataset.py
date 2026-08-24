"""
Generate Balanced 5-Class CAPTCHA Dataset
==========================================
Creates comprehensive training dataset with:
- CAPTCHA samples (classes 0-3): Each CAPTCHA on 4 unique backgrounds
- Robot checkbox samples (class 4): Random backgrounds
- Negative samples: Pure backgrounds, no annotations

Total: 1,200 samples (880 CAPTCHA + 200 Robot + 120 Negative)
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import random
import json
from datetime import datetime
import shutil
import cv2
import pytesseract
import numpy as np

class Balanced5ClassGenerator:
    def __init__(self):
        # Directories
        self.background_dir = Path("screenshots_unique_many")
        self.captcha_dir = Path("captcha_images")
        self.robot_image = Path("I_am_not_a_robot.png")

        # Output directory
        self.output_dir = Path("balanced_5class_dataset")
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.verification_dir = self.output_dir / "verification"

        # Create directories
        for dir_path in [self.images_dir, self.labels_dir, self.verification_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Class definitions
        self.class_names = {
            0: "captcha_area",
            1: "cell",
            2: "reload_button",
            3: "submit_button",
            4: "robot_checkbox"
        }

        # Button coordinates 
        self.reload_button = {
            'class': 2,
            'center_x': 0.073,
            'center_y': 0.954,  
            'width': 0.063,
            'height': 0.059,
        }

        self.submit_button = {
            'class': 3,
            'center_x': 0.854, 
            'center_y': 0.945,
            'width': 0.247,    
            'height': 0.070,
        }

        # Scale factors for augmentation
        self.scale_factors = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        # Grid cell templates (from actual CAPTCHA measurements)
        self.grid_templates = {
            "3x3": {
                "coordinates": [
                    # Row 0
                    [0.180272, 0.329953, 0.292517, 0.198128],  # Cell (0,0)
                    [0.502268, 0.329953, 0.297052, 0.198128],  # Cell (0,1)
                    [0.823129, 0.329953, 0.290249, 0.198128],  # Cell (0,2)
                    # Row 1
                    [0.180272, 0.549922, 0.292517, 0.204368],  # Cell (1,0)
                    [0.502268, 0.549922, 0.297052, 0.204368],  # Cell (1,1)
                    [0.823129, 0.549922, 0.290249, 0.204368],  # Cell (1,2)
                    # Row 2
                    [0.180272, 0.769891, 0.292517, 0.198128],  # Cell (2,0)
                    [0.502268, 0.769891, 0.297052, 0.198128],  # Cell (2,1)
                    [0.823129, 0.769891, 0.290249, 0.198128],  # Cell (2,2)
                ]
            },
            "4x4": {
                "coordinates": [
                    # Row 0
                    [0.138322, 0.301092, 0.222222, 0.149766],  # Cell (0,0)
                    [0.377551, 0.301092, 0.229025, 0.149766],  # Cell (0,1)
                    [0.620181, 0.301092, 0.229025, 0.149766],  # Cell (0,2)
                    [0.861678, 0.301092, 0.226757, 0.149766],  # Cell (0,3)
                    # Row 1
                    [0.138322, 0.464119, 0.222222, 0.157566],  # Cell (1,0)
                    [0.377551, 0.464119, 0.229025, 0.157566],  # Cell (1,1)
                    [0.620181, 0.464119, 0.229025, 0.157566],  # Cell (1,2)
                    [0.861678, 0.464119, 0.226757, 0.157566],  # Cell (1,3)
                    # Row 2
                    [0.138322, 0.631045, 0.222222, 0.157566],  # Cell (2,0)
                    [0.377551, 0.631045, 0.229025, 0.157566],  # Cell (2,1)
                    [0.620181, 0.631045, 0.229025, 0.157566],  # Cell (2,2)
                    [0.861678, 0.631045, 0.226757, 0.157566],  # Cell (2,3)
                    # Row 3
                    [0.138322, 0.796412, 0.222222, 0.154446],  # Cell (3,0)
                    [0.377551, 0.796412, 0.229025, 0.154446],  # Cell (3,1)
                    [0.620181, 0.796412, 0.229025, 0.154446],  # Cell (3,2)
                    [0.861678, 0.796412, 0.226757, 0.154446],  # Cell (3,3)
                ]
            }
        }

        # Statistics
        self.stats = {
            'captcha_samples': 0,
            'robot_samples': 0,
            'negative_samples': 0,
            'failed': 0
        }

        print("Balanced 5-Class Generator Initialized")
        print(f"Backgrounds: {self.background_dir}")
        print(f"CAPTCHAs: {self.captcha_dir}")
        print(f"Output: {self.output_dir}")

    def detect_grid_type(self, captcha_path):
        # OCR-based text pattern detection
        grid_type = self.detect_by_text_patterns(captcha_path)
        if grid_type:
            return grid_type, 9 if grid_type == "3x3" else 16

        # Visual grid counting
        grid_type = self.count_grid_cells(captcha_path)
        return grid_type, 9 if grid_type == "3x3" else 16

    def detect_by_text_patterns(self, captcha_path):
        """Detect grid type using OCR to find text patterns """
        try:
            # Load image
            img = cv2.imread(str(captcha_path))
            if img is None:
                return None

            height, width = img.shape[:2]

            # Extract header
            header_height = int(height * 0.25)
            header = img[:header_height, :]

            # Convert to grayscale
            gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)

            # Invert for white text on blue
            inverted = 255 - gray

            # Extract text
            try:
                text = pytesseract.image_to_string(inverted, config='--psm 6')
                text_lower = text.lower()

                # Check for 4x4 indicators
                if "squares" in text_lower or "square" in text_lower:
                    return "4x4"  # Definitive 4x4
                if "skip" in text_lower:
                    return "4x4"  # 4x4 has skip button

                # Check for 3x3 indicators
                if "none left" in text_lower:
                    return "3x3"  # 3x3 specific text
                if "verify" in text_lower and "none" in text_lower:
                    return "3x3"  # 3x3 pattern
                if "images" in text_lower and ("with" in text_lower or "none" in text_lower):
                    return "3x3"  # Definitive 3x3

            except Exception as e:
                pass  

        except Exception as e:
            pass

        return None  

    def count_grid_cells(self, captcha_path):
        """Count actual grid cells visually using edge detection"""
        try:
            img = cv2.imread(str(captcha_path))
            if img is None:
                return "3x3" 

            height, width = img.shape[:2]

            # Skip header area
            header_height = int(height * 0.22)
            grid_area = img[header_height:, :]

            # Convert to grayscale
            gray = cv2.cvtColor(grid_area, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                   minLineLength=width*0.3, maxLineGap=10)

            if lines is not None:
                # Count horizontal and vertical lines
                h_lines = []
                v_lines = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                    if angle < 10 or angle > 170:  # Horizontal
                        h_lines.append(y1)
                    elif 80 < angle < 100:  # Vertical
                        v_lines.append(x1)

                # Remove duplicates (lines close to each other)
                h_lines = list(set([h // 20 for h in h_lines]))
                v_lines = list(set([v // 20 for v in v_lines]))

                # Estimate grid size
                if len(h_lines) <= 4 and len(v_lines) <= 4:
                    return "3x3"
                else:
                    return "4x4"

            # Fallback: check grid area dimensions
            grid_height = grid_area.shape[0]
            estimated_cell_height = grid_height / 3

            if estimated_cell_height > 130:  # Large cells = 3x3
                return "3x3"
            else:
                return "4x4"

        except Exception as e:
            print(f"Grid detection error: {e}")
            return "3x3"  # Default

    def generate_captcha_sample(self, background_path, captcha_path, output_name, scale_factor=1.0):
        """Generate CAPTCHA sample with corrected button coordinates"""

        try:
            # Load images
            background = Image.open(background_path).convert('RGB')
            captcha_img = Image.open(captcha_path).convert('RGBA')

            # Detect grid type
            grid_type, num_cells = self.detect_grid_type(captcha_path)

            # Scale CAPTCHA
            if scale_factor != 1.0:
                orig_w, orig_h = captcha_img.size
                new_w = int(orig_w * scale_factor)
                new_h = int(orig_h * scale_factor)
                captcha_img = captcha_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            bg_width, bg_height = background.size
            captcha_width, captcha_height = captcha_img.size

            # Ensure CAPTCHA fits
            if captcha_width > bg_width or captcha_height > bg_height:
                scale = min(bg_width / captcha_width, bg_height / captcha_height) * 0.9
                new_w = int(captcha_width * scale)
                new_h = int(captcha_height * scale)
                captcha_img = captcha_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                captcha_width, captcha_height = new_w, new_h

            # Random position
            max_x = bg_width - captcha_width
            max_y = bg_height - captcha_height
            x_offset = random.randint(0, max_x)
            y_offset = random.randint(0, max_y)

            # Add shadow
            shadow = Image.new('RGBA', captcha_img.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rectangle([0, 0, captcha_width, captcha_height], fill=(0, 0, 0, 60))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))

            # Create composite
            synthetic = background.copy()
            synthetic.paste(shadow, (x_offset + 5, y_offset + 5), shadow)
            synthetic.paste(captcha_img, (x_offset, y_offset), captcha_img)

            # Save image
            image_path = self.images_dir / f"{output_name}.jpg"
            synthetic.save(image_path, quality=95)

            # Create annotations
            self.create_captcha_annotations(
                (bg_width, bg_height),
                (captcha_width, captcha_height),
                (x_offset, y_offset),
                grid_type,
                num_cells,
                output_name
            )

            # Create verification
            self.create_captcha_verification(
                synthetic,
                (captcha_width, captcha_height),
                (x_offset, y_offset),
                grid_type,
                num_cells,
                output_name
            )

            return True

        except Exception as e:
            print(f"Error generating CAPTCHA sample: {e}")
            return False

    def create_captcha_annotations(self, image_size, captcha_size, captcha_pos, grid_type, num_cells, output_name):
        """Create YOLO annotations for CAPTCHA with corrected buttons"""

        img_width, img_height = image_size
        captcha_width, captcha_height = captcha_size
        captcha_x, captcha_y = captcha_pos

        label_path = self.labels_dir / f"{output_name}.txt"

        with open(label_path, 'w') as f:
            # CAPTCHA area
            captcha_x_center = (captcha_x + captcha_width / 2) / img_width
            captcha_y_center = (captcha_y + captcha_height / 2) / img_height
            captcha_norm_width = captcha_width / img_width
            captcha_norm_height = captcha_height / img_height

            f.write(f"0 {captcha_x_center:.6f} {captcha_y_center:.6f} "
                   f"{captcha_norm_width:.6f} {captcha_norm_height:.6f}\n")

            # Grid cells (using correct templates from actual CAPTCHAs)
            if grid_type in self.grid_templates:
                template_coords = self.grid_templates[grid_type]["coordinates"]

                for coord in template_coords:
                    template_x_center, template_y_center, template_width, template_height = coord

                    # Convert template coordinates (relative to CAPTCHA) to absolute pixel coordinates
                    abs_x_center = template_x_center * captcha_width
                    abs_y_center = template_y_center * captcha_height
                    abs_width = template_width * captcha_width
                    abs_height = template_height * captcha_height

                    # Translate to position within synthetic screenshot
                    final_x_center = captcha_x + abs_x_center
                    final_y_center = captcha_y + abs_y_center

                    # Normalize to screenshot dimensions
                    norm_x_center = final_x_center / img_width
                    norm_y_center = final_y_center / img_height
                    norm_width = abs_width / img_width
                    norm_height = abs_height / img_height

                    f.write(f"1 {norm_x_center:.6f} {norm_y_center:.6f} "
                           f"{norm_width:.6f} {norm_height:.6f}\n")

            # Reload button (CORRECTED coordinates)
            reload_x = captcha_x + self.reload_button['center_x'] * captcha_width
            reload_y = captcha_y + self.reload_button['center_y'] * captcha_height
            reload_w = self.reload_button['width'] * captcha_width
            reload_h = self.reload_button['height'] * captcha_height

            reload_x_norm = reload_x / img_width
            reload_y_norm = reload_y / img_height
            reload_w_norm = reload_w / img_width
            reload_h_norm = reload_h / img_height

            f.write(f"2 {reload_x_norm:.6f} {reload_y_norm:.6f} "
                   f"{reload_w_norm:.6f} {reload_h_norm:.6f}\n")

            # Class 3: Submit button (CORRECTED coordinates)
            submit_x = captcha_x + self.submit_button['center_x'] * captcha_width
            submit_y = captcha_y + self.submit_button['center_y'] * captcha_height
            submit_w = self.submit_button['width'] * captcha_width
            submit_h = self.submit_button['height'] * captcha_height

            submit_x_norm = submit_x / img_width
            submit_y_norm = submit_y / img_height
            submit_w_norm = submit_w / img_width
            submit_h_norm = submit_h / img_height

            f.write(f"3 {submit_x_norm:.6f} {submit_y_norm:.6f} "
                   f"{submit_w_norm:.6f} {submit_h_norm:.6f}\n")

    def create_captcha_verification(self, synthetic_img, captcha_size, captcha_pos, grid_type, num_cells, output_name):
        """Create verification image for CAPTCHA"""

        verify_img = synthetic_img.copy()
        draw = ImageDraw.Draw(verify_img)

        captcha_width, captcha_height = captcha_size
        captcha_x, captcha_y = captcha_pos

        # Draw CAPTCHA area (red)
        for i in range(3):
            draw.rectangle(
                [captcha_x - i, captcha_y - i,
                 captcha_x + captcha_width + i, captcha_y + captcha_height + i],
                outline="red", fill=None
            )

        # Draw grid cells (blue) using CORRECT templates
        if grid_type in self.grid_templates:
            template_coords = self.grid_templates[grid_type]["coordinates"]

            for coord in template_coords:
                template_x_center, template_y_center, template_width, template_height = coord

                # Convert to absolute coordinates
                abs_x_center = captcha_x + (template_x_center * captcha_width)
                abs_y_center = captcha_y + (template_y_center * captcha_height)
                abs_width = template_width * captcha_width
                abs_height = template_height * captcha_height

                # Calculate corners
                cell_x1 = abs_x_center - abs_width / 2
                cell_y1 = abs_y_center - abs_height / 2
                cell_x2 = abs_x_center + abs_width / 2
                cell_y2 = abs_y_center + abs_height / 2

                draw.rectangle([cell_x1, cell_y1, cell_x2, cell_y2], outline="blue", width=2)

        # Draw buttons
        # Reload (green)
        reload_x = captcha_x + self.reload_button['center_x'] * captcha_width
        reload_y = captcha_y + self.reload_button['center_y'] * captcha_height
        reload_w = self.reload_button['width'] * captcha_width
        reload_h = self.reload_button['height'] * captcha_height

        draw.rectangle([
            reload_x - reload_w/2, reload_y - reload_h/2,
            reload_x + reload_w/2, reload_y + reload_h/2
        ], outline="green", width=3)

        # Submit (orange)
        submit_x = captcha_x + self.submit_button['center_x'] * captcha_width
        submit_y = captcha_y + self.submit_button['center_y'] * captcha_height
        submit_w = self.submit_button['width'] * captcha_width
        submit_h = self.submit_button['height'] * captcha_height

        draw.rectangle([
            submit_x - submit_w/2, submit_y - submit_h/2,
            submit_x + submit_w/2, submit_y + submit_h/2
        ], outline="orange", width=3)

        # Save
        verification_path = self.verification_dir / f"{output_name}.jpg"
        verify_img.save(verification_path, quality=95)

    def generate_robot_sample(self, background_path, output_name, scale_factor=1.0):
        """Generate robot checkbox sample"""

        try:
            # Load images
            background = Image.open(background_path).convert('RGB')
            robot_img = Image.open(self.robot_image).convert('RGBA')

            # Scale
            if scale_factor != 1.0:
                orig_w, orig_h = robot_img.size
                new_w = int(orig_w * scale_factor)
                new_h = int(orig_h * scale_factor)
                robot_img = robot_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            bg_width, bg_height = background.size
            robot_width, robot_height = robot_img.size

            # Ensure if it fits
            if robot_width > bg_width or robot_height > bg_height:
                scale = min(bg_width / robot_width, bg_height / robot_height) * 0.9
                new_w = int(robot_width * scale)
                new_h = int(robot_height * scale)
                robot_img = robot_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                robot_width, robot_height = new_w, new_h

            # Random position
            max_x = bg_width - robot_width
            max_y = bg_height - robot_height
            x_offset = random.randint(0, max_x)
            y_offset = random.randint(0, max_y)

            # Composite
            synthetic = background.copy()
            synthetic.paste(robot_img, (x_offset, y_offset), robot_img)

            # Save image
            image_path = self.images_dir / f"{output_name}.jpg"
            synthetic.save(image_path, quality=95)

            # Create annotation
            label_path = self.labels_dir / f"{output_name}.txt"
            with open(label_path, 'w') as f:
                robot_x_center = (x_offset + robot_width / 2) / bg_width
                robot_y_center = (y_offset + robot_height / 2) / bg_height
                robot_w_norm = robot_width / bg_width
                robot_h_norm = robot_height / bg_height

                f.write(f"4 {robot_x_center:.6f} {robot_y_center:.6f} "
                       f"{robot_w_norm:.6f} {robot_h_norm:.6f}\n")

            # Verification
            verify_img = synthetic.copy()
            draw = ImageDraw.Draw(verify_img)
            for i in range(4):
                draw.rectangle([
                    x_offset - i, y_offset - i,
                    x_offset + robot_width + i, y_offset + robot_height + i
                ], outline="purple", fill=None)

            verification_path = self.verification_dir / f"{output_name}.jpg"
            verify_img.save(verification_path, quality=95)

            return True

        except Exception as e:
            print(f"Error generating robot sample: {e}")
            return False

    def generate_negative_sample(self, background_path, output_name):
        """Generate negative sample (pure background, no annotations)"""

        try:
            # Just copy the background
            background = Image.open(background_path).convert('RGB')

            # Save image
            image_path = self.images_dir / f"{output_name}.jpg"
            background.save(image_path, quality=95)

            # Create EMPTY annotation file (crucial for YOLO)
            label_path = self.labels_dir / f"{output_name}.txt"
            label_path.write_text("")  # Empty file

            # Verification (background with "NEGATIVE" text)
            verify_img = background.copy()
            draw = ImageDraw.Draw(verify_img)

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
            except:
                font = ImageFont.load_default()

            # Add watermark
            draw.text((50, 50), "NEGATIVE SAMPLE", fill="red", font=font)
            draw.text((50, 100), "(No CAPTCHA, No Checkbox)", fill="red", font=font)

            verification_path = self.verification_dir / f"{output_name}.jpg"
            verify_img.save(verification_path, quality=95)

            return True

        except Exception as e:
            print(f"Error generating negative sample: {e}")
            return False

    def generate_dataset(self):
        """Generate complete balanced dataset"""

        print("\n" + "=" * 70)
        print("GENERATING BALANCED 5-CLASS DATASET")
        print("=" * 70)

        # Get resources
        background_files = list(self.background_dir.glob("*.jpg")) + \
                          list(self.background_dir.glob("*.png"))
        background_files = [f for f in background_files if not f.name.endswith("Zone.Identifier")]

        captcha_files = list(self.captcha_dir.glob("*.png"))
        captcha_files = [f for f in captcha_files if not f.name.endswith("Zone.Identifier")]

        print(f"\nResources:")
        print(f"   Backgrounds: {len(background_files)}")
        print(f"   CAPTCHAs: {len(captcha_files)}")

        if len(background_files) < 1200:
            print(f"Warning: Only {len(background_files)} backgrounds available")

        if not self.robot_image.exists():
            print(f"Robot image not found: {self.robot_image}")
            return

        # Shuffle backgrounds for randomness
        random.shuffle(background_files)

        # Strategy:
        # - 880 CAPTCHA samples: Each CAPTCHA on 4 unique backgrounds
        # - 200 Robot samples
        # - 120 Negative samples

        backgrounds_per_captcha = 4
        robot_samples = 200
        negative_samples = 120

        print(f"Generation Plan:")
        print(f"   CAPTCHA samples: {len(captcha_files)} × {backgrounds_per_captcha} = {len(captcha_files) * backgrounds_per_captcha}")
        print(f"   Robot samples: {robot_samples}")
        print(f"   Negative samples: {negative_samples}")
        print(f"   TOTAL: {len(captcha_files) * backgrounds_per_captcha + robot_samples + negative_samples}\n")

        # Track which backgrounds are used
        bg_index = 0

        # 1. Generate CAPTCHA samples
        print("=" * 70)
        print("Generating CAPTCHA Samples")
        print("=" * 70)

        for captcha_idx, captcha_file in enumerate(captcha_files, 1):
            print(f"\n[{captcha_idx}/{len(captcha_files)}] Processing: {captcha_file.name}")

            for bg_num in range(backgrounds_per_captcha):
                # Get unique background
                if bg_index >= len(background_files):
                    bg_index = 0
                    random.shuffle(background_files)

                background_file = background_files[bg_index]
                bg_index += 1

                # Random scale
                scale = random.choice(self.scale_factors)

                # Output name
                output_name = f"captcha_{captcha_idx:04d}_bg{bg_num+1:02d}_{captcha_file.stem}_s{scale:.1f}"

                if self.generate_captcha_sample(background_file, captcha_file, output_name, scale):
                    self.stats['captcha_samples'] += 1

                    if self.stats['captcha_samples'] % 100 == 0:
                        print(f"Generated {self.stats['captcha_samples']} CAPTCHA samples...")
                else:
                    self.stats['failed'] += 1

        print(f"CAPTCHA samples complete: {self.stats['captcha_samples']}")

        # 2. Generate Robot samples
        print("\n" + "=" * 70)
        print("Generating Robot Checkbox Samples")
        print("=" * 70)

        for i in range(robot_samples):
            # Random background
            background_file = random.choice(background_files)
            scale = random.choice(self.scale_factors)

            output_name = f"robot_{i+1:04d}_scale{scale:.1f}_bg{hash(background_file.name) % 1000:03d}"

            if self.generate_robot_sample(background_file, output_name, scale):
                self.stats['robot_samples'] += 1

                if (i + 1) % 50 == 0:
                    print(f"Generated {i + 1}/{robot_samples} robot samples...")
            else:
                self.stats['failed'] += 1

        print(f"Robot samples complete: {self.stats['robot_samples']}")

        # 3. Generate Negative samples
        print("\n" + "=" * 70)
        print("Generating Negative Samples (No Objects)")
        print("=" * 70)

        for i in range(negative_samples):
            # Random background
            background_file = random.choice(background_files)

            output_name = f"negative_{i+1:04d}_bg{hash(background_file.name) % 1000:03d}"

            if self.generate_negative_sample(background_file, output_name):
                self.stats['negative_samples'] += 1

                if (i + 1) % 30 == 0:
                    print(f"Generated {i + 1}/{negative_samples} negative samples...")
            else:
                self.stats['failed'] += 1

        print(f"Negative samples complete: {self.stats['negative_samples']}")

        # Create YAML
        self.create_dataset_yaml()

        # Final summary
        print("\n" + "=" * 70)
        print("DATASET GENERATION COMPLETE!")
        print("=" * 70)
        print(f"Final Statistics:")
        print(f"   CAPTCHA samples: {self.stats['captcha_samples']}")
        print(f"   Robot samples: {self.stats['robot_samples']}")
        print(f"   Negative samples: {self.stats['negative_samples']}")
        print(f"   Failed: {self.stats['failed']}")
        print(f"   TOTAL: {self.stats['captcha_samples'] + self.stats['robot_samples'] + self.stats['negative_samples']}")

        print(f"Output: {self.output_dir}")
        print(f"Ready for training:")
        print(f"   yolo train data={self.output_dir}/dataset.yaml model=yolo11n.pt epochs=100\n")

    def create_dataset_yaml(self):
        """Create dataset.yaml"""

        yaml_content = """path: .
train: images
val: images

nc: 5
names:
  0: captcha_area
  1: cell
  2: reload_button
  3: submit_button
  4: robot_checkbox
"""

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"Created dataset.yaml")


def main():
    generator = Balanced5ClassGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main()
