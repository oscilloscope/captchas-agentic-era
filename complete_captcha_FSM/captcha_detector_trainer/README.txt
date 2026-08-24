1. Download background screenshots from kaggle - https://www.kaggle.com/datasets/pooriamst/website-screenshots
2. Place screenshots in 'screenshots_unique_many/' directory
3. Ensure CAPTCHA images are in 'captcha_images/' directory
4. Verify 'I_am_not_a_robot.png' exists


Run the generation script: python generate_balanced_5class_dataset.py to generate syntetic samples.

After samples for training are generated, run train_ui_detector.py script to train YOLO CAPTCHA UI Detector

See balanced_5class_dataset folder for 10 annotated samples, demonstrating the annotation format.