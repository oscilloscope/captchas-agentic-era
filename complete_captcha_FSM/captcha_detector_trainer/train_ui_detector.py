#!/usr/bin/env python3
"""
Fine-tune OmniParser YOLO model on CAPTCHA dataset (STABLE VERSION)

Key improvements:
- Lower learning rate (0.001 vs 0.005) for fine-tuning stability
- Cosine LR scheduler for smoother convergence
- Reduced data augmentation (mosaic 0.7, scale 0.3)
- Larger batch size (16 vs 12) for more stable gradients
- Longer warmup (5 epochs vs 3)
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OmniParser YOLO (Stable)")
    parser.add_argument("--dataset", default="balanced_5class_dataset/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--weights", default="weights/icon_detect/icon_detect/model.pt", help="Path to OmniParser YOLO weights")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 for stability, reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (lower for stability)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Fine-tuning OmniParser YOLO (STABLE VERSION)")
    print("="*60)
    print(f"Dataset YAML   : {args.dataset}")
    print(f"Pretrained Wts : {args.weights}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch Size     : {args.batch}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Learning Rate  : {args.lr0}")
    print(f"LR Scheduler   : Cosine annealing")
    print(f"Device         : {args.device.upper()}")
    print("="*60 + "\n")

    # Verify dataset.yaml
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset YAML not found at: {dataset_path}")
        return

    # Load OmniParser YOLOv8-based model
    model = YOLO(args.weights)

    # STABLE training configuration (reduced oscillations)
    results = model.train(
        data=str(dataset_path.absolute()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="captcha_training",
        name="omniparser_stable",
        device=args.device,

        # Optimizer & Learning Rate
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        momentum=0.937,

        # Warmup
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Early Stopping
        patience=30,

        # Augmentation (REDUCED for stability)
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.0,

        # Training Settings
        multi_scale=True,
        rect=False,
        val=True,
        plots=True,
        verbose=True,
        amp=True,

        # Validation
        save=True,
        save_period=-1,
    )

    print("\n" + "="*60)
    print("Stable fine-tuning completed!")
    print("="*60)
    if hasattr(results, "save_dir"):
        best_model = Path(results.save_dir) / "weights" / "best.pt"
        if best_model.exists():
            print(f"Best model: {best_model}")
            print(f"\nCompare results:")
            print(f"   Results plot: {Path(results.save_dir) / 'results.png'}")
            print(f"\nValidation:")
            print(f"   yolo val model={best_model} data={args.dataset}")
            print(f"\nExport:")
            print(f"   yolo export model={best_model} format=onnx")
        else:
            print("WARNING: No best.pt found - check logs for details.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
