"""
Step 3: Train a YOLO model.

Usage:
    python 3_train_model.py
    python 3_train_model.py --data data.yaml --model yolo11s.pt --epochs 60 --imgsz 640
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def get_device() -> int | str:
    """Auto-select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = 0
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {name}")
    else:
        device = "cpu"
        print("[INFO] No GPU found — training on CPU (this will be slow for large datasets).")
    return device


def train_model(
    data_yaml: str = "data.yaml",
    model_name: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640,
    patience: int = 15,
    batch: int = -1,  # -1 = auto-batch
    project: str | None = None,
    name: str = "train",
) -> None:
    """
    Train a YOLO model.

    Args:
        data_yaml:  Path to the data configuration YAML file.
        model_name: YOLO model weights to start from (e.g. yolo11s.pt, yolov8m.pt).
        epochs:     Maximum training epochs.
        imgsz:      Input image size.
        patience:   Early-stopping patience (epochs without improvement).
        batch:      Batch size. -1 lets Ultralytics auto-select based on VRAM.
        project:    Base output folder for YOLO runs. Defaults to model/runs/detect.
        name:       Run name folder inside project (default: train).
    """
    print("=" * 50)
    print("YOLO Training")
    print(f"  Data YAML  : {data_yaml}")
    print(f"  Model      : {model_name}")
    print(f"  Epochs     : {epochs}  (early stop after {patience} with no gain)")
    print(f"  Image size : {imgsz}x{imgsz}")
    print("=" * 50)

    device = get_device()
    model = YOLO(model_name)

    if project is None:
        project = str(Path(__file__).resolve().parent / "runs" / "detect")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,   # stop early if validation stops improving
        device=device,
        project=project,
        name=name,
        exist_ok=True,       # don't error if run folder already exists
    )

    print("\nTraining complete.")
    print(f"Best weights saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("--data",    default="data.yaml",  help="Path to data.yaml")
    parser.add_argument("--model",   default="yolo11s.pt", help="YOLO model to use")
    parser.add_argument("--epochs",  type=int, default=60, help="Max training epochs")
    parser.add_argument("--imgsz",   type=int, default=640,help="Input image size")
    parser.add_argument("--patience",type=int, default=15, help="Early stopping patience")
    parser.add_argument("--batch",   type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument("--project", default=None, help="YOLO output project dir (default: model/runs/detect)")
    parser.add_argument("--name",    default="train", help="Run name folder (default: train)")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
