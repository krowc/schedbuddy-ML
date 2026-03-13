"""
Full YOLO pipeline runner — executes all 5 steps in sequence.

Usage:
    python run_pipeline.py --datapath /path/to/dataset --classes classes.txt

Steps:
    1  Split dataset into train / validation folders
    2  Generate data.yaml
    3  Train YOLO model
    4  Run inference on validation set
    5  Crop detected objects (e.g. tables) from results

Skip steps with --skip:
    python run_pipeline.py --datapath /path/to/dataset --skip 1 2

Run a single step:
    python run_pipeline.py --only 4
"""

import argparse
import sys
from pathlib import Path

# ── Import each step ─────────────────────────────────────────────────────────
from train_val_split  import split_dataset      # step 1
from create_yaml      import create_data_yaml   # step 2
from train_model      import train_model        # step 3
from test_model       import run_predict, display_results  # step 4
from crop_predict     import crop_predictions    # step 5


def banner(step: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*60}")


def run_pipeline(
    datapath:    str   = "",
    classes_txt: str   = "classes.txt",
    data_yaml:   str   = "data.yaml",
    model:       str   = "yolo11s.pt",
    epochs:      int   = 60,
    imgsz:       int   = 640,
    train_pct:   float = 0.8,
    patience:    int   = 15,
    conf:        float = 0.25,
    class_id:    int   = 0,
    padding:     int   = 0,
    skip:        list  = (),
    only:        int   = None,
) -> None:

    model_dir = Path(__file__).resolve().parent
    runs_detect_dir = model_dir / "runs" / "detect"
    train_weights = runs_detect_dir / "train" / "weights" / "best.pt"

    def should_run(step: int) -> bool:
        if only is not None:
            return step == only
        return step not in skip

    # ── Step 1: Split ─────────────────────────────────────────────────────────
    if should_run(1):
        banner(1, "Train / Validation Split")
        if not datapath:
            print("[ERROR] --datapath is required for step 1.")
            sys.exit(1)
        split_dataset(datapath, train_pct)

    # ── Step 2: Create YAML ───────────────────────────────────────────────────
    if should_run(2):
        banner(2, "Create data.yaml")
        create_data_yaml(Path(classes_txt), Path(data_yaml))

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    if should_run(3):
        banner(3, "Train YOLO Model")
        train_model(
            data_yaml=data_yaml,
            model_name=model,
            epochs=epochs,
            imgsz=imgsz,
            patience=patience,
        )

    # ── Step 4: Predict ───────────────────────────────────────────────────────
    if should_run(4):
        banner(4, "Run Inference on Validation Set")
        predict_dir = run_predict(
            weights=str(train_weights),
            source="data/validation/images",
            conf=conf,
            project=str(runs_detect_dir),
            name="predict",
        )
        display_results(predict_dir)

    # ── Step 5: Crop ──────────────────────────────────────────────────────────
    if should_run(5):
        banner(5, "Crop Detected Objects")
        crop_predictions(
            image_folder=runs_detect_dir / "predict",
            output_folder=runs_detect_dir / "cropped_tables",
            table_class_id=class_id,
            padding=padding,
        )

    print("\n✅ Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full YOLO detection pipeline.")
    parser.add_argument("--datapath",  default="",           help="Path to raw dataset (images/ + labels/)")
    parser.add_argument("--classes",   default="classes.txt",help="Path to classes.txt")
    parser.add_argument("--yaml",      default="data.yaml",  help="Output YAML path")
    parser.add_argument("--model",     default="yolo11s.pt", help="YOLO model weights")
    parser.add_argument("--epochs",    type=int, default=60)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--patience",  type=int, default=15)
    parser.add_argument("--conf",      type=float, default=0.25)
    parser.add_argument("--class_id",  type=int, default=0,  help="Class index to crop in step 5")
    parser.add_argument("--padding",   type=int, default=0,  help="Pixel padding for crops")
    parser.add_argument("--skip",  nargs="+", type=int, default=[], help="Steps to skip (e.g. --skip 1 2)")
    parser.add_argument("--only",  type=int, default=None,           help="Run only this step number")
    args = parser.parse_args()

    run_pipeline(
        datapath=args.datapath,
        classes_txt=args.classes,
        data_yaml=args.yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        train_pct=args.train_pct,
        patience=args.patience,
        conf=args.conf,
        class_id=args.class_id,
        padding=args.padding,
        skip=args.skip,
        only=args.only,
    )
