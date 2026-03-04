"""
Step 4: Run inference on validation images and display results.

Usage:
    python 4_test_model.py
    python 4_test_model.py --weights runs/detect/train/weights/best.pt --source data/validation/images
    python 4_test_model.py --conf 0.4 --show 20
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_predict(
    weights: str | None = None,
    source:  str = "data/validation/images",
    conf:    float = 0.25,
    project: str | None = None,
    name: str = "predict",
    exist_ok: bool = True,
) -> Path:
    """Run YOLO prediction and save results with bounding boxes + label txt files."""
    model_dir = Path(__file__).resolve().parent
    if project is None:
        project = str(model_dir / "runs" / "detect")
    if weights is None:
        weights = str(Path(project) / "train" / "weights" / "best.pt")

    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Run step 3 (train_model.py) first."
        )

    print(f"Running inference with {weights} on {source}")
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,       # save annotated images
        save_txt=True,   # save label txt files (needed for step 5)
        project=project,
        name=name,
        exist_ok=exist_ok,
    )

    predict_dir = Path(results[0].save_dir)
    print(f"Results saved to: {predict_dir}")
    return predict_dir


def display_results(predict_dir: Path, max_images: int = 10) -> None:
    """Display predicted images inline (works in Jupyter / IPython)."""
    try:
        from IPython.display import Image, display
        in_ipython = True
    except ImportError:
        in_ipython = False

    images = sorted(predict_dir.glob("*.jpg"))[:max_images]
    if not images:
        images = sorted(predict_dir.glob("*.png"))[:max_images]

    if not images:
        print(f"[INFO] No result images found in {predict_dir}")
        return

    print(f"\nShowing {len(images)} prediction result(s):")
    for img_path in images:
        print(f"  {img_path}")
        if in_ipython:
            display(Image(filename=str(img_path), height=400))


if __name__ == "__main__":
    default_project = Path(__file__).resolve().parent / "runs" / "detect"
    default_weights = default_project / "train" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="Run YOLO inference on validation images.")
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument("--source",  default="data/validation/images")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show",    type=int,   default=10,   help="Max images to display")
    parser.add_argument("--project", default=str(default_project), help="Prediction output project folder")
    parser.add_argument("--name",    default="predict", help="Prediction run folder name")
    args = parser.parse_args()

    predict_dir = run_predict(args.weights, args.source, args.conf, args.project, args.name)
    display_results(predict_dir, args.show)
