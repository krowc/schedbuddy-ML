import argparse
import cv2
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
RUNS_DETECT_DIR = MODEL_DIR / "runs" / "detect"


def crop_predictions(
    image_folder: str | Path | None = None,
    output_folder: str | Path | None = None,
    table_class_id: int = 0,
    padding: int = 0,
) -> None:
    if image_folder is None:
        image_folder = RUNS_DETECT_DIR / "predict"
    else:
        image_folder = Path(image_folder)

    if output_folder is None:
        output_folder = RUNS_DETECT_DIR / "cropped_tables"
    else:
        output_folder = Path(output_folder)

    label_folder = image_folder / "labels"
    padding = max(0, int(padding))

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    if not label_folder.exists():
        raise FileNotFoundError(
            f"Label folder not found: {label_folder}. "
            "Run predict with save_txt=True to create exact box coordinates."
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        path for path in image_folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        label_path = label_folder / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipped (no label file): {label_path}")
            continue

        height, width = image.shape[:2]
        with label_path.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        table_count = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            if cls_id != table_class_id:
                continue

            x_center_n, y_center_n, box_width_n, box_height_n = map(float, parts[1:5])

            box_width = box_width_n * width
            box_height = box_height_n * height
            x_center = x_center_n * width
            y_center = y_center_n * height

            x1 = int(round(x_center - box_width / 2))
            y1 = int(round(y_center - box_height / 2))
            x2 = int(round(x_center + box_width / 2))
            y2 = int(round(y_center + box_height / 2))

            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            x1 = max(0, min(x1, image.shape[1]))
            x2 = max(0, min(x2, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            y2 = max(0, min(y2, image.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            table_count += 1
            output_path = output_folder / f"{image_path.stem}_table_{table_count}{image_path.suffix}"
            cv2.imwrite(str(output_path), cropped)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop predicted table detections from YOLO output.")
    parser.add_argument("--predict_dir", default=str(RUNS_DETECT_DIR / "predict"), help="YOLO predict folder")
    parser.add_argument("--output_dir", default=str(RUNS_DETECT_DIR / "cropped_tables"), help="Output crops folder")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID to crop")
    parser.add_argument("--padding", type=int, default=0, help="Padding in pixels around each crop")
    args = parser.parse_args()

    crop_predictions(args.predict_dir, args.output_dir, args.class_id, args.padding)