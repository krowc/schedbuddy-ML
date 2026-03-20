"""Entry point for table detection and extraction pipeline."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import asdict

from detector import BorderlessTableDetector
from extraction import extract_table
from config import TESSERACT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    IMAGE_PATH = base_dir / "87ef5a9f-25_table_1.jpg"
    OUTPUT_IMAGE = base_dir / "output.png"
    DETECTIONS_JSON = base_dir / "detections.json"
    TABLE_JSON = base_dir / "extracted_table.json"

    detector = BorderlessTableDetector(IMAGE_PATH, OUTPUT_IMAGE)

    # Run structure recognition
    detections, _ = detector.process(
        model_type="structure", threshold=0.9, show_plot=True, save_plot=True
    )

    # Save raw detections
    rows_det = [asdict(d) for d in detections if "row" in d.label.lower()]
    cols_det = [asdict(d) for d in detections if "column" in d.label.lower()]
    Path(DETECTIONS_JSON).write_text(
        json.dumps({"rows": rows_det, "columns": cols_det}, indent=2), encoding="utf-8"
    )
    logger.info("Detections saved as: %s (rows = %d, columns = %d)", DETECTIONS_JSON, len(rows_det), len(cols_det))

    # Extract table data via OCR
    table_data = extract_table(detector, detections)
    Path(TABLE_JSON).write_text(
        json.dumps(
            {
                "image file:": str(IMAGE_PATH),
                "ocr configuration:": TESSERACT_CONFIG,
                "headers": table_data.headers,
                "rows": table_data.rows,
                "cells": table_data.cells
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )
    logger.info("Table JSON saved: %s", TABLE_JSON)

    # Preview first two rows
    print("\nFirst two rows:")
    for row in table_data.rows[:2]:
        print(row)


if __name__ == "__main__":
    main()
