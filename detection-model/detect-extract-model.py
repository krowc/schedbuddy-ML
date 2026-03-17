"""
Borderless Table Detection & OCR Extraction
Uses Microsoft Table Transformer for detection + Tesseract for OCR.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import pytesseract

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TESSERACT_CONFIG = "--oem 3 --psm 6"
COLORS =[
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
]

# Data classes
@dataclass
class Detection:
    label_id: int
    label: str
    score: float
    bbox: list[float] # [xmin, ymin, xmax, ymax]
    bbox_xywh: list[float] # [x, y, w, h]

@dataclass
class CellRecord:
    row: int
    columm: int
    bbox: Optional[list[int]]
    text: str

@dataclass
class TableData: 
    headers: list[str]
    rows: list[dict[str, str]]
    cells: list[dict]

# Helpers
def bbox_intersection(
        box_a: list[float], box_b: list[float]
    ) -> Optional[list[int]]:

    # return: integer intersection of two [xmin, ymin, xmax, ymax] boxes or None

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(c)) for c in (x1, y1, x2, y2)]


def ocr_crop(image: Image.Image, box: list[int]) -> str:
    # Crop image to box and run Tesseract OCR

    crop = image.crop(tuple(box))
    return pytesseract.image_to_string(crop, config=TESSERACT_CONFIG)

# Main class
class BorderlessTableDetector:
    # Detect and extract data from the bordeless tables in images

    _DETECTION_MODEL = "microsoft/table-transformer-detection"
    _STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"

    def __init__(
            self,
            image_path: str | Path,
            output_path: str | Path,
            detection_model: str = _DETECTION_MODEL,
            structure_model: str = _STRUCTURE_MODEL
    ) -> None: 
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)

        logger.info("Loading models...")
        self.processor = DetrImageProcessor()
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(detection_model)
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model)
        logger.info("Models loaded.")

        self.image: Optional[Image.Image] = None
        self._encoding: Optional[dict] = None
    
    # Pipeline
    def load_image(self) -> None:
        # Load and convert image to RGB

        try: 
            self.image = Image.open(self.image_path).convert("RGB")
            logger.info("Image loaded: %s %s", self.image_path.name, self.image.size)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file: {self.image_path}")
        
    def _encode(self) -> None:
        # Encode loaded image for the transformer

        if self.image is None:
            raise RuntimeError("Call load_image() first.")
        self._encoding = self.processor(self.image, return_tensors="pt")

    def _run_model(self, model_type: str) -> object:
        # Run detection or structure model. 
        # return: raw model outputs

        if self._encoding is None:
            raise RuntimeError("Internal encoding is missing. Call _encode() first.")
        model = self.detection_model if model_type == "detection" else self.structure_model
        with torch.no_grad():
            return model(**self._encoding)
        
    def _post_process(self, outputs, threshold: float) -> dict:
        # Filter outputs by confidence threshold
        if self.image is None:
            raise RuntimeError("Image not loaded. Call load_image() first.")
        w, h = self.image.size
        return self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(h, w)]
        )[0]
    
    def build_detections(self, results: dict, model_type: str) -> list[Detection]:
        # Convert raw results into Detection classes
        model = self.detection_model if model_type == "detection" else self.structure_model
        detections = []
        for score, label, (xmin, ymin, xmax, ymax) in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
        ):
            detections.append(Detection(
                label_id=int(label),
                label=model.config.id2label.get(label, "Unknown"),
                score=float(score),
                bbox=[xmin, ymin, xmax, ymax],
                bbox_xywh=[xmin, ymin, xmax - xmin, ymax - ymin]
            ))
        return detections
    
    def _plot(
            self,
            detections: list[Detection],
            model_type: str,
            show: bool,
            save: bool
    ) -> plt.Figure:
        # Draw bounding boxes on the image

        fig, ax = plt.subplots(1, figsize=(16, 10))
        ax.imshow(self.image)
        ax.axis("off")

        cycled = (COLORS * (len(detections) // len(COLORS) + 1))[:len(detections)]
        for det, color in zip(detections, cycled):
            xmin, ymin, xmax, ymax = det.bbox
            ax.add_patch(
                mpatches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    fill=False, color=color, linewidth=1
                )
            )

            ax.text(xmin, ymin, f"{det.label}: {det.score:.2f}",
                    fontsize=10, bbox=dict(facecolor="yellow", alpha=0.5))
            
        fig.tight_layout()
        if save:
            fig.savefig(self.output_path, dpi=600)
            logger.info("Plot save: %s", self.output_path)
        if show:
            plt.show()
        return fig
    
    # Public API
    def process(
            self,
            model_type: str = "detection",
            threshold: float = 0.7,
            show_plot: bool = True,
            save_plot: bool = True
    ) -> tuple[list[Detection], plt.Figure]:
        # Run the full detection pipeline
        # return:
        #   detections: list of Detection objects
        #   figure: Matplotlib figure with annotated image

        self.load_image()
        self._encode()
        outputs = self._run_model(model_type)
        results = self._post_process(outputs, threshold)
        detections = self.build_detections(results, model_type)
        figure = self._plot(detections, model_type, show=show_plot, save=save_plot)
        return detections, figure
    
    # OCR / Extraction
    def extract_table(self, detections: list[Detection]) -> TableData:
        # Extract strcutured table data from structure-model detections via OCR
        # Args: detections: output of process() with model_type="structure"
        # Returns: TableData with headers, rows, and individual cell records

        if self.image is None:
            raise RuntimeError("Call process() before extract_table().")
        
        rows = sorted(
            [d for d in detections if "row" in d.label.lower()],
            key=lambda d: d.bbox[1] # sort by ymin
        )

        columns = sorted(
            [d for d in detections if d.label.lower() == "table column"],
            key=lambda d: d.bbox[0] # sort by xmin
        )

        header_dets = [d for d in detections if "header" in d.label.lower()]

        n_cols = len(columns)
        header_names = [f"col{i + 1}" for i in range(n_cols)]

        # Build cell grid
        cell_records: list[CellRecord] = []
        rows_as_dicts: list[dict] = []

        for r_idx, row in enumerate(rows, 1):
            row_dict: dict[str, str] = {}
            for c_idx, col in enumerate(columns, 1):
                box = bbox_intersection(row.bbox, col.bbox)
                text = ocr_crop(self.image, box) if box else ""
                col_name = header_names[c_idx - 1]
                row_dict[col_name] = text
                cell_records.append(CellRecord(row=r_idx, columm=c_idx, bbox=box, text=text))
            rows_as_dicts.append(row_dict)
        
        # Attempt to rename columns from detected headeer region
        if header_dets:
            header_box = header_dets[0].bbox
            extracted = []
            for col in columns:
                header_cell = bbox_intersection(header_box, col.bbox)
                extracted.append(ocr_crop(self.image, header_cell).strip() 
                                 if header_cell else "")
            
            if any(extracted):
                clean = [t or f"col_{i + 1}" for i, t in enumerate(extracted)]
                rows_as_dicts = [
                    {clean[i]: row[header_names[i]] for i in range(n_cols)}
                    for row in rows_as_dicts
                ]
                header_names = clean
        
        logger.info("Extracted %d rows × %d columns", len(rows), n_cols)
        return TableData(
            headers=header_names,
            rows=rows_as_dicts,
            cells=[asdict(c) for c in cell_records]
        )
    
# Entry point
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    IMAGE_PATH = base_dir / "3eec9544-151_table_1.jpg"
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
    table_data = detector.extract_table(detections)
    Path(TABLE_JSON).write_text(
        json.dumps(
            {"headers": table_data.headers, "rows": table_data.rows, "cells": table_data.cells},
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

