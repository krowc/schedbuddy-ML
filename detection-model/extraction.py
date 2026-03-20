"""Table data extraction and OCR workflow."""

from __future__ import annotations
import logging
from dataclasses import asdict
from PIL import Image

from models import Detection, CellRecord, TableData
from utils import bbox_intersection, ocr_crop
from config import TESSERACT_CONFIG

logger = logging.getLogger(__name__)


def extract_table(detector, detections: list[Detection]) -> TableData:
    """Extract structured table data from structure-model detections via OCR.
    
    Args:
        detector: BorderlessTableDetector instance with loaded image
        detections: output of process() with model_type="structure"
    
    Returns:
        TableData with headers, rows, and individual cell records
    """

    if detector.image is None:
        raise RuntimeError("Call process() before extract_table().")

    rows = sorted(
        [d for d in detections if "row" in d.label.lower()],
        key=lambda d: d.bbox[1]  # sort by ymin
    )

    columns = sorted(
        [d for d in detections if d.label.lower() == "table column"],
        key=lambda d: d.bbox[0]  # sort by xmin
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
            text = ocr_crop(detector.image, box) if box else ""
            col_name = header_names[c_idx - 1]
            row_dict[col_name] = text
            cell_records.append(CellRecord(row=r_idx, column=c_idx, bbox=box, text=text))
        rows_as_dicts.append(row_dict)

    # Attempt to rename columns from detected header region
    if header_dets:
        header_box = header_dets[0].bbox
        extracted = []
        for col in columns:
            header_cell = bbox_intersection(header_box, col.bbox)
            extracted.append(ocr_crop(detector.image, header_cell)
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
