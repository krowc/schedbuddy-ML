"""Table data extraction and OCR workflow."""

from __future__ import annotations
import logging
from dataclasses import asdict
import re

from models import Detection, CellRecord, TableData
from utils import bbox_intersection, ocr_crop

logger = logging.getLogger(__name__)

def parse_units_cell(text: str) -> dict[str, float]:
    """
    Parse OCR text for units into numeric Credit/Lec/Lab values.

    Expected format is similar to "3.0 2.0 1.0". Comma decimals from noisy OCR
    (for example "3,0 2,0 1,0") are normalized to periods.

    Returns:
        Dictionary with keys "Credit", "Lec", and "Lab" as floats.
        Missing or invalid values default to 0.0.

    TODO: Parse extracted text and use it as subheader names.
    FIXME: Current implementation uses hardcoded subcolumns. 
    """
    sub_columns = ("Credit", "Lec", "Lab")
    units = re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))
    default = dict.fromkeys(sub_columns, 0.0)
    default.update(zip(sub_columns, map(float, units)))
    return default

def expand_multiline_rows(row: dict[str, str]) -> list[dict[str, str]]:
    """
    Given rows with multiline values separated by newlinein some columns, 
    expland into per-schedule-entry dicts. If a column has fewer lines than
    the max, the last known value is carried forward.
    
    FIXME: Current implementation repeats the entire row data changing only 
    the values on multiline columns.

    Current output:
        {
            "Code": "code",
            "Subject": "subject",
            "Units\nCredit Lee Lab": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "class",
            "Days": "TTh",
            "Time": "04:00 PM - 07:00 PM",
            "Room": "CS-02-104",
            "Faculty": "faculty"
            },
            {
            "Code": "code",
            "Subject": "subject",
            "Units\nCredit Lee Lab": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "class",
            "Days": "T",
            "Time": "10:00 AM - 12:00 PM",
            "Room": "CS-02-104",
            "Faculty": "faculty"
        }
    Goal output:
        {
            "Code": "code",
            "Subject": "subject",
            "Units": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "BSCS-3A",
            "Schedules": [
                {
                    "Days": "day",
                    "Time": "01:00 PM - 04:00 PM",
                    "Room": "CS-02-201 CS-02-105",
                    "Faculty": "class"
                },
                {
                    "Days": "day",
                    "Time": "01:00 PM - 04:00 PM",
                    "Room": "CS-02-201 CS-02-105",
                    "Faculty": "class"
                }
            ]
        }
    """

    split_rows = {
        col: [line.strip() for line in val.split("\n") if line.strip()]
        for col, val in row.items()
    }

    max_lines = max(len(lines) for lines in split_rows.values())

    entries = []
    last_entry = {} # carry forward the last non-empty value per column

    for i in range(max_lines):
        entry = {}
        for col, lines in split_rows.items():
            if i < len(lines):
                entry[col] = lines[i]
                last_entry[col] = lines[i]
            else: 
                entry[col] = last_entry.get(col, "")
        
        entries.append(entry)

    return entries

def extract_table(detector, detections: list[Detection]) -> TableData:
    """Extract structured table data from structure-model detections via OCR.
    
    Args:
        detector: BorderlessTableDetector instance with loaded image
        detections: output of process() with model_type="structure"
    
    Returns:
        TableData with headers, rows, and individual cell records

    FIXME: "Units" is hardcoded. Improve column checking for passing units cell text
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

            if "col3" in col_name:
                row_dict[col_name] = parse_units_cell(text)
            else:
                row_dict[col_name] = text

            cell_records.append(CellRecord(row=r_idx, column=c_idx, bbox=box, text=text))
        
        units = row_dict.pop("col3")

        # Expand multiline columns with carry-forward
        expanded = expand_multiline_rows(row_dict)

        for entry in expanded:
            entry["col3"] = units

        rows_as_dicts.extend(expanded)

    # Temporarily mode  header naming after the data extraction  as too many hardcoding is expected. 
    # TODO: Find a way to parse Unit/Credit/Lec/Lab for sub-columning
    if header_dets:
        header_box = header_dets[0].bbox
        extracted = []
        for col in columns:
            header_cell = bbox_intersection(header_box, col.bbox)
            extracted.append(ocr_crop(detector.image, header_cell).strip())

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
