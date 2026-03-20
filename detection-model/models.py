"""Data classes for table detection and extraction."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    label_id: int
    label: str
    score: float
    bbox: list[float]  # [xmin, ymin, xmax, ymax]
    bbox_xywh: list[float]  # [x, y, w, h]


@dataclass
class CellRecord:
    row: int
    column: int
    bbox: Optional[list[int]]
    text: str


@dataclass
class TableData:
    headers: list[str]
    rows: list[dict[str, str]]
    cells: list[dict]
