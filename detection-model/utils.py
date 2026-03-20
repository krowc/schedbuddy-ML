"""Utility functions for image processing and OCR."""

from __future__ import annotations
import re
from typing import Optional
from PIL import Image
import pytesseract

from config import TESSERACT_CONFIG


def bbox_intersection(
        box_a: list[float], box_b: list[float]
    ) -> Optional[list[int]]:
    """Return integer intersection of two [xmin, ymin, xmax, ymax] boxes or None."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(c)) for c in (x1, y1, x2, y2)]


def ocr_crop(image: Image.Image, box: list[int]) -> str:
    """Crop image to box and run Tesseract OCR."""

    crop = image.crop(tuple(box))
    raw_text = pytesseract.image_to_string(crop, config=TESSERACT_CONFIG)
    # Flatten OCR output so JSON values do not contain embedded newlines.
    return re.sub(r"\s+", " ", raw_text).strip()
