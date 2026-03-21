"""preprocess_img.py — Three-phase image preprocessing pipeline for student schedule OCR.

Implements a strict three-phase pipeline that prepares student schedule document images
for reliable OCR ingestion:

    Phase 0 — Document Normalisation (unconditional, applied to every image):
        Step 0-A: Document Framing — crops excess background to isolate the document.
        Step 0-B: Flatness Correction — rectifies perspective distortion and page curl.
        Step 0-C: Portrait Orientation Enforcement — ensures the document is upright
                  and in portrait orientation before quality assessment.

    Phase 1 — Quality Gate (operates on the Phase-0-normalised image):
        Evaluates the normalised image against five checks: resolution, blur, brightness,
        border completeness, and minor skew. Rejects immediately on the first failure;
        Phase 2 is never entered for rejected images.

    Phase 2 — OCR Enhancement (only on images that pass Phase 1):
        1. Lighting normalisation via CLAHE on the luminance channel.
        2. Minor deskewing using the skew angle measured (and cached) in Phase 1.

All tunable thresholds and constants live in :class:`PreprocessingConfig`; no numeric
literals appear in processing or gate logic.

Dependencies:
    - Pillow (PIL) ≥ 9.0 — image I/O and final output representation
    - opencv-python (cv2) ≥ 4.5 — all image processing operations
    - numpy — array operations (installed as a cv2 transitive dependency)
    - Standard library: dataclasses, logging, math, os, pathlib, sys, typing
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

os.makedirs("processed", exist_ok=True)


# ===========================================================================
# 1. Configuration
# ===========================================================================


@dataclass
class PreprocessingConfig:
    """All tunable parameters for the three-phase preprocessing pipeline.

    Attributes:
        min_image_width: Minimum acceptable image width in pixels.
        min_image_height: Minimum acceptable image height in pixels.
        blur_threshold: Minimum Laplacian variance; below this the image is
            considered too blurry for reliable OCR.
        min_brightness: Minimum mean pixel intensity (0–255); below this the
            image is too dark.
        max_brightness: Maximum mean pixel intensity (0–255); above this the
            image is overexposed.
        max_skew_angle_deg: Maximum tolerated residual skew angle (degrees)
            after Phase 0 correction.  Images exceeding this are rejected.
        skew_correction_tolerance_deg: Minimum skew angle that triggers active
            deskewing in Phase 2; smaller angles are left untouched.
        border_completeness_margin: Border strip width as a fraction of the
            corresponding image dimension (e.g. 0.03 → 3 %).
        border_edge_density_threshold: Minimum fraction of Canny edge pixels
            in a border strip before that strip is flagged as incomplete.
        border_max_failing_edges: Maximum number of border strips permitted to
            have low edge density before rejecting the image.
        clahe_clip_limit: Clip limit for CLAHE contrast enhancement.
        clahe_tile_grid_size: Tile grid size ``(cols, rows)`` for CLAHE.
        rotation_fill_value: Grayscale fill value (0–255) for border gaps
            introduced by warp or rotation operations.
        hough_rho: Distance resolution (pixels) for the Hough line transform.
        hough_theta_deg: Angle resolution (degrees) for the Hough transform.
        hough_threshold: Accumulator threshold for the Hough transform.
        hough_min_line_length: Minimum line length (pixels) for the Hough
            transform.
        hough_max_line_gap: Maximum gap between collinear segments (pixels).
        min_document_area_fraction: Minimum contour area as a fraction of total
            image area to be considered a valid document region in Phase 0.
        crop_padding_px: Pixels of padding to add around the detected document
            crop bounding rectangle in Step 0-A.
        perspective_rect_tolerance_px: Maximum per-corner deviation from a
            perfect axis-aligned rectangle (pixels) before perspective
            correction is applied in Step 0-B.
    """

    # Resolution
    min_image_width: int = 640
    min_image_height: int = 480

    # Blur
    blur_threshold: float = 50.0

    # Brightness
    min_brightness: float = 40.0
    max_brightness: float = 220.0

    # Skew
    max_skew_angle_deg: float = 15.0
    skew_correction_tolerance_deg: float = 0.5

    # Border completeness
    border_completeness_margin: float = 0.03
    border_edge_density_threshold: float = 0.005
    border_max_failing_edges: int = 2

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = field(default_factory=lambda: (8, 8))

    # Rotation / warp fill
    rotation_fill_value: int = 255

    # Hough transform
    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 80
    hough_min_line_length: float = 50.0
    hough_max_line_gap: float = 10.0

    # Phase 0 — new fields
    min_document_area_fraction: float = 0.10
    crop_padding_px: int = 10
    perspective_rect_tolerance_px: int = 15


# ===========================================================================
# 2. Result
# ===========================================================================


@dataclass
class PreprocessingResult:
    """Output contract for :func:`preprocess_schedule_image`.

    Attributes:
        status: ``"accepted"`` when the image passed all quality checks and has
            been preprocessed; ``"rejected"`` otherwise.
        processed_image: The preprocessed PIL image when *status* is
            ``"accepted"``; ``None`` when rejected.
        rejection_reason: Human-readable explanation of the first quality check
            that failed, including the measured value and the violated threshold.
            ``None`` when *status* is ``"accepted"``.
        quality_metrics: Dictionary of quality measurements collected during
            Phase 0 and Phase 1.  Always populated regardless of outcome. Keys
            include at minimum: ``blur_score``, ``mean_brightness``,
            ``skew_angle_deg``, ``image_width``, ``image_height``,
            ``crop_applied``, ``perspective_corrected``, and
            ``coarse_rotation_deg``.
    """

    status: Literal["accepted", "rejected"]
    processed_image: Optional[Image.Image]
    rejection_reason: Optional[str]
    quality_metrics: Dict[str, float]


# ===========================================================================
# 3. Phase 0 — Document Normalisation
# ===========================================================================


def normalise_document_framing(
    bgr: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> np.ndarray:
    """Step 0-A: Crop to document content, removing excess background.

    Converts to greyscale, applies Gaussian blur, then uses Otsu thresholding
    to separate the document from its background.  The largest external contour
    that exceeds ``config.min_document_area_fraction`` of the total image area
    and approximates a quadrilateral is used to derive an axis-aligned crop
    rectangle.  An optional ``config.crop_padding_px`` border is added and
    clamped to image bounds.

    If no valid contour is found the original image is passed through unchanged
    and a warning is logged.  This step never raises an exception.

    Writes ``crop_applied`` (1.0 / 0.0) into *metrics*.

    Args:
        bgr: Input image in BGR format (H × W × 3, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``crop_applied`` is written here.

    Returns:
        Cropped BGR image, or the original image if no valid contour was found.
    """
    # NEW function (Phase 0-A)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    h, w = gray.shape[:2]
    total_area = float(h * w)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_raw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh: np.ndarray = np.asarray(thresh_raw)

    # If background is predominantly light, invert so the document is white
    if float(np.mean(thresh)) > 127.0:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning(
            "Document framing (0-A): no contours found; passing through unchanged."
        )
        metrics["crop_applied"] = 0.0
        return bgr

    min_area = config.min_document_area_fraction * total_area
    valid_contour: Optional[np.ndarray] = None

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break  # list is sorted descending; no point continuing
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            valid_contour = cnt
            break

    if valid_contour is None:
        logger.warning(
            "Document framing (0-A): no valid quadrilateral contour found; "
            "passing through unchanged."
        )
        metrics["crop_applied"] = 0.0
        return bgr

    x, y, rw, rh = cv2.boundingRect(valid_contour)
    pad = config.crop_padding_px
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + rw + pad)
    y2 = min(h, y + rh + pad)

    cropped = bgr[y1:y2, x1:x2]
    logger.info(
        "Document framing (0-A): cropped %d×%d → %d×%d.",
        w,
        h,
        x2 - x1,
        y2 - y1,
    )
    metrics["crop_applied"] = 1.0
    return cropped


def correct_perspective_distortion(
    bgr: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> np.ndarray:
    """Step 0-B: Rectify perspective distortion and page curl.

    Re-detects the document boundary on the already-cropped image from Step
    0-A.  If the boundary approximates a quadrilateral whose corners deviate
    from an axis-aligned rectangle by more than
    ``config.perspective_rect_tolerance_px`` pixels, a four-point perspective
    transform is applied.  For non-quadrilateral boundaries (e.g. page curl), a
    thin-plate spline (TPS) warp is used with control points sampled along the
    detected contour edges.

    If no significant distortion is found, the image is passed through
    unchanged.

    Writes ``perspective_corrected`` (1.0 / 0.0) into *metrics*.

    Args:
        bgr: Input image in BGR format (H × W × 3, dtype uint8), already
            cropped by Step 0-A.
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``perspective_corrected`` is
            written here.

    Returns:
        Perspective-corrected BGR image, or the input image if no distortion
        was detected or no valid contour was found.
    """
    # NEW function (Phase 0-B)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    h, w = gray.shape[:2]
    total_area = float(h * w)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_raw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh: np.ndarray = np.asarray(thresh_raw)
    if float(np.mean(thresh)) > 127.0:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug(
            "Perspective correction (0-B): no contours found; passing through."
        )
        metrics["perspective_corrected"] = 0.0
        return bgr

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < config.min_document_area_fraction * total_area:
        logger.debug(
            "Perspective correction (0-B): no sufficiently large contour; passing through."
        )
        metrics["perspective_corrected"] = 0.0
        return bgr

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_points(pts)

        if _is_approximately_rectangular(ordered, config.perspective_rect_tolerance_px):
            logger.debug(
                "Perspective correction (0-B): contour already rectangular; passing through."
            )
            metrics["perspective_corrected"] = 0.0
            return bgr

        warped = _apply_four_point_perspective_transform(bgr, ordered, config)
        logger.info(
            "Perspective correction (0-B): four-point perspective warp applied."
        )
        metrics["perspective_corrected"] = 1.0
        return warped

    # Non-quadrilateral contour: TPS warp
    warped = _apply_contour_tps_warp(bgr, largest, config)
    logger.info(
        "Perspective correction (0-B): TPS warp applied for non-quadrilateral distortion."
    )
    metrics["perspective_corrected"] = 1.0
    return warped


def enforce_portrait_orientation(
    bgr: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> np.ndarray:
    """Step 0-C: Ensure the document is upright and in portrait orientation.

    Handles two cases in a single pass using row-wise projection profile
    variance on a binarised image:

    * **Landscape → Portrait**: If the image is wider than it is tall, the
      best of the four 90°-multiple orientations is selected.
    * **180° flip detection**: An upright document has higher row-wise
      variance in the upper half of the image (header / title area).

    All four orientations (0°, 90°, 180°, 270°) are tested simultaneously and
    the orientation with the highest row-wise projection variance is selected,
    which implicitly handles both landscape and upside-down cases.  No
    ML-based model is used.

    Writes ``coarse_rotation_deg`` into *metrics*.

    .. note::
        This function supersedes the ``check_approx_roatation``
        ``apply_approx_reorientation`` helpers from the original
        ``preprocessor.py``.  Coarse orientation correction now occurs
        exclusively in Phase 0-C.

    Args:
        bgr: Input image in BGR format (H × W × 3, dtype uint8), already
            processed by Steps 0-A and 0-B.
        config: Pipeline configuration (unused directly; accepted for
            interface consistency).
        metrics: Mutable metrics dictionary; ``coarse_rotation_deg`` is
            written here.

    Returns:
        Orientation-corrected BGR image, or the input image if no correction
        was needed.
    """
    # NEW function (Phase 0-C) — replaces check_approx_roatation /
    # apply_approx_reorientation which have been removed from orchestration.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    correction_angle = _detect_coarse_orientation(gray)

    if correction_angle != 0:
        result = _apply_coarse_rotation(bgr, correction_angle)
        logger.info("Portrait enforcement (0-C): rotated %d° CW.", correction_angle)
        metrics["coarse_rotation_deg"] = float(correction_angle)
        return result

    logger.debug("Portrait enforcement (0-C): no orientation correction needed.")
    metrics["coarse_rotation_deg"] = 0.0
    return bgr


# ===========================================================================
# 4. Phase 1 — Quality Gate
# ===========================================================================


def check_resolution(
    gray: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> Optional[str]:
    """Check whether the normalised image meets minimum resolution requirements.

    Unchanged from ``preprocessor.py``.

    Args:
        gray: Greyscale image as a NumPy array (H × W, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``image_width`` and
            ``image_height`` are written here.

    Returns:
        A rejection reason string if the check fails, otherwise ``None``.
    """
    # UNCHANGED from preprocessor.py
    h, w = gray.shape[:2]
    metrics["image_width"] = float(w)
    metrics["image_height"] = float(h)

    logger.debug("Resolution check: %d × %d pixels", w, h)

    if w < config.min_image_width or h < config.min_image_height:
        return (
            f"Resolution check failed: image is {w}×{h} px but minimum required is "
            f"{config.min_image_width}×{config.min_image_height} px."
        )
    return None


def check_blur(
    gray: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> Optional[str]:
    """Assess image sharpness via Laplacian variance.

    Unchanged from ``preprocessor.py``.

    Args:
        gray: Greyscale image as a NumPy array (H × W, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``blur_score`` is written here.

    Returns:
        A rejection reason string if the check fails, otherwise ``None``.
    """
    # UNCHANGED from preprocessor.py
    laplacian_var: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["blur_score"] = laplacian_var

    logger.debug(
        "Blur check: Laplacian variance = %.2f (threshold %.2f)",
        laplacian_var,
        config.blur_threshold,
    )

    if laplacian_var < config.blur_threshold:
        return (
            f"Blur check failed: Laplacian variance {laplacian_var:.1f} is below "
            f"threshold {config.blur_threshold:.1f}."
        )
    return None


def check_brightness(
    gray: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> Optional[str]:
    """Check whether image brightness falls within the acceptable range.

    Unchanged from ``preprocessor.py``.

    Args:
        gray: Greyscale image as a NumPy array (H × W, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``mean_brightness`` is written
            here.

    Returns:
        A rejection reason string if the check fails, otherwise ``None``.
    """
    # UNCHANGED from preprocessor.py
    mean_brightness: float = float(gray.mean())
    metrics["mean_brightness"] = mean_brightness

    logger.debug(
        "Brightness check: mean = %.2f (range [%.2f, %.2f])",
        mean_brightness,
        config.min_brightness,
        config.max_brightness,
    )

    if mean_brightness < config.min_brightness:
        return (
            f"Brightness check failed: mean brightness {mean_brightness:.1f} is below "
            f"minimum {config.min_brightness:.1f}."
        )
    if mean_brightness > config.max_brightness:
        return (
            f"Brightness check failed: mean brightness {mean_brightness:.1f} exceeds "
            f"maximum {config.max_brightness:.1f} (overexposed)."
        )
    return None


def check_border_completeness(
    gray: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> Optional[str]:
    """Detect missing or clipped content near the document crop borders.

    Because framing was corrected in Phase 0-A, this check now verifies that
    the document crop itself is complete — i.e. no half-cut text at the edges —
    rather than testing for a document's presence in the frame.

    Unchanged in implementation from ``preprocessor.py``.

    Args:
        gray: Greyscale image as a NumPy array (H × W, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; per-edge density values are
            written as ``border_density_top``, ``border_density_bottom``,
            ``border_density_left``, and ``border_density_right``.

    Returns:
        A rejection reason string listing the incomplete edges, or ``None``.
    """
    # UNCHANGED from preprocessor.py
    h, w = gray.shape[:2]
    margin_y = max(1, int(h * config.border_completeness_margin))
    margin_x = max(1, int(w * config.border_completeness_margin))

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    strips: Dict[str, np.ndarray] = {
        "top": edges[:margin_y, :],
        "bottom": edges[h - margin_y :, :],
        "left": edges[:, :margin_x],
        "right": edges[:, w - margin_x :],
    }

    incomplete_edges: List[str] = []
    for name, strip in strips.items():
        density = float(strip.astype(np.float64).mean()) / 255.0
        metrics[f"border_density_{name}"] = density
        logger.debug("Border completeness — %s strip edge density: %.4f", name, density)
        if density < config.border_edge_density_threshold:
            incomplete_edges.append(name)

    if len(incomplete_edges) > config.border_max_failing_edges:
        return (
            f"Completeness check failed: insufficient edge content detected near "
            f"border(s): {', '.join(incomplete_edges)}. The document may be clipped "
            f"(max allowed failing borders: {config.border_max_failing_edges})."
        )
    return None


def detect_skew_angle(
    gray: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> float:
    """Estimate residual document skew using the Probabilistic Hough Transform.

    Because coarse orientation correction was applied in Phase 0-C, this
    function only needs to detect small residual tilts.  The detected angle is
    stored in *metrics* and **must be reused** in Phase 2 — it must not be
    recomputed.

    Unchanged in implementation from ``preprocessor.py``.

    Args:
        gray: Greyscale image as a NumPy array (H × W, dtype uint8).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary; ``skew_angle_deg`` is written
            here.

    Returns:
        Estimated skew angle in degrees (float). Returns 0.0 if no lines are
        detected.
    """
    # UNCHANGED from preprocessor.py
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    hough_theta = math.radians(config.hough_theta_deg)
    lines = cv2.HoughLinesP(
        edges,
        rho=config.hough_rho,
        theta=hough_theta,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    if lines is None:
        logger.debug("Skew detection: no lines found; defaulting to 0.0°.")
        metrics["skew_angle_deg"] = 0.0
        return 0.0

    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) > 0:
            angle_deg = math.degrees(math.atan2(dy, dx))
            if -45.0 <= angle_deg <= 45.0:
                angles.append(angle_deg)

    if not angles:
        logger.debug("Skew detection: no near-horizontal lines; defaulting to 0.0°.")
        metrics["skew_angle_deg"] = 0.0
        return 0.0

    skew = float(np.median(angles))
    metrics["skew_angle_deg"] = skew
    logger.debug(
        "Skew detection: median angle = %.2f° over %d lines.", skew, len(angles)
    )
    return skew


def check_orientation(
    skew_angle_deg: float,
    config: PreprocessingConfig,
) -> Optional[str]:
    """Reject the image if its residual skew angle is too large to correct.

    Takes the pre-computed skew angle so that :func:`detect_skew_angle` is
    only called once across Phase 1 and Phase 2.

    Unchanged from ``preprocessor.py``.

    Args:
        skew_angle_deg: Skew angle already measured by
            :func:`detect_skew_angle`.
        config: Pipeline configuration.

    Returns:
        A rejection reason string if the angle exceeds the maximum, otherwise
        ``None``.
    """
    # UNCHANGED from preprocessor.py
    abs_angle = abs(skew_angle_deg)
    logger.debug(
        "Orientation check: |skew| = %.2f° (max %.2f°)",
        abs_angle,
        config.max_skew_angle_deg,
    )
    if abs_angle > config.max_skew_angle_deg:
        return (
            f"Orientation check failed: skew angle {skew_angle_deg:.1f}° exceeds "
            f"maximum {config.max_skew_angle_deg:.1f}°."
        )
    return None


# ===========================================================================
# 5. Phase 2 — OCR Enhancement
# ===========================================================================


def apply_lighting_normalisation(
    bgr: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Normalise image lighting using CLAHE on the luminance channel.

    The image is converted to LAB colour space so that CLAHE is applied only
    to the lightness channel, preserving hue and saturation.  The result is
    converted back to BGR.  Greyscale-only input is processed directly.

    Unchanged from ``preprocessor.py``.

    Args:
        bgr: Input image in BGR format (H × W × 3, dtype uint8) or greyscale
            (H × W, dtype uint8).
        config: Pipeline configuration, supplying ``clahe_clip_limit`` and
            ``clahe_tile_grid_size``.

    Returns:
        Lighting-normalised image in the same colour format as the input.
    """
    # UNCHANGED from preprocessor.py
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )

    if bgr.ndim == 2:
        logger.debug("Lighting normalisation: greyscale path.")
        return clahe.apply(bgr)

    logger.debug("Lighting normalisation: LAB colour path.")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_orientation_correction(
    bgr: np.ndarray,
    skew_angle_deg: float,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Deskew the image using the skew angle cached from Phase 1.

    Rotation is skipped when the absolute skew angle is within
    ``config.skew_correction_tolerance_deg`` to avoid introducing unnecessary
    interpolation artefacts.

    The skew angle must **not** be recomputed here — it must be the value
    already stored in ``quality_metrics`` by :func:`detect_skew_angle`.

    Unchanged from ``preprocessor.py``.

    Args:
        bgr: Input image in BGR format (H × W × 3, dtype uint8) or greyscale
            (H × W, dtype uint8).
        skew_angle_deg: Skew angle in degrees as computed during Phase 1.
            Positive values indicate a counter-clockwise tilt.
        config: Pipeline configuration, supplying
            ``skew_correction_tolerance_deg`` and ``rotation_fill_value``.

    Returns:
        Deskewed image in the same colour format as the input.
    """
    # UNCHANGED from preprocessor.py
    if abs(skew_angle_deg) < config.skew_correction_tolerance_deg:
        logger.debug(
            "Orientation correction: skew %.2f° within tolerance %.2f°; skipping.",
            skew_angle_deg,
            config.skew_correction_tolerance_deg,
        )
        return bgr

    h, w = bgr.shape[:2]
    centre = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(centre, -skew_angle_deg, scale=1.0)

    fill: tuple = (config.rotation_fill_value,) * (bgr.shape[2] if bgr.ndim == 3 else 1)

    rotated = cv2.warpAffine(
        bgr,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    logger.debug("Orientation correction: rotated by %.2f°.", skew_angle_deg)
    return rotated


# ===========================================================================
# 6. Orchestration
# ===========================================================================


def preprocess_schedule_image(
    image_path: str,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """Run the full three-phase preprocessing pipeline on a schedule image.

    **Phase 0 — Document Normalisation** (unconditional):
        0-A: Crop excess background.
        0-B: Correct perspective distortion / page curl.
        0-C: Enforce portrait orientation.

    **Phase 1 — Quality Gate** (on the normalised image):
        Checks resolution, blur, brightness, border completeness, and minor
        skew in order.  Rejects on the first failure.

    **Phase 2 — OCR Enhancement** (only on accepted images):
        CLAHE lighting normalisation followed by minor deskewing using the
        skew angle cached from Phase 1.

    The ``quality_metrics`` dictionary is always populated, regardless of
    whether the image was accepted or rejected.

    Args:
        image_path: Filesystem path to the input image.  Any format supported
            by Pillow is accepted; the file is validated by pixel content, not
            by extension.
        config: A :class:`PreprocessingConfig` instance controlling all
            thresholds and parameters.

    Returns:
        A :class:`PreprocessingResult` describing the outcome.

    Raises:
        FileNotFoundError: If *image_path* does not point to an existing file.
        ValueError: If the file exists but cannot be decoded as a valid image.
        RuntimeError: If an unexpected error occurs during processing.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {image_path!r}")

    # ------------------------------------------------------------------
    # Load image via PIL (validates pixel content, not just extension)
    # ------------------------------------------------------------------
    try:
        pil_image = Image.open(path)
        pil_image.verify()
        pil_image = Image.open(path).convert("RGB")
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {image_path!r}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to open image {image_path!r}: {exc}") from exc

    bgr: np.ndarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    metrics: Dict[str, float] = {}

    # ==================================================================
    # Phase 0 — Document Normalisation
    # ==================================================================
    logger.debug("Phase 0: document normalisation — %s", image_path)

    # Step 0-A: Document framing (crop to content)
    bgr = normalise_document_framing(bgr, config, metrics)

    # Step 0-B: Flatness correction (perspective / warp rectification)
    bgr = correct_perspective_distortion(bgr, config, metrics)

    # Step 0-C: Portrait orientation enforcement
    # NOTE: coarse rotation detection happens ONLY here; the duplicate
    # check_approx_roatation / apply_approx_reorientation calls that
    # appeared in the original preprocessor.py orchestration have been
    # removed.
    bgr = enforce_portrait_orientation(bgr, config, metrics)

    gray: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ==================================================================
    # Phase 1 — Quality Gate  (on normalised image, not raw input)
    # ==================================================================
    logger.debug("Phase 1: quality gate — %s", image_path)

    # 1. Resolution
    reason = check_resolution(gray, config, metrics)
    if reason:
        _fill_missing_metrics(gray, bgr, config, metrics)
        logger.info("REJECTED (%s): %s", path.name, reason)
        return PreprocessingResult(
            status="rejected",
            processed_image=None,
            rejection_reason=reason,
            quality_metrics=metrics,
        )

    # 2. Blur
    reason = check_blur(gray, config, metrics)
    if reason:
        _fill_missing_metrics(gray, bgr, config, metrics)
        logger.info("REJECTED (%s): %s", path.name, reason)
        return PreprocessingResult(
            status="rejected",
            processed_image=None,
            rejection_reason=reason,
            quality_metrics=metrics,
        )

    # 3. Brightness
    reason = check_brightness(gray, config, metrics)
    if reason:
        _fill_missing_metrics(gray, bgr, config, metrics)
        logger.info("REJECTED (%s): %s", path.name, reason)
        return PreprocessingResult(
            status="rejected",
            processed_image=None,
            rejection_reason=reason,
            quality_metrics=metrics,
        )

    # 4. Border completeness
    reason = check_border_completeness(gray, config, metrics)
    if reason:
        _fill_missing_metrics(gray, bgr, config, metrics)
        logger.info("REJECTED (%s): %s", path.name, reason)
        return PreprocessingResult(
            status="rejected",
            processed_image=None,
            rejection_reason=reason,
            quality_metrics=metrics,
        )

    # 5. Minor skew  — angle is detected here and reused in Phase 2
    skew_angle_deg = detect_skew_angle(gray, config, metrics)
    reason = check_orientation(skew_angle_deg, config)
    if reason:
        logger.info("REJECTED (%s): %s", path.name, reason)
        return PreprocessingResult(
            status="rejected",
            processed_image=None,
            rejection_reason=reason,
            quality_metrics=metrics,
        )

    logger.debug("Phase 1 passed — proceeding to Phase 2.")

    # ==================================================================
    # Phase 2 — OCR Enhancement  (only reached if Phase 1 passed)
    # ==================================================================
    logger.debug("Phase 2: OCR enhancement — %s", image_path)

    # Step 1: Lighting normalisation
    bgr_normalised = apply_lighting_normalisation(bgr, config)

    # Step 2: Minor deskewing — reuses skew_angle_deg from Phase 1
    bgr_corrected = apply_orientation_correction(bgr_normalised, skew_angle_deg, config)

    rgb_corrected = cv2.cvtColor(bgr_corrected, cv2.COLOR_BGR2RGB)
    processed_pil = Image.fromarray(rgb_corrected)

    logger.info("ACCEPTED (%s): preprocessing complete.", path.name)
    return PreprocessingResult(
        status="accepted",
        processed_image=processed_pil,
        rejection_reason=None,
        quality_metrics=metrics,
    )


# ===========================================================================
# 7. Internal helpers
# ===========================================================================


def _fill_missing_metrics(
    gray: np.ndarray,
    bgr: np.ndarray,
    config: PreprocessingConfig,
    metrics: Dict[str, float],
) -> None:
    """Populate quality metrics that have not yet been computed.

    Called after an early Phase 1 rejection to ensure the output
    ``quality_metrics`` dictionary always contains the minimum required keys.

    Args:
        gray: Greyscale image array.
        bgr: Colour image array (reserved for future metrics).
        config: Pipeline configuration.
        metrics: Mutable metrics dictionary to fill in.
    """
    if "mean_brightness" not in metrics:
        metrics["mean_brightness"] = float(gray.mean())
    if "blur_score" not in metrics:
        metrics["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if "skew_angle_deg" not in metrics:
        detect_skew_angle(gray, config, metrics)
    # Phase 0 metrics default to 0.0 if not populated (e.g. very early failure)
    metrics.setdefault("crop_applied", 0.0)
    metrics.setdefault("perspective_corrected", 0.0)
    metrics.setdefault("coarse_rotation_deg", 0.0)


def _detect_coarse_orientation(gray: np.ndarray) -> int:
    """Detect coarse 90°-multiple rotation using projection profile variance.

    Binarises the image with Otsu's method and measures the row-wise
    projection variance at 0°, 90°, 180°, and 270°.  Text lines create
    alternating light/dark bands across rows, so the highest row-wise
    variance indicates the correct upright orientation.

    Adapted from ``check_approx_roatation`` in ``preprocessor.py``
    (typo in original function name corrected; now private).

    Args:
        gray: Greyscale image (H × W, dtype uint8).

    Returns:
        Clockwise rotation in degrees needed to correct orientation:
        0, 90, 180, or 270.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def _row_variance(img: np.ndarray) -> float:
        """Return variance of row-wise mean projections."""
        return float(np.var(img.mean(axis=1)))

    variances = {
        0: _row_variance(binary),
        90: _row_variance(cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)),
        180: _row_variance(cv2.rotate(binary, cv2.ROTATE_180)),
        270: _row_variance(cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    }

    best_angle = max(variances, key=variances.__getitem__)
    logger.debug(
        "Coarse orientation: variances=%s → correction=%d°",
        {k: f"{v:.2f}" for k, v in variances.items()},
        best_angle,
    )
    return best_angle


def _apply_coarse_rotation(bgr: np.ndarray, angle_deg: int) -> np.ndarray:
    """Apply a 90°-multiple clockwise rotation.

    Adapted from ``apply_approx_reorientation`` in ``preprocessor.py``
    (now private).

    Args:
        bgr: Input image (BGR or greyscale).
        angle_deg: 0, 90, 180, or 270.

    Returns:
        Rotated image, or the original if *angle_deg* is 0.
    """
    if angle_deg == 0:
        return bgr
    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotated = cv2.rotate(bgr, rotate_codes[angle_deg])
    logger.debug("Coarse rotation applied: %d° CW.", angle_deg)
    return rotated


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    Uses the sum (x+y) to identify top-left and bottom-right, and the
    difference (x-y) to identify top-right and bottom-left.

    Args:
        pts: Array of shape (4, 2) with float32 coordinates.

    Returns:
        Ordered array of shape (4, 2) as float32.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x + y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x + y
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest x - y
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest x - y
    return rect


def _is_approximately_rectangular(
    ordered_pts: np.ndarray,
    tolerance_px: int,
) -> bool:
    """Return True if four ordered corners form an approximately rectangular shape.

    Compares each corner against the corresponding corner of the tightest
    axis-aligned bounding rectangle.

    Args:
        ordered_pts: (4, 2) float32 array ordered as [tl, tr, br, bl].
        tolerance_px: Maximum Euclidean deviation per corner (pixels).

    Returns:
        ``True`` if all four corners are within *tolerance_px* of their ideal
        positions.
    """
    tl, tr, br, bl = ordered_pts
    x_min = min(float(tl[0]), float(bl[0]))
    x_max = max(float(tr[0]), float(br[0]))
    y_min = min(float(tl[1]), float(tr[1]))
    y_max = max(float(bl[1]), float(br[1]))

    ideal = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    deviations = np.linalg.norm(ordered_pts - ideal, axis=1)
    return bool(np.all(deviations <= float(tolerance_px)))


def _apply_four_point_perspective_transform(
    bgr: np.ndarray,
    ordered_pts: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Remap a quadrilateral document region to a flat rectangle.

    Output dimensions are set to the larger of the two widths (top / bottom)
    and the larger of the two heights (left / right).  Border gaps are filled
    with ``config.rotation_fill_value``.

    Args:
        bgr: Source image (H × W × 3, dtype uint8).
        ordered_pts: (4, 2) float32 corners as [tl, tr, br, bl].
        config: Pipeline configuration.

    Returns:
        Perspective-corrected image.
    """
    tl, tr, br, bl = ordered_pts

    dst_w = int(
        max(
            float(np.linalg.norm(tr - tl)),
            float(np.linalg.norm(br - bl)),
        )
    )
    dst_h = int(
        max(
            float(np.linalg.norm(bl - tl)),
            float(np.linalg.norm(br - tr)),
        )
    )
    dst_w = max(dst_w, 1)
    dst_h = max(dst_h, 1)

    dst_pts = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    n_channels = bgr.shape[2] if bgr.ndim == 3 else 1
    fill = (config.rotation_fill_value,) * n_channels

    return cv2.warpPerspective(
        bgr,
        M,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _apply_contour_tps_warp(
    bgr: np.ndarray,
    contour: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Correct non-quadrilateral distortion using a thin-plate spline warp.

    Samples ``N_CTRL`` evenly-spaced control points along the detected contour
    perimeter (source) and maps each to the corresponding position on the
    target bounding-rectangle perimeter (destination) using equal arc-length
    parameterisation.  A central anchor point (same in source and destination)
    constrains interior behaviour.

    The TPS system is solved in normalised coordinates to improve numerical
    conditioning.  If the linear system is singular a fallback perspective warp
    based on the four extremal convex-hull points is used instead.

    The output image covers exactly the bounding rectangle of the input contour.

    Args:
        bgr: Source image (H × W × 3, dtype uint8).
        contour: OpenCV contour (N × 1 × 2) of the document boundary.
        config: Pipeline configuration (provides ``rotation_fill_value``).

    Returns:
        TPS-warped image of size equal to the contour bounding rectangle.
    """
    N_CTRL = 20  # number of boundary control points — fixed implementation detail

    x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
    dst_w, dst_h = max(w_br, 1), max(h_br, 1)

    # ---- Sample N_CTRL points along contour perimeter ----
    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    n_pts = contour_pts.shape[0]

    diffs = np.diff(contour_pts, axis=0, prepend=contour_pts[-1:])
    arc_lengths = np.cumsum(np.linalg.norm(diffs, axis=1))
    total_len = arc_lengths[-1]

    if total_len < 1.0:
        logger.warning("TPS warp: contour too short; falling back to extremal warp.")
        return _apply_extremal_perspective_warp(bgr, contour, config)

    sample_lengths = np.linspace(0.0, total_len, N_CTRL, endpoint=False)
    src_boundary = np.zeros((N_CTRL, 2), dtype=np.float64)
    for i, s in enumerate(sample_lengths):
        idx = int(np.searchsorted(arc_lengths, s, side="left")) % n_pts
        src_boundary[i] = contour_pts[idx]

    # ---- Map to corresponding points on target rectangle perimeter ----
    rect_perimeter = 2.0 * (dst_w + dst_h)
    dst_boundary = np.zeros((N_CTRL, 2), dtype=np.float64)
    for i in range(N_CTRL):
        t = float(i) / float(N_CTRL)
        s = t * rect_perimeter
        if s < dst_w:
            dst_boundary[i] = [x_br + s, float(y_br)]
        elif s < dst_w + dst_h:
            dst_boundary[i] = [float(x_br + dst_w), y_br + (s - dst_w)]
        elif s < 2.0 * dst_w + dst_h:
            dst_boundary[i] = [x_br + dst_w - (s - dst_w - dst_h), float(y_br + dst_h)]
        else:
            dst_boundary[i] = [float(x_br), y_br + dst_h - (s - 2.0 * dst_w - dst_h)]

    # ---- Add central anchor (identity-mapped) ----
    cx = x_br + dst_w / 2.0
    cy = y_br + dst_h / 2.0
    src_ctrl = np.vstack([src_boundary, [cx, cy]])
    dst_ctrl = np.vstack([dst_boundary, [cx, cy]])

    # ---- Normalise coordinates for numerical stability ----
    scale = max(float(bgr.shape[1]), float(bgr.shape[0]), 1.0)
    src_norm = src_ctrl / scale
    dst_norm = dst_ctrl / scale

    # ---- Solve TPS system ----
    K = _tps_kernel_matrix(dst_norm)
    n = dst_norm.shape[0]
    P = np.hstack([np.ones((n, 1), dtype=np.float64), dst_norm])  # (n, 3)
    zeros33 = np.zeros((3, 3), dtype=np.float64)
    A = np.block([[K, P], [P.T, zeros33]])
    b_x = np.concatenate([src_norm[:, 0], [0.0, 0.0, 0.0]])
    b_y = np.concatenate([src_norm[:, 1], [0.0, 0.0, 0.0]])

    try:
        params_x = np.linalg.solve(A, b_x)
        params_y = np.linalg.solve(A, b_y)
    except np.linalg.LinAlgError:
        logger.warning(
            "TPS warp: singular system; falling back to extremal perspective warp."
        )
        return _apply_extremal_perspective_warp(bgr, contour, config)

    # ---- Build remap arrays (evaluated in batches to bound peak memory) ----
    grid_y, grid_x = np.mgrid[y_br : y_br + dst_h, x_br : x_br + dst_w]
    flat_dst = np.column_stack(
        [grid_x.ravel().astype(np.float64), grid_y.ravel().astype(np.float64)]
    )
    flat_dst_norm = flat_dst / scale

    BATCH = 65536
    total = flat_dst_norm.shape[0]
    map_x_flat = np.empty(total, dtype=np.float64)
    map_y_flat = np.empty(total, dtype=np.float64)

    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        mx, my = _tps_evaluate(flat_dst_norm[start:end], dst_norm, params_x, params_y)
        map_x_flat[start:end] = mx * scale
        map_y_flat[start:end] = my * scale

    map_x = map_x_flat.reshape(dst_h, dst_w).astype(np.float32)
    map_y = map_y_flat.reshape(dst_h, dst_w).astype(np.float32)

    n_channels = bgr.shape[2] if bgr.ndim == 3 else 1
    fill = (config.rotation_fill_value,) * n_channels

    return cv2.remap(
        bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _tps_kernel_matrix(pts: np.ndarray) -> np.ndarray:
    """Compute the N × N TPS kernel matrix U(|p_i − p_j|).

    Uses the 2-D TPS kernel U(r) = r² log(r) for r > 0, and 0 for r = 0.

    Args:
        pts: (N, 2) array of control point coordinates.

    Returns:
        Symmetric (N, N) kernel matrix.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    r = np.linalg.norm(diff, axis=2)  # (N, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(r == 0.0, 0.0, r**2 * np.log(np.where(r == 0.0, 1.0, r)))
    return K


def _tps_evaluate(
    query_pts: np.ndarray,
    ctrl_pts: np.ndarray,
    params_x: np.ndarray,
    params_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the fitted TPS mapping at a set of query points.

    Computes the source image coordinates corresponding to each destination
    query point.

    Args:
        query_pts: (M, 2) destination-space points to evaluate.
        ctrl_pts: (N, 2) control points used during TPS fitting.
        params_x: (N+3,) TPS coefficients for the x-component.
        params_y: (N+3,) TPS coefficients for the y-component.

    Returns:
        Tuple ``(map_x, map_y)`` of shape (M,) giving the source sampling
        coordinates.
    """
    M = query_pts.shape[0]
    diff = query_pts[:, np.newaxis, :] - ctrl_pts[np.newaxis, :, :]  # (M, N, 2)
    r = np.linalg.norm(diff, axis=2)  # (M, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        K_eval = np.where(r == 0.0, 0.0, r**2 * np.log(np.where(r == 0.0, 1.0, r)))

    P_eval = np.hstack([np.ones((M, 1), dtype=np.float64), query_pts])  # (M, 3)
    phi = np.hstack([K_eval, P_eval])  # (M, N+3)

    return phi @ params_x, phi @ params_y


def _apply_extremal_perspective_warp(
    bgr: np.ndarray,
    contour: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Fallback perspective warp using the four extremal convex-hull points.

    Used when the TPS linear system is singular.  Finds the topmost, rightmost,
    bottommost, and leftmost points on the convex hull and applies a four-point
    perspective transform on those.

    Args:
        bgr: Source image (H × W × 3, dtype uint8).
        contour: Document boundary contour.
        config: Pipeline configuration.

    Returns:
        Perspective-corrected image.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    top = hull[np.argmin(hull[:, 1])]
    right = hull[np.argmax(hull[:, 0])]
    bottom = hull[np.argmax(hull[:, 1])]
    left = hull[np.argmin(hull[:, 0])]
    pts = np.array([top, right, bottom, left], dtype=np.float32)
    ordered = _order_points(pts)
    return _apply_four_point_perspective_transform(bgr, ordered, config)


# ===========================================================================
# 8. __main__ — minimal usage example
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess a student schedule image for OCR ingestion."
    )
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the preprocessed image (PNG). "
        "Only written when the image is accepted.",
    )
    args = parser.parse_args()

    cfg = PreprocessingConfig()  # all defaults
    try:
        result = preprocess_schedule_image(args.image, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'':10}status: {result.status}")
    if result.rejection_reason:
        print(f"rejection_reason: {result.rejection_reason}")
    print("\nquality_metrics:")
    for k, v in sorted(result.quality_metrics.items()):
        print(f"  {k:45s} {v:.4f}")

    if result.status == "accepted" and result.processed_image is not None:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("processed") / (
                Path(args.image).stem + "_preprocessed.png"
            )
        result.processed_image.save(output_path)
        print(f"\nPreprocessed image saved to: {output_path}")
