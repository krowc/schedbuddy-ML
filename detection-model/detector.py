"""Core table detection model."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from config import COLORS, DETECTION_MODEL, STRUCTURE_MODEL
from models import Detection

logger = logging.getLogger(__name__)


class BorderlessTableDetector:
    """Detect and extract data from borderless tables in images."""

    def __init__(
            self,
            image_path: str | Path,
            output_path: str | Path,
            detection_model: str = DETECTION_MODEL,
            structure_model: str = STRUCTURE_MODEL
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
        """Load and convert image to RGB."""

        try:
            self.image = Image.open(self.image_path).convert("RGB")
            logger.info("Image loaded: %s %s", self.image_path.name, self.image.size)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file: {self.image_path}")

    def _encode(self) -> None:
        """Encode loaded image for the transformer."""

        if self.image is None:
            raise RuntimeError("Call load_image() first.")
        self._encoding = self.processor(self.image, return_tensors="pt")

    def _run_model(self, model_type: str) -> object:
        """Run detection or structure model.
        
        return: raw model outputs
        """

        if self._encoding is None:
            raise RuntimeError("Internal encoding is missing. Call _encode() first.")
        model = self.detection_model if model_type == "detection" else self.structure_model
        with torch.no_grad():
            return model(**self._encoding)

    def _post_process(self, outputs, threshold: float) -> dict:
        """Filter outputs by confidence threshold."""
        if self.image is None:
            raise RuntimeError("Image not loaded. Call load_image() first.")
        w, h = self.image.size
        return self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(h, w)]
        )[0]

    def build_detections(self, results: dict, model_type: str) -> list[Detection]:
        """Convert raw results into Detection classes."""
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
        """Draw bounding boxes on the image."""

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
            logger.info("Plot saved: %s", self.output_path)
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
        """Run the full detection pipeline.
        
        return:
            detections: list of Detection objects
            figure: Matplotlib figure with annotated image
        """

        self.load_image()
        self._encode()
        outputs = self._run_model(model_type)
        results = self._post_process(outputs, threshold)
        detections = self.build_detections(results, model_type)
        figure = self._plot(detections, model_type, show=show_plot, save=save_plot)
        return detections, figure
