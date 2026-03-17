
from PIL import Image, UnidentifiedImageError
import json
import torch
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

import pytesseract
from pathlib import Path

class BorderlessTableDetection:
    def __init__(self, image_path, output_path, detection_model="microsoft/table-transformer-detection",
                 structure_model="microsoft/table-transformer-structure-recognition"):
        """
        Initializes the BorderlessTableDetection class with models, feature extractor, and image paths.
        
        @param image_path: The path of the image to process.
        @param output_path: The path to save the output image.
        @param detection_model: Pre-trained model to detect tables.
        @param structure_model: Pre-trained model to recognize the structure of tables.
        """
        self.image_path = image_path
        self.output_path = output_path
        try:
            self.feature_extractor = DetrImageProcessor()
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(detection_model)
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model)
        except Exception as e:
            print(f"Error loading models or feature extractor: {e}")
            raise
        
        self.image = None
        self.encoding = None

    def load_image(self):
        """
        Loads the image from the given path and resizes it.

        @return: None
        """
        try:
            self.image = Image.open(self.image_path).convert("RGB")
            width, height = self.image.size
            # self.image = self.image.resize((int(width * 0.5), int(height * 0.5)))
        except FileNotFoundError:
            print(f"Error: The file {self.image_path} was not found.")
            raise
        except UnidentifiedImageError:
            print(f"Error: Unable to identify the image file at {self.image_path}.")
            raise
        except Exception as e:
            print(f"Unexpected error while loading image: {e}")
            raise
    
    def extract_features(self):
        """
        Extracts the features of the image using the DetrFeatureExtractor.

        @return: None
        """
        try:
            self.encoding = self.feature_extractor(self.image, return_tensors="pt")
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise
    
    def detect_tables(self, model_type="detection"):
        """
        Detects tables or table structures from the image using the appropriate model.

        @param model_type: Specifies whether to use the detection model or structure model. Defaults to "detection".
        @return: The output predictions from the model.
        """
        model = self.detection_model if model_type == "detection" else self.structure_model
        try:
            with torch.no_grad():
                outputs = model(**self.encoding)
            return outputs
        except Exception as e:
            print(f"Error during table detection: {e}")
            raise

    def post_process(self, outputs, threshold=0.7):
        """
        Post-processes the model outputs to apply a confidence threshold.

        @param outputs: The raw outputs from the model.
        @param threshold: The confidence threshold for filtering results. Defaults to 0.7.
        @return: Processed results containing scores, labels, and bounding boxes.
        """
        try:
            width, height = self.image.size
            results = self.feature_extractor.post_process_object_detection(outputs, threshold=threshold, 
                                                                          target_sizes=[(height, width)])[0]
            return results
        except Exception as e:
            print(f"Error during post-processing: {e}")
            raise

    def plot_results(self, results, scores, labels, boxes, model_type="detection", show_plot=True, save_plot=True):
        """
        Plots the detected tables on the image and saves the result.

        @param results: The processed results from the model, including scores, labels, and bounding boxes.
        @param scores: Confidence scores for the detected tables.
        @param labels: Labels for the detected tables.
        @param boxes: Bounding boxes for the detected tables.
        @param model_type: Specifies which model's labels are being used (detection or structure). Defaults to "detection".
        @param show_plot: Whether to render the matplotlib window.
        @param save_plot: Whether to save the plotted image to output_path.
        @return: Dictionary containing matplotlib objects and normalized detections.
        """
        try:
            COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                      [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
            
            fig = plt.figure(figsize=(16, 10))
            plt.imshow(self.image)
            ax = plt.gca()
            colors = COLORS * 100
            detections = []

            model = self.detection_model if model_type == "detection" else self.structure_model

            for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=1))
                try:
                    text = f'{model.config.id2label[label]}: {score:0.2f}'
                except KeyError:
                    text = f'Unknown Label: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
                detections.append({
                    "label_id": int(label),
                    "label": model.config.id2label.get(label, "Unknown Label"),
                    "score": float(score),
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "bbox_xywh": [
                        float(xmin),
                        float(ymin),
                        float(xmax - xmin),
                        float(ymax - ymin),
                    ],
                })
            
            plt.axis('off')
            if save_plot:
                plt.savefig(self.output_path, dpi=1000)
            if show_plot:
                plt.show()

            return {
                "figure": fig,
                "axis": ax,
                "detections": detections,
            }
        except Exception as e:
            print(f"Error during plotting or saving the results: {e}")
            raise

    def process_image(self, model_type="detection", threshold=0.7, show_plot=True, save_plot=True):
        """
        The main function that loads the image, processes it, detects tables, and plots the results.

        @param model_type: Specifies whether to use the detection model or structure model. Defaults to "detection".
        @param threshold: The confidence threshold for filtering results. Defaults to 0.7.
        @param show_plot: Whether to render the matplotlib window.
        @param save_plot: Whether to save the plotted image to output_path.
        @return: Dictionary with raw model outputs, post-processed results, and plot payload.
        """
        try:
            self.load_image()
            self.extract_features()
            outputs = self.detect_tables(model_type)
            results = self.post_process(outputs, threshold=threshold)
            plot_payload = self.plot_results(
                results,
                results['scores'],
                results['labels'],
                results['boxes'],
                model_type,
                show_plot=show_plot,
                save_plot=save_plot,
            )
            return {
                "outputs": outputs,
                "results": results,
                "plot": plot_payload,
            }
        except Exception as e:
            print(f"Error during image processing: {e}")
            raise

image_path = "3eec9544-151_table_1.jpg"
detector = BorderlessTableDetection(image_path, "output.png")
pipeline_output = detector.process_image(model_type="structure", threshold=0.9, show_plot=True, save_plot=True)

# ------------------------------------
# Extract structure
# ------------------------------------
detections = pipeline_output["plot"]["detections"]
detections_output_path = "detections.json"

# Explicit arrays
rows = [item for item in detections if "row" in item["label"].lower()]
columns = [item for item in detections if "column" in item["label"].lower()]

with open(detections_output_path, "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "columns": columns}, f, indent=2)

print(f"Detected rows: {len(rows)}")
print(f"Detected columns: {len(columns)}")

print(f"Detections saved to: {detections_output_path}")

# ------------------------------------
# Extract data
# ------------------------------------
try:
    import pytesseract
    OCR_READY = True
except ImportError:
    OCR_READY = False

# Keep only true table columns (exclude column-header box from column list)
table_rows = sorted(rows, key=lambda item: item["bbox"][1])
table_columns = sorted(
    [item for item in columns if item["label"].lower() == "table column"],
    key=lambda item: item["bbox"][0],
)
column_headers = [item for item in columns if "header" in item["label"].lower()]

def bbox_intersection(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def ocr_image(pil_image):
    if not OCR_READY:
        return ""
    return pytesseract.image_to_string(pil_image, config="--oem 3 --psm 6")

# Build row-wise table text while preserving column index
header_names = [f"col_{idx + 1}" for idx in range(len(table_columns))]
rows_as_dicts = []
cell_records = []

for row_index, row_item in enumerate(table_rows, start=1):
    current_row = {}
    for col_index, col_item in enumerate(table_columns, start=1):
        cell_box = bbox_intersection(row_item["bbox"], col_item["bbox"])
        if cell_box is None:
            cell_text = ""
        else:
            crop = detector.image.crop(tuple(cell_box))
            cell_text = ocr_image(crop)
        
        col_name = header_names[col_index - 1]
        current_row[col_name] = cell_text
        cell_records.append(
            {
                "row": row_index,
                "column": col_index,
                "bbox": cell_box,
                "text": cell_text
            }
        )

    rows_as_dicts.append(current_row)

# If a column-header region exists, try OCR there to rename columns
if column_headers:
    header_box = column_headers[0]["bbox"]
    extracted_headers = []
    for col_item in table_columns:
        hdr_cell = bbox_intersection(header_box, col_item["bbox"])
        if hdr_cell is None:
            extracted_headers.append("")
        else:
            header_crop = detector.image.crop(tuple(hdr_cell))
            extracted_headers.append(ocr_image(header_crop))

    if any(text.strip() for text in extracted_headers):
        clean_headers = [
            (text.strip() if text.strip() else f"col_{idx + 1}")
            for idx, text in enumerate(extracted_headers)
        ]
        remapped_rows = []
        for row_dict in rows_as_dicts:
            remapped_rows.append(
                {
                    clean_headers[idx]: row_dict[header_names[idx]]
                    for idx in range(len(header_names))
                }
            )
        header_names = clean_headers
        rows_as_dicts = remapped_rows

# Save row-wise output
output_dir = Path("extracted_data")
output_dir.mkdir(parents=True, exist_ok=True)
row_json_path = "table_text_by_row.json"

with open(row_json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "headers": header_names,
            "rows": rows_as_dicts,
            "cells": cell_records,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Rows: {len(table_rows)} | Columns: {len(table_columns)}")
print(f"Saved JSON: {row_json_path}")

if not OCR_READY:
    print("pytesseract not installed. Install it to extract text:")
    print("  %pip install pytesseract")
    print("Also install Tesseract OCR app on Windows and set pytesseract.pytesseract.tesseract_cmd.")

rows_as_dicts[:2]