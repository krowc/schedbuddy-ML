# SchedBuddy-ML

**Project Overview**
- **Purpose:**: Schedule builder that integrates machine learning to automatically generate timetable, specifically for Bicol University students.
- **Machine Learning resources:** Label Studio, YOLO11s, Hugging Face Table Transformer: microsoft/table-detection-model and microsoft/table-detection-structure-recognition, Tesseract, [Borderless Tables Detection](https://github.com/ShakilMahmudShuvo/Borderless-Tables-Detection)
---
## **Workflow**
- `detection-model/`: Table Transformer + Tesseract OCR extraction pipeline for structured JSON output
- `model/`: YOLO training and inference pipeline for table detection.
---
## **Approach**
**1. Image Processing and Loading**

The input image is loaded using the Pillow (PIL) library. Since the model expects the image in RGB format, the image is converted from any format (e.g., grayscale, CMYK) to RGB.

The image is then passed through the `DetrImageProcessor` provided by Hugging Face, which converts the image into a tensor format compatible with PyTorch. This extracted feature tensor serves as the input to the table detection and structure recognition models.

**2. Model Selection and Loading**

The project currently uses two model from Hugging Face's Table Transformer suite, specifically designed for table detection and table structure recognition:
- **Table Detection Model**
    
    The table detection model (`microsoft/table-detection-model`) functions similarly with `model/`, which is custom trained using YOLOv11s. It looks at the image and identifies the schedule table. The model processes the extracted feature tensors and outputs a bounding box around the table. In comparison, `model/` relies on the existing grid present in the COR while the table detection model finds edges among the text to define the entire table. 

        The model architecture includes a ResNet backbone followed by a transformer encoder-decoder structure. The ResNet backbone captures local features (like edges or corners), while the transformer layers model long-range dependencies, which are crucial for understanding table layouts, especially when no borders are present.

    Disclaimer: This model is not used in this project as the `model/` already functions accurately. It is only included in this approach serving as an option and for future purposes.
- **Table Detection Structure Recognition**
    
    Once the tables are identified, the table structure recognition model (`microsoft/table-detection-structure-recognition`) is employed to detect the internal structure of the table. It predicts the arrangement of rows, columns, and potentially spanning cells within each table. It outputs bounding boxes and labels corresponding to a specific element in the table.
---
## **Next Improvements**
- Do changes in row structure to exclude the unit summary; may be done hardcoded, however, conflicts may arise when the institution changes their formatting.
- Fine-tune OCR to accurately extract text. Suggested solutions:
    - Explore other OCR configurations (current: PSM = 6)
        - `--oem 3 --psm 7`   for single text line (good for single-row cells)
        - `--oem 3 --psm 8`  for  single word (good for numeric/code cells)
        - `--oem 3 --psm 6`  for block of text (good for multi-line cells)
        - `--oem 3 --psm 13` for raw line, no layout analysis (sometimes better for small crops)
    - Preprocess each cell before OCR
    - Train a custom Tesseract model
