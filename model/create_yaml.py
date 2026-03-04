"""
Step 2: Generate data.yaml config file from a classes.txt labelmap.

Usage:
    python 2_create_yaml.py
    python 2_create_yaml.py --classes path/to/classes.txt --output data.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def create_data_yaml(classes_txt: Path, output_yaml: Path) -> None:
    if not classes_txt.exists():
        print(f"[ERROR] classes.txt not found at: {classes_txt}")
        print("  Create a text file with one class name per line.")
        sys.exit(1)

    classes = [
        line.strip()
        for line in classes_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        print("[ERROR] classes.txt is empty.")
        sys.exit(1)

    data = {
        "path":  "./data",            # base folder containing train/ and validation/
        "train": "train/images",
        "val":   "validation/images",
        "nc":    len(classes),
        "names": classes,
    }

    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created {output_yaml} with {len(classes)} class(es): {classes}")

    # Pretty-print for confirmation
    print("\nFile contents:")
    print(output_yaml.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data.yaml for YOLO training.")
    parser.add_argument("--classes", default="..\\table-schedule\\classes.txt",  help="Path to classes.txt (default: classes.txt)")
    parser.add_argument("--output",  default="data.yaml",    help="Output YAML path (default: data.yaml)")
    args = parser.parse_args()
    create_data_yaml(Path(args.classes), Path(args.output))
