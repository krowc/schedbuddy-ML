from pathlib import Path
import random
import os
import sys
import shutil
import argparse


def split_dataset(data_path, train_percent=0.8, output_root=None):
    if not os.path.isdir(data_path):
        raise ValueError(
            "Directory specified by data_path not found. Verify the path is correct."
        )
    if train_percent < 0.01 or train_percent > 0.99:
        raise ValueError("Invalid train_percent. Please enter a number between 0.01 and 0.99.")

    input_image_path = os.path.join(data_path, 'images')
    input_label_path = os.path.join(data_path, 'labels')

    if output_root is None:
        output_root = os.getcwd()

    train_img_path = os.path.join(output_root, 'data', 'train', 'images')
    train_txt_path = os.path.join(output_root, 'data', 'train', 'labels')
    val_img_path = os.path.join(output_root, 'data', 'validation', 'images')
    val_txt_path = os.path.join(output_root, 'data', 'validation', 'labels')

    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created folder at {dir_path}")

    img_file_list = [path for path in Path(input_image_path).rglob('*') if path.is_file()]
    txt_file_list = [path for path in Path(input_label_path).rglob('*') if path.is_file()]

    print(f"Number of image files: {len(img_file_list)}")
    print(f"Number of annotation files: {len(txt_file_list)}")

    file_num = len(img_file_list)
    train_num = int(file_num * train_percent)
    val_num = file_num - train_num
    print(f"Images moving to train: {train_num}")
    print(f"Images moving to validation: {val_num}")

    for i, set_num in enumerate([train_num, val_num]):
        for _ in range(set_num):
            img_path = random.choice(img_file_list)
            img_fn = img_path.name
            base_fn = img_path.stem
            txt_fn = base_fn + '.txt'
            txt_path = os.path.join(input_label_path, txt_fn)

            if i == 0:
                new_img_path, new_txt_path = train_img_path, train_txt_path
            else:
                new_img_path, new_txt_path = val_img_path, val_txt_path

            shutil.copy(img_path, os.path.join(new_img_path, img_fn))
            if os.path.exists(txt_path):
                shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

            img_file_list.remove(img_path)

    return {
        'train_images': train_num,
        'val_images': val_num,
        'train_image_path': train_img_path,
        'train_label_path': train_txt_path,
        'val_image_path': val_img_path,
        'val_label_path': val_txt_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        help='Path to data folder containing image and annotation files',
        required=True,
    )
    parser.add_argument(
        '--train_pct',
        help='Ratio of images to go to train folder; the rest go to validation folder (example: 0.8)',
        default=0.8,
        type=float,
    )

    args = parser.parse_args()

    try:
        split_dataset(args.datapath, args.train_pct)
    except ValueError as error:
        print(error)
        sys.exit(1)


if __name__ == '__main__':
    main()

