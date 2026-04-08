#!/usr/bin/env python3
# Create deterministic train and validation splits from a flat image directory.

"""Create train/validation split in phone and PC datasets."""

import os
import shutil
import random
from pathlib import Path

def create_train_val_split(dataset_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train/validation folders.

    Args:
        dataset_dir: Directory containing images to split
        train_ratio: Ratio of images for training (default: 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)


    # Create destination folders next to source images.
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validation"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)


    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in dataset_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if len(image_files) == 0:
        print(f"No images found in {dataset_dir}")
        return


    # Shuffle once so the split is random but reproducible with the same seed.
    random.seed(seed)
    random.shuffle(image_files)


    train_count = int(len(image_files) * train_ratio)

    print(f"\nProcessing {dataset_dir}...")
    print(f"Total images: {len(image_files)}")
    print(f"Train: {train_count} ({train_ratio*100:.0f}%)")
    print(f"Validation: {len(image_files) - train_count} ({(1-train_ratio)*100:.0f}%)")


    # Move selected images into the train folder.
    for idx, img_file in enumerate(image_files[:train_count], 1):
        dest_file = train_dir / img_file.name
        shutil.move(str(img_file), str(dest_file))
        if idx % 1000 == 0:
            print(f"  Moving to train: {idx}/{train_count}...", end='\r')
    print(f"  Moved {train_count} images to train/")


    # Move the remaining images into validation.
    for idx, img_file in enumerate(image_files[train_count:], 1):
        dest_file = val_dir / img_file.name
        shutil.move(str(img_file), str(dest_file))
        if idx % 1000 == 0:
            print(f"  Moving to validation: {idx}/{len(image_files) - train_count}...", end='\r')
    print(f"  Moved {len(image_files) - train_count} images to validation/")

    print(f"✓ Completed: {dataset_dir}")


if __name__ == "__main__":

    PHONE_DIR = "dataset/vae_dataset_phone"
    PC_DIR = "dataset/vae_dataset_pc"
    TRAIN_RATIO = 0.8
    SEED = 42


    create_train_val_split(PHONE_DIR, TRAIN_RATIO, SEED)
    create_train_val_split(PC_DIR, TRAIN_RATIO, SEED)

    print("\n" + "="*50)
    print("All datasets split successfully!")
    print("="*50)
