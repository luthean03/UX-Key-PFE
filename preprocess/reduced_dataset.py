#!/usr/bin/env python3
"""Create a reduced dataset by sampling images from train_scaled and validation_scaled.

This script:
- Collects all PNG images from vae_dataset_pc/train_scaled and validation_scaled
- Randomly samples 5000 images for training and 1250 for validation (80/20 split)
- Copies them to reduced_dataset/train and reduced_dataset/validation
"""

import random
import shutil
import pathlib
import sys
import argparse


def create_reduced_dataset(
    train_src="dataset/vae_dataset_pc/train_scaled",
    val_src="dataset/vae_dataset_pc/validation_scaled",
    out_dir="dataset/vae_dataset_pc/reduced_dataset",
    train_size=5000,
    val_ratio=0.20,
    seed=42
):
    """Create reduced dataset with specified train/val split.
    
    Args:
        train_src: Source directory for training images
        val_src: Source directory for validation images
        out_dir: Output directory for reduced dataset
        train_size: Number of training images to sample
        val_ratio: Validation ratio (e.g., 0.20 for 20%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    train_src = pathlib.Path(train_src)
    val_src = pathlib.Path(val_src)
    out_dir = pathlib.Path(out_dir)
    
    # Gather all PNG images from both source directories
    print(f"Gathering images from {train_src} and {val_src}...")
    imgs = list(train_src.rglob("*.png")) + list(val_src.rglob("*.png"))
    
    if len(imgs) == 0:
        print(f"ERROR: No PNG images found in {train_src} or {val_src}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(imgs)} total images")
    
    # Shuffle images for random sampling
    random.shuffle(imgs)
    
    # Compute validation size based on ratio (80/20 means val = train * 0.25)
    val_size = int(train_size * val_ratio / (1.0 - val_ratio))
    
    total_needed = train_size + val_size
    if len(imgs) < total_needed:
        print(f"WARNING: Only {len(imgs)} images available, need {total_needed}")
        print(f"Reducing train_size to maintain {int((1-val_ratio)*100)}/{int(val_ratio*100)} split")
        train_size = int(len(imgs) * (1 - val_ratio))
        val_size = len(imgs) - train_size
    
    print(f"Creating reduced dataset: {train_size} train, {val_size} val")
    
    # Create output directories
    train_out = out_dir / "train"
    val_out = out_dir / "validation"
    
    # Clean and create directories
    if out_dir.exists():
        print(f"Removing existing {out_dir}...")
        shutil.rmtree(out_dir)
    
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    print(f"Copying {train_size} training images to {train_out}...")
    for i, img_path in enumerate(imgs[:train_size]):
        dest = train_out / img_path.name
        shutil.copy2(img_path, dest)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{train_size} train images copied")
    
    # Copy validation images
    print(f"Copying {val_size} validation images to {val_out}...")
    for i, img_path in enumerate(imgs[train_size:train_size + val_size]):
        dest = val_out / img_path.name
        shutil.copy2(img_path, dest)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{val_size} val images copied")
    
    print(f"\n✓ Reduced dataset created successfully in {out_dir}")
    print(f"  Train: {len(list(train_out.glob('*.png')))} images")
    print(f"  Validation: {len(list(val_out.glob('*.png')))} images")
    print(f"  Split: {int((1-val_ratio)*100)}/{int(val_ratio*100)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create reduced dataset from train_scaled and validation_scaled"
    )
    parser.add_argument(
        "--train-src",
        default="dataset/vae_dataset_pc/train_scaled",
        help="Source directory for training images"
    )
    parser.add_argument(
        "--val-src",
        default="dataset/vae_dataset_pc/validation_scaled",
        help="Source directory for validation images"
    )
    parser.add_argument(
        "--out-dir",
        default="reduced_dataset",
        help="Output directory for reduced dataset"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
        help="Number of training images to sample (default: 5000)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio, e.g., 0.20 for 80/20 split (default: 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_reduced_dataset(
        train_src=args.train_src,
        val_src=args.val_src,
        out_dir=args.out_dir,
        train_size=args.train_size,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
