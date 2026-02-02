#!/usr/bin/env python3
"""Split dataset into phone and PC folders based on image width."""

import os
import shutil
from pathlib import Path
from PIL import Image

def split_dataset_by_width(source_dir, phone_dir, pc_dir, width_threshold=800, min_width=320, min_height=480):
    """
    Split images into phone/PC folders based on width.
    
    Args:
        source_dir: Source dataset directory
        phone_dir: Output directory for narrow images (< threshold)
        pc_dir: Output directory for wide images (>= threshold)
        width_threshold: Width threshold in pixels (default: 800)
        min_width: Minimum width to keep (default: 320)
        min_height: Minimum height to keep (default: 480)
    """
    source_path = Path(source_dir)
    phone_path = Path(phone_dir)
    pc_path = Path(pc_dir)
    
    # Create output directories
    phone_path.mkdir(parents=True, exist_ok=True)
    pc_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in source_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    phone_count = 0
    pc_count = 0
    skipped_count = 0
    errors = 0
    
    print(f"Processing {len(image_files)} images from {source_dir}...")
    print(f"Width threshold: {width_threshold}px")
    print(f"Minimum dimensions: {min_width}x{min_height}px (width x height)")
    print()
    
    for idx, img_file in enumerate(image_files, 1):
        try:
            # Open image to get dimensions (faster: just read header)
            with Image.open(img_file) as img:
                width, height = img.width, img.height
            
            # Skip images smaller than minimum dimensions
            if width < min_width or height < min_height:
                skipped_count += 1
                continue
            
            # Determine destination based on width
            if width < width_threshold:
                dest_dir = phone_path
                phone_count += 1
            else:
                dest_dir = pc_path
                pc_count += 1
            
            # Copy file to destination
            dest_file = dest_dir / img_file.name
            shutil.copy2(img_file, dest_file)
            
            # Progress every 100 images
            if idx % 100 == 0:
                percent = (idx / len(image_files)) * 100
                print(f"Progress: {idx}/{len(image_files)} ({percent:.1f}%) - Phone: {phone_count}, PC: {pc_count}, Skipped: {skipped_count}", end='\r')
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Show only first 5 errors
                print(f"\nError processing {img_file.name}: {e}")
            continue
    
    print(f"\n\nCompleted!")
    print(f"Phone images (< {width_threshold}px): {phone_count} -> {phone_dir}")
    print(f"PC images (>= {width_threshold}px): {pc_count} -> {pc_dir}")
    print(f"Skipped (< {min_width}x{min_height}px): {skipped_count}")
    print(f"Total processed: {phone_count + pc_count}")
    if errors > 0:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "dataset/vae_dataset/png"
    PHONE_DIR = "dataset/vae_dataset_phone/all"
    PC_DIR = "dataset/vae_dataset_pc/all"
    WIDTH_THRESHOLD = 800
    
    split_dataset_by_width(SOURCE_DIR, PHONE_DIR, PC_DIR, WIDTH_THRESHOLD)
