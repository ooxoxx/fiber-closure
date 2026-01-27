#!/usr/bin/env python3
"""
Generate train.txt and valid.txt file lists for darknet training.

Usage:
    python scripts/prepare_training.py \
        --sliced-dir data/sliced \
        --output-dir data \
        --train-ratio 0.9
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Generate train.txt and valid.txt for darknet'
    )
    parser.add_argument(
        '--sliced-dir', type=str, required=True,
        help='Directory containing sliced images and labels'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for train.txt and valid.txt'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.9,
        help='Ratio of training data (default: 0.9)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    sliced_dir = Path(args.sliced_dir)
    images_dir = sliced_dir / 'images'
    labels_dir = sliced_dir / 'labels'
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Find all images with corresponding labels
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                image_files.append(img_path.resolve())

    if not image_files:
        print("Error: No image-label pairs found")
        return

    print(f"Found {len(image_files)} image-label pairs")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * args.train_ratio)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]

    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(valid_files)} images")

    # Write train.txt
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'train.txt'
    with open(train_path, 'w') as f:
        for img_path in train_files:
            f.write(f"{img_path}\n")
    print(f"Saved: {train_path}")

    # Write valid.txt
    valid_path = output_dir / 'valid.txt'
    with open(valid_path, 'w') as f:
        for img_path in valid_files:
            f.write(f"{img_path}\n")
    print(f"Saved: {valid_path}")


if __name__ == '__main__':
    main()
