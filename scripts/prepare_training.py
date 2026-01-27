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
from typing import List, Tuple


def classify_samples(
    images_dir: Path, labels_dir: Path
) -> Tuple[List[Path], List[Path]]:
    """
    Classify image-label pairs into positive and negative samples.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files

    Returns:
        Tuple of (positives, negatives) where each is a list of image paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    positives = []
    negatives = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Check if label file has content (non-whitespace)
        label_content = label_path.read_text().strip()
        if label_content:
            positives.append(img_path.resolve())
        else:
            negatives.append(img_path.resolve())

    return positives, negatives


def sample_negatives(
    negatives: List[Path],
    positives: List[Path],
    ratio: float,
    seed: int = 42
) -> List[Path]:
    """
    Sample negative samples to achieve target ratio.

    Args:
        negatives: List of negative sample paths
        positives: List of positive sample paths
        ratio: Target ratio of negatives in final dataset (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of sampled negative paths
    """
    if ratio <= 0.0 or len(positives) == 0:
        return []

    if ratio >= 1.0:
        return list(negatives)

    # Calculate needed negatives: neg / (pos + neg) = ratio
    # So: neg = pos * ratio / (1 - ratio)
    num_positives = len(positives)
    needed_negatives = int(num_positives * ratio / (1 - ratio))

    if needed_negatives >= len(negatives):
        return list(negatives)

    # Sample randomly
    rng = random.Random(seed)
    return rng.sample(negatives, needed_negatives)


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
    parser.add_argument(
        '--negative-ratio', type=float, default=None,
        help='Ratio of negative samples in final dataset (0.0-1.0). '
             'If not set, all samples are used.'
    )

    args = parser.parse_args()

    sliced_dir = Path(args.sliced_dir)
    images_dir = sliced_dir  # Images and labels are in the same directory
    labels_dir = sliced_dir
    output_dir = Path(args.output_dir)

    if not sliced_dir.exists():
        print(f"Error: Sliced directory not found: {sliced_dir}")
        return

    # Classify samples into positive (with labels) and negative (empty labels)
    positives, negatives = classify_samples(images_dir, labels_dir)

    if not positives and not negatives:
        print("Error: No image-label pairs found")
        return

    print(f"Found {len(positives)} positive samples (with labels)")
    print(f"Found {len(negatives)} negative samples (empty labels)")

    # Sample negatives if ratio is specified
    if args.negative_ratio is not None:
        sampled_negatives = sample_negatives(
            negatives, positives, args.negative_ratio, args.seed
        )
        print(f"Sampled {len(sampled_negatives)} negative samples "
              f"(target ratio: {args.negative_ratio:.1%})")
    else:
        sampled_negatives = negatives

    # Combine and shuffle
    image_files = positives + sampled_negatives
    random.seed(args.seed)
    random.shuffle(image_files)

    total = len(image_files)
    actual_neg_ratio = len(sampled_negatives) / total if total > 0 else 0
    print(f"Total samples: {total} (negative ratio: {actual_neg_ratio:.1%})")

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
