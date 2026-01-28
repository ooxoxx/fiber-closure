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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


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


def count_class_instances(
    image_paths: List[Path], labels_dir: Path
) -> Dict[int, int]:
    """
    Count instances of each class from label files.

    Args:
        image_paths: List of image paths
        labels_dir: Directory containing label files

    Returns:
        Dict mapping class_id -> count
    """
    counts: Dict[int, int] = defaultdict(int)

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        content = label_path.read_text().strip()
        if not content:
            continue

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                class_id = int(parts[0])
                counts[class_id] += 1

    return dict(counts)


def validate_min_class_count(
    class_counts: Dict[int, int], min_count: int, num_classes: int
) -> bool:
    """
    Check if all classes have at least min_count instances.

    Args:
        class_counts: Dict mapping class_id -> count
        min_count: Minimum required instances per class
        num_classes: Total number of classes expected

    Returns:
        True if all classes meet minimum, False otherwise
    """
    if min_count <= 0:
        return True

    for class_id in range(num_classes):
        if class_counts.get(class_id, 0) < min_count:
            return False

    return True


def split_with_retry(
    samples: List[Path],
    train_ratio: float,
    min_valid_count: int,
    labels_dir: Path,
    num_classes: int,
    seed: int,
    max_attempts: int = 10,
) -> Tuple[List[Path], List[Path]]:
    """
    Split samples into train/valid sets with retry if validation minimum not met.

    Args:
        samples: List of sample paths to split
        train_ratio: Ratio of samples for training (0.0-1.0)
        min_valid_count: Minimum instances per class in validation set
        labels_dir: Directory containing label files
        num_classes: Number of classes
        seed: Random seed
        max_attempts: Maximum retry attempts

    Returns:
        Tuple of (train_samples, valid_samples)

    Raises:
        RuntimeError: If max_attempts exceeded without meeting minimum
    """
    for attempt in range(1, max_attempts + 1):
        current_seed = seed + attempt - 1
        rng = random.Random(current_seed)

        shuffled = list(samples)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        valid = shuffled[split_idx:]

        # Skip validation if min_valid_count is 0
        if min_valid_count <= 0:
            return train, valid

        valid_counts = count_class_instances(valid, labels_dir)
        if validate_min_class_count(valid_counts, min_valid_count, num_classes):
            print(f"Splitting dataset (attempt {attempt}/{max_attempts})...")
            print("  Split successful!")
            return train, valid

        # Log the failure
        print(f"Splitting dataset (attempt {attempt}/{max_attempts})...")
        for class_id in range(num_classes):
            count = valid_counts.get(class_id, 0)
            if count < min_valid_count:
                print(f"  Validation set class {class_id}: {count} < {min_valid_count}")

    raise RuntimeError(
        f"Failed to achieve minimum class count after max attempts ({max_attempts})"
    )


def print_class_distribution(
    train_counts: Dict[int, int],
    valid_counts: Dict[int, int],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print class distribution table for train and valid sets.

    Args:
        train_counts: Class counts for training set
        valid_counts: Class counts for validation set
        class_names: Optional list of class names
    """
    all_classes = sorted(set(train_counts.keys()) | set(valid_counts.keys()))

    if not all_classes:
        print("No class instances found.")
        return

    # Determine column widths
    if class_names:
        max_name_len = max(len(name) for name in class_names)
    else:
        max_name_len = max(len(f"class_{c}") for c in all_classes)
    max_name_len = max(max_name_len, 5)  # Minimum width for "Class"

    print("\nClass Distribution:")
    print(f"  {'Class':<{max_name_len}} | Train | Valid")
    print(f"  {'-' * max_name_len}-|-------|------")

    train_total = 0
    valid_total = 0

    for class_id in all_classes:
        if class_names and class_id < len(class_names):
            name = class_names[class_id]
        else:
            name = f"class_{class_id}"

        train_count = train_counts.get(class_id, 0)
        valid_count = valid_counts.get(class_id, 0)
        train_total += train_count
        valid_total += valid_count

        print(f"  {name:<{max_name_len}} | {train_count:>5} | {valid_count:>5}")

    print(f"  {'-' * max_name_len}-|-------|------")
    print(f"  {'Total':<{max_name_len}} | {train_total:>5} | {valid_total:>5}")


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
    parser.add_argument(
        '--min-valid-class-count', type=int, default=0,
        help='Minimum instances per class in validation set. '
             'Re-shuffles if not met (default: 0, no minimum).'
    )
    parser.add_argument(
        '--num-classes', type=int, default=5,
        help='Number of classes in the dataset (default: 5)'
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

    total = len(image_files)
    actual_neg_ratio = len(sampled_negatives) / total if total > 0 else 0
    print(f"Total samples: {total} (negative ratio: {actual_neg_ratio:.1%})")

    # Split with retry if min_valid_class_count is specified
    if args.min_valid_class_count > 0:
        train_files, valid_files = split_with_retry(
            samples=image_files,
            train_ratio=args.train_ratio,
            min_valid_count=args.min_valid_class_count,
            labels_dir=labels_dir,
            num_classes=args.num_classes,
            seed=args.seed,
        )
    else:
        random.seed(args.seed)
        random.shuffle(image_files)
        split_idx = int(len(image_files) * args.train_ratio)
        train_files = image_files[:split_idx]
        valid_files = image_files[split_idx:]

    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(valid_files)} images")

    # Count and print class distribution
    train_counts = count_class_instances(train_files, labels_dir)
    valid_counts = count_class_instances(valid_files, labels_dir)
    print_class_distribution(train_counts, valid_counts)

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
