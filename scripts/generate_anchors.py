#!/usr/bin/env python3
"""
K-Means clustering to generate anchors for YOLOv4.

Usage:
    python scripts/generate_anchors.py \
        --labels-dir data/sliced/labels \
        --num-clusters 9 \
        --input-size 1024
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_boxes(labels_dir: Path, input_size: int) -> np.ndarray:
    """Load all bounding boxes from label files.

    Returns:
        Array of shape (N, 2) with (width, height) in pixels
    """
    boxes = []
    label_files = list(labels_dir.glob('*.txt'))

    for label_path in label_files:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    w = float(parts[3]) * input_size
                    h = float(parts[4]) * input_size
                    if w > 0 and h > 0:
                        boxes.append([w, h])

    return np.array(boxes)


def iou(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Calculate IoU between a box and cluster centroids.

    Args:
        box: Single box (w, h)
        clusters: Cluster centroids (K, 2)

    Returns:
        IoU values (K,)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection
    return intersection / union


def avg_iou(boxes: np.ndarray, clusters: np.ndarray) -> float:
    """Calculate average IoU between boxes and their nearest clusters."""
    return np.mean([np.max(iou(box, clusters)) for box in boxes])


def kmeans_anchors(
    boxes: np.ndarray,
    num_clusters: int,
    max_iter: int = 300
) -> np.ndarray:
    """K-Means clustering using IoU distance metric.

    Args:
        boxes: Array of (N, 2) with box dimensions
        num_clusters: Number of anchor clusters
        max_iter: Maximum iterations

    Returns:
        Cluster centroids (K, 2)
    """
    n = boxes.shape[0]
    if n < num_clusters:
        raise ValueError(f"Not enough boxes ({n}) for {num_clusters} clusters")

    # Initialize clusters randomly
    indices = np.random.choice(n, num_clusters, replace=False)
    clusters = boxes[indices].copy()

    prev_assignments = np.zeros(n)

    for iteration in range(max_iter):
        # Assign boxes to nearest cluster (using 1 - IoU as distance)
        distances = np.zeros((n, num_clusters))
        for i, box in enumerate(boxes):
            distances[i] = 1 - iou(box, clusters)

        assignments = np.argmin(distances, axis=1)

        # Check convergence
        if np.array_equal(assignments, prev_assignments):
            print(f"Converged at iteration {iteration}")
            break

        prev_assignments = assignments.copy()

        # Update cluster centroids
        for k in range(num_clusters):
            mask = assignments == k
            if np.sum(mask) > 0:
                clusters[k] = np.mean(boxes[mask], axis=0)

    return clusters


def main():
    parser = argparse.ArgumentParser(
        description='Generate YOLOv4 anchors using K-Means clustering'
    )
    parser.add_argument(
        '--labels-dir', type=str, required=True,
        help='Directory containing YOLO label files'
    )
    parser.add_argument(
        '--num-clusters', type=int, default=9,
        help='Number of anchor clusters (default: 9)'
    )
    parser.add_argument(
        '--input-size', type=int, default=1024,
        help='Network input size (default: 1024)'
    )

    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    print(f"Loading boxes from {labels_dir}...")
    boxes = load_boxes(labels_dir, args.input_size)

    if len(boxes) == 0:
        print("Error: No bounding boxes found")
        return

    print(f"Loaded {len(boxes)} bounding boxes")
    print(f"Box size range: {boxes.min(axis=0)} to {boxes.max(axis=0)}")
    print()

    print(f"Running K-Means with {args.num_clusters} clusters...")
    clusters = kmeans_anchors(boxes, args.num_clusters)

    # Sort by area
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_indices = np.argsort(areas)
    clusters = clusters[sorted_indices]

    # Calculate average IoU
    avg = avg_iou(boxes, clusters)
    print(f"\nAverage IoU: {avg:.4f}")

    # Format anchors for darknet config
    anchors_str = ', '.join([f"{int(w)},{int(h)}" for w, h in clusters])

    print(f"\nAnchors (sorted by area):")
    for i, (w, h) in enumerate(clusters):
        print(f"  {i+1}: {w:.1f} x {h:.1f} (area: {w*h:.0f})")

    print(f"\nDarknet config format:")
    print(f"anchors = {anchors_str}")

    # YOLOv4 mask assignments (3 scales)
    print(f"\nYOLOv4 mask assignments:")
    print(f"  Large objects (mask = 6,7,8):  {clusters[6:9]}")
    print(f"  Medium objects (mask = 3,4,5): {clusters[3:6]}")
    print(f"  Small objects (mask = 0,1,2):  {clusters[0:3]}")


if __name__ == '__main__':
    main()
