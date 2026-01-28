#!/usr/bin/env python3
"""
Slice 4K images into tiles with label coordinate conversion using SAHI.

Usage:
    python scripts/slice_dataset.py \
        --input-images data/raw/images \
        --input-labels data/raw/labels \
        --output-dir data/sliced \
        --tile-size 928 \
        --overlap 0.2 \
        --min-area 0.6
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sahi.slicing import slice_image as sahi_slice_image
from tqdm import tqdm


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO format label file.

    Returns:
        List of (class_id, x_center, y_center, width, height) in normalized coords
    """
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
    return labels


def convert_label_to_tile(
    label: Tuple[int, float, float, float, float],
    img_width: int,
    img_height: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_area_ratio: float
) -> Optional[Tuple[int, float, float, float, float]]:
    """Convert global label coordinates to tile-local coordinates.

    Args:
        label: (class_id, x_center, y_center, width, height) in normalized coords
        img_width: Original image width
        img_height: Original image height
        tile_x: Tile top-left x coordinate
        tile_y: Tile top-left y coordinate
        tile_size: Size of the tile
        min_area_ratio: Minimum area ratio to keep the label

    Returns:
        Converted label or None if filtered out
    """
    class_id, x_center, y_center, w, h = label

    # Convert normalized to absolute coordinates
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    abs_w = w * img_width
    abs_h = h * img_height

    # Calculate bounding box in absolute coordinates
    x1 = abs_x_center - abs_w / 2
    y1 = abs_y_center - abs_h / 2
    x2 = abs_x_center + abs_w / 2
    y2 = abs_y_center + abs_h / 2

    # Calculate tile boundaries
    tile_x2 = tile_x + tile_size
    tile_y2 = tile_y + tile_size

    # Check if box intersects with tile
    if x2 <= tile_x or x1 >= tile_x2 or y2 <= tile_y or y1 >= tile_y2:
        return None

    # Clip box to tile boundaries
    clipped_x1 = max(x1, tile_x)
    clipped_y1 = max(y1, tile_y)
    clipped_x2 = min(x2, tile_x2)
    clipped_y2 = min(y2, tile_y2)

    # Calculate area ratio
    original_area = abs_w * abs_h
    clipped_area = (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)

    if original_area <= 0:
        return None

    area_ratio = clipped_area / original_area

    # Filter by minimum area ratio
    if area_ratio < min_area_ratio:
        return None

    # Convert to tile-local coordinates
    local_x1 = clipped_x1 - tile_x
    local_y1 = clipped_y1 - tile_y
    local_x2 = clipped_x2 - tile_x
    local_y2 = clipped_y2 - tile_y

    # Convert to YOLO format (normalized center coordinates)
    local_w = local_x2 - local_x1
    local_h = local_y2 - local_y1
    local_x_center = (local_x1 + local_x2) / 2 / tile_size
    local_y_center = (local_y1 + local_y2) / 2 / tile_size
    local_w_norm = local_w / tile_size
    local_h_norm = local_h / tile_size

    # Clamp values to [0, 1]
    local_x_center = max(0, min(1, local_x_center))
    local_y_center = max(0, min(1, local_y_center))
    local_w_norm = max(0, min(1, local_w_norm))
    local_h_norm = max(0, min(1, local_h_norm))

    return (class_id, local_x_center, local_y_center, local_w_norm, local_h_norm)


def slice_image_with_sahi(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: float,
    min_area_ratio: float
) -> int:
    """Slice a single image and its labels into tiles using SAHI.

    Args:
        image_path: Path to input image
        label_path: Path to YOLO label file
        output_dir: Output directory for tiles
        tile_size: Size of output tiles (square)
        overlap: Overlap ratio between tiles (0.0-1.0)
        min_area_ratio: Minimum area ratio to keep a label

    Returns:
        Number of tiles created
    """
    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return 0

    img_height, img_width = img.shape[:2]

    # Parse labels
    labels = parse_yolo_label(label_path)

    # Use SAHI to slice the image (don't pass output_dir - we save files ourselves)
    base_name = image_path.stem
    slice_result = sahi_slice_image(
        image=str(image_path),
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=False,
    )

    # Process each sliced image and create corresponding labels
    tile_count = 0
    for idx, sliced_image in enumerate(slice_result.sliced_image_list):
        # Get tile position from SAHI result
        tile_x = sliced_image.starting_pixel[0]
        tile_y = sliced_image.starting_pixel[1]

        # Generate output filename (similar to SAHI's naming convention)
        tile_name = f"{base_name}_{tile_x}_{tile_y}_{tile_x + tile_size}_{tile_y + tile_size}"
        tile_image_path = output_dir / f"{tile_name}.jpg"
        tile_label_path = output_dir / f"{tile_name}.txt"

        # Save tile image (SAHI stores image data in memory)
        tile_img = sliced_image.image
        if tile_img is not None:
            # Convert RGB to BGR for OpenCV
            if len(tile_img.shape) == 3 and tile_img.shape[2] == 3:
                tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tile_image_path), tile_img)

        # Convert labels for this tile
        tile_labels = []
        for label in labels:
            converted = convert_label_to_tile(
                label, img_width, img_height,
                tile_x, tile_y, tile_size, min_area_ratio
            )
            if converted is not None:
                tile_labels.append(converted)

        # Save tile labels
        with open(tile_label_path, 'w') as f:
            for lbl in tile_labels:
                class_id, xc, yc, w, h = lbl
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        tile_count += 1

    return tile_count


# Keep old function name as alias for backward compatibility
slice_image = slice_image_with_sahi


def main():
    parser = argparse.ArgumentParser(
        description='Slice 4K images into tiles with label conversion'
    )
    parser.add_argument(
        '--input-images', type=str, required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--input-labels', type=str, required=True,
        help='Directory containing input YOLO labels'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for sliced data'
    )
    parser.add_argument(
        '--tile-size', type=int, default=928,
        help='Size of output tiles (default: 928)'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.2,
        help='Overlap ratio between tiles (default: 0.2)'
    )
    parser.add_argument(
        '--min-area', type=float, default=0.6,
        help='Minimum area ratio to keep a label (default: 0.6)'
    )

    args = parser.parse_args()

    # Setup paths
    input_images_dir = Path(args.input_images)
    input_labels_dir = Path(args.input_labels)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in input_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_images_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap * 100:.0f}%")
    print(f"Step size: {int(args.tile_size * (1 - args.overlap))}px")
    print(f"Min area ratio: {args.min_area * 100:.0f}%")
    print()

    total_tiles = 0
    for image_path in tqdm(image_files, desc="Slicing images"):
        # Find corresponding label file
        label_path = input_labels_dir / f"{image_path.stem}.txt"

        tiles = slice_image(
            image_path, label_path,
            output_dir,
            args.tile_size, args.overlap, args.min_area
        )
        total_tiles += tiles

    print(f"\nTotal tiles created: {total_tiles}")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
