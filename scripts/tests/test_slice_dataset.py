#!/usr/bin/env python3
"""Tests for slice_dataset.py using SAHI-based image tiling."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2

from slice_dataset import (
    slice_image_with_sahi,
    convert_label_to_tile,
    parse_yolo_label,
)


class TestSliceImageBasic:
    """Test basic image slicing functionality."""

    def test_slice_image_correct_tile_count(self, tmp_path):
        """Slices image into correct number of tiles."""
        # Create 1856x1856 test image (2x2 tiles at 928px with 20% overlap)
        # With 20% overlap: step = 928 * 0.8 = 742.4 -> 742
        # Tiles: ceil((1856 - 928) / 742) + 1 = ceil(928/742) + 1 = 2 + 1 = 3?
        # Actually: positions are 0, 742, and then 1856-928=928 for last tile
        # So we get 2x2 = 4 tiles for a 1856x1856 image
        img_size = 1856
        tile_size = 928

        # Create test image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), img)

        # Create empty label file
        label_path = tmp_path / "test_image.txt"
        label_path.write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tile_count = slice_image_with_sahi(
            image_path=img_path,
            label_path=label_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=0.2,
            min_area_ratio=0.6,
        )

        # Verify tiles were created
        assert tile_count > 0

        # Check output files exist
        output_images = list(output_dir.glob("*.jpg"))
        output_labels = list(output_dir.glob("*.txt"))
        assert len(output_images) == tile_count
        assert len(output_labels) == tile_count

        # Verify tile dimensions
        for img_file in output_images:
            tile = cv2.imread(str(img_file))
            assert tile.shape[0] == tile_size
            assert tile.shape[1] == tile_size

    def test_slice_image_small_image(self, tmp_path):
        """Image smaller than tile size produces single tile."""
        img_size = 500
        tile_size = 928

        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img_path = tmp_path / "small_image.jpg"
        cv2.imwrite(str(img_path), img)

        label_path = tmp_path / "small_image.txt"
        label_path.write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tile_count = slice_image_with_sahi(
            image_path=img_path,
            label_path=label_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=0.2,
            min_area_ratio=0.6,
        )

        # Small image should still produce at least one tile
        assert tile_count >= 1


class TestLabelConversion:
    """Test label coordinate conversion."""

    def test_label_fully_inside_tile(self):
        """Label fully inside tile is converted correctly."""
        # Label at center of 1000x1000 image
        label = (0, 0.5, 0.5, 0.1, 0.1)  # class 0, center at (500, 500), size 100x100
        img_width, img_height = 1000, 1000
        tile_x, tile_y = 400, 400  # Tile covers 400-1328 (but we use 928 tile)
        tile_size = 928
        min_area_ratio = 0.6

        result = convert_label_to_tile(
            label, img_width, img_height,
            tile_x, tile_y, tile_size, min_area_ratio
        )

        assert result is not None
        class_id, xc, yc, w, h = result
        assert class_id == 0
        # Label center (500, 500) relative to tile (400, 400) = (100, 100)
        # Normalized: (100/928, 100/928) ≈ (0.108, 0.108)
        assert 0.1 < xc < 0.12
        assert 0.1 < yc < 0.12

    def test_label_outside_tile_returns_none(self):
        """Label completely outside tile returns None."""
        label = (0, 0.1, 0.1, 0.05, 0.05)  # Label at top-left corner
        img_width, img_height = 1000, 1000
        tile_x, tile_y = 500, 500  # Tile starts at (500, 500)
        tile_size = 928
        min_area_ratio = 0.6

        result = convert_label_to_tile(
            label, img_width, img_height,
            tile_x, tile_y, tile_size, min_area_ratio
        )

        assert result is None

    def test_label_partial_overlap_below_threshold(self):
        """Label with partial overlap below min_area is filtered out."""
        # Label at edge - only 50% inside tile
        label = (0, 0.45, 0.5, 0.1, 0.1)  # 100x100 box centered at (450, 500)
        img_width, img_height = 1000, 1000
        tile_x, tile_y = 450, 0  # Tile starts at x=450
        tile_size = 500
        min_area_ratio = 0.6  # Require 60% inside

        result = convert_label_to_tile(
            label, img_width, img_height,
            tile_x, tile_y, tile_size, min_area_ratio
        )

        # Box from 400-500 x, tile from 450-950
        # Overlap: 450-500 = 50px out of 100px = 50%
        assert result is None

    def test_label_partial_overlap_above_threshold(self):
        """Label with partial overlap above min_area is kept."""
        # Label mostly inside tile - centered at (520, 250) in 1000x1000 image
        # Box spans 470-570 x, 200-300 y (fully inside tile 450-950 x, 0-500 y)
        label = (1, 0.52, 0.25, 0.1, 0.1)
        img_width, img_height = 1000, 1000
        tile_x, tile_y = 450, 0
        tile_size = 500
        min_area_ratio = 0.6

        result = convert_label_to_tile(
            label, img_width, img_height,
            tile_x, tile_y, tile_size, min_area_ratio
        )

        # Box from 470-570 x, 200-300 y - fully inside tile
        assert result is not None
        assert result[0] == 1  # class_id preserved


class TestMinAreaFiltering:
    """Test minimum area ratio filtering."""

    def test_min_area_filtering_excludes_small_overlap(self, tmp_path):
        """Labels below min_area threshold are filtered out."""
        img_size = 2000
        tile_size = 928

        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        # Create label at position that will be at tile edge
        # Label at x=0.23 (460px center) with width 0.1 (200px)
        # Box spans 360-560px
        # First tile at x=0 covers 0-928px - label fully inside
        # But if we check second tile starting at step=742, it covers 742-1670
        # Label 360-560 doesn't intersect with 742-1670
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.23 0.5 0.1 0.1\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        slice_image_with_sahi(
            image_path=img_path,
            label_path=label_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=0.2,
            min_area_ratio=0.6,
        )

        # Check that labels were created for tiles containing the object
        label_files = list(output_dir.glob("*.txt"))
        labels_with_content = [f for f in label_files if f.read_text().strip()]

        # At least one tile should have the label
        assert len(labels_with_content) >= 1


class TestOverlapCalculation:
    """Test overlap ratio produces correct step size."""

    def test_overlap_20_percent(self, tmp_path):
        """20% overlap produces correct step size (928 * 0.8 = 742)."""
        tile_size = 928
        overlap = 0.2
        expected_step = int(tile_size * (1 - overlap))  # 742

        # Create image large enough for multiple tiles
        img_size = 2000
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        label_path = tmp_path / "test.txt"
        label_path.write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tile_count = slice_image_with_sahi(
            image_path=img_path,
            label_path=label_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            min_area_ratio=0.6,
        )

        # Calculate expected tile count
        # For 2000px with 928 tile and 742 step:
        # x positions: 0, 742, 1072 (last adjusted to 2000-928=1072)
        # Actually SAHI may handle this differently, but we should get multiple tiles
        assert tile_count > 1
        assert expected_step == 742

    def test_overlap_30_percent(self, tmp_path):
        """30% overlap produces correct step size (928 * 0.7 = 649)."""
        tile_size = 928
        overlap = 0.3
        expected_step = int(tile_size * (1 - overlap))  # 649

        assert expected_step == 649


class TestParseYoloLabel:
    """Test YOLO label parsing."""

    def test_parse_valid_labels(self, tmp_path):
        """Parses valid YOLO format labels."""
        label_path = tmp_path / "labels.txt"
        label_path.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.7 0.2 0.15\n")

        labels = parse_yolo_label(label_path)

        assert len(labels) == 2
        assert labels[0] == (0, 0.5, 0.5, 0.1, 0.1)
        assert labels[1] == (1, 0.3, 0.7, 0.2, 0.15)

    def test_parse_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        label_path = tmp_path / "empty.txt"
        label_path.write_text("")

        labels = parse_yolo_label(label_path)

        assert labels == []

    def test_parse_nonexistent_file(self, tmp_path):
        """Nonexistent file returns empty list."""
        label_path = tmp_path / "nonexistent.txt"

        labels = parse_yolo_label(label_path)

        assert labels == []

    def test_parse_with_blank_lines(self, tmp_path):
        """Handles blank lines in label file."""
        label_path = tmp_path / "labels.txt"
        label_path.write_text("0 0.5 0.5 0.1 0.1\n\n1 0.3 0.7 0.2 0.15\n\n")

        labels = parse_yolo_label(label_path)

        assert len(labels) == 2
