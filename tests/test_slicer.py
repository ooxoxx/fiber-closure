"""Tests for the image slicer module."""

import numpy as np
import pytest

from inference.slicer import ImageSlicer, Tile


class TestImageSlicer:
    """Test cases for ImageSlicer."""

    def test_init_default_values(self):
        """Test default initialization."""
        slicer = ImageSlicer()
        assert slicer.tile_size == 1024
        assert slicer.overlap == 0.3
        assert slicer.step == 716  # 1024 * 0.7

    def test_init_custom_values(self):
        """Test custom initialization."""
        slicer = ImageSlicer(tile_size=512, overlap=0.5)
        assert slicer.tile_size == 512
        assert slicer.overlap == 0.5
        assert slicer.step == 256

    def test_slice_small_image(self):
        """Test slicing image smaller than tile size."""
        slicer = ImageSlicer(tile_size=1024, overlap=0.3)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        tiles = slicer.slice(image)

        assert len(tiles) == 1
        assert tiles[0].x == 0
        assert tiles[0].y == 0
        assert tiles[0].image.shape == (1024, 1024, 3)

    def test_slice_4k_image(self):
        """Test slicing a 4K image."""
        slicer = ImageSlicer(tile_size=1024, overlap=0.3)
        # 4K image: 3840 x 2160
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tiles = slicer.slice(image)

        # With step=716, we expect multiple tiles
        assert len(tiles) > 1
        # All tiles should have correct shape
        for tile in tiles:
            assert tile.image.shape == (1024, 1024, 3)

    def test_tile_offsets(self):
        """Test that tile offsets are correct."""
        slicer = ImageSlicer(tile_size=1024, overlap=0.3)
        image = np.zeros((2048, 2048, 3), dtype=np.uint8)
        tiles = slicer.slice(image)

        # First tile should be at origin
        assert tiles[0].x == 0
        assert tiles[0].y == 0

        # Check step increments
        step = slicer.step
        for tile in tiles:
            assert tile.x % step == 0 or tile.x == 0
            assert tile.y % step == 0 or tile.y == 0
