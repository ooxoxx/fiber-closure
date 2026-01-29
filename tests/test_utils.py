"""Tests for the utils module."""

import pytest
import numpy as np
from pathlib import Path

from inference.utils import load_image, save_image, resize_image


class TestLoadImage:
    """Test cases for load_image."""

    def test_load_valid_image(self, tmp_path):
        """Test loading a valid image file."""
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [128, 64, 32]
        img_path = tmp_path / "test.png"

        import cv2
        cv2.imwrite(str(img_path), img)

        loaded = load_image(img_path)
        assert loaded.shape == (100, 100, 3)

    def test_load_nonexistent_raises_error(self, tmp_path):
        """Test that loading nonexistent file raises ValueError."""
        with pytest.raises(ValueError, match="Failed to load"):
            load_image(tmp_path / "nonexistent.png")


class TestSaveImage:
    """Test cases for save_image."""

    def test_save_image(self, tmp_path):
        """Test saving an image to disk."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img_path = tmp_path / "output.png"

        save_image(img, img_path)
        assert img_path.exists()


class TestResizeImage:
    """Test cases for resize_image."""

    def test_no_resize_when_no_dims(self):
        """Test that image is unchanged when no dimensions given."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img)
        assert result.shape == (100, 200, 3)

    def test_resize_both_dims(self):
        """Test resizing with both width and height."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img, width=50, height=25)
        assert result.shape == (25, 50, 3)

    def test_resize_width_only(self):
        """Test resizing with width only maintains aspect ratio."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img, width=100)
        assert result.shape[1] == 100
        assert result.shape[0] == 50  # aspect ratio maintained

    def test_resize_height_only(self):
        """Test resizing with height only maintains aspect ratio."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img, height=50)
        assert result.shape[0] == 50
        assert result.shape[1] == 100  # aspect ratio maintained
