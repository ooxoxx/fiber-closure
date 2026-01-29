"""Shared pytest fixtures for fiber-closure tests."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def test_image_small() -> np.ndarray:
    """Create a small test image (512x512)."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 128]  # gray background
    return img


@pytest.fixture
def test_image_1k() -> np.ndarray:
    """Create a 1K test image (1024x1024)."""
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img[:, :] = [100, 100, 100]
    # Add some features for detection
    cv2.rectangle(img, (400, 400), (600, 600), (255, 255, 255), -1)
    return img


@pytest.fixture
def test_image_4k() -> np.ndarray:
    """Create a 4K test image (3840x2160)."""
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    img[:, :] = [80, 80, 80]
    # Add some rectangular features
    cv2.rectangle(img, (500, 500), (700, 800), (200, 200, 200), -1)
    cv2.rectangle(img, (2000, 1000), (2200, 1300), (180, 180, 180), -1)
    return img


@pytest.fixture
def test_image_bytes(test_image_small) -> io.BytesIO:
    """Create test image as bytes for upload testing."""
    _, encoded = cv2.imencode('.jpg', test_image_small)
    return io.BytesIO(encoded.tobytes())


@pytest.fixture
def test_image_4k_bytes(test_image_4k) -> io.BytesIO:
    """Create 4K test image as bytes for upload testing."""
    _, encoded = cv2.imencode('.jpg', test_image_4k)
    return io.BytesIO(encoded.tobytes())


def create_test_image(width: int, height: int) -> np.ndarray:
    """Helper function to create test images of arbitrary size."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 128]
    return img


def create_image_with_objects(
    width: int,
    height: int,
    num_objects: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Create a test image with rectangular features that may trigger detections.

    Args:
        width: Image width.
        height: Image height.
        num_objects: Number of rectangular objects to draw.
        seed: Random seed for reproducibility.

    Returns:
        Image with rectangular features on varying background.
    """
    rng = np.random.default_rng(seed)

    # Create image with slight gradient background (more realistic than solid)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        gray_val = int(80 + 40 * (y / height))  # gradient from 80 to 120
        img[y, :] = [gray_val, gray_val, gray_val]

    # Add some noise for texture
    noise = rng.integers(-10, 10, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add rectangular objects (resembling closures/racks)
    for i in range(num_objects):
        # Random position (ensure object fits within image)
        obj_w = int(rng.integers(50, 150))
        obj_h = int(rng.integers(80, 200))
        x1 = int(rng.integers(0, max(1, width - obj_w)))
        y1 = int(rng.integers(0, max(1, height - obj_h)))
        x2 = x1 + obj_w
        y2 = y1 + obj_h

        # Random gray color (darker or lighter than background)
        color_val = int(rng.choice([60, 180]))  # contrast with background
        cv2.rectangle(img, (x1, y1), (x2, y2), (color_val, color_val, color_val), -1)

        # Add inner rectangle for detail
        margin = 10
        if obj_w > 2 * margin and obj_h > 2 * margin:
            inner_color = 255 - color_val
            cv2.rectangle(
                img,
                (x1 + margin, y1 + margin),
                (x2 - margin, y2 - margin),
                (inner_color, inner_color, inner_color),
                2,
            )

    return img


def image_to_bytes(image: np.ndarray, format: str = '.jpg') -> io.BytesIO:
    """Convert numpy image to bytes."""
    _, encoded = cv2.imencode(format, image)
    return io.BytesIO(encoded.tobytes())
