"""Utility functions for inference service."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an image to disk."""
    cv2.imwrite(str(path), image)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Resize image maintaining aspect ratio if only one dimension given."""
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None and height is not None:
        return cv2.resize(image, (width, height))

    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h))

    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(image, (new_w, height))
