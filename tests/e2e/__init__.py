"""End-to-end tests for the full detection pipeline.

These tests verify the complete workflow from image input to detection output,
including 4K image processing, coordinate transformations, and service health.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from tests.conftest import PROJECT_ROOT, create_test_image, image_to_bytes


def model_available() -> bool:
    """Check if model files are available."""
    cfg_path = PROJECT_ROOT / "cfg" / "fiber-tiny.cfg"
    weights_path = PROJECT_ROOT / "weights" / "fiber-tiny_best.weights"
    data_path = PROJECT_ROOT / "cfg" / "fiber.data"
    return cfg_path.exists() and weights_path.exists() and data_path.exists()


pytestmark = pytest.mark.skipif(
    not model_available(),
    reason="Model files not available for E2E tests"
)
