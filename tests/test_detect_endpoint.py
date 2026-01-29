"""Integration tests for /detect endpoint with real model.

These tests verify the full detection pipeline including:
- Image upload and decoding
- Tiling for large images
- Detection with real darknet model
- Post-processing and response format
"""

import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from pathlib import Path

from tests.conftest import PROJECT_ROOT, create_test_image, image_to_bytes


def darknet_available() -> bool:
    """Check if darknet library and model files are available."""
    cfg_path = PROJECT_ROOT / "cfg" / "fiber-tiny.cfg"
    weights_path = PROJECT_ROOT / "weights" / "fiber-tiny_best.weights"
    data_path = PROJECT_ROOT / "cfg" / "fiber.data"
    if not (cfg_path.exists() and weights_path.exists() and data_path.exists()):
        return False
    try:
        from inference.detector import DarknetDetector
        DarknetDetector._instance = None
        DarknetDetector(
            cfg_path=cfg_path,
            weights_path=weights_path,
            data_path=data_path,
        )
        DarknetDetector._instance = None
        return True
    except (RuntimeError, OSError, ImportError):
        return False


# Skip if darknet not available
pytestmark = pytest.mark.skipif(
    not darknet_available(),
    reason="Darknet library not available"
)


@pytest.fixture(scope="module")
def integration_client():
    """Create test client with real model loaded."""
    from inference.main import app
    from inference.detector import DarknetDetector

    # Reset detector singleton
    DarknetDetector._instance = None

    with TestClient(app) as client:
        yield client

    # Cleanup
    DarknetDetector._instance = None


class TestDetectEndpointIntegration:
    """Integration tests for /detect endpoint."""

    def test_detect_valid_image_returns_200(self, integration_client):
        """Test that valid image returns 200."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = integration_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_detect_response_structure(self, integration_client):
        """Test response has correct structure."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = integration_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        data = response.json()

        assert "detections" in data
        assert "image_size" in data
        assert "inference_time_ms" in data
        assert "tile_count" in data
        assert isinstance(data["detections"], list)
        assert data["image_size"]["width"] == 512
        assert data["image_size"]["height"] == 512

    def test_detect_with_custom_threshold(self, integration_client):
        """Test detection with custom threshold."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = integration_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.8"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data

    def test_detect_large_image_uses_tiling(self, integration_client):
        """Test that large images are tiled."""
        # Create 4K image
        img = create_test_image(3840, 2160)
        img_bytes = image_to_bytes(img)

        response = integration_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # 4K image should produce multiple tiles
        assert data["tile_count"] > 1
        assert data["image_size"]["width"] == 3840
        assert data["image_size"]["height"] == 2160

    def test_detect_png_image(self, integration_client):
        """Test detection with PNG image format."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img, '.png')

        response = integration_client.post(
            "/detect",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_detection_result_format(self, integration_client):
        """Test that detection results have correct format."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = integration_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.1"},  # Low threshold to get detections
        )
        data = response.json()

        for det in data["detections"]:
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert "is_edge" in det
            assert "x" in det["bbox"]
            assert "y" in det["bbox"]
            assert "w" in det["bbox"]
            assert "h" in det["bbox"]
