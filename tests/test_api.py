"""Tests for the FastAPI endpoints."""

import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from inference.main import app
from inference import __version__


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response has correct format."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == __version__


class TestDetectEndpoint:
    """Test cases for /detect endpoint."""

    def _create_test_image(self, width=100, height=100):
        """Create a simple test image."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [128, 128, 128]  # gray
        _, encoded = cv2.imencode('.jpg', img)
        return io.BytesIO(encoded.tobytes())

    def test_detect_invalid_image_returns_400(self, client):
        """Test that invalid image data returns 400."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_detect_empty_file_returns_400(self, client):
        """Test that empty file returns 400."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_detect_threshold_validation_too_high(self, client):
        """Test that threshold > 1.0 is rejected."""
        img_bytes = self._create_test_image()
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "1.5"},
        )
        assert response.status_code == 422  # validation error

    def test_detect_threshold_validation_negative(self, client):
        """Test that negative threshold is rejected."""
        img_bytes = self._create_test_image()
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "-0.1"},
        )
        assert response.status_code == 422  # validation error


class TestDetectEndpointWithMock:
    """Test /detect endpoint with mocked detector."""

    def _create_test_image(self, width=512, height=512):
        """Create a test image."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [128, 128, 128]
        _, encoded = cv2.imencode('.jpg', img)
        return io.BytesIO(encoded.tobytes())

    def test_detect_with_mocked_detector(self, client):
        """Test /detect endpoint with mocked detector."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            ("closure_normal", "95.5", (256.0, 256.0, 100.0, 100.0))
        ]
        mock_detector.thresh = 0.5
        mock_detector.get_network_size.return_value = (416, 416)

        with patch('inference.main.get_detector', return_value=mock_detector):
            img_bytes = self._create_test_image()
            response = client.post(
                "/detect",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "tile_count" in data
        assert data["image_size"]["width"] == 512
        assert data["image_size"]["height"] == 512

    def test_detect_large_image_tiling(self, client):
        """Test that large images are tiled correctly."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_detector.thresh = 0.5

        with patch('inference.main.get_detector', return_value=mock_detector):
            # Create 4K image
            img = np.zeros((2160, 3840, 3), dtype=np.uint8)
            _, encoded = cv2.imencode('.jpg', img)
            img_bytes = io.BytesIO(encoded.tobytes())

            response = client.post(
                "/detect",
                files={"file": ("4k.jpg", img_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["tile_count"] > 1
        assert data["image_size"]["width"] == 3840
        assert data["image_size"]["height"] == 2160

    def test_detect_with_custom_threshold(self, client):
        """Test /detect with custom threshold."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_detector.thresh = 0.5

        with patch('inference.main.get_detector', return_value=mock_detector):
            img_bytes = self._create_test_image()
            response = client.post(
                "/detect",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                data={"threshold": "0.8"},
            )

        assert response.status_code == 200
        # Verify threshold was set
        assert mock_detector.thresh == 0.8

    def test_health_with_model_loaded(self, client):
        """Test health endpoint when model is loaded."""
        from inference.detector import DarknetDetector

        mock_instance = MagicMock()
        mock_instance.get_network_size.return_value = (416, 416)

        with patch.object(DarknetDetector, 'is_loaded', return_value=True):
            with patch('inference.main.get_detector', return_value=mock_instance):
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["network_size"]["width"] == 416
        assert data["network_size"]["height"] == 416
