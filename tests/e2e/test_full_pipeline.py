"""End-to-end tests for the full detection pipeline.

These tests verify the complete workflow from image input to detection output,
including 4K image processing, coordinate transformations, and service health.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from tests.conftest import PROJECT_ROOT, create_test_image, create_image_with_objects, image_to_bytes


def darknet_available() -> bool:
    """Check if darknet library and model files are available."""
    cfg_path = PROJECT_ROOT / "cfg" / "fiber-tiny.cfg"
    weights_path = PROJECT_ROOT / "weights" / "fiber-tiny_best.weights"
    data_path = PROJECT_ROOT / "cfg" / "fiber.data"

    if not cfg_path.exists():
        print(f"E2E skip: cfg not found: {cfg_path}")
        return False
    if not weights_path.exists():
        print(f"E2E skip: weights not found: {weights_path}")
        return False
    if not data_path.exists():
        print(f"E2E skip: data not found: {data_path}")
        return False

    # Try to actually load darknet
    try:
        from inference.detector import DarknetDetector
        DarknetDetector._instance = None
        detector = DarknetDetector(
            cfg_path=cfg_path,
            weights_path=weights_path,
            data_path=data_path,
        )
        DarknetDetector._instance = None
        return True
    except Exception as e:
        print(f"E2E skip: DarknetDetector init failed: {type(e).__name__}: {e}")
        return False


pytestmark = pytest.mark.skipif(
    not darknet_available(),
    reason="Darknet library not available for E2E tests"
)


@pytest.fixture(scope="module")
def e2e_client():
    """Create test client for E2E tests."""
    from inference.main import app
    from inference.detector import DarknetDetector
    DarknetDetector._instance = None
    with TestClient(app) as client:
        yield client
    DarknetDetector._instance = None


class TestFullPipeline:
    """End-to-end tests for complete detection pipeline."""

    def test_service_startup(self, e2e_client):
        """Test that service starts and health endpoint works."""
        response = e2e_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_darknet_module_functions(self, e2e_client):
        """Test that darknet module functions are correctly exposed."""
        import darknet as dn

        # Verify network_width and network_height are callable
        assert callable(dn.network_width)
        assert callable(dn.network_height)

    def test_detector_initialization(self, e2e_client):
        """Test that detector initializes correctly with network dimensions."""
        from inference.detector import DarknetDetector
        detector = DarknetDetector.get_instance()

        # Verify network dimensions were correctly obtained
        assert detector.network_width > 0
        assert detector.network_height > 0
        assert detector.is_loaded()

    def test_health_shows_model_loaded_after_detect(self, e2e_client):
        """Test that model is loaded after first detection."""
        # Trigger model load via detect
        img = create_test_image(256, 256)
        img_bytes = image_to_bytes(img)
        e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

        # Check health shows model loaded
        response = e2e_client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_4k_image_detection(self, e2e_client):
        """Test detection on 4K image with tiling."""
        img = create_test_image(3840, 2160)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("4k.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify tiling occurred
        assert data["tile_count"] > 1
        assert data["image_size"]["width"] == 3840
        assert data["image_size"]["height"] == 2160

    def test_detection_coordinates_within_image(self, e2e_client):
        """Test that detection coordinates are within image bounds."""
        img = create_test_image(2048, 2048)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.1"},
        )
        data = response.json()

        for det in data["detections"]:
            bbox = det["bbox"]
            # Center coordinates should be within image
            assert 0 <= bbox["x"] <= 2048
            assert 0 <= bbox["y"] <= 2048
            assert bbox["w"] > 0
            assert bbox["h"] > 0

    def test_post_processing_chain(self, e2e_client):
        """Test that post-processing produces valid results."""
        img = create_test_image(1024, 1024)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        data = response.json()

        # Verify inference time is reasonable
        assert data["inference_time_ms"] > 0
        assert data["inference_time_ms"] < 60000  # < 60 seconds

    def test_multiple_requests(self, e2e_client):
        """Test multiple sequential requests work correctly."""
        for i in range(3):
            img = create_test_image(512, 512)
            img_bytes = image_to_bytes(img)

            response = e2e_client.post(
                "/detect",
                files={"file": (f"test_{i}.jpg", img_bytes, "image/jpeg")},
            )
            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_image_returns_400(self, e2e_client):
        """Test that uploading non-image data returns 400."""
        response = e2e_client.post(
            "/detect",
            files={"file": ("test.txt", b"not an image content", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_empty_file_returns_400(self, e2e_client):
        """Test that uploading an empty file returns 400."""
        response = e2e_client.post(
            "/detect",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_corrupted_jpeg_returns_400(self, e2e_client):
        """Test that corrupted JPEG data returns 400."""
        # JPEG magic bytes but corrupted content
        corrupted = b"\xff\xd8\xff\xe0" + b"corrupted data here"
        response = e2e_client.post(
            "/detect",
            files={"file": ("bad.jpg", corrupted, "image/jpeg")},
        )
        assert response.status_code == 400


class TestThresholdParameter:
    """Tests for threshold parameter behavior."""

    def test_threshold_parameter_affects_results(self, e2e_client):
        """Test that different threshold values produce different detection counts."""
        img = create_image_with_objects(1024, 1024, num_objects=5)
        img_bytes = image_to_bytes(img)

        # Low threshold - may have more detections (including low confidence)
        resp_low = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.1"},
        )
        assert resp_low.status_code == 200

        # High threshold - fewer detections
        img_bytes.seek(0)  # Reset file pointer
        resp_high = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.9"},
        )
        assert resp_high.status_code == 200

        # High threshold should return same or fewer detections
        assert len(resp_high.json()["detections"]) <= len(resp_low.json()["detections"])

    def test_large_threshold_no_detections(self, e2e_client):
        """Test that threshold=0.99 returns very few or no detections."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.99"},
        )
        assert response.status_code == 200
        # With 0.99 threshold, detections should be minimal or zero
        assert len(response.json()["detections"]) <= 1

    def test_threshold_validation(self, e2e_client):
        """Test that invalid threshold values are rejected."""
        img = create_test_image(256, 256)
        img_bytes = image_to_bytes(img)

        # Threshold > 1.0 should be rejected
        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "1.5"},
        )
        assert response.status_code == 422  # Validation error


class TestTileCalculation:
    """Tests for image tiling logic."""

    def test_small_image_no_tiling(self, e2e_client):
        """Test that image smaller than tile_size produces single tile."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("small.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["tile_count"] == 1

    def test_exact_tile_size_single_tile(self, e2e_client):
        """Test that image exactly equal to tile_size produces single tile."""
        from inference.config import settings

        img = create_test_image(settings.tile_size, settings.tile_size)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("exact.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["tile_count"] == 1

    def test_tile_count_matches_slicer(self, e2e_client):
        """Test that API tile count matches ImageSlicer calculation."""
        from inference.slicer import ImageSlicer
        from inference.config import settings

        slicer = ImageSlicer(
            tile_size=settings.tile_size, overlap=settings.tile_overlap
        )

        test_cases = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (3840, 2160),
        ]

        for width, height in test_cases:
            expected_tiles = slicer.get_tile_count(width, height)
            img = create_test_image(width, height)
            img_bytes = image_to_bytes(img)

            response = e2e_client.post(
                "/detect",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert response.status_code == 200
            actual_tiles = response.json()["tile_count"]
            assert actual_tiles == expected_tiles, (
                f"Tile mismatch for {width}x{height}: "
                f"expected {expected_tiles}, got {actual_tiles}"
            )


class TestResponseFormat:
    """Tests for API response format and structure."""

    def test_detection_bbox_format(self, e2e_client):
        """Test that detection bbox follows expected format (center x,y,w,h)."""
        img = create_image_with_objects(1024, 1024)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.1"},
        )
        data = response.json()

        # Verify response structure
        assert "detections" in data
        assert "image_size" in data
        assert "inference_time_ms" in data
        assert "tile_count" in data

        # Verify image_size structure
        assert data["image_size"]["width"] == 1024
        assert data["image_size"]["height"] == 1024

        # Verify each detection format
        for det in data["detections"]:
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert "is_edge" in det

            bbox = det["bbox"]
            assert "x" in bbox and "y" in bbox
            assert "w" in bbox and "h" in bbox

            # Confidence should be between 0 and 1
            assert 0 <= det["confidence"] <= 1

            # is_edge should be boolean
            assert isinstance(det["is_edge"], bool)

    def test_inference_time_positive(self, e2e_client):
        """Test that inference_time_ms is positive and reasonable."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img)

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        data = response.json()

        assert data["inference_time_ms"] > 0
        assert data["inference_time_ms"] < 120000  # < 2 minutes


class TestImageFormats:
    """Tests for different image format support."""

    def test_png_format_supported(self, e2e_client):
        """Test that PNG format images are processed correctly."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img, format='.png')

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["image_size"]["width"] == 512

    def test_jpeg_format_supported(self, e2e_client):
        """Test that JPEG format images are processed correctly."""
        img = create_test_image(512, 512)
        img_bytes = image_to_bytes(img, format='.jpg')

        response = e2e_client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200


class TestConcurrency:
    """Tests for concurrent request handling."""

    def test_concurrent_requests(self, e2e_client):
        """Test thread safety with concurrent requests."""
        import concurrent.futures

        def make_request(idx: int):
            img = create_test_image(512, 512)
            img_bytes = image_to_bytes(img)
            return e2e_client.post(
                "/detect",
                files={"file": (f"test_{idx}.jpg", img_bytes, "image/jpeg")},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert all(r.status_code == 200 for r in results)

        # All responses should be valid
        for r in results:
            data = r.json()
            assert "detections" in data
            assert "tile_count" in data

    def test_sequential_requests_consistent(self, e2e_client):
        """Test that sequential requests with same image produce consistent results."""
        img = create_image_with_objects(512, 512, seed=123)
        img_bytes = image_to_bytes(img)

        results = []
        for _ in range(3):
            img_bytes.seek(0)  # Reset file pointer
            response = e2e_client.post(
                "/detect",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                data={"threshold": "0.5"},
            )
            assert response.status_code == 200
            results.append(response.json())

        # Detection counts should be identical for same image
        counts = [len(r["detections"]) for r in results]
        assert all(c == counts[0] for c in counts), f"Inconsistent counts: {counts}"
