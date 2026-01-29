"""Integration tests using real darknet library.

These tests require the actual darknet library and model files to be available.
They are marked with pytest.mark.integration and can be skipped in CI if needed.
"""

import pytest
import numpy as np
from pathlib import Path

from tests.conftest import PROJECT_ROOT


# Skip all tests in this module if darknet is not available
pytestmark = pytest.mark.integration


def darknet_available() -> bool:
    """Check if darknet library and model files are available."""
    try:
        cfg_path = PROJECT_ROOT / "cfg" / "fiber-tiny.cfg"
        weights_path = PROJECT_ROOT / "weights" / "fiber-tiny_best.weights"
        data_path = PROJECT_ROOT / "cfg" / "fiber.data"
        return cfg_path.exists() and weights_path.exists() and data_path.exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def detector():
    """Load real detector (module-level, only load once)."""
    if not darknet_available():
        pytest.skip("Darknet model files not available")

    from inference.detector import DarknetDetector

    # Reset singleton for clean test
    DarknetDetector._instance = None

    try:
        det = DarknetDetector(
            cfg_path=PROJECT_ROOT / "cfg" / "fiber-tiny.cfg",
            weights_path=PROJECT_ROOT / "weights" / "fiber-tiny_best.weights",
            data_path=PROJECT_ROOT / "cfg" / "fiber.data",
            thresh=0.25,
        )
        yield det
    except RuntimeError as e:
        pytest.skip(f"Failed to load darknet: {e}")
    finally:
        DarknetDetector._instance = None


class TestDarknetIntegration:
    """Integration tests for DarknetDetector with real library."""

    def test_load_network_success(self, detector):
        """Test that network loads successfully."""
        assert detector is not None
        assert detector.network is not None

    def test_network_dimensions(self, detector):
        """Test network input dimensions are correct."""
        w, h = detector.get_network_size()
        assert w > 0
        assert h > 0
        # fiber-tiny uses 416x416 or 608x608
        assert w in [416, 608, 1024]
        assert h in [416, 608, 1024]

    def test_class_names_loaded(self, detector):
        """Test that class names are loaded correctly."""
        assert len(detector.class_names) == 5
        assert "closure_normal" in detector.class_names
        assert "closure_dropped" in detector.class_names

    def test_detect_returns_list(self, detector):
        """Test that detect returns a list."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert isinstance(result, list)

    def test_detect_output_format(self, detector):
        """Test detection output format is correct."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :] = [128, 128, 128]
        result = detector.detect(image)

        # Even if no detections, format should be correct
        for det in result:
            assert len(det) == 3  # (class_name, confidence, bbox)
            class_name, confidence, bbox = det
            assert isinstance(class_name, str)
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4  # (x, y, w, h)

    def test_detect_with_real_image_size(self, detector):
        """Test detection with various image sizes."""
        for size in [(416, 416), (1024, 1024), (800, 600)]:
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            image[:, :] = [100, 100, 100]
            result = detector.detect(image)
            assert isinstance(result, list)

    def test_threshold_affects_results(self, detector):
        """Test that threshold parameter works."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Store original threshold
        original_thresh = detector.thresh

        # High threshold should give fewer or equal detections
        detector.thresh = 0.9
        high_thresh_result = detector.detect(image)

        detector.thresh = 0.1
        low_thresh_result = detector.detect(image)

        # Restore original
        detector.thresh = original_thresh

        assert len(high_thresh_result) <= len(low_thresh_result)
