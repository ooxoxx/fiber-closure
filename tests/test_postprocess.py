"""Tests for the postprocess module."""

import pytest

from inference.postprocess import (
    Detection,
    compute_iou,
    compute_iom,
    containment_suppression,
    global_nms,
)


class TestDetection:
    """Test cases for Detection dataclass."""

    def test_area(self):
        """Test area calculation."""
        det = Detection("test", 90.0, 100, 100, 50, 30)
        assert det.area() == 1500

    def test_to_xyxy(self):
        """Test conversion to xyxy format."""
        det = Detection("test", 90.0, 100, 100, 50, 30)
        x1, y1, x2, y2 = det.to_xyxy()
        assert x1 == 75
        assert y1 == 85
        assert x2 == 125
        assert y2 == 115


class TestIoU:
    """Test cases for IoU computation."""

    def test_no_overlap(self):
        """Test IoU with no overlap."""
        det1 = Detection("a", 90, 0, 0, 10, 10)
        det2 = Detection("a", 90, 100, 100, 10, 10)
        assert compute_iou(det1, det2) == 0.0

    def test_full_overlap(self):
        """Test IoU with identical boxes."""
        det1 = Detection("a", 90, 50, 50, 20, 20)
        det2 = Detection("a", 90, 50, 50, 20, 20)
        assert compute_iou(det1, det2) == 1.0


class TestIoM:
    """Test cases for IoM computation."""

    def test_contained_box(self):
        """Test IoM when small box is inside large box."""
        small = Detection("a", 90, 50, 50, 10, 10)
        large = Detection("a", 90, 50, 50, 100, 100)
        assert compute_iom(small, large) == 1.0


class TestGlobalNMS:
    """Test cases for global NMS."""

    def test_empty_list(self):
        """Test NMS with empty list."""
        result = global_nms([], 0.5)
        assert result == []

    def test_single_detection(self):
        """Test NMS with single detection."""
        det = Detection("a", 90, 50, 50, 20, 20)
        result = global_nms([det], 0.5)
        assert len(result) == 1
