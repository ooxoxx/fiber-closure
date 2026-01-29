"""Tests for the postprocess module."""

import pytest

from inference.postprocess import (
    Detection,
    compute_iou,
    compute_iom,
    containment_suppression,
    global_nms,
    mark_edge_boxes,
    restore_coordinates,
)
from inference.slicer import Tile
import numpy as np


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

    def test_zero_area_box(self):
        """Test IoU with zero area box returns 0."""
        det1 = Detection("a", 90, 50, 50, 0, 0)
        det2 = Detection("a", 90, 50, 50, 20, 20)
        assert compute_iou(det1, det2) == 0.0


class TestIoM:
    """Test cases for IoM computation."""

    def test_contained_box(self):
        """Test IoM when small box is inside large box."""
        small = Detection("a", 90, 50, 50, 10, 10)
        large = Detection("a", 90, 50, 50, 100, 100)
        assert compute_iom(small, large) == 1.0

    def test_zero_area_small_box(self):
        """Test IoM with zero area small box returns 0."""
        small = Detection("a", 90, 50, 50, 0, 0)
        large = Detection("a", 90, 50, 50, 100, 100)
        assert compute_iom(small, large) == 0.0


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

    def test_suppresses_overlapping_same_class(self):
        """Test NMS suppresses overlapping boxes of same class."""
        det1 = Detection("a", 95, 50, 50, 20, 20)
        det2 = Detection("a", 80, 52, 52, 20, 20)  # overlapping, lower conf
        result = global_nms([det1, det2], 0.5)
        assert len(result) == 1
        assert result[0].confidence == 95

    def test_keeps_different_classes(self):
        """Test NMS keeps overlapping boxes of different classes."""
        det1 = Detection("a", 90, 50, 50, 20, 20)
        det2 = Detection("b", 90, 50, 50, 20, 20)  # same location, different class
        result = global_nms([det1, det2], 0.5)
        assert len(result) == 2


class TestContainmentSuppression:
    """Test cases for containment suppression."""

    def test_empty_list(self):
        """Test with empty list."""
        result = containment_suppression([], 0.7)
        assert result == []

    def test_single_detection(self):
        """Test with single detection."""
        det = Detection("a", 90, 50, 50, 20, 20)
        result = containment_suppression([det], 0.7)
        assert len(result) == 1

    def test_suppresses_contained_box(self):
        """Test that smaller contained box is suppressed."""
        large = Detection("a", 90, 50, 50, 100, 100)
        small = Detection("a", 85, 50, 50, 20, 20)  # contained in large
        result = containment_suppression([large, small], 0.7)
        assert len(result) == 1
        assert result[0].area() == large.area()

    def test_keeps_non_overlapping(self):
        """Test that non-overlapping boxes are kept."""
        det1 = Detection("a", 90, 0, 0, 20, 20)
        det2 = Detection("a", 85, 200, 200, 20, 20)
        result = containment_suppression([det1, det2], 0.7)
        assert len(result) == 2

    def test_edge_box_loses_to_non_edge(self):
        """Test that edge box is suppressed when contained in non-edge box."""
        large = Detection("a", 90, 50, 50, 100, 100, is_edge=False)
        small = Detection("a", 95, 50, 50, 20, 20, is_edge=True)
        result = containment_suppression([large, small], 0.7)
        assert len(result) == 1


class TestMarkEdgeBoxes:
    """Test cases for mark_edge_boxes."""

    def test_center_box_not_marked(self):
        """Test that box in center is not marked as edge."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        det = Detection("a", 90, 512, 512, 50, 50)  # center of tile
        result = mark_edge_boxes([det], tile, 1024, margin=2)
        assert result[0].is_edge is False

    def test_left_edge_box_marked(self):
        """Test that box near left edge is marked."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        det = Detection("a", 90, 10, 512, 20, 20)  # left edge: x-w/2 = 0
        result = mark_edge_boxes([det], tile, 1024, margin=2)
        assert result[0].is_edge is True

    def test_right_edge_box_marked(self):
        """Test that box near right edge is marked."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        det = Detection("a", 90, 1014, 512, 20, 20)  # right edge: x+w/2 = 1024
        result = mark_edge_boxes([det], tile, 1024, margin=2)
        assert result[0].is_edge is True

    def test_top_edge_box_marked(self):
        """Test that box near top edge is marked."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        det = Detection("a", 90, 512, 10, 20, 20)  # top edge
        result = mark_edge_boxes([det], tile, 1024, margin=2)
        assert result[0].is_edge is True

    def test_bottom_edge_box_marked(self):
        """Test that box near bottom edge is marked."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        det = Detection("a", 90, 512, 1014, 20, 20)  # bottom edge
        result = mark_edge_boxes([det], tile, 1024, margin=2)
        assert result[0].is_edge is True


class TestRestoreCoordinates:
    """Test cases for restore_coordinates."""

    def test_restores_offset(self):
        """Test that tile offset is added to coordinates."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=100, y=200, index=0)
        raw_dets = [("class_a", "95.5", (50, 60, 20, 30))]
        result = restore_coordinates(raw_dets, tile)
        assert len(result) == 1
        assert result[0].x == 150  # 50 + 100
        assert result[0].y == 260  # 60 + 200
        assert result[0].w == 20
        assert result[0].h == 30

    def test_preserves_class_and_confidence(self):
        """Test that class name and confidence are preserved."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=0, y=0, index=0)
        raw_dets = [("fiber_closure", "87.3", (100, 100, 50, 50))]
        result = restore_coordinates(raw_dets, tile)
        assert result[0].class_name == "fiber_closure"
        assert result[0].confidence == 87.3

    def test_multiple_detections(self):
        """Test with multiple detections."""
        tile = Tile(image=np.zeros((1024, 1024, 3)), x=500, y=500, index=0)
        raw_dets = [
            ("a", "90", (10, 10, 5, 5)),
            ("b", "80", (20, 20, 10, 10)),
        ]
        result = restore_coordinates(raw_dets, tile)
        assert len(result) == 2
        assert result[0].x == 510
        assert result[1].x == 520
