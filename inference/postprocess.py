"""Post-processing utilities for detection results."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .slicer import Tile


@dataclass
class Detection:
    """Internal detection representation."""

    class_name: str
    confidence: float  # 0-100 percentage
    x: float  # Center X in global coordinates
    y: float  # Center Y in global coordinates
    w: float  # Width
    h: float  # Height
    is_edge: bool = False  # Near tile edge flag

    def area(self) -> float:
        """Calculate bounding box area."""
        return self.w * self.h

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        return x1, y1, x2, y2


def restore_coordinates(
    detections: List[Tuple[str, str, Tuple[float, float, float, float]]],
    tile: Tile,
) -> List[Detection]:
    """
    Convert tile-local coordinates to global image coordinates.

    Args:
        detections: List of (class_name, confidence, (x, y, w, h)) from darknet.
        tile: The tile these detections came from.

    Returns:
        List of Detection objects with global coordinates.
    """
    results = []
    for class_name, confidence, bbox in detections:
        x, y, w, h = bbox
        # Add tile offset to center coordinates
        global_x = x + tile.x
        global_y = y + tile.y
        results.append(
            Detection(
                class_name=class_name,
                confidence=float(confidence),
                x=global_x,
                y=global_y,
                w=w,
                h=h,
            )
        )
    return results


def mark_edge_boxes(
    detections: List[Detection],
    tile: Tile,
    tile_size: int,
    margin: int = 2,
) -> List[Detection]:
    """
    Mark detections that are near tile edges as low-quality.

    Args:
        detections: List of detections with global coordinates.
        tile: The tile these detections came from.
        tile_size: Size of the tile.
        margin: Distance from edge to consider as "near edge".

    Returns:
        List of detections with is_edge flag set appropriately.
    """
    for det in detections:
        # Convert to tile-local coordinates
        local_x = det.x - tile.x
        local_y = det.y - tile.y

        # Calculate box edges in local coordinates
        left = local_x - det.w / 2
        right = local_x + det.w / 2
        top = local_y - det.h / 2
        bottom = local_y + det.h / 2

        # Check if any edge is within margin of tile boundary
        near_left = left < margin
        near_right = right > tile_size - margin
        near_top = top < margin
        near_bottom = bottom > tile_size - margin

        det.is_edge = near_left or near_right or near_top or near_bottom

    return detections


def compute_iou(det1: Detection, det2: Detection) -> float:
    """Compute Intersection over Union between two detections."""
    x1_1, y1_1, x2_1, y2_1 = det1.to_xyxy()
    x1_2, y1_2, x2_2, y2_2 = det2.to_xyxy()

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = det1.area()
    area2 = det2.area()
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_iom(small: Detection, large: Detection) -> float:
    """Compute Intersection over Minimum (smaller box area)."""
    x1_s, y1_s, x2_s, y2_s = small.to_xyxy()
    x1_l, y1_l, x2_l, y2_l = large.to_xyxy()

    xi1 = max(x1_s, x1_l)
    yi1 = max(y1_s, y1_l)
    xi2 = min(x2_s, x2_l)
    yi2 = min(y2_s, y2_l)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    small_area = small.area()
    if small_area <= 0:
        return 0.0
    return inter_area / small_area


def containment_suppression(
    detections: List[Detection],
    iom_thresh: float = 0.7,
) -> List[Detection]:
    """
    Suppress smaller boxes that are largely contained within larger boxes.

    When IoM > threshold, the larger non-edge box wins.

    Args:
        detections: List of detections.
        iom_thresh: IoM threshold for suppression.

    Returns:
        Filtered list of detections.
    """
    if len(detections) <= 1:
        return detections

    # Sort by area descending
    sorted_dets = sorted(detections, key=lambda d: d.area(), reverse=True)
    keep = []
    suppressed = set()

    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue

        keep.append(det_i)

        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue

            det_j = sorted_dets[j]

            # Only compare same class
            if det_i.class_name != det_j.class_name:
                continue

            iom = compute_iom(det_j, det_i)  # smaller over larger
            if iom > iom_thresh:
                # Larger box wins if it's not an edge box
                # or if smaller is also an edge box
                if not det_i.is_edge or det_j.is_edge:
                    suppressed.add(j)

    return keep


def global_nms(
    detections: List[Detection],
    iou_thresh: float = 0.5,
) -> List[Detection]:
    """
    Apply global NMS to merge duplicate detections across tiles.

    Args:
        detections: List of detections.
        iou_thresh: IoU threshold for suppression.

    Returns:
        Filtered list of detections.
    """
    if len(detections) <= 1:
        return detections

    # Group by class
    by_class: dict[str, List[Detection]] = {}
    for det in detections:
        if det.class_name not in by_class:
            by_class[det.class_name] = []
        by_class[det.class_name].append(det)

    results = []
    for class_name, class_dets in by_class.items():
        # Sort by confidence descending
        sorted_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, det_i in enumerate(sorted_dets):
            if i in suppressed:
                continue
            keep.append(det_i)

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue
                det_j = sorted_dets[j]
                iou = compute_iou(det_i, det_j)
                if iou > iou_thresh:
                    suppressed.add(j)

        results.extend(keep)

    return results
