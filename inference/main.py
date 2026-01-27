"""FastAPI inference service for fiber optic closure detection."""

import time
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .postprocess import (
    Detection,
    containment_suppression,
    global_nms,
    mark_edge_boxes,
    restore_coordinates,
)
from .schemas import (
    BoundingBox,
    DetectionResult,
    HealthResponse,
    ImageSize,
    InferenceResponse,
)
from .slicer import ImageSlicer

# Global detector instance (lazy loaded)
_detector = None


def get_detector():
    """Get or create detector instance."""
    global _detector
    if _detector is None:
        from .detector import DarknetDetector
        _detector = DarknetDetector.get_instance()
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: optionally preload model
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Fiber Closure Detection API",
    description="4K drone image detection for fiber optic closures and cable racks",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from .detector import DarknetDetector

    network_size = None
    if DarknetDetector.is_loaded():
        detector = get_detector()
        w, h = detector.get_network_size()
        network_size = ImageSize(width=w, height=h)

    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=DarknetDetector.is_loaded(),
        network_size=network_size,
    )


@app.post("/detect", response_model=InferenceResponse)
async def detect(
    file: UploadFile = File(..., description="Image file to process"),
    threshold: float = Form(
        default=settings.confidence_threshold,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
    ),
):
    """
    Detect fiber optic closures and cable racks in an image.

    Uses SAHI-style high-overlap slicing for 4K images.
    """
    start_time = time.perf_counter()

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = image.shape[:2]

    # Get detector
    detector = get_detector()
    detector.thresh = threshold

    # Create slicer
    slicer = ImageSlicer(
        tile_size=settings.tile_size,
        overlap=settings.tile_overlap,
    )

    # Slice image
    tiles = slicer.slice(image)

    # Run detection on each tile
    all_detections: List[Detection] = []
    for tile in tiles:
        raw_dets = detector.detect(tile.image)
        dets = restore_coordinates(raw_dets, tile)
        dets = mark_edge_boxes(dets, tile, settings.tile_size, settings.edge_margin)
        all_detections.extend(dets)

    # Post-processing
    all_detections = containment_suppression(
        all_detections, settings.iom_threshold
    )
    all_detections = global_nms(
        all_detections, settings.global_nms_threshold
    )

    # Convert to response format
    results = [
        DetectionResult(
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=BoundingBox(x=det.x, y=det.y, w=det.w, h=det.h),
            is_edge=det.is_edge,
        )
        for det in all_detections
    ]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return InferenceResponse(
        detections=results,
        image_size=ImageSize(width=w, height=h),
        inference_time_ms=elapsed_ms,
        tile_count=len(tiles),
    )
