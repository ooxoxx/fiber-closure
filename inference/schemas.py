"""Pydantic models for API request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in global pixel coordinates (center format)."""

    x: float = Field(..., description="Center X coordinate")
    y: float = Field(..., description="Center Y coordinate")
    w: float = Field(..., description="Width")
    h: float = Field(..., description="Height")


class DetectionResult(BaseModel):
    """Single detection result."""

    class_name: str = Field(..., description="Detected class name")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage")
    bbox: BoundingBox = Field(..., description="Bounding box in global coordinates")
    is_edge: bool = Field(False, description="Whether detection is near tile edge")


class ImageSize(BaseModel):
    """Image dimensions."""

    width: int
    height: int


class InferenceResponse(BaseModel):
    """Response model for detection endpoint."""

    detections: List[DetectionResult]
    image_size: ImageSize
    inference_time_ms: float
    tile_count: int = Field(..., description="Number of tiles processed")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    model_config = {"protected_namespaces": ()}

    status: str
    version: str
    model_loaded: bool
    network_size: Optional[ImageSize] = None
