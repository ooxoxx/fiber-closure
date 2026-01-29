"""Configuration management for inference service."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model paths
    model_cfg: Path = Path("cfg/fiber-tiny.cfg")
    model_weights: Path = Path("weights/fiber-tiny_best.weights")
    model_data: Path = Path("cfg/fiber.data")

    # Detection thresholds
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    hier_threshold: float = 0.5

    # Slicer settings
    tile_size: int = 928
    tile_overlap: float = 0.3  # 30% overlap

    # Post-processing
    edge_margin: int = 2  # pixels from edge to mark as low-quality
    iom_threshold: float = 0.7  # IoM threshold for containment suppression
    global_nms_threshold: float = 0.5

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1

    # Darknet library path
    darknet_lib_path: str = os.environ.get(
        "DARKNET_LIB_PATH", "libdarknet.so"
    )

    model_config = {
        "env_prefix": "FIBER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }


settings = Settings()
