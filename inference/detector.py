"""Darknet detector wrapper with thread-safe inference."""

import os
import sys
import threading
from ctypes import CDLL, RTLD_GLOBAL
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import settings


class DarknetDetector:
    """Thread-safe wrapper for Darknet inference."""

    _instance: Optional["DarknetDetector"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        data_path: Path,
        thresh: float = 0.5,
        nms: float = 0.45,
        hier_thresh: float = 0.5,
    ):
        """
        Initialize the detector.

        Args:
            cfg_path: Path to .cfg file.
            weights_path: Path to .weights file.
            data_path: Path to .data file.
            thresh: Detection confidence threshold.
            nms: NMS threshold.
            hier_thresh: Hierarchical threshold.
        """
        self.thresh = thresh
        self.nms = nms
        self.hier_thresh = hier_thresh
        self._inference_lock = threading.Lock()

        # Validate model files exist
        self._validate_model_files(cfg_path, weights_path, data_path)

        # Load darknet library
        self._load_darknet_lib()

        # Load network
        self.network = self._load_network(cfg_path, weights_path)
        self.class_names = self._load_class_names(data_path)
        self.network_width, self.network_height = self._get_network_size()

    def _validate_model_files(
        self, cfg_path: Path, weights_path: Path, data_path: Path
    ) -> None:
        """Validate that all required model files exist."""
        if not Path(cfg_path).exists():
            raise FileNotFoundError(
                f"Model cfg file not found: {cfg_path}"
            )
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model weights file not found: {weights_path}"
            )
        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"Model data file not found: {data_path}"
            )

    def _load_darknet_lib(self) -> None:
        """Load the darknet shared library."""
        lib_path = settings.darknet_lib_path

        # Add darknet python path to sys.path
        darknet_python_path = Path(__file__).parent.parent / "darknet" / "src-python"
        if darknet_python_path.exists():
            sys.path.insert(0, str(darknet_python_path))

        # Set library path for ctypes
        os.environ["DARKNET_LIB_PATH"] = lib_path

        # Import darknet module with error handling
        try:
            import darknet as dn
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import darknet module: {e}. "
                "Ensure darknet Python bindings are available."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Failed to load libdarknet shared library: {e}. "
                f"Check that DARKNET_LIB_PATH ({lib_path}) points to a valid library."
            ) from e

        self.dn = dn

    def _load_network(self, cfg_path: Path, weights_path: Path):
        """Load the neural network."""
        return self.dn.load_network(
            str(cfg_path),
            None,  # data_file not needed for load_network
            str(weights_path),
            batch_size=1,
        )

    def _load_class_names(self, data_path: Path) -> List[str]:
        """Load class names from .data file."""
        names_path = None
        with open(data_path, "r") as f:
            for line in f:
                if line.startswith("names"):
                    names_path = line.split("=")[1].strip()
                    break

        if names_path is None:
            raise ValueError(f"No 'names' entry found in {data_path}")

        # Handle relative paths
        if not Path(names_path).is_absolute():
            names_path = data_path.parent / names_path

        with open(names_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _get_network_size(self) -> Tuple[int, int]:
        """Get network input dimensions."""
        w = self.dn.network_width(self.network)
        h = self.dn.network_height(self.network)
        return w, h

    def _array_to_image(self, arr: np.ndarray):
        """Convert numpy array to darknet IMAGE."""
        arr = arr.astype(np.float32) / 255.0
        h, w, c = arr.shape
        arr = arr.transpose(2, 0, 1).flatten()
        image = self.dn.make_image(w, h, c)
        self.dn.copy_image_from_bytes(image, arr.tobytes())
        return image

    def detect(
        self, image: np.ndarray
    ) -> List[Tuple[str, str, Tuple[float, float, float, float]]]:
        """
        Run detection on an image.

        Args:
            image: BGR image as numpy array.

        Returns:
            List of (class_name, confidence, (x, y, w, h)) tuples.
        """
        # Resize to network size
        resized = cv2.resize(image, (self.network_width, self.network_height))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        with self._inference_lock:
            darknet_image = self._array_to_image(rgb)
            detections = self.dn.detect_image(
                self.network,
                self.class_names,
                darknet_image,
                thresh=self.thresh,
                hier_thresh=self.hier_thresh,
                nms=self.nms,
            )
            self.dn.free_image(darknet_image)

        # Scale coordinates back to original image size
        h, w = image.shape[:2]
        scale_x = w / self.network_width
        scale_y = h / self.network_height

        scaled_detections = []
        for class_name, confidence, bbox in detections:
            x, y, bw, bh = bbox
            scaled_detections.append((
                class_name,
                confidence,
                (x * scale_x, y * scale_y, bw * scale_x, bh * scale_y),
            ))

        return scaled_detections

    def get_network_size(self) -> Tuple[int, int]:
        """Return network input dimensions."""
        return self.network_width, self.network_height

    @classmethod
    def get_instance(cls) -> "DarknetDetector":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cfg_path=settings.model_cfg,
                        weights_path=settings.model_weights,
                        data_path=settings.model_data,
                        thresh=settings.confidence_threshold,
                        nms=settings.nms_threshold,
                        hier_thresh=settings.hier_threshold,
                    )
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._instance is not None
