"""Image slicer for SAHI-style high-overlap tiling."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Tile:
    """A tile extracted from a larger image."""

    image: np.ndarray
    x: int  # Global X offset (top-left corner)
    y: int  # Global Y offset (top-left corner)
    index: int  # Tile index for tracking


class ImageSlicer:
    """Slice large images into overlapping tiles for inference."""

    def __init__(self, tile_size: int = 928, overlap: float = 0.3):
        """
        Initialize the slicer.

        Args:
            tile_size: Size of each tile (square).
            overlap: Overlap ratio between tiles (0.3 = 30%).
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.step = int(tile_size * (1 - overlap))

    def slice(self, image: np.ndarray) -> List[Tile]:
        """
        Slice an image into overlapping tiles.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            List of Tile objects with image data and global offsets.
        """
        h, w = image.shape[:2]
        tiles = []
        index = 0

        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate actual tile boundaries
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)

                # Extract tile
                tile_img = image[y:y_end, x:x_end]

                # Pad if necessary (for edge tiles)
                if tile_img.shape[0] < self.tile_size or tile_img.shape[1] < self.tile_size:
                    padded = np.zeros(
                        (self.tile_size, self.tile_size, image.shape[2]),
                        dtype=image.dtype,
                    )
                    padded[: tile_img.shape[0], : tile_img.shape[1]] = tile_img
                    tile_img = padded

                tiles.append(Tile(image=tile_img, x=x, y=y, index=index))
                index += 1

                # Move to next column
                if x + self.tile_size >= w:
                    break
                x += self.step

            # Move to next row
            if y + self.tile_size >= h:
                break
            y += self.step

        return tiles

    def get_tile_count(self, width: int, height: int) -> int:
        """Calculate the number of tiles for a given image size."""
        # If image is smaller than or equal to tile size, only 1 tile needed
        if width <= self.tile_size and height <= self.tile_size:
            return 1

        # Calculate columns (matching slice() logic)
        if width <= self.tile_size:
            cols = 1
        else:
            cols = 1
            x = self.step
            while x + self.tile_size < width:
                cols += 1
                x += self.step
            # Add one more if there's remaining space
            if x < width:
                cols += 1

        # Calculate rows (matching slice() logic)
        if height <= self.tile_size:
            rows = 1
        else:
            rows = 1
            y = self.step
            while y + self.tile_size < height:
                rows += 1
                y += self.step
            # Add one more if there's remaining space
            if y < height:
                rows += 1

        return cols * rows
