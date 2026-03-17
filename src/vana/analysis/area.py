"""Area calculation: convert pixel counts to hectares of forest loss."""

from __future__ import annotations

import numpy as np


class AreaCalculator:
    """Convert pixel-level detections into real-world area measurements.

    Args:
        resolution_m: Spatial resolution in meters per pixel side.
    """

    def __init__(self, resolution_m: float = 10.0) -> None:
        self.resolution_m = resolution_m

    @property
    def pixel_area_m2(self) -> float:
        """Area of a single pixel in square meters."""
        return self.resolution_m ** 2

    @property
    def pixel_area_hectares(self) -> float:
        """Area of a single pixel in hectares (1 hectare = 10,000 m^2)."""
        return self.pixel_area_m2 / 10_000.0

    def pixels_to_hectares(self, pixel_count: int) -> float:
        """Convert a number of pixels to hectares.

        Args:
            pixel_count: Number of affected pixels.

        Returns:
            Area in hectares.
        """
        return pixel_count * self.pixel_area_hectares

    def mask_to_hectares(self, mask: np.ndarray) -> float:
        """Compute total area in hectares from a boolean mask.

        Args:
            mask: 2-D boolean array where True indicates affected pixels.

        Returns:
            Total affected area in hectares.
        """
        return int(np.sum(mask)) * self.pixel_area_hectares

    def loss_summary(self, mask: np.ndarray) -> dict:
        """Return a summary dict with pixel count, hectares, and percentage.

        Args:
            mask: 2-D boolean array of affected pixels.

        Returns:
            Dictionary with keys: affected_pixels, total_pixels,
            affected_hectares, total_hectares, percent_affected.
        """
        affected = int(np.sum(mask))
        total = mask.size
        return {
            "affected_pixels": affected,
            "total_pixels": total,
            "affected_hectares": affected * self.pixel_area_hectares,
            "total_hectares": total * self.pixel_area_hectares,
            "percent_affected": (affected / total * 100.0) if total > 0 else 0.0,
        }
