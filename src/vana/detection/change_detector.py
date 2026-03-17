"""Change detection by comparing satellite images over time using NDVI."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from vana.analysis.ndvi import compute_ndvi, ndvi_difference
from vana.analysis.area import AreaCalculator
from vana.models import SatelliteImage, DeforestationEvent


class ChangeDetector:
    """Detect deforestation by comparing NDVI between two satellite images.

    A pixel is flagged as deforested when:
      1. The NDVI drop exceeds ``ndvi_threshold`` (default -0.2).
      2. The earlier NDVI was above ``min_forest_ndvi`` (i.e., it was forest).

    Args:
        ndvi_threshold: Maximum allowed NDVI decrease before flagging
            (negative value; more negative = stricter).
        min_forest_ndvi: Minimum NDVI in the earlier image for a pixel
            to be considered forest.
    """

    def __init__(
        self,
        ndvi_threshold: float = -0.2,
        min_forest_ndvi: float = 0.4,
    ) -> None:
        self.ndvi_threshold = ndvi_threshold
        self.min_forest_ndvi = min_forest_ndvi

    def detect(
        self,
        before: SatelliteImage,
        after: SatelliteImage,
    ) -> DeforestationEvent:
        """Compare two images and return a DeforestationEvent.

        Args:
            before: Earlier satellite image.
            after: Later satellite image.

        Returns:
            A DeforestationEvent describing detected forest loss.
        """
        ndvi_before = compute_ndvi(before.nir, before.red)
        ndvi_after = compute_ndvi(after.nir, after.red)
        diff = ndvi_difference(ndvi_before, ndvi_after)

        # Build deforestation mask.
        was_forest = ndvi_before >= self.min_forest_ndvi
        significant_drop = diff <= self.ndvi_threshold
        change_mask = was_forest & significant_drop

        affected_pixels = int(np.sum(change_mask))
        calculator = AreaCalculator(resolution_m=before.resolution_m)
        hectares_lost = calculator.pixels_to_hectares(affected_pixels)

        mean_drop = float(np.mean(diff[change_mask])) if affected_pixels > 0 else 0.0

        return DeforestationEvent(
            region_id=before.region_id,
            start_date=before.timestamp,
            end_date=after.timestamp,
            hectares_lost=hectares_lost,
            mean_ndvi_drop=mean_drop,
            affected_pixels=affected_pixels,
            change_mask=change_mask,
        )

    def detect_series(
        self,
        images: list[SatelliteImage],
    ) -> list[DeforestationEvent]:
        """Detect deforestation across a time series of images.

        Compares each consecutive pair of images.

        Args:
            images: Chronologically ordered satellite images for one region.

        Returns:
            List of DeforestationEvent objects, one per consecutive pair.
        """
        events: list[DeforestationEvent] = []
        for i in range(len(images) - 1):
            events.append(self.detect(images[i], images[i + 1]))
        return events
