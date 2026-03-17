"""Tests for change detection."""

from datetime import datetime

import numpy as np
import pytest

from vana.detection.change_detector import ChangeDetector
from vana.models import SatelliteImage


def _make_image(
    red: np.ndarray, nir: np.ndarray, region_id: str = "test", ts: datetime | None = None
) -> SatelliteImage:
    return SatelliteImage(
        region_id=region_id,
        timestamp=ts or datetime(2025, 1, 1),
        red=red,
        nir=nir,
        resolution_m=10.0,
    )


class TestChangeDetector:
    def test_no_change(self):
        """Identical images should detect zero deforestation."""
        red = np.full((10, 10), 0.05)
        nir = np.full((10, 10), 0.5)
        before = _make_image(red, nir, ts=datetime(2025, 1, 1))
        after = _make_image(red, nir, ts=datetime(2025, 2, 1))

        detector = ChangeDetector()
        event = detector.detect(before, after)
        assert event.affected_pixels == 0
        assert event.hectares_lost == 0.0

    def test_full_deforestation(self):
        """Clearing all forest pixels should be detected."""
        red_before = np.full((10, 10), 0.05)
        nir_before = np.full((10, 10), 0.5)  # NDVI ~ 0.82 (forest)
        red_after = np.full((10, 10), 0.2)
        nir_after = np.full((10, 10), 0.22)  # NDVI ~ 0.05 (cleared)

        before = _make_image(red_before, nir_before, ts=datetime(2025, 1, 1))
        after = _make_image(red_after, nir_after, ts=datetime(2025, 2, 1))

        detector = ChangeDetector()
        event = detector.detect(before, after)

        assert event.affected_pixels == 100
        assert event.hectares_lost == pytest.approx(1.0)
        assert event.mean_ndvi_drop < -0.5
        assert event.change_mask is not None
        assert event.change_mask.shape == (10, 10)

    def test_partial_deforestation(self):
        """Only forest pixels that drop significantly should be flagged."""
        h, w = 10, 10
        red_before = np.full((h, w), 0.05)
        nir_before = np.full((h, w), 0.5)

        # Clear only the top half.
        red_after = red_before.copy()
        nir_after = nir_before.copy()
        red_after[:5, :] = 0.2
        nir_after[:5, :] = 0.22

        before = _make_image(red_before, nir_before, ts=datetime(2025, 1, 1))
        after = _make_image(red_after, nir_after, ts=datetime(2025, 2, 1))

        detector = ChangeDetector()
        event = detector.detect(before, after)

        assert event.affected_pixels == 50
        assert event.hectares_lost == pytest.approx(0.5)

    def test_detect_series(self):
        """Series detection should return n-1 events for n images."""
        red = np.full((5, 5), 0.05)
        nir = np.full((5, 5), 0.5)
        images = [
            _make_image(red, nir, ts=datetime(2025, i, 1)) for i in range(1, 5)
        ]
        detector = ChangeDetector()
        events = detector.detect_series(images)
        assert len(events) == 3

    def test_water_not_flagged(self):
        """Water pixels (low NDVI) should not be flagged as deforestation."""
        red_before = np.full((5, 5), 0.08)
        nir_before = np.full((5, 5), 0.02)  # NDVI ~ -0.6 (water)
        red_after = np.full((5, 5), 0.1)
        nir_after = np.full((5, 5), 0.01)

        before = _make_image(red_before, nir_before, ts=datetime(2025, 1, 1))
        after = _make_image(red_after, nir_after, ts=datetime(2025, 2, 1))

        detector = ChangeDetector()
        event = detector.detect(before, after)
        assert event.affected_pixels == 0
