"""Tests for area calculation."""

import numpy as np
import pytest

from vana.analysis.area import AreaCalculator


class TestAreaCalculator:
    def test_pixel_area_default(self):
        calc = AreaCalculator(resolution_m=10.0)
        assert calc.pixel_area_m2 == pytest.approx(100.0)
        assert calc.pixel_area_hectares == pytest.approx(0.01)

    def test_pixels_to_hectares(self):
        calc = AreaCalculator(resolution_m=10.0)
        assert calc.pixels_to_hectares(100) == pytest.approx(1.0)

    def test_mask_to_hectares(self):
        calc = AreaCalculator(resolution_m=10.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True  # 25 pixels
        assert calc.mask_to_hectares(mask) == pytest.approx(0.25)

    def test_loss_summary(self):
        calc = AreaCalculator(resolution_m=10.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        summary = calc.loss_summary(mask)
        assert summary["affected_pixels"] == 1
        assert summary["total_pixels"] == 100
        assert summary["affected_hectares"] == pytest.approx(0.01)
        assert summary["percent_affected"] == pytest.approx(1.0)

    def test_different_resolution(self):
        calc = AreaCalculator(resolution_m=30.0)
        assert calc.pixel_area_m2 == pytest.approx(900.0)
        assert calc.pixels_to_hectares(1) == pytest.approx(0.09)
