"""Tests for NDVI calculation."""

import numpy as np
import pytest

from vana.analysis.ndvi import compute_ndvi, ndvi_difference


class TestComputeNDVI:
    def test_perfect_vegetation(self):
        """NIR >> Red should give NDVI close to 1."""
        nir = np.array([[0.9]])
        red = np.array([[0.1]])
        ndvi = compute_ndvi(nir, red)
        assert ndvi[0, 0] == pytest.approx(0.8, abs=1e-10)

    def test_water_body(self):
        """Red > NIR should give negative NDVI."""
        nir = np.array([[0.02]])
        red = np.array([[0.08]])
        ndvi = compute_ndvi(nir, red)
        assert ndvi[0, 0] < 0

    def test_zero_bands(self):
        """Both bands zero should return 0, not NaN."""
        nir = np.zeros((3, 3))
        red = np.zeros((3, 3))
        ndvi = compute_ndvi(nir, red)
        assert np.all(ndvi == 0.0)

    def test_clipping(self):
        """Values should be clipped to [-1, 1]."""
        ndvi = compute_ndvi(np.array([[1.0]]), np.array([[0.0]]))
        assert ndvi[0, 0] <= 1.0
        ndvi = compute_ndvi(np.array([[0.0]]), np.array([[1.0]]))
        assert ndvi[0, 0] >= -1.0

    def test_shape_preserved(self):
        nir = np.random.rand(10, 15)
        red = np.random.rand(10, 15)
        ndvi = compute_ndvi(nir, red)
        assert ndvi.shape == (10, 15)


class TestNDVIDifference:
    def test_no_change(self):
        a = np.array([[0.6, 0.3]])
        diff = ndvi_difference(a, a)
        np.testing.assert_array_almost_equal(diff, 0.0)

    def test_loss_is_negative(self):
        before = np.array([[0.8]])
        after = np.array([[0.2]])
        diff = ndvi_difference(before, after)
        assert diff[0, 0] < 0

    def test_growth_is_positive(self):
        before = np.array([[0.2]])
        after = np.array([[0.7]])
        diff = ndvi_difference(before, after)
        assert diff[0, 0] > 0
