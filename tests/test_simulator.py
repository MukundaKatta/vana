"""Tests for the synthetic data simulator."""

from datetime import datetime

import numpy as np
import pytest

from vana.simulator import Simulator
from vana.analysis.ndvi import compute_ndvi


class TestSimulator:
    def test_generate_time_series(self):
        sim = Simulator(height=32, width=32, seed=0)
        region, images = sim.generate_time_series(n_steps=4)

        assert region.region_id == "region_0"
        assert len(images) == 4
        for img in images:
            assert img.red.shape == (32, 32)
            assert img.nir.shape == (32, 32)
            assert np.all(img.red >= 0) and np.all(img.red <= 1)
            assert np.all(img.nir >= 0) and np.all(img.nir <= 1)

    def test_deforestation_reduces_ndvi(self):
        """Later images should have lower average NDVI due to deforestation."""
        sim = Simulator(height=64, width=64, seed=1)
        _, images = sim.generate_time_series(n_steps=5, deforestation_rate=0.1)

        ndvi_first = compute_ndvi(images[0].nir, images[0].red)
        ndvi_last = compute_ndvi(images[-1].nir, images[-1].red)

        assert np.mean(ndvi_last) < np.mean(ndvi_first)

    def test_timestamps_increase(self):
        sim = Simulator(seed=2)
        _, images = sim.generate_time_series(n_steps=6)
        for i in range(len(images) - 1):
            assert images[i].timestamp < images[i + 1].timestamp

    def test_custom_region_id(self):
        sim = Simulator(seed=3)
        region, images = sim.generate_time_series(region_id="amazon_west")
        assert region.region_id == "amazon_west"
        assert all(img.region_id == "amazon_west" for img in images)
