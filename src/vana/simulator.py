"""Generate synthetic satellite data for testing."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from vana.models import SatelliteImage, Region


class Simulator:
    """Generate synthetic multispectral satellite images with realistic land cover.

    The simulator creates a base image with forest, water, and urban zones,
    then progressively clears forest pixels over time to simulate deforestation.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        resolution_m: Spatial resolution in meters per pixel.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        resolution_m: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.height = height
        self.width = width
        self.resolution_m = resolution_m
        self.rng = np.random.default_rng(seed)

    def _generate_base(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a base scene with forest, water, and urban zones.

        Returns:
            (red, nir, zone_map) where zone_map encodes:
              0 = water, 1 = urban, 2 = forest.
        """
        h, w = self.height, self.width
        zone_map = np.full((h, w), 2, dtype=np.int32)  # default forest

        # Place a water body (elliptical blob).
        cy, cx = h // 4, w // 4
        yy, xx = np.ogrid[:h, :w]
        water_mask = ((yy - cy) ** 2 / (h * 0.05) ** 2 + (xx - cx) ** 2 / (w * 0.08) ** 2) < 1
        zone_map[water_mask] = 0

        # Place an urban area (rectangular block).
        zone_map[int(h * 0.7):int(h * 0.85), int(w * 0.6):int(w * 0.85)] = 1

        red = np.zeros((h, w), dtype=np.float64)
        nir = np.zeros((h, w), dtype=np.float64)

        # Water: high red absorption, low NIR.
        water = zone_map == 0
        red[water] = self.rng.uniform(0.02, 0.06, size=int(np.sum(water)))
        nir[water] = self.rng.uniform(0.01, 0.04, size=int(np.sum(water)))

        # Urban: moderate both.
        urban = zone_map == 1
        red[urban] = self.rng.uniform(0.15, 0.25, size=int(np.sum(urban)))
        nir[urban] = self.rng.uniform(0.18, 0.28, size=int(np.sum(urban)))

        # Forest: low red, high NIR (high NDVI).
        forest = zone_map == 2
        red[forest] = self.rng.uniform(0.03, 0.08, size=int(np.sum(forest)))
        nir[forest] = self.rng.uniform(0.35, 0.55, size=int(np.sum(forest)))

        return red, nir, zone_map

    def _deforest(
        self,
        red: np.ndarray,
        nir: np.ndarray,
        zone_map: np.ndarray,
        fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clear a fraction of forest pixels, simulating deforestation.

        Cleared pixels get spectral signatures similar to bare soil.
        """
        red = red.copy()
        nir = nir.copy()
        forest_pixels = np.argwhere(zone_map == 2)
        n_clear = max(1, int(len(forest_pixels) * fraction))
        chosen = self.rng.choice(len(forest_pixels), size=n_clear, replace=False)
        for idx in chosen:
            r, c = forest_pixels[idx]
            red[r, c] = self.rng.uniform(0.15, 0.25)
            nir[r, c] = self.rng.uniform(0.18, 0.28)
            zone_map[r, c] = 3  # mark as cleared
        return red, nir

    def generate_time_series(
        self,
        region_id: str = "region_0",
        n_steps: int = 6,
        start_date: datetime | None = None,
        interval_days: int = 30,
        deforestation_rate: float = 0.03,
    ) -> tuple[Region, list[SatelliteImage]]:
        """Generate a synthetic time series of satellite images.

        Args:
            region_id: Identifier for the region.
            n_steps: Number of time steps.
            start_date: First observation date (defaults to 2025-01-01).
            interval_days: Days between observations.
            deforestation_rate: Fraction of remaining forest cleared per step.

        Returns:
            A (Region, list[SatelliteImage]) tuple.
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1)

        red_base, nir_base, zone_map = self._generate_base()
        region = Region(
            region_id=region_id,
            name=f"Simulated Region {region_id}",
            latitude=-3.0 + self.rng.uniform(-2, 2),
            longitude=-60.0 + self.rng.uniform(-5, 5),
        )

        images: list[SatelliteImage] = []
        red, nir = red_base.copy(), nir_base.copy()

        for step in range(n_steps):
            ts = start_date + timedelta(days=step * interval_days)
            # Add slight sensor noise.
            noisy_red = np.clip(red + self.rng.normal(0, 0.005, red.shape), 0, 1)
            noisy_nir = np.clip(nir + self.rng.normal(0, 0.005, nir.shape), 0, 1)

            images.append(
                SatelliteImage(
                    region_id=region_id,
                    timestamp=ts,
                    red=noisy_red,
                    nir=noisy_nir,
                    resolution_m=self.resolution_m,
                )
            )

            # Deforest for next step.
            if step < n_steps - 1:
                red, nir = self._deforest(red, nir, zone_map, deforestation_rate)

        return region, images
