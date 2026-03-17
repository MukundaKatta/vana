"""NDVI (Normalized Difference Vegetation Index) calculation from multispectral bands."""

from __future__ import annotations

import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI from near-infrared and red reflectance bands.

    NDVI = (NIR - Red) / (NIR + Red)

    Values range from -1.0 to 1.0:
      - High positive (>0.4): Dense vegetation / healthy forest
      - Low positive (0.1-0.3): Sparse vegetation / cleared land
      - Near zero: Bare soil / urban
      - Negative: Water bodies

    Args:
        nir: Near-infrared band reflectance array (values 0.0-1.0).
        red: Red band reflectance array (values 0.0-1.0).

    Returns:
        NDVI array with values in [-1.0, 1.0].
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)

    denominator = nir + red
    # Avoid division by zero; set NDVI to 0 where both bands are zero.
    ndvi = np.where(
        denominator > 0,
        (nir - red) / denominator,
        0.0,
    )
    return np.clip(ndvi, -1.0, 1.0)


def ndvi_difference(ndvi_before: np.ndarray, ndvi_after: np.ndarray) -> np.ndarray:
    """Compute per-pixel NDVI change between two time periods.

    Negative values indicate vegetation loss (potential deforestation).

    Args:
        ndvi_before: NDVI array at the earlier time.
        ndvi_after: NDVI array at the later time.

    Returns:
        Difference array (after - before). Negative = loss.
    """
    return np.asarray(ndvi_after, dtype=np.float64) - np.asarray(
        ndvi_before, dtype=np.float64
    )
