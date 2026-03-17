"""Land cover classification: forest, cleared, water, urban."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRFC

from vana.analysis.ndvi import compute_ndvi
from vana.models import LandCoverType, SatelliteImage


# Integer labels matching LandCoverType.
_LABEL_MAP: dict[int, LandCoverType] = {
    0: LandCoverType.WATER,
    1: LandCoverType.URBAN,
    2: LandCoverType.CLEARED,
    3: LandCoverType.FOREST,
}

_INVERSE_LABEL_MAP: dict[LandCoverType, int] = {v: k for k, v in _LABEL_MAP.items()}


class ForestClassifier:
    """Classify each pixel of a satellite image into land cover types.

    Uses a scikit-learn RandomForestClassifier trained on features derived
    from the red band, NIR band, and NDVI.

    The classifier can operate in two modes:
      - **Rule-based** (default, no training needed): applies NDVI thresholds.
      - **Trained**: after calling ``fit()`` with labelled data.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self._model = SklearnRFC(
            n_estimators=n_estimators, random_state=random_state
        )
        self._is_trained = False

    # ------------------------------------------------------------------
    # Rule-based classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_rules(image: SatelliteImage) -> np.ndarray:
        """Classify land cover using simple NDVI thresholds.

        Returns an integer array with the same shape as the image bands:
          0 = water, 1 = urban, 2 = cleared, 3 = forest.
        """
        ndvi = compute_ndvi(image.nir, image.red)
        result = np.full(ndvi.shape, _INVERSE_LABEL_MAP[LandCoverType.CLEARED], dtype=np.int32)
        result[ndvi < -0.1] = _INVERSE_LABEL_MAP[LandCoverType.WATER]
        result[(ndvi >= -0.1) & (ndvi < 0.15)] = _INVERSE_LABEL_MAP[LandCoverType.URBAN]
        result[(ndvi >= 0.15) & (ndvi < 0.4)] = _INVERSE_LABEL_MAP[LandCoverType.CLEARED]
        result[ndvi >= 0.4] = _INVERSE_LABEL_MAP[LandCoverType.FOREST]
        return result

    # ------------------------------------------------------------------
    # ML-based classification
    # ------------------------------------------------------------------

    def _build_features(self, image: SatelliteImage) -> np.ndarray:
        """Build a feature matrix (n_pixels, 3) from an image."""
        ndvi = compute_ndvi(image.nir, image.red)
        red_flat = image.red.ravel()
        nir_flat = image.nir.ravel()
        ndvi_flat = ndvi.ravel()
        return np.column_stack([red_flat, nir_flat, ndvi_flat])

    def fit(self, images: list[SatelliteImage], labels: list[np.ndarray]) -> None:
        """Train the classifier on labelled satellite images.

        Args:
            images: List of SatelliteImage objects.
            labels: Corresponding integer label arrays (same shape as bands).
        """
        X_parts, y_parts = [], []
        for img, lbl in zip(images, labels):
            X_parts.append(self._build_features(img))
            y_parts.append(lbl.ravel())
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, image: SatelliteImage) -> np.ndarray:
        """Classify an image using the trained model, or fall back to rules.

        Returns an integer label array with the same shape as the image bands.
        """
        if not self._is_trained:
            return self.classify_rules(image)
        X = self._build_features(image)
        preds = self._model.predict(X)
        return preds.reshape(image.red.shape).astype(np.int32)

    @staticmethod
    def label_to_type(label: int) -> LandCoverType:
        """Map an integer label back to a LandCoverType enum."""
        return _LABEL_MAP[label]

    @staticmethod
    def class_fractions(label_map: np.ndarray) -> dict[str, float]:
        """Compute the fraction of each land cover class in a label map."""
        total = label_map.size
        fractions: dict[str, float] = {}
        for code, lc_type in _LABEL_MAP.items():
            fractions[lc_type.value] = float(np.sum(label_map == code)) / total
        return fractions
