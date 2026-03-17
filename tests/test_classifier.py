"""Tests for land cover classification."""

from datetime import datetime

import numpy as np

from vana.detection.classifier import ForestClassifier
from vana.models import SatelliteImage, LandCoverType


def _make_image(red: np.ndarray, nir: np.ndarray) -> SatelliteImage:
    return SatelliteImage(
        region_id="test",
        timestamp=datetime(2025, 1, 1),
        red=red,
        nir=nir,
    )


class TestForestClassifier:
    def test_rule_based_forest(self):
        """High NDVI pixels should be classified as forest."""
        red = np.full((5, 5), 0.05)
        nir = np.full((5, 5), 0.5)  # NDVI ~ 0.82
        img = _make_image(red, nir)
        labels = ForestClassifier.classify_rules(img)
        assert np.all(labels == 3)  # 3 = forest

    def test_rule_based_water(self):
        """Negative NDVI should be water."""
        red = np.full((5, 5), 0.08)
        nir = np.full((5, 5), 0.02)  # NDVI ~ -0.6
        img = _make_image(red, nir)
        labels = ForestClassifier.classify_rules(img)
        assert np.all(labels == 0)  # 0 = water

    def test_rule_based_urban(self):
        """Low positive NDVI should be urban."""
        red = np.full((5, 5), 0.20)
        nir = np.full((5, 5), 0.22)  # NDVI ~ 0.05
        img = _make_image(red, nir)
        labels = ForestClassifier.classify_rules(img)
        assert np.all(labels == 1)  # 1 = urban

    def test_class_fractions(self):
        labels = np.array([[0, 0, 3, 3, 3]])
        fractions = ForestClassifier.class_fractions(labels)
        assert fractions["water"] == 0.4
        assert fractions["forest"] == 0.6
        assert fractions["cleared"] == 0.0
        assert fractions["urban"] == 0.0

    def test_label_to_type(self):
        assert ForestClassifier.label_to_type(0) == LandCoverType.WATER
        assert ForestClassifier.label_to_type(3) == LandCoverType.FOREST

    def test_predict_falls_back_to_rules(self):
        """Untrained classifier should use rule-based classification."""
        clf = ForestClassifier()
        red = np.full((3, 3), 0.05)
        nir = np.full((3, 3), 0.5)
        img = _make_image(red, nir)
        labels = clf.predict(img)
        assert np.all(labels == 3)

    def test_trained_classifier(self):
        """Trained classifier should produce valid label arrays."""
        clf = ForestClassifier(n_estimators=10)
        # Create training data with known classes.
        images, labels = [], []
        for _ in range(3):
            red = np.random.rand(10, 10).astype(np.float64)
            nir = np.random.rand(10, 10).astype(np.float64)
            img = _make_image(red, nir)
            lbl = ForestClassifier.classify_rules(img)
            images.append(img)
            labels.append(lbl)

        clf.fit(images, labels)
        pred = clf.predict(images[0])
        assert pred.shape == (10, 10)
        assert set(np.unique(pred)).issubset({0, 1, 2, 3})
