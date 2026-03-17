"""Tests for pydantic models."""

from datetime import datetime

import numpy as np
import pytest

from vana.models import (
    SatelliteImage,
    Region,
    DeforestationEvent,
    Alert,
    AlertSeverity,
    LandCoverType,
)


class TestSatelliteImage:
    def test_creation(self):
        img = SatelliteImage(
            region_id="r1",
            timestamp=datetime(2025, 6, 1),
            red=np.zeros((10, 20)),
            nir=np.ones((10, 20)),
        )
        assert img.height == 10
        assert img.width == 20
        assert img.resolution_m == 10.0


class TestRegion:
    def test_defaults(self):
        r = Region(region_id="test")
        assert r.name == ""
        assert r.area_hectares is None


class TestDeforestationEvent:
    def test_validation(self):
        ev = DeforestationEvent(
            region_id="r1",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 2, 1),
            hectares_lost=5.0,
            mean_ndvi_drop=-0.3,
            affected_pixels=500,
        )
        assert ev.hectares_lost == 5.0

    def test_negative_hectares_rejected(self):
        with pytest.raises(Exception):
            DeforestationEvent(
                region_id="r1",
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2025, 2, 1),
                hectares_lost=-1.0,
                mean_ndvi_drop=-0.3,
                affected_pixels=0,
            )


class TestAlert:
    def test_creation(self):
        alert = Alert(
            region_id="r1",
            severity=AlertSeverity.HIGH,
            hectares_lost=25.0,
            message="Test alert",
        )
        assert alert.severity == AlertSeverity.HIGH


class TestEnums:
    def test_land_cover_values(self):
        assert LandCoverType.FOREST.value == "forest"
        assert LandCoverType.WATER.value == "water"

    def test_alert_severity_values(self):
        assert AlertSeverity.CRITICAL.value == "critical"
