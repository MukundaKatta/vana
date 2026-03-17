"""Tests for the alert system."""

from datetime import datetime

import pytest

from vana.detection.alert import AlertSystem
from vana.models import AlertSeverity, DeforestationEvent


def _make_event(hectares: float) -> DeforestationEvent:
    return DeforestationEvent(
        region_id="test",
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 1),
        hectares_lost=hectares,
        mean_ndvi_drop=-0.3,
        affected_pixels=int(hectares * 100),
    )


class TestAlertSystem:
    def test_no_alert_below_threshold(self):
        system = AlertSystem(low_threshold=1.0)
        alert = system.evaluate(_make_event(0.5))
        assert alert is None

    def test_low_alert(self):
        system = AlertSystem()
        alert = system.evaluate(_make_event(2.0))
        assert alert is not None
        assert alert.severity == AlertSeverity.LOW

    def test_medium_alert(self):
        system = AlertSystem()
        alert = system.evaluate(_make_event(10.0))
        assert alert is not None
        assert alert.severity == AlertSeverity.MEDIUM

    def test_high_alert(self):
        system = AlertSystem()
        alert = system.evaluate(_make_event(30.0))
        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH

    def test_critical_alert(self):
        system = AlertSystem()
        alert = system.evaluate(_make_event(100.0))
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_evaluate_many(self):
        system = AlertSystem()
        events = [_make_event(h) for h in [0.1, 2.0, 10.0]]
        alerts = system.evaluate_many(events)
        assert len(alerts) == 2  # 0.1 is below threshold

    def test_alert_history(self):
        system = AlertSystem()
        system.evaluate(_make_event(5.0))
        system.evaluate(_make_event(25.0))
        assert len(system.alerts) == 2

    def test_clear(self):
        system = AlertSystem()
        system.evaluate(_make_event(5.0))
        system.clear()
        assert len(system.alerts) == 0

    def test_alert_message_contains_region(self):
        system = AlertSystem()
        alert = system.evaluate(_make_event(5.0))
        assert "test" in alert.message
