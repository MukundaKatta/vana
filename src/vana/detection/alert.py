"""Alert system for triggering deforestation notifications."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from vana.models import Alert, AlertSeverity, DeforestationEvent


class AlertSystem:
    """Evaluate deforestation events and emit alerts when thresholds are exceeded.

    Args:
        low_threshold: Hectares lost to trigger a LOW alert.
        medium_threshold: Hectares lost to trigger a MEDIUM alert.
        high_threshold: Hectares lost to trigger a HIGH alert.
        critical_threshold: Hectares lost to trigger a CRITICAL alert.
    """

    def __init__(
        self,
        low_threshold: float = 1.0,
        medium_threshold: float = 5.0,
        high_threshold: float = 20.0,
        critical_threshold: float = 50.0,
    ) -> None:
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self._alerts: list[Alert] = []

    def _classify_severity(self, hectares: float) -> AlertSeverity | None:
        """Determine alert severity based on hectares lost, or None if below threshold."""
        if hectares >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        if hectares >= self.high_threshold:
            return AlertSeverity.HIGH
        if hectares >= self.medium_threshold:
            return AlertSeverity.MEDIUM
        if hectares >= self.low_threshold:
            return AlertSeverity.LOW
        return None

    def evaluate(self, event: DeforestationEvent) -> Alert | None:
        """Evaluate a single deforestation event and return an alert if warranted.

        Args:
            event: A detected deforestation event.

        Returns:
            An Alert if the loss exceeds the low threshold, else None.
        """
        severity = self._classify_severity(event.hectares_lost)
        if severity is None:
            return None

        message = (
            f"[{severity.value.upper()}] {event.hectares_lost:.1f} hectares of forest "
            f"lost in region {event.region_id} between "
            f"{event.start_date:%Y-%m-%d} and {event.end_date:%Y-%m-%d}."
        )

        alert = Alert(
            region_id=event.region_id,
            timestamp=datetime.now(),
            severity=severity,
            hectares_lost=event.hectares_lost,
            message=message,
            event=event,
        )
        self._alerts.append(alert)
        return alert

    def evaluate_many(self, events: Sequence[DeforestationEvent]) -> list[Alert]:
        """Evaluate multiple events, returning all triggered alerts."""
        alerts: list[Alert] = []
        for event in events:
            alert = self.evaluate(event)
            if alert is not None:
                alerts.append(alert)
        return alerts

    @property
    def alerts(self) -> list[Alert]:
        """All alerts generated so far."""
        return list(self._alerts)

    def clear(self) -> None:
        """Clear the alert history."""
        self._alerts.clear()
