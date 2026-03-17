"""Trend analysis: track deforestation rate over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class TrendResult:
    """Result of a deforestation trend analysis."""

    slope_hectares_per_day: float
    intercept: float
    r_squared: float
    p_value: float
    dates: list[datetime]
    cumulative_loss: list[float]
    rolling_avg: list[float]


class TrendAnalyzer:
    """Analyze deforestation rate trends over time.

    Args:
        window_size: Number of observations for the rolling average.
    """

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size

    def analyze(
        self,
        dates: Sequence[datetime],
        hectares_lost: Sequence[float],
    ) -> TrendResult:
        """Fit a linear trend to cumulative deforestation data.

        Args:
            dates: Ordered sequence of observation dates.
            hectares_lost: Hectares of forest lost at each date (incremental).

        Returns:
            TrendResult with slope, r-squared, and rolling statistics.
        """
        dates = list(dates)
        losses = np.asarray(hectares_lost, dtype=np.float64)
        cumulative = np.cumsum(losses).tolist()

        # Convert dates to ordinal days for regression.
        day_numbers = np.array([d.toordinal() for d in dates], dtype=np.float64)
        day_numbers -= day_numbers[0]  # relative to first observation

        cum_arr = np.array(cumulative)
        slope, intercept, r_value, p_value, _ = stats.linregress(day_numbers, cum_arr)

        # Rolling average of incremental losses.
        rolling = self._rolling_mean(losses, self.window_size)

        return TrendResult(
            slope_hectares_per_day=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            dates=dates,
            cumulative_loss=cumulative,
            rolling_avg=rolling,
        )

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> list[float]:
        """Compute a simple rolling mean, padding the start with partial windows."""
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(float(np.mean(values[start : i + 1])))
        return result
