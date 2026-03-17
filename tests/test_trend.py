"""Tests for trend analysis."""

from datetime import datetime, timedelta

import pytest

from vana.analysis.trend import TrendAnalyzer


class TestTrendAnalyzer:
    def test_linear_trend(self):
        """A constant loss per period should give a strong linear fit."""
        analyzer = TrendAnalyzer(window_size=2)
        dates = [datetime(2025, 1, 1) + timedelta(days=30 * i) for i in range(6)]
        losses = [10.0] * 6  # constant 10 hectares per period
        result = analyzer.analyze(dates, losses)

        assert result.r_squared > 0.99
        assert result.slope_hectares_per_day > 0
        assert len(result.cumulative_loss) == 6
        assert result.cumulative_loss[-1] == pytest.approx(60.0)

    def test_rolling_average(self):
        analyzer = TrendAnalyzer(window_size=3)
        dates = [datetime(2025, 1, 1) + timedelta(days=30 * i) for i in range(4)]
        losses = [0.0, 6.0, 12.0, 6.0]
        result = analyzer.analyze(dates, losses)

        assert result.rolling_avg[0] == pytest.approx(0.0)
        assert result.rolling_avg[1] == pytest.approx(3.0)  # avg(0, 6)
        assert result.rolling_avg[2] == pytest.approx(6.0)  # avg(0, 6, 12)
        assert result.rolling_avg[3] == pytest.approx(8.0)  # avg(6, 12, 6)

    def test_two_points(self):
        """Minimum viable input: two data points."""
        analyzer = TrendAnalyzer()
        dates = [datetime(2025, 1, 1), datetime(2025, 2, 1)]
        losses = [5.0, 10.0]
        result = analyzer.analyze(dates, losses)
        assert result.r_squared == pytest.approx(1.0)
