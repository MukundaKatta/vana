"""Analysis modules for deforestation monitoring."""

from vana.analysis.ndvi import compute_ndvi
from vana.analysis.area import AreaCalculator
from vana.analysis.trend import TrendAnalyzer

__all__ = ["compute_ndvi", "AreaCalculator", "TrendAnalyzer"]
