"""Detection modules for deforestation monitoring."""

from vana.detection.change_detector import ChangeDetector
from vana.detection.classifier import ForestClassifier
from vana.detection.alert import AlertSystem

__all__ = ["ChangeDetector", "ForestClassifier", "AlertSystem"]
