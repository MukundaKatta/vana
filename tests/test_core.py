"""Tests for Vana."""
from src.core import Vana
def test_init(): assert Vana().get_stats()["ops"] == 0
def test_op(): c = Vana(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Vana(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Vana(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Vana(); r = c.detect(); assert r["service"] == "vana"
