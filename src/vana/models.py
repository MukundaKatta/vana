"""Pydantic data models for VANA deforestation monitoring."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class LandCoverType(str, Enum):
    """Land cover classification categories."""

    FOREST = "forest"
    CLEARED = "cleared"
    WATER = "water"
    URBAN = "urban"


class AlertSeverity(str, Enum):
    """Severity levels for deforestation alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SatelliteImage(BaseModel):
    """A multispectral satellite image capture."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    region_id: str = Field(description="Identifier for the geographic region")
    timestamp: datetime = Field(description="Acquisition timestamp")
    red: np.ndarray = Field(description="Red band reflectance array")
    nir: np.ndarray = Field(description="Near-infrared band reflectance array")
    resolution_m: float = Field(
        default=10.0, description="Spatial resolution in meters per pixel"
    )

    @property
    def height(self) -> int:
        return self.red.shape[0]

    @property
    def width(self) -> int:
        return self.red.shape[1]


class Region(BaseModel):
    """A geographic region being monitored."""

    region_id: str = Field(description="Unique identifier for this region")
    name: str = Field(default="", description="Human-readable region name")
    latitude: float = Field(default=0.0, description="Center latitude")
    longitude: float = Field(default=0.0, description="Center longitude")
    area_hectares: Optional[float] = Field(
        default=None, description="Total area in hectares"
    )


class DeforestationEvent(BaseModel):
    """A detected deforestation event between two time periods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    region_id: str
    start_date: datetime
    end_date: datetime
    hectares_lost: float = Field(ge=0, description="Hectares of forest lost")
    mean_ndvi_drop: float = Field(description="Average NDVI decrease in affected area")
    affected_pixels: int = Field(ge=0, description="Number of pixels classified as lost")
    change_mask: Optional[np.ndarray] = Field(
        default=None, description="Boolean mask of deforested pixels"
    )


class Alert(BaseModel):
    """A deforestation alert triggered by the monitoring system."""

    region_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: AlertSeverity
    hectares_lost: float = Field(ge=0)
    message: str = Field(description="Human-readable alert message")
    event: Optional[DeforestationEvent] = Field(
        default=None, description="Associated deforestation event"
    )
