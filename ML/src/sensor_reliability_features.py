"""Canonical feature contract for RoadWeave virtual-sensor reliability."""

from __future__ import annotations

from typing import Any, Dict, Mapping
import math


SCHEMA_VERSION = "roadweave.sensor-reliability/1.0"
SENSOR_TYPES = ("CAMERA", "LIDAR", "RADAR")
WEATHER_CONDITIONS = ("DRY", "RAIN", "SNOW", "FOG")

NUMERIC_FEATURES = (
    "dropout_rate",
    "message_age_mean",
    "message_age_max",
    "detection_count_mean",
    "detection_count_std",
    "confidence_mean",
    "confidence_std",
    "track_continuity",
    "range_variance",
    "velocity_variance",
    "innovation_mean",
    "innovation_std",
    "cross_sensor_disagreement",
    "ego_speed_mean",
    "ego_speed_std",
    "yaw_rate_mean",
    "yaw_rate_std",
)
FEATURE_COLUMNS = NUMERIC_FEATURES + tuple(
    "weather_{}".format(condition.lower()) for condition in WEATHER_CONDITIONS
)


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_sensor_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {"CAM": "CAMERA", "LASER": "LIDAR", "RAD": "RADAR"}
    text = aliases.get(text, text)
    if text not in SENSOR_TYPES:
        raise ValueError("Unsupported virtual sensor type {!r}".format(value))
    return text


def normalize_weather(value: Any) -> str:
    text = str(value or "DRY").strip().upper()
    aliases = {"CLEAR": "DRY", "RAINY": "RAIN", "SNOWY": "SNOW", "HAZE": "FOG"}
    text = aliases.get(text, text)
    return text if text in WEATHER_CONDITIONS else "DRY"


def canonical_features(observation: Mapping[str, Any]) -> Dict[str, float]:
    result = {
        feature: finite_number(observation.get(feature), 0.0)
        for feature in NUMERIC_FEATURES
    }
    weather = normalize_weather(observation.get("weather", "DRY"))
    for condition in WEATHER_CONDITIONS:
        result["weather_{}".format(condition.lower())] = float(condition == weather)
    return result


def reliability_status(value: Any) -> str:
    reliability = max(0.0, min(1.0, finite_number(value)))
    if reliability >= 0.90:
        return "HEALTHY"
    if reliability >= 0.70:
        return "ACCEPTABLE"
    if reliability >= 0.40:
        return "DEGRADED"
    return "FAILED"
