"""Canonical features shared by Extreme Driving conversion and live inference."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping
import math

import numpy as np


SCHEMA_VERSION = "roadweave.weather-observation/1.0"
WEATHER_CONDITIONS = ("DRY", "RAIN", "SNOW", "FOG")
HISTORY_SIGNALS = ("ego_speed", "ego_accel", "ego_yaw_rate")
SUMMARY_SUFFIXES = ("now", "min", "max", "mean", "std", "trend")
FEATURE_COLUMNS = tuple(
    f"{signal}_{suffix}"
    for signal in HISTORY_SIGNALS
    for suffix in SUMMARY_SUFFIXES
) + tuple(f"weather_{condition.lower()}" for condition in WEATHER_CONDITIONS)


def finite_number(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_condition(value: object) -> str:
    text = str(value or "DRY").strip().upper()
    aliases = {
        "CLEAR": "DRY",
        "NORMAL": "DRY",
        "WET": "RAIN",
        "RAINY": "RAIN",
        "SNOWY": "SNOW",
        "HAZE": "FOG",
        "HAZY": "FOG",
        "FOGGY": "FOG",
    }
    text = aliases.get(text, text)
    return text if text in WEATHER_CONDITIONS else "DRY"


def new_weather_state(
    timestamp: float,
    ego_speed_mps: float,
    ego_accel_mps2: float,
    ego_yaw_rate_degrees_per_second: float,
) -> Dict[str, float]:
    return {
        "timestamp": finite_number(timestamp, 0.0),
        "ego_speed": finite_number(ego_speed_mps),
        "ego_accel": finite_number(ego_accel_mps2),
        "ego_yaw_rate": finite_number(ego_yaw_rate_degrees_per_second),
    }


def summarize_weather_history(
    states: Iterable[Mapping[str, float]],
    condition: object,
) -> Dict[str, float]:
    history: List[Mapping[str, float]] = list(states)
    if not history:
        raise ValueError("Cannot summarize an empty weather history.")
    history.sort(key=lambda state: finite_number(state.get("timestamp"), 0.0))

    result: Dict[str, float] = {}
    for signal in HISTORY_SIGNALS:
        values = np.asarray(
            [finite_number(state.get(signal)) for state in history],
            dtype=np.float64,
        )
        valid_indices = np.flatnonzero(np.isfinite(values))
        if len(valid_indices) == 0:
            for suffix in SUMMARY_SUFFIXES:
                result[f"{signal}_{suffix}"] = math.nan
            continue

        valid_values = values[valid_indices]
        first_index = int(valid_indices[0])
        last_index = int(valid_indices[-1])
        first_time = finite_number(history[first_index].get("timestamp"), 0.0)
        last_time = finite_number(history[last_index].get("timestamp"), first_time)
        duration = max(0.001, last_time - first_time)
        result[f"{signal}_now"] = float(valid_values[-1])
        result[f"{signal}_min"] = float(np.min(valid_values))
        result[f"{signal}_max"] = float(np.max(valid_values))
        result[f"{signal}_mean"] = float(np.mean(valid_values))
        result[f"{signal}_std"] = float(np.std(valid_values))
        result[f"{signal}_trend"] = float(
            (valid_values[-1] - valid_values[0]) / duration
        )

    normalized = normalize_condition(condition)
    for name in WEATHER_CONDITIONS:
        result[f"weather_{name.lower()}"] = float(name == normalized)
    return result
