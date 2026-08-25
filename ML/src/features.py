from __future__ import annotations

from typing import Dict, Iterable, List
import math
import numpy as np


SLOTS = (
    "front",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
    "pedestrian",
)

BASE_SIGNALS = (
    "ego_speed",
    "ego_accel",
    "ego_yaw_rate",
    "left_clear",
    "right_clear",
) + tuple(
    f"{slot}_{field}"
    for slot in SLOTS
    for field in ("present", "gap", "closing_speed", "ttc", "speed")
)


def safe_number(value, default=np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def capped_ttc(gap: float, closing_speed: float) -> float:
    if not math.isfinite(gap) or not math.isfinite(closing_speed):
        return np.nan

    if gap <= 0.0:
        return 0.0

    if closing_speed <= 0.05:
        return 20.0

    return min(20.0, gap / closing_speed)


def new_state(
    timestamp: float,
    ego_speed: float,
    ego_accel: float,
    ego_yaw_rate: float,
) -> Dict[str, float]:
    state: Dict[str, float] = {
        "timestamp": float(timestamp),
        "ego_speed": safe_number(ego_speed),
        "ego_accel": safe_number(ego_accel),
        "ego_yaw_rate": safe_number(ego_yaw_rate),
        "left_clear": 1.0,
        "right_clear": 1.0,
    }

    for slot in SLOTS:
        state[f"{slot}_present"] = 0.0
        state[f"{slot}_gap"] = np.nan
        state[f"{slot}_closing_speed"] = np.nan
        state[f"{slot}_ttc"] = np.nan
        state[f"{slot}_speed"] = np.nan

    return state


def set_slot(
    state: Dict[str, float],
    slot: str,
    gap: float,
    closing_speed: float,
    actor_speed: float,
) -> None:
    gap = max(0.0, safe_number(gap, 0.0))
    closing_speed = safe_number(closing_speed, 0.0)

    state[f"{slot}_present"] = 1.0
    state[f"{slot}_gap"] = gap
    state[f"{slot}_closing_speed"] = closing_speed
    state[f"{slot}_ttc"] = capped_ttc(gap, closing_speed)
    state[f"{slot}_speed"] = safe_number(actor_speed)


def summarize_history(
    states: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    history: List[Dict[str, float]] = list(states)

    if not history:
        raise ValueError("Cannot summarize an empty history.")

    history.sort(key=lambda state: state["timestamp"])

    result: Dict[str, float] = {}

    for signal in BASE_SIGNALS:
        values = np.asarray(
            [safe_number(state.get(signal)) for state in history],
            dtype=np.float64,
        )

        valid_indices = np.flatnonzero(np.isfinite(values))

        if len(valid_indices) == 0:
            for suffix in ("now", "min", "max", "mean", "std", "trend"):
                result[f"{signal}_{suffix}"] = np.nan
            continue

        valid_values = values[valid_indices]
        first_index = int(valid_indices[0])
        last_index = int(valid_indices[-1])

        first_time = history[first_index]["timestamp"]
        last_time = history[last_index]["timestamp"]
        duration = max(0.001, last_time - first_time)

        result[f"{signal}_now"] = float(valid_values[-1])
        result[f"{signal}_min"] = float(np.min(valid_values))
        result[f"{signal}_max"] = float(np.max(valid_values))
        result[f"{signal}_mean"] = float(np.mean(valid_values))
        result[f"{signal}_std"] = float(np.std(valid_values))
        result[f"{signal}_trend"] = float(
            (valid_values[-1] - valid_values[0]) / duration
        )

    return result