"""Loading and inference helpers for the RoadWeave weather-speed model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    from ML.src.weather_features import SCHEMA_VERSION, normalize_condition
except ModuleNotFoundError:
    from weather_features import SCHEMA_VERSION, normalize_condition  # type: ignore


def load_weather_artifact(path: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Weather model not found: {}".format(resolved))
    artifact = joblib.load(resolved)
    if not isinstance(artifact, dict):
        raise ValueError("{} is not a RoadWeave weather artifact.".format(resolved))
    required = {
        "schema_version",
        "task",
        "feature_columns",
        "model",
        "minimum_factor",
        "maximum_factor",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError("Weather artifact is missing: {}".format(", ".join(missing)))
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported weather schema: {}".format(artifact["schema_version"]))
    if artifact["task"] != "weather_speed":
        raise ValueError("Expected weather_speed artifact, found {}".format(artifact["task"]))
    trained = str(artifact.get("sklearn_version", ""))
    if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        warnings.warn(
            "Weather model used scikit-learn {}, runtime uses {}.".format(
                trained, sklearn.__version__
            ),
            RuntimeWarning,
        )
    return artifact


def weather_feature_frame(
    values: Mapping[str, Any], feature_columns: Sequence[str]
) -> pd.DataFrame:
    row = {}
    for column in feature_columns:
        try:
            number = float(values.get(column, math.nan))
        except (TypeError, ValueError):
            number = math.nan
        row[column] = number if math.isfinite(number) else np.nan
    return pd.DataFrame([row], columns=list(feature_columns), dtype=np.float64)


def predict_weather_factor(
    artifact: Mapping[str, Any],
    summary: Mapping[str, Any],
    condition: object,
) -> float:
    values = dict(summary)
    normalized = normalize_condition(condition)
    for name in ("DRY", "RAIN", "SNOW", "FOG"):
        values["weather_{}".format(name.lower())] = float(name == normalized)
    frame = weather_feature_frame(values, artifact["feature_columns"])
    prediction = float(artifact["model"].predict(frame)[0])
    return float(np.clip(
        prediction,
        float(artifact["minimum_factor"]),
        float(artifact["maximum_factor"]),
    ))
