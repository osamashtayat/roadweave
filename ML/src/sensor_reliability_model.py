"""Load and run RoadWeave virtual-sensor reliability regressors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    from ML.src.sensor_reliability_features import (
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        canonical_features,
        normalize_sensor_type,
        reliability_status,
    )
except ModuleNotFoundError:
    from sensor_reliability_features import (  # type: ignore
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        canonical_features,
        normalize_sensor_type,
        reliability_status,
    )


def load_sensor_reliability_artifact(path: Path, sensor_type: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Sensor reliability model not found: {}".format(resolved))
    artifact = joblib.load(resolved)
    if not isinstance(artifact, dict):
        raise ValueError("{} is not a RoadWeave reliability artifact.".format(resolved))
    required = {"schema_version", "task", "sensor_type", "feature_columns", "model"}
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError("Reliability artifact is missing: {}".format(", ".join(missing)))
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported reliability schema {}".format(artifact["schema_version"]))
    if artifact["task"] != "sensor_reliability":
        raise ValueError("Expected sensor_reliability artifact")
    expected = normalize_sensor_type(sensor_type)
    if normalize_sensor_type(artifact["sensor_type"]) != expected:
        raise ValueError("Expected {} artifact, found {}".format(expected, artifact["sensor_type"]))
    trained = str(artifact.get("sklearn_version", ""))
    if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        warnings.warn("Reliability model used scikit-learn {}, runtime uses {}.".format(trained, sklearn.__version__), RuntimeWarning)
    return artifact


def predict_sensor_reliability(
    artifact: Mapping[str, Any], observation: Mapping[str, Any]
) -> Dict[str, Any]:
    values = canonical_features(observation)
    columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
    row = []
    for column in columns:
        try:
            number = float(values.get(column, math.nan))
        except (TypeError, ValueError):
            number = math.nan
        row.append(number if math.isfinite(number) else np.nan)
    frame = pd.DataFrame([row], columns=columns, dtype=np.float64)
    prediction = float(np.clip(artifact["model"].predict(frame)[0], 0.0, 1.0))
    return {
        "reliability": prediction,
        "status": reliability_status(prediction),
    }
