from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn


SCHEMA_VERSION = "roadweave.ml-observation/1.0"

TASK_LABELS: Dict[str, Tuple[str, ...]] = {
    "risk": ("LOW", "MODERATE", "HIGH", "EXTREME"),
    "policy": (
        "KEEP",
        "ACCELERATE",
        "DECELERATE",
        "CHANGE_LEFT",
        "CHANGE_RIGHT",
    ),
}

TASK_FILES: Dict[str, Tuple[str, ...]] = {
    "risk": ("risk_nuscenes.parquet", "risk_krisk.parquet"),
    "policy": ("policy_nuscenes.parquet", "policy_krisk.parquet"),
}

METADATA_COLUMNS = {
    "source",
    "group_id",
    "timestamp",
    "target",
    "scene_token",
    "sample_token",
    "event_id",
    "split",
    "_split_group",
}

# These fields are labels, future information, or source-specific shortcuts.
# Letting any of them into the feature matrix would make offline scores look
# better while producing a model that cannot work online.
FORBIDDEN_FEATURE_NAMES = {
    "action",
    "action_id",
    "action_label",
    "folder",
    "folder_name",
    "risk",
    "risk_label",
    "risk_level",
    "risk_value",
    "total_risk",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ml_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expected_data_paths(task: str) -> List[Path]:
    validate_task(task)
    return [ml_root() / "data" / "processed" / name for name in TASK_FILES[task]]


def validate_task(task: str) -> None:
    if task not in TASK_LABELS:
        raise ValueError(
            "Unknown task {!r}. Expected one of: {}.".format(
                task,
                ", ".join(sorted(TASK_LABELS)),
            )
        )


def normalize_target(value: Any, task: str) -> str:
    validate_task(task)
    labels = TASK_LABELS[task]

    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Target label is missing.")

    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(labels):
            return labels[index]

    if isinstance(value, float) and value.is_integer():
        index = int(value)
        if 0 <= index < len(labels):
            return labels[index]

    text = str(value).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(labels):
            return labels[index]
    if text in labels:
        return text

    raise ValueError(
        "Invalid {} target {!r}. Expected one of {} or its zero-based ID.".format(
            task,
            value,
            ", ".join(labels),
        )
    )


def normalize_targets(values: Iterable[Any], task: str) -> pd.Series:
    return pd.Series([normalize_target(value, task) for value in values], dtype="object")


def _validate_required_columns(data: pd.DataFrame, path: Path) -> None:
    required = {"source", "group_id", "target"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "{} is missing required column(s): {}.".format(
                path,
                ", ".join(missing),
            )
        )
    if data.empty:
        raise ValueError("{} contains no rows.".format(path))
    if data["source"].isna().any():
        raise ValueError("{} contains missing source values.".format(path))
    if data["group_id"].isna().any():
        raise ValueError("{} contains missing group_id values.".format(path))


def load_task_data(
    task: str,
    paths: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    validate_task(task)
    selected_paths = list(paths) if paths is not None else expected_data_paths(task)
    if not selected_paths:
        raise ValueError("No input data files were supplied.")

    frames: List[pd.DataFrame] = []
    for raw_path in selected_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Prepared dataset not found: {}. Run the relevant converter first.".format(
                    path
                )
            )
        frame = pd.read_parquet(path)
        _validate_required_columns(frame, path)
        frame = frame.copy()
        frame["target"] = normalize_targets(frame["target"].tolist(), task).to_numpy()
        frame["source"] = frame["source"].astype(str).str.strip().str.lower()
        frame["group_id"] = frame["group_id"].astype(str).str.strip()
        frame["_split_group"] = frame["source"] + "::" + frame["group_id"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["_split_group"].eq("::").any():
        raise ValueError("At least one row has an empty source and group_id.")
    return combined


def select_feature_columns(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    selected: List[str] = []
    rejected: List[str] = []

    for column in sorted(data.columns):
        lower = column.lower()
        is_forbidden = (
            column in METADATA_COLUMNS
            or lower in FORBIDDEN_FEATURE_NAMES
            or lower.startswith("future_")
            or lower.endswith("_target")
            or lower.endswith("_label")
        )
        if is_forbidden:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            rejected.append(column)
            continue
        if data[column].isna().all():
            rejected.append(column)
            continue
        selected.append(column)

    if not selected:
        raise ValueError("No usable numeric feature columns were found.")
    return selected, rejected


def clean_feature_frame(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    aligned = data.reindex(columns=list(feature_columns)).copy()
    for column in feature_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def load_artifact(path: Path, expected_task: Optional[str] = None) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Model artifact not found: {}".format(resolved))

    artifact = joblib.load(resolved)
    if not isinstance(artifact, dict):
        raise ValueError("{} is not a RoadWeave model artifact.".format(resolved))

    required = {"schema_version", "task", "feature_columns", "label_order", "model"}
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(
            "{} is missing artifact field(s): {}.".format(
                resolved,
                ", ".join(missing),
            )
        )

    task = str(artifact["task"])
    validate_task(task)
    if expected_task is not None and task != expected_task:
        raise ValueError(
            "Expected a {} model, but {} contains a {} model.".format(
                expected_task,
                resolved,
                task,
            )
        )
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported model schema {!r}; expected {!r}.".format(
                artifact["schema_version"],
                SCHEMA_VERSION,
            )
        )
    if not artifact["feature_columns"]:
        raise ValueError("Model artifact contains an empty feature list.")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("Model artifact does not support probability predictions.")

    trained_version = str(artifact.get("sklearn_version", ""))
    current_version = sklearn.__version__
    if trained_version and trained_version.split(".")[:2] != current_version.split(".")[:2]:
        warnings.warn(
            "Model used scikit-learn {}, but runtime uses {}. Recreate the environment "
            "from requirements-lock.txt or retrain the artifact.".format(
                trained_version,
                current_version,
            ),
            RuntimeWarning,
        )
    return artifact


def predict_one(
    artifact: Mapping[str, Any],
    features: Mapping[str, Any],
) -> Tuple[str, float, Dict[str, float]]:
    columns = list(artifact["feature_columns"])
    row = pd.DataFrame([{column: features.get(column, np.nan) for column in columns}])
    row = clean_feature_frame(row, columns)
    model = artifact["model"]
    probabilities = np.asarray(model.predict_proba(row)[0], dtype=np.float64)
    classes = [str(value).strip().upper().rsplit(".", 1)[-1] for value in model.classes_]
    if len(classes) != len(probabilities):
        raise ValueError("Model returned a probability vector with the wrong length.")
    best_index = int(np.argmax(probabilities))
    probability_map = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    return classes[best_index], float(probabilities[best_index]), probability_map
