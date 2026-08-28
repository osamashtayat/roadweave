#!/usr/bin/env python3
"""Train and evaluate one RoadWeave reliability regressor per virtual sensor."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    r2_score,
)

try:
    from ML.src.sensor_reliability_features import (
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        SENSOR_TYPES,
        reliability_status,
    )
except ModuleNotFoundError:
    from sensor_reliability_features import (  # type: ignore
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        SENSOR_TYPES,
        reliability_status,
    )


ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ML_ROOT / "data" / "processed" / "sensor_reliability.parquet"
DEFAULT_MODEL_DIR = ML_ROOT / "models"
DEFAULT_REPORT = ML_ROOT / "reports" / "sensor_reliability_metrics.json"
DEFAULT_PREDICTIONS = ML_ROOT / "reports" / "sensor_reliability_predictions.csv"
STATUS_ORDER = ["FAILED", "DEGRADED", "ACCEPTABLE", "HEALTHY"]


def feature_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = data.reindex(columns=list(columns)).copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def create_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=260,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def calibration_error(expected: np.ndarray, predicted: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (predicted >= edges[index]) & (predicted < edges[index + 1])
        if index == bins - 1:
            mask |= predicted == 1.0
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(expected[mask])) - float(np.mean(predicted[mask])))
    return total


def metrics(expected: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    clipped = np.clip(predicted, 0.0, 1.0)
    expected_status = [reliability_status(value) for value in expected]
    predicted_status = [reliability_status(value) for value in clipped]
    return {
        "rows": int(len(expected)),
        "mae": float(mean_absolute_error(expected, clipped)),
        "rmse": float(np.sqrt(mean_squared_error(expected, clipped))),
        "r2": float(r2_score(expected, clipped)) if len(expected) > 1 else 0.0,
        "within_0_10": float(np.mean(np.abs(expected - clipped) <= 0.10)),
        "status_macro_f1": float(f1_score(expected_status, predicted_status, labels=STATUS_ORDER, average="macro", zero_division=0)),
        "failure_recall": float(
            recall_score(
                expected_status,
                predicted_status,
                labels=["FAILED"],
                average="micro",
                zero_division=0,
            )
        ),
        "calibration_error": calibration_error(expected, clipped),
        "confusion_matrix": confusion_matrix(expected_status, predicted_status, labels=STATUS_ORDER).tolist(),
        "status_order": STATUS_ORDER,
    }


def evaluate(model: HistGradientBoostingRegressor, data: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    expected = data["target_reliability"].to_numpy(dtype=np.float64)
    predicted = np.clip(model.predict(feature_frame(data, FEATURE_COLUMNS)), 0.0, 1.0)
    confidence_baseline = np.clip(data["confidence_mean"].to_numpy(dtype=np.float64), 0.0, 1.0)
    result = metrics(expected, predicted)
    result["confidence_baseline"] = metrics(expected, confidence_baseline)
    result["by_weather"] = {}
    for weather, indices in data.groupby("weather").groups.items():
        positions = data.index.get_indexer(indices)
        result["by_weather"][str(weather)] = metrics(expected[positions], predicted[positions])
    result["by_fault"] = {}
    for fault, indices in data.groupby("fault_type").groups.items():
        positions = data.index.get_indexer(indices)
        result["by_fault"][str(fault)] = metrics(expected[positions], predicted[positions])
    predictions = data[["scenario_id", "split", "sensor_type", "weather", "fault_type", "fault_severity", "target_reliability", "target_status"]].copy()
    predictions["prediction"] = predicted
    predictions["predicted_status"] = [reliability_status(value) for value in predicted]
    predictions["absolute_error"] = np.abs(expected - predicted)
    return result, predictions


def leave_one_fault_out(data: pd.DataFrame, seed: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for fault in sorted(data["fault_type"].unique()):
        train = data[data["fault_type"] != fault]
        test = data[data["fault_type"] == fault]
        if train.empty or test.empty:
            continue
        model = create_model(seed)
        model.fit(feature_frame(train, FEATURE_COLUMNS), train["target_reliability"].to_numpy(dtype=np.float64))
        results[str(fault)] = metrics(
            test["target_reliability"].to_numpy(dtype=np.float64),
            model.predict(feature_frame(test, FEATURE_COLUMNS)),
        )
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-leave-one-fault-out", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    data = pd.read_parquet(args.data.expanduser().resolve())
    required = {"scenario_id", "split", "sensor_type", "weather", "fault_type", "target_reliability"} | set(FEATURE_COLUMNS)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Reliability data is missing: {}".format(", ".join(missing)))

    model_dir = args.model_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "sensor_reliability",
        "created_utc": now,
        "input_path": str(args.data.expanduser().resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "excluded_metadata": ["fault_type", "fault_severity", "target_status", "label_precision", "label_recall", "label_range_rmse", "label_velocity_rmse"],
        "sensors": {},
    }
    prediction_frames: List[pd.DataFrame] = []

    for sensor_index, sensor in enumerate(SENSOR_TYPES):
        subset = data[data["sensor_type"] == sensor].reset_index(drop=True)
        train = subset[subset["split"] == "train"].reset_index(drop=True)
        validation = subset[subset["split"] == "validation"].reset_index(drop=True)
        test = subset[subset["split"] == "test"].reset_index(drop=True)
        model = create_model(args.seed + sensor_index)
        model.fit(feature_frame(train, FEATURE_COLUMNS), train["target_reliability"].to_numpy(dtype=np.float64))
        validation_metrics, validation_predictions = evaluate(model, validation)
        test_metrics, test_predictions = evaluate(model, test)
        validation_predictions["evaluation"] = "validation"
        test_predictions["evaluation"] = "test"
        prediction_frames.extend([validation_predictions, test_predictions])

        importance = permutation_importance(
            model,
            feature_frame(test, FEATURE_COLUMNS),
            test["target_reliability"].to_numpy(dtype=np.float64),
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=args.seed,
            n_jobs=1,
        )
        important = sorted(
            [
                {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
                for feature, mean, std in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std)
            ],
            key=lambda item: item["importance_mean"],
            reverse=True,
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "task": "sensor_reliability",
            "sensor_type": sensor,
            "model_version": "{}-reliability-1.0".format(sensor.lower()),
            "created_utc": now,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "feature_columns": list(FEATURE_COLUMNS),
            "model": model,
        }
        model_path = model_dir / "sensor_{}_reliability_model.joblib".format(sensor.lower())
        joblib.dump(artifact, model_path)
        sensor_report: Dict[str, Any] = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation": validation_metrics,
            "test": test_metrics,
            "important_features": important,
            "model_path": str(model_path),
        }
        if not args.skip_leave_one_fault_out:
            sensor_report["leave_one_fault_out"] = leave_one_fault_out(train, args.seed + sensor_index)
        report["sensors"][sensor] = sensor_report
        print("{} test MAE={:.4f}, macro-F1={:.4f}".format(sensor, test_metrics["mae"], test_metrics["status_macro_f1"]))

    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    report["predictions_path"] = str(predictions_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
