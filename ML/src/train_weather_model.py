#!/usr/bin/env python3
"""Train and evaluate RoadWeave's Extreme Driving weather-speed regressor."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from ML.src.weather_features import FEATURE_COLUMNS, SCHEMA_VERSION
except ModuleNotFoundError:
    from weather_features import FEATURE_COLUMNS, SCHEMA_VERSION  # type: ignore


ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ML_ROOT / "data" / "processed" / "weather_extreme.parquet"
DEFAULT_MODEL = ML_ROOT / "models" / "weather_model.joblib"
DEFAULT_REPORT = ML_ROOT / "reports" / "weather_metrics.json"
DEFAULT_PREDICTIONS = ML_ROOT / "reports" / "weather_test_predictions.csv"
RULE_FACTORS = {"DRY": 1.0, "RAIN": 0.90, "SNOW": 0.78, "FOG": 0.82}
# Current speed is deliberately excluded.  Otherwise predicting the next
# second's speed becomes a near-identity shortcut and produces impressive but
# meaningless scores.  The deployed model is a speed *cap* conditioned on
# weather and recent acceleration/turning dynamics, not a speed follower.
MODEL_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if not column.startswith("ego_speed_")
)


def feature_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = data.reindex(columns=list(columns)).copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def condition_weights(data: pd.DataFrame) -> np.ndarray:
    counts = data["condition"].value_counts()
    total = float(len(data))
    condition_count = max(1, len(counts))
    return np.asarray(
        [total / (condition_count * float(counts[value])) for value in data["condition"]],
        dtype=np.float64,
    )


def create_model(arguments: argparse.Namespace) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=arguments.learning_rate,
        max_iter=arguments.max_iter,
        max_leaf_nodes=arguments.max_leaf_nodes,
        min_samples_leaf=arguments.min_samples_leaf,
        l2_regularization=1.0,
        loss="squared_error",
        early_stopping=False,
        random_state=arguments.seed,
    )


def metrics(expected: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "rows": int(len(expected)),
        "mae": float(mean_absolute_error(expected, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(expected, predicted))),
        "r2": float(r2_score(expected, predicted)) if len(expected) > 1 else 0.0,
        "within_0_05": float(np.mean(np.abs(expected - predicted) <= 0.05)),
        "within_0_10": float(np.mean(np.abs(expected - predicted) <= 0.10)),
    }


def evaluate(
    model: HistGradientBoostingRegressor,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    minimum_factor: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    expected = data["target_speed_factor"].to_numpy(dtype=np.float64)
    predicted = np.clip(
        model.predict(feature_frame(data, feature_columns)), minimum_factor, 1.0
    )
    baseline = np.asarray(
        [RULE_FACTORS.get(str(value), 1.0) for value in data["condition"]],
        dtype=np.float64,
    )
    result: Dict[str, Any] = metrics(expected, predicted)
    result["fixed_rule_baseline"] = metrics(expected, baseline)
    result["by_condition"] = {}
    for condition, indices in data.groupby("condition").groups.items():
        positions = data.index.get_indexer(indices)
        result["by_condition"][str(condition)] = metrics(
            expected[positions], predicted[positions]
        )

    predictions = data[
        ["source", "group_id", "split", "condition", "timestamp", "future_speed_kph", "target_speed_factor"]
    ].copy()
    predictions["prediction"] = predicted
    predictions["fixed_rule_prediction"] = baseline
    predictions["absolute_error"] = np.abs(expected - predicted)
    return result, predictions


def cross_validate(
    data: pd.DataFrame,
    arguments: argparse.Namespace,
    feature_columns: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    group_conditions = data.groupby("group_id")["condition"].first()
    minimum_groups = int(group_conditions.value_counts().min())
    folds = min(arguments.folds, minimum_groups)
    if folds < 2:
        raise ValueError("At least two episodes per weather condition are required.")

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=arguments.seed,
    )
    out_of_fold: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    x = feature_frame(data, feature_columns)
    y = data["target_speed_factor"].to_numpy(dtype=np.float64)
    conditions = data["condition"].astype(str)
    groups = data["group_id"].astype(str)

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(x, conditions, groups), 1
    ):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices].reset_index(drop=True)
        model = create_model(arguments)
        model.fit(
            feature_frame(train, feature_columns),
            train["target_speed_factor"].to_numpy(dtype=np.float64),
            sample_weight=condition_weights(train),
        )
        fold_result, predictions = evaluate(
            model, test, feature_columns, arguments.minimum_factor
        )
        fold_result["fold"] = fold
        fold_result["groups"] = int(test["group_id"].nunique())
        fold_metrics.append(fold_result)
        predictions["fold"] = fold
        out_of_fold.append(predictions)

    combined = pd.concat(out_of_fold, ignore_index=True)
    overall = metrics(
        combined["target_speed_factor"].to_numpy(dtype=np.float64),
        combined["prediction"].to_numpy(dtype=np.float64),
    )
    overall["fixed_rule_baseline"] = metrics(
        combined["target_speed_factor"].to_numpy(dtype=np.float64),
        combined["fixed_rule_prediction"].to_numpy(dtype=np.float64),
    )
    overall["folds"] = fold_metrics
    overall["fold_count"] = folds
    return overall, combined


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--minimum-factor", type=float, default=0.35)
    parser.add_argument("--importance-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    data_path = arguments.data.expanduser().resolve()
    data = pd.read_parquet(data_path)
    required = {
        "source", "group_id", "split", "condition", "target_speed_factor"
    } | set(FEATURE_COLUMNS)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Weather data is missing: {}".format(", ".join(missing)))
    train = data[data["split"] == "train"].reset_index(drop=True)
    held_out = data[data["split"] == "val"].reset_index(drop=True)
    if train.empty or held_out.empty:
        raise ValueError("Both official train and val rows are required.")

    cross_validation, oof_predictions = cross_validate(
        train, arguments, MODEL_FEATURE_COLUMNS
    )
    model = create_model(arguments)
    model.fit(
        feature_frame(train, MODEL_FEATURE_COLUMNS),
        train["target_speed_factor"].to_numpy(dtype=np.float64),
        sample_weight=condition_weights(train),
    )
    held_out_metrics, held_out_predictions = evaluate(
        model, held_out, MODEL_FEATURE_COLUMNS, arguments.minimum_factor
    )

    importance = permutation_importance(
        model,
        feature_frame(held_out, MODEL_FEATURE_COLUMNS),
        held_out["target_speed_factor"].to_numpy(dtype=np.float64),
        scoring="neg_mean_absolute_error",
        n_repeats=max(1, arguments.importance_repeats),
        random_state=arguments.seed,
        n_jobs=1,
    )
    important_features = sorted(
        [
            {
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(
                MODEL_FEATURE_COLUMNS,
                importance.importances_mean,
                importance.importances_std,
            )
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )

    now = datetime.now(timezone.utc).isoformat()
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": "weather-speed-extreme-driving-1.0",
        "task": "weather_speed",
        "created_utc": now,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "minimum_factor": arguments.minimum_factor,
        "maximum_factor": 1.0,
        "reference_speed_kph": 50.0,
        "conditions": ["DRY", "RAIN", "SNOW", "FOG"],
        "model": model,
        "training": {
            "input_path": str(data_path),
            "rows": len(train),
            "groups": int(train["group_id"].nunique()),
            "seed": arguments.seed,
        },
    }

    model_path = arguments.model_output.expanduser().resolve()
    report_path = arguments.report_output.expanduser().resolve()
    predictions_path = arguments.predictions_output.expanduser().resolve()
    for path in (model_path, report_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    oof_predictions["evaluation"] = "group_cross_validation"
    held_out_predictions["evaluation"] = "official_validation"
    pd.concat([oof_predictions, held_out_predictions], ignore_index=True).to_csv(
        predictions_path, index=False
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "task": "weather_speed",
        "created_utc": now,
        "input_path": str(data_path),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "excluded_shortcut_features": [
            column for column in FEATURE_COLUMNS if column.startswith("ego_speed_")
        ],
        "train_rows": len(train),
        "train_groups": int(train["group_id"].nunique()),
        "validation_rows": len(held_out),
        "validation_groups": int(held_out["group_id"].nunique()),
        "train_condition_rows": {
            str(key): int(value) for key, value in train["condition"].value_counts().items()
        },
        "validation_condition_rows": {
            str(key): int(value) for key, value in held_out["condition"].value_counts().items()
        },
        "cross_validation": cross_validation,
        "official_validation": held_out_metrics,
        "important_features": important_features,
        "fixed_rule_factors": RULE_FACTORS,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Cross-validation MAE: {:.4f} (fixed rule {:.4f})".format(
        cross_validation["mae"], cross_validation["fixed_rule_baseline"]["mae"]
    ))
    print("Official validation MAE: {:.4f} (fixed rule {:.4f})".format(
        held_out_metrics["mae"], held_out_metrics["fixed_rule_baseline"]["mae"]
    ))
    print("Model: {}".format(model_path))
    print("Report: {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
