#!/usr/bin/env python3

"""Train and evaluate one RoadWeave risk or driving-policy model."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit

try:
    from ML.src.model_support import (
        SCHEMA_VERSION,
        TASK_LABELS,
        clean_feature_frame,
        expected_data_paths,
        load_task_data,
        ml_root,
        select_feature_columns,
    )
except ModuleNotFoundError:
    from model_support import (  # type: ignore
        SCHEMA_VERSION,
        TASK_LABELS,
        clean_feature_frame,
        expected_data_paths,
        load_task_data,
        ml_root,
        select_feature_columns,
    )


def _distribution(values: pd.Series, labels: Sequence[str]) -> np.ndarray:
    counts = values.value_counts(normalize=True)
    return np.asarray([float(counts.get(label, 0.0)) for label in labels])


def best_group_holdout(
    data: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    candidates: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint holdout with a close class distribution."""

    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError("holdout_fraction must be between 0.05 and 0.5.")
    if data["_split_group"].nunique() < 3:
        raise ValueError("At least three independent groups are required for splitting.")

    labels = sorted(data["target"].astype(str).unique())
    overall = _distribution(data["target"], labels)
    splitter = GroupShuffleSplit(
        n_splits=max(20, candidates),
        test_size=holdout_fraction,
        random_state=seed,
    )

    best_score = float("inf")
    best_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None
    groups = data["_split_group"].astype(str)

    for train_indices, holdout_indices in splitter.split(data, data["target"], groups):
        train_targets = set(data.iloc[train_indices]["target"].astype(str))
        holdout_targets = set(data.iloc[holdout_indices]["target"].astype(str))
        if train_targets != set(labels):
            continue

        holdout_distribution = _distribution(
            data.iloc[holdout_indices]["target"], labels
        )
        missing_holdout_classes = len(set(labels) - holdout_targets)
        actual_fraction = len(holdout_indices) / len(data)
        score = (
            float(np.abs(overall - holdout_distribution).sum())
            + abs(actual_fraction - holdout_fraction) * 2.0
            + missing_holdout_classes * 2.0
        )
        if score < best_score:
            best_score = score
            best_indices = (train_indices, holdout_indices)

    if best_indices is None:
        raise ValueError(
            "Could not make a group-disjoint split containing every class in training. "
            "Add more independent groups for the rare classes."
        )
    return best_indices


def make_splits(
    data: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation_indices, test_indices = best_group_holdout(
        data,
        holdout_fraction=0.20,
        seed=seed,
    )
    train_validation = data.iloc[train_validation_indices].reset_index(drop=True)
    test = data.iloc[test_indices].reset_index(drop=True)

    train_indices, validation_indices = best_group_holdout(
        train_validation,
        holdout_fraction=0.125,
        seed=seed + 1,
    )
    train = train_validation.iloc[train_indices].reset_index(drop=True)
    validation = train_validation.iloc[validation_indices].reset_index(drop=True)

    split_groups = {
        "train": set(train["_split_group"]),
        "validation": set(validation["_split_group"]),
        "test": set(test["_split_group"]),
    }
    if split_groups["train"] & split_groups["validation"]:
        raise AssertionError("Group leakage between train and validation splits.")
    if split_groups["train"] & split_groups["test"]:
        raise AssertionError("Group leakage between train and test splits.")
    if split_groups["validation"] & split_groups["test"]:
        raise AssertionError("Group leakage between validation and test splits.")
    return train, validation, test


def _class_counts(data: pd.DataFrame, labels: Sequence[str]) -> Dict[str, int]:
    counts = data["target"].value_counts()
    return {label: int(counts.get(label, 0)) for label in labels}


def _evaluate(
    model: HistGradientBoostingClassifier,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    label_order: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    x_values = clean_feature_frame(data, feature_columns)
    expected = data["target"].astype(str)
    predicted = model.predict(x_values)
    probabilities = model.predict_proba(x_values)

    metrics: Dict[str, Any] = {
        "rows": int(len(data)),
        "groups": int(data["_split_group"].nunique()),
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(expected, predicted, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=list(label_order),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=list(label_order),
        ).tolist(),
    }

    # sklearn orders probability columns using model.classes_, not label_order.
    metrics["log_loss"] = float(
        log_loss(expected, probabilities, labels=list(model.classes_))
    )

    predictions = data[["source", "group_id", "_split_group", "target"]].copy()
    if "timestamp" in data.columns:
        predictions["timestamp"] = data["timestamp"].to_numpy()
    predictions["prediction"] = predicted
    predictions["correct"] = predictions["target"] == predictions["prediction"]
    for index, label in enumerate(model.classes_):
        predictions["probability_{}".format(label)] = probabilities[:, index]
    return metrics, predictions


def _domain_metrics(
    model: HistGradientBoostingClassifier,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    label_order: Sequence[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for source, subset in data.groupby("source"):
        metrics, _ = _evaluate(model, subset, feature_columns, label_order)
        result[str(source)] = metrics
    return result


def leave_one_source_out(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    labels: Sequence[str],
    task: str,
    seed: int,
    krisk_policy_weight: float,
) -> Dict[str, Any]:
    """Train on every source except one, then evaluate on the held-out source.

    A source-confounded model scores well on a random group split because the
    test rows look like the training rows. Leave-one-source-out instead exposes
    whether the learned cue actually transfers across domains.
    """

    results: Dict[str, Any] = {}
    sources = sorted(data["source"].astype(str).unique())
    for held_out_source in sources:
        train = data[data["source"].astype(str) != held_out_source]
        test = data[data["source"].astype(str) == held_out_source]
        if train.empty or test.empty:
            continue

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
        x_train = clean_feature_frame(train, feature_columns)
        y_train = train["target"].astype(str)
        sample_weight = np.ones(len(train), dtype=np.float64)
        if task == "policy":
            krisk_rows = train["source"].astype(str).eq("krisk").to_numpy()
            sample_weight[krisk_rows] = max(1.0, krisk_policy_weight)
        model.fit(x_train, y_train, sample_weight=sample_weight)

        metrics, _ = _evaluate(model, test, feature_columns, labels)
        results[str(held_out_source)] = {
            "trained_on": [s for s in sources if s != held_out_source],
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one RoadWeave HistGradientBoosting baseline model."
    )
    parser.add_argument("--task", required=True, choices=tuple(TASK_LABELS))
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Prepared Parquet input. Repeat for multiple files; defaults to both task files.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--krisk-policy-weight", type=float, default=3.0)
    parser.add_argument(
        "--importance-jobs",
        type=int,
        default=1,
        help=(
            "Worker processes for permutation importance. Default 1 is "
            "portable on macOS and restricted research machines."
        ),
    )
    parser.add_argument(
        "--importance-sample-size",
        type=int,
        default=1000,
        help="Maximum held-out rows used for permutation importance. Default: 1000.",
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=2,
        help="Permutations per feature. Default: 2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and group splits without fitting a model.",
    )
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    task = arguments.task
    labels = TASK_LABELS[task]
    input_paths = arguments.data or expected_data_paths(task)
    data = load_task_data(task, input_paths)
    feature_columns, rejected_columns = select_feature_columns(data)

    train, validation, test = make_splits(data, seed=arguments.seed)
    print("Task: {}".format(task))
    print("Rows: {} | Features: {} | Groups: {}".format(
        len(data), len(feature_columns), data["_split_group"].nunique()
    ))
    print("Train: {} | Validation: {} | Test: {}".format(
        len(train), len(validation), len(test)
    ))
    print("Train classes: {}".format(_class_counts(train, labels)))
    print("Validation classes: {}".format(_class_counts(validation, labels)))
    print("Test classes: {}".format(_class_counts(test, labels)))

    if arguments.dry_run:
        print("Dry run passed. No model was trained or written.")
        return 0

    x_train = clean_feature_frame(train, feature_columns)
    y_train = train["target"].astype(str)
    sample_weight = np.ones(len(train), dtype=np.float64)
    if task == "policy":
        krisk_rows = train["source"].astype(str).eq("krisk").to_numpy()
        sample_weight[krisk_rows] = max(1.0, arguments.krisk_policy_weight)

    model = HistGradientBoostingClassifier(
        learning_rate=arguments.learning_rate,
        max_iter=arguments.max_iter,
        max_leaf_nodes=arguments.max_leaf_nodes,
        min_samples_leaf=arguments.min_samples_leaf,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=arguments.seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    validation_metrics, validation_predictions = _evaluate(
        model, validation, feature_columns, labels
    )
    test_metrics, test_predictions = _evaluate(
        model, test, feature_columns, labels
    )

    leave_one_out = leave_one_source_out(
        data,
        feature_columns,
        labels,
        task,
        arguments.seed,
        arguments.krisk_policy_weight,
    )

    importance_sample_size = max(1, arguments.importance_sample_size)
    importance_sample = test
    if len(test) > importance_sample_size:
        importance_sample = test.sample(
            importance_sample_size,
            random_state=arguments.seed,
        )
    importance = permutation_importance(
        model,
        clean_feature_frame(importance_sample, feature_columns),
        importance_sample["target"].astype(str),
        n_repeats=max(1, arguments.importance_repeats),
        random_state=arguments.seed,
        scoring="f1_macro",
        n_jobs=arguments.importance_jobs,
    )
    important_features = sorted(
        [
            {
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(
                feature_columns,
                importance.importances_mean,
                importance.importances_std,
            )
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )[:40]

    now = datetime.now(timezone.utc).isoformat()
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": "{}-baseline-1.0".format(task),
        "task": task,
        "created_utc": now,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "feature_columns": feature_columns,
        "label_order": list(labels),
        "model": model,
        "training": {
            "seed": arguments.seed,
            "input_paths": [str(Path(path).expanduser().resolve()) for path in input_paths],
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    }

    default_model = ml_root() / "models" / "{}_model.joblib".format(task)
    default_report = ml_root() / "reports" / "{}_metrics.json".format(task)
    default_predictions = ml_root() / "reports" / "{}_test_predictions.csv".format(task)
    model_path = (arguments.model_output or default_model).expanduser().resolve()
    report_path = (arguments.report_output or default_report).expanduser().resolve()
    predictions_path = (
        arguments.predictions_output or default_predictions
    ).expanduser().resolve()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    validation_predictions["split"] = "validation"
    test_predictions["split"] = "test"
    pd.concat(
        [validation_predictions, test_predictions],
        ignore_index=True,
    ).to_csv(predictions_path, index=False)

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "created_utc": now,
        "input_paths": artifact["training"]["input_paths"],
        "rows": len(data),
        "groups": int(data["_split_group"].nunique()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "rejected_non_numeric_or_empty_columns": rejected_columns,
        "class_counts": {
            "all": _class_counts(data, labels),
            "train": _class_counts(train, labels),
            "validation": _class_counts(validation, labels),
            "test": _class_counts(test, labels),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_by_source": _domain_metrics(
            model, test, feature_columns, labels
        ),
        "leave_one_source_out": leave_one_out,
        "important_features": important_features,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Validation macro F1: {:.4f}".format(validation_metrics["macro_f1"]))
    print("Test macro F1: {:.4f}".format(test_metrics["macro_f1"]))
    for source, lodo in leave_one_out.items():
        print(
            "Leave-one-source-out {}: macro F1 {:.4f} | accuracy {:.4f}".format(
                source,
                lodo["macro_f1"],
                lodo["accuracy"],
            )
        )
    print("Model: {}".format(model_path))
    print("Metrics: {}".format(report_path))
    print("Predictions: {}".format(predictions_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
