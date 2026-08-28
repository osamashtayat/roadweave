#!/usr/bin/env python3
"""Measure dataset identity leakage and feature drift in RoadWeave tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from ML.src.model_support import (
        clean_feature_frame,
        expected_data_paths,
        load_task_data,
        ml_root,
        select_feature_columns,
        select_transfer_feature_columns,
    )
    from ML.src.train_model import best_group_holdout
except ModuleNotFoundError:
    from model_support import (  # type: ignore
        clean_feature_frame,
        expected_data_paths,
        load_task_data,
        ml_root,
        select_feature_columns,
        select_transfer_feature_columns,
    )
    from train_model import best_group_holdout  # type: ignore


def _distribution(data: pd.DataFrame, field: str) -> Dict[str, Dict[str, int]]:
    table = pd.crosstab(data[field], data["target"])
    return {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in table.iterrows()
    }


def classifier_audit(
    data: pd.DataFrame,
    features: List[str],
    field: str,
    seed: int,
) -> Dict[str, Any]:
    train_indices, test_indices = best_group_holdout(data, 0.20, seed)
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )
    model.fit(clean_feature_frame(train, features), train[field].astype(str))
    expected = test[field].astype(str)
    predicted = model.predict(clean_feature_frame(test, features))
    return {
        "field": field,
        "classes": sorted(data[field].astype(str).unique()),
        "chance_accuracy_majority": float(data[field].value_counts(normalize=True).max()),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(
            expected, predicted, output_dict=True, zero_division=0
        ),
    }


def feature_drift(data: pd.DataFrame, features: List[str]) -> List[Dict[str, Any]]:
    sources = sorted(data["source"].astype(str).unique())
    if len(sources) != 2:
        return []
    left = data[data["source"].astype(str) == sources[0]]
    right = data[data["source"].astype(str) == sources[1]]
    results: List[Dict[str, Any]] = []
    for feature in features:
        left_values = pd.to_numeric(left[feature], errors="coerce")
        right_values = pd.to_numeric(right[feature], errors="coerce")
        left_finite = left_values.dropna().to_numpy(dtype=np.float64)
        right_finite = right_values.dropna().to_numpy(dtype=np.float64)
        if len(left_finite) == 0 or len(right_finite) == 0:
            normalized_distance = float("inf")
        else:
            combined = np.concatenate([left_finite, right_finite])
            scale = max(1e-6, float(np.quantile(combined, 0.75) - np.quantile(combined, 0.25)))
            normalized_distance = float(
                wasserstein_distance(left_finite, right_finite) / scale
            )
        missing_gap = abs(float(left_values.isna().mean() - right_values.isna().mean()))
        score = normalized_distance + 2.0 * missing_gap
        results.append(
            {
                "feature": feature,
                "sources": sources,
                "normalized_wasserstein": normalized_distance,
                "missing_rate_gap": missing_gap,
                "drift_score": score,
            }
        )
    return sorted(results, key=lambda item: item["drift_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("risk", "policy"))
    parser.add_argument("--data", type=Path, action="append")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    data = load_task_data(arguments.task, arguments.data or expected_data_paths(arguments.task))
    features, rejected = select_transfer_feature_columns(data)
    report = {
        "format": "roadweave.domain-audit/1.0",
        "task": arguments.task,
        "rows": len(data),
        "groups": int(data["_split_group"].nunique()),
        "feature_count": len(features),
        "rejected_features": rejected,
        "target_by_source": _distribution(data, "source"),
        "target_by_domain": _distribution(data, "domain"),
        "source_classifier": classifier_audit(data, features, "source", arguments.seed),
        "domain_classifier": classifier_audit(data, features, "domain", arguments.seed + 1),
        "top_feature_drift": feature_drift(data, features)[:60],
    }
    output = arguments.output or ml_root() / "reports" / "{}_domain_audit.json".format(arguments.task)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Source classifier macro F1: {:.4f}".format(report["source_classifier"]["macro_f1"]))
    print("Domain classifier macro F1: {:.4f}".format(report["domain_classifier"]["macro_f1"]))
    print("Report: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
