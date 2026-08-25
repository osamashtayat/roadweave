#!/usr/bin/env python3

"""Load a saved RoadWeave model and explain one prepared-data prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    from ML.src.model_support import (
        clean_feature_frame,
        expected_data_paths,
        load_artifact,
        load_task_data,
        ml_root,
    )
except ModuleNotFoundError:
    from model_support import (  # type: ignore
        clean_feature_frame,
        expected_data_paths,
        load_artifact,
        load_task_data,
        ml_root,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prediction from a saved RoadWeave model."
    )
    parser.add_argument("--task", choices=("risk", "policy"), default="policy")
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Prepared Parquet input. Repeat to combine files.",
    )
    parser.add_argument(
        "--row",
        type=int,
        help="Exact zero-based row. Without it, choose a reproducible random row.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable prediction record.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    model_path = arguments.model or (
        ml_root() / "models" / "{}_model.joblib".format(arguments.task)
    )
    artifact = load_artifact(model_path, expected_task=arguments.task)
    data = load_task_data(
        arguments.task,
        arguments.data or expected_data_paths(arguments.task),
    )

    if arguments.row is None:
        row_index = int(
            np.random.default_rng(arguments.seed).integers(0, len(data))
        )
    else:
        row_index = arguments.row
    if row_index < 0 or row_index >= len(data):
        raise IndexError(
            "Row {} is outside the valid range 0 through {}.".format(
                row_index,
                len(data) - 1,
            )
        )

    row = data.iloc[row_index]
    feature_columns = list(artifact["feature_columns"])
    feature_frame = clean_feature_frame(
        pd.DataFrame([row]),
        feature_columns,
    )
    model = artifact["model"]
    prediction = str(model.predict(feature_frame)[0])
    probabilities = model.predict_proba(feature_frame)[0]
    probability_map = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    result: Dict[str, Any] = {
        "task": arguments.task,
        "model_version": artifact.get("model_version", "unknown"),
        "row": row_index,
        "source": str(row["source"]),
        "group_id": str(row["group_id"]),
        "expected": str(row["target"]),
        "predicted": prediction,
        "correct": prediction == str(row["target"]),
        "probabilities": dict(
            sorted(
                probability_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }
    if "timestamp" in row.index and pd.notna(row["timestamp"]):
        result["timestamp"] = float(row["timestamp"])

    print("Model: {}".format(Path(model_path).expanduser().resolve()))
    print("Source: {} | Group: {}".format(result["source"], result["group_id"]))
    print("Expected: {}".format(result["expected"]))
    print("Predicted: {}".format(result["predicted"]))
    print("Probabilities:")
    for label, probability in result["probabilities"].items():
        print("  {:15s} {:.4f}".format(label, probability))

    if arguments.json_output:
        output_path = arguments.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("JSON result: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
