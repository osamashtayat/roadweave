#!/usr/bin/env python3

"""Validate prepared RoadWeave ML tables before training.

This script is intentionally read-only. It reports schema problems, missing
values, impossible physical values, class balance, and possible label leakage.
It does not silently repair the prepared data; corrections belong in the
dataset converter so training and later re-runs stay reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ML.src.model_support import (
        FORBIDDEN_FEATURE_NAMES,
        METADATA_COLUMNS,
        TASK_FILES,
        TASK_LABELS,
        expected_data_paths,
        normalize_targets,
        select_feature_columns,
    )
except ModuleNotFoundError:
    from model_support import (  # type: ignore
        FORBIDDEN_FEATURE_NAMES,
        METADATA_COLUMNS,
        TASK_FILES,
        TASK_LABELS,
        expected_data_paths,
        normalize_targets,
        select_feature_columns,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _task_from_filename(path: Path) -> Optional[str]:
    lower = path.name.lower()
    if lower.startswith("risk_"):
        return "risk"
    if lower.startswith("policy_"):
        return "policy"
    return None


def _physical_bounds(column: str) -> Optional[Tuple[float, float]]:
    lower = column.lower()
    summary_suffixes = ("_now", "_min", "_max", "_mean")
    if not lower.endswith(summary_suffixes):
        if lower.endswith("_std"):
            return (0.0, float("inf"))
        return None

    if "_present_" in lower or lower.startswith("left_clear_") or lower.startswith("right_clear_"):
        return (0.0, 1.0)
    if lower.startswith("ego_accel_"):
        return (-15.0, 15.0)
    if lower.startswith("ego_yaw_rate_"):
        return (-180.0, 180.0)
    if "_gap_" in lower:
        return (0.0, 150.0)
    if "_closing_speed_" in lower:
        return (-70.0, 70.0)
    if lower.startswith("ego_speed_") or "_speed_" in lower:
        return (0.0, 70.0)
    if "_ttc_" in lower:
        return (0.0, 20.0)
    return None


def inspect_file(path: Path, task: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    report: Dict[str, Any] = {
        "path": str(resolved),
        "task": task,
        "errors": [],
        "warnings": [],
    }

    if not resolved.is_file():
        report["errors"].append("File does not exist. Run its converter first.")
        return report

    try:
        data = pd.read_parquet(resolved)
    except Exception as exception:
        report["errors"].append("Could not read Parquet: {}".format(exception))
        return report

    report["rows"] = int(len(data))
    report["columns"] = int(len(data.columns))
    report["column_names"] = list(data.columns)
    report["duplicate_complete_rows"] = int(data.duplicated().sum())

    required = {"source", "group_id", "target"}
    missing_required = sorted(required - set(data.columns))
    if missing_required:
        report["errors"].append(
            "Missing required column(s): {}".format(", ".join(missing_required))
        )
        return report
    if data.empty:
        report["errors"].append("The table contains no rows.")
        return report

    missing_source = int(data["source"].isna().sum())
    missing_groups = int(data["group_id"].isna().sum())
    missing_targets = int(data["target"].isna().sum())
    report["missing_metadata"] = {
        "source": missing_source,
        "group_id": missing_groups,
        "target": missing_targets,
    }
    if missing_source or missing_groups or missing_targets:
        report["errors"].append("Required metadata contains missing values.")

    report["groups"] = int(data["group_id"].nunique(dropna=True))
    group_sizes = data.groupby(["source", "group_id"], dropna=True).size()
    report["rows_per_group"] = {
        "minimum": int(group_sizes.min()) if len(group_sizes) else 0,
        "median": float(group_sizes.median()) if len(group_sizes) else 0.0,
        "maximum": int(group_sizes.max()) if len(group_sizes) else 0,
    }
    report["source_counts"] = {
        str(key): int(value)
        for key, value in data["source"].astype(str).value_counts().items()
    }

    try:
        normalized = normalize_targets(data["target"].tolist(), task)
        report["target_counts"] = {
            label: int((normalized == label).sum())
            for label in TASK_LABELS[task]
        }
        absent = [label for label, count in report["target_counts"].items() if count == 0]
        if absent:
            # A single physical source may legitimately omit a rare class;
            # completeness is checked across all files for the task.
            report["warnings"].append(
                "This source has no examples for target class(es): {}".format(
                    ", ".join(absent)
                )
            )
    except ValueError as exception:
        report["errors"].append(str(exception))

    numeric = data.select_dtypes(include=[np.number])
    report["infinite_numeric_values"] = int(np.isinf(numeric.to_numpy()).sum())
    if report["infinite_numeric_values"]:
        report["errors"].append("Numeric features contain infinite values.")

    missing_rates = numeric.isna().mean().sort_values(ascending=False)
    report["highest_missing_rates"] = {
        str(column): float(rate)
        for column, rate in missing_rates.head(20).items()
    }

    try:
        features, rejected = select_feature_columns(data)
        report["usable_feature_count"] = len(features)
        report["rejected_non_numeric_or_empty_columns"] = rejected
    except ValueError as exception:
        features = []
        report["errors"].append(str(exception))

    suspicious = []
    for column in data.columns:
        lower = column.lower()
        if column in METADATA_COLUMNS:
            continue
        if (
            lower in FORBIDDEN_FEATURE_NAMES
            or lower.startswith("future_")
            or lower.endswith("_label")
            or lower.endswith("_target")
        ) and lower != "target":
            suspicious.append(column)
    report["possible_leakage_columns"] = sorted(suspicious)
    if suspicious:
        report["warnings"].append(
            "Possible label/future columns were found. Training excludes them, but the converter should remove them: {}".format(
                ", ".join(sorted(suspicious))
            )
        )

    invalid_ranges: Dict[str, int] = {}
    for column in features:
        bounds = _physical_bounds(column)
        if bounds is None:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        finite = values[np.isfinite(values)]
        minimum, maximum = bounds
        invalid = (finite < minimum) | (finite > maximum)
        count = int(invalid.sum())
        if count:
            invalid_ranges[column] = count
    report["out_of_range_counts"] = invalid_ranges
    if invalid_ranges:
        report["errors"].append(
            "Physical range checks failed for {} feature column(s). Fix these in the converter.".format(
                len(invalid_ranges)
            )
        )

    if report["duplicate_complete_rows"]:
        report["warnings"].append(
            "Complete duplicate rows were found; verify that events were not imported twice."
        )
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect prepared RoadWeave ML Parquet files without changing them."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional Parquet paths. With no paths, inspect all four expected files.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_LABELS),
        help="Required only when a custom filename does not begin with risk_ or policy_.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "data_inspection.json",
        help="JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    paths: Sequence[Path]
    if arguments.paths:
        paths = arguments.paths
    else:
        paths = [path for task in TASK_FILES for path in expected_data_paths(task)]

    reports: List[Dict[str, Any]] = []
    for path in paths:
        task = arguments.task or _task_from_filename(path)
        if task is None:
            reports.append(
                {
                    "path": str(path.expanduser().resolve()),
                    "task": None,
                    "errors": ["Could not infer task. Pass --task risk or --task policy."],
                    "warnings": [],
                }
            )
            continue
        reports.append(inspect_file(path, task))

    combined_target_counts: Dict[str, Dict[str, int]] = {}
    combined_errors: List[str] = []
    for task, labels in TASK_LABELS.items():
        task_reports = [report for report in reports if report.get("task") == task]
        if not task_reports:
            continue
        combined_target_counts[task] = {
            label: sum(
                int(report.get("target_counts", {}).get(label, 0))
                for report in task_reports
            )
            for label in labels
        }
        # The default four-file audit is the pre-training gate. A custom
        # single-source audit may legitimately omit classes and reports that
        # fact as a warning above.
        if not arguments.paths:
            missing = [
                label
                for label, count in combined_target_counts[task].items()
                if count == 0
            ]
            if missing:
                combined_errors.append(
                    "Combined {} data has no examples for: {}".format(
                        task,
                        ", ".join(missing),
                    )
                )

    summary = {
        "files": reports,
        "combined_target_counts": combined_target_counts,
        "errors": combined_errors,
        "error_count": (
            sum(len(report["errors"]) for report in reports)
            + len(combined_errors)
        ),
        "warning_count": sum(len(report["warnings"]) for report in reports),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    for report in reports:
        print()
        print("=" * 78)
        print(report["path"])
        print("=" * 78)
        if "rows" in report:
            print("Rows: {} | Groups: {} | Columns: {}".format(
                report["rows"], report.get("groups", 0), report["columns"]
            ))
            print("Targets: {}".format(report.get("target_counts", {})))
            print("Usable features: {}".format(report.get("usable_feature_count", 0)))
        for error in report["errors"]:
            print("ERROR: {}".format(error))
        for warning in report["warnings"]:
            print("WARNING: {}".format(warning))

    for error in combined_errors:
        print("ERROR: {}".format(error))

    print()
    print("Report written to {}".format(arguments.output.resolve()))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
