#!/usr/bin/env python3
"""Generate RoadWeave Experiment 5 model-evaluation tables and figures.

This analysis reuses the frozen risk and policy validation/test predictions and
metrics. It does not retrain or modify either model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "ML" / "reports"
DEFAULT_OUTPUT = PROJECT_ROOT / "ExperimentResults" / "experiment5"

TASKS: Mapping[str, Dict[str, Any]] = {
    "risk": {
        "title": "Risk model",
        "labels": ("LOW", "MODERATE", "HIGH", "EXTREME"),
        "display": ("Low", "Moderate", "High", "Extreme"),
        "metrics": REPORT_ROOT / "risk_metrics.json",
        "predictions": REPORT_ROOT / "risk_test_predictions.csv",
        "color": "#3569B0",
    },
    "policy": {
        "title": "Policy model",
        "labels": ("KEEP", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"),
        "display": ("Keep", "Accelerate", "Decelerate", "Change left", "Change right"),
        "metrics": REPORT_ROOT / "policy_metrics.json",
        "predictions": REPORT_ROOT / "policy_test_predictions.csv",
        "color": "#E07A30",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty Experiment 5 table: {}".format(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_and_validate() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, pd.DataFrame] = {}
    checks: List[Dict[str, Any]] = []

    for task, settings in TASKS.items():
        metrics_path: Path = settings["metrics"]
        predictions_path: Path = settings["predictions"]
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError("Missing {} Experiment 5 source files.".format(task))

        task_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        all_predictions = pd.read_csv(predictions_path)
        required = {"source", "target", "prediction", "split"}
        missing = required - set(all_predictions.columns)
        if missing:
            raise ValueError("{} predictions are missing columns: {}".format(task, sorted(missing)))

        test_predictions = all_predictions[
            all_predictions["split"].astype(str).str.lower() == "test"
        ].copy()
        labels: Tuple[str, ...] = settings["labels"]
        unknown_targets = set(test_predictions["target"].astype(str)) - set(labels)
        unknown_predictions = set(test_predictions["prediction"].astype(str)) - set(labels)
        if unknown_targets or unknown_predictions:
            raise ValueError(
                "{} contains unknown target/prediction labels: {} / {}".format(
                    task, sorted(unknown_targets), sorted(unknown_predictions)
                )
            )

        expected_test = task_metrics["test"]
        calculated_matrix = confusion_matrix(
            test_predictions["target"], test_predictions["prediction"], labels=list(labels)
        )
        calculated_macro_f1 = f1_score(
            test_predictions["target"],
            test_predictions["prediction"],
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
        checks.extend(
            [
                {
                    "task": task,
                    "check": "test_row_count_matches_metrics",
                    "passed": len(test_predictions) == int(expected_test["rows"]),
                    "observed": len(test_predictions),
                    "expected": int(expected_test["rows"]),
                },
                {
                    "task": task,
                    "check": "confusion_matrix_matches_metrics",
                    "passed": calculated_matrix.tolist() == expected_test["confusion_matrix"],
                    "observed": json.dumps(calculated_matrix.tolist()),
                    "expected": json.dumps(expected_test["confusion_matrix"]),
                },
                {
                    "task": task,
                    "check": "macro_f1_matches_metrics",
                    "passed": math.isclose(
                        float(calculated_macro_f1),
                        float(expected_test["macro_f1"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    "observed": float(calculated_macro_f1),
                    "expected": float(expected_test["macro_f1"]),
                },
            ]
        )
        metrics[task] = task_metrics
        predictions[task] = test_predictions

    failed = [check for check in checks if not check["passed"]]
    if failed:
        raise ValueError("Experiment 5 validation failed: {}".format(failed))
    return metrics, predictions, checks


def _confusion_figure(
    task: str,
    data: pd.DataFrame,
    output_dir: Path,
    graph_number: int,
) -> np.ndarray:
    settings = TASKS[task]
    labels = list(settings["labels"])
    display = list(settings["display"])
    matrix = confusion_matrix(data["target"], data["prediction"], labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    size = 7.4 if task == "risk" else 8.3
    fig, ax = plt.subplots(figsize=(size, size - 0.8))
    image = ax.imshow(normalized * 100.0, vmin=0.0, vmax=100.0, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            percent = normalized[row, column] * 100.0
            color = "white" if percent >= 52.0 else "#202020"
            ax.text(
                column,
                row,
                "{:.1f}%\n(n={:,})".format(percent, int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=9.2,
                color=color,
            )
    ax.set_xticks(range(len(labels)), display, rotation=24, ha="right")
    ax.set_yticks(range(len(labels)), display)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Graph {}. {} confusion matrix (test set)".format(graph_number, settings["title"]))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage of actual class")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"graph{graph_number}_{task}_confusion_matrix")
    return matrix


def _per_class_rows(metrics: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, settings in TASKS.items():
        report = metrics[task]["test"]["classification_report"]
        for label, display in zip(settings["labels"], settings["display"]):
            values = report[label]
            rows.append(
                {
                    "task": task,
                    "class": label,
                    "class_display": display,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
    return rows


def _graph18(metrics: Mapping[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), gridspec_kw={"width_ratios": [4, 5]})
    for ax, (task, settings) in zip(axes, TASKS.items()):
        selected = [row for row in rows if row["task"] == task]
        values = [float(row["f1"]) for row in selected]
        supports = [int(row["support"]) for row in selected]
        bars = ax.bar(settings["display"], values, color=settings["color"], alpha=0.86)
        macro_f1 = float(metrics[task]["test"]["macro_f1"])
        ax.axhline(
            macro_f1,
            color="#333333",
            linestyle="--",
            linewidth=1.25,
            label="Macro-F1 = {:.3f}".format(macro_f1),
        )
        for bar, value, support in zip(bars, values, supports):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                "{:.3f}\nn={:,}".format(value, support),
                ha="center",
                va="bottom",
                fontsize=8.7,
            )
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("F1 score")
        ax.set_title(settings["title"])
        ax.tick_params(axis="x", rotation=24)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(loc="lower left")
    fig.suptitle("Graph 18. Per-class test F1 scores")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output_dir, "graph18_per_class_f1")


def _cross_source_rows(metrics: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mappings = (
        ("nuScenes → K-Risk", "krisk", "nuscenes", "krisk"),
        ("K-Risk → nuScenes", "nuscenes", "krisk", "nuscenes"),
    )
    for direction, held_out_key, training_source, evaluation_source in mappings:
        rows.append(
            {
                "transfer_direction": direction,
                "training_source": training_source,
                "evaluation_source": evaluation_source,
                "risk_macro_f1": metrics["risk"]["leave_one_source_out"][held_out_key]["macro_f1"],
                "policy_macro_f1": metrics["policy"]["leave_one_source_out"][held_out_key]["macro_f1"],
            }
        )
    return rows


def _graph19(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    directions = [row["transfer_direction"] for row in rows]
    risk = [float(row["risk_macro_f1"]) for row in rows]
    policy = [float(row["policy_macro_f1"]) for row in rows]
    x_values = np.arange(len(directions))
    width = 0.33
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    risk_bars = ax.bar(x_values - width / 2, risk, width, label="Risk model", color="#3569B0")
    policy_bars = ax.bar(x_values + width / 2, policy, width, label="Policy model", color="#E07A30")
    for bars in (risk_bars, policy_bars):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center")
    ax.set_xticks(x_values, directions)
    ax.set_ylim(0.0, 0.65)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Graph 19. Cross-source transfer performance")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "graph19_cross_source_transfer")


def _distribution_rows(predictions: Mapping[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, settings in TASKS.items():
        data = predictions[task]
        for source in ("nuscenes", "krisk"):
            selected = data[data["source"].astype(str).str.lower() == source]
            total = len(selected)
            if total == 0:
                raise ValueError("No {} test rows found for {}.".format(source, task))
            counts = selected["target"].value_counts()
            for label, display in zip(settings["labels"], settings["display"]):
                count = int(counts.get(label, 0))
                rows.append(
                    {
                        "task": task,
                        "source": source,
                        "class": label,
                        "class_display": display,
                        "count": count,
                        "source_test_rows": total,
                        "percentage_within_source": count / total,
                    }
                )
    return rows


def _graph20(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [4, 5]})
    width = 0.36
    source_styles = (("nuscenes", "nuScenes", "#3569B0"), ("krisk", "K-Risk", "#6BAED6"))
    for ax, (task, settings) in zip(axes, TASKS.items()):
        positions = np.arange(len(settings["labels"]))
        for offset_index, (source, source_display, color) in enumerate(source_styles):
            selected = {
                row["class"]: row
                for row in rows
                if row["task"] == task and row["source"] == source
            }
            values = [float(selected[label]["percentage_within_source"]) * 100.0 for label in settings["labels"]]
            counts = [int(selected[label]["count"]) for label in settings["labels"]]
            offset = (-width / 2) if offset_index == 0 else (width / 2)
            bars = ax.bar(positions + offset, values, width, label=source_display, color=color)
            for bar, value, count in zip(bars, values, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.2,
                    "{:.1f}%\nn={:,}".format(value, count),
                    ha="center",
                    va="bottom",
                    fontsize=7.7,
                )
        ax.set_xticks(positions, settings["display"], rotation=24, ha="right")
        ax.set_ylim(0.0, 94.0)
        ax.set_ylabel("Percentage within source test set")
        ax.set_title(settings["title"])
        ax.grid(axis="y", alpha=0.22)
        ax.legend()
    fig.suptitle("Graph 20. Held-out test class distributions by data source")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output_dir, "graph20_dataset_class_distributions")


def analyze(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions, checks = _load_and_validate()

    model_rows = []
    confusion_rows: List[Dict[str, Any]] = []
    for graph_number, task in ((16, "risk"), (17, "policy")):
        task_metrics = metrics[task]["test"]
        model_rows.append(
            {
                "task": task,
                "test_rows": int(task_metrics["rows"]),
                "accuracy": task_metrics["accuracy"],
                "balanced_accuracy": task_metrics["balanced_accuracy"],
                "macro_f1": task_metrics["macro_f1"],
                "weighted_f1": task_metrics["weighted_f1"],
                "log_loss": task_metrics["log_loss"],
            }
        )
        matrix = _confusion_figure(task, predictions[task], output_dir, graph_number)
        labels = TASKS[task]["labels"]
        for actual_index, actual in enumerate(labels):
            actual_total = int(matrix[actual_index].sum())
            for predicted_index, predicted in enumerate(labels):
                count = int(matrix[actual_index, predicted_index])
                confusion_rows.append(
                    {
                        "task": task,
                        "actual_class": actual,
                        "predicted_class": predicted,
                        "count": count,
                        "actual_class_total": actual_total,
                        "row_percentage": count / actual_total if actual_total else math.nan,
                    }
                )

    per_class = _per_class_rows(metrics)
    cross_source = _cross_source_rows(metrics)
    distributions = _distribution_rows(predictions)
    _graph18(metrics, per_class, output_dir)
    _graph19(cross_source, output_dir)
    _graph20(distributions, output_dir)

    _write_csv(output_dir / "experiment5_model_summary.csv", model_rows)
    _write_csv(output_dir / "experiment5_confusion_matrices.csv", confusion_rows)
    _write_csv(output_dir / "experiment5_per_class_metrics.csv", per_class)
    _write_csv(output_dir / "experiment5_cross_source_transfer.csv", cross_source)
    _write_csv(output_dir / "experiment5_class_distributions.csv", distributions)
    _write_csv(output_dir / "experiment5_validation_checks.csv", checks)

    provenance = {
        "experiment": 5,
        "analysis": "existing risk and policy model reports; no retraining",
        "test_split_filter": "split == test",
        "source_file_hashes_sha256": {
            str(settings[key].relative_to(PROJECT_ROOT)): _sha256(settings[key])
            for settings in TASKS.values()
            for key in ("metrics", "predictions")
        },
        "all_validation_checks_passed": all(check["passed"] for check in checks),
    }
    (output_dir / "experiment5_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Experiment 5 complete: {}".format(output_dir))
    print("Validated test rows: risk={}, policy={}".format(len(predictions["risk"]), len(predictions["policy"])))
    print("Generated Graphs 16–20 as PNG and PDF.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyze(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
