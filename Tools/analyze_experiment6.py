#!/usr/bin/env python3
"""Generate the evidence-backed offline figures for RoadWeave Experiment 6.

Graphs 21 and 22 use the frozen sensor-reliability test split. Graph 24 uses
the frozen official weather validation split. Graph 23 is intentionally not
created here because it requires a new closed-loop controller experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = PROJECT_ROOT / "ML"
DEFAULT_OUTPUT = PROJECT_ROOT / "ExperimentResults" / "experiment6"
SENSOR_DATA = ML_ROOT / "data" / "processed" / "sensor_reliability.parquet"
SENSOR_METRICS = ML_ROOT / "reports" / "sensor_reliability_metrics.json"
SENSOR_PREDICTIONS = ML_ROOT / "reports" / "sensor_reliability_predictions.csv"
WEATHER_METRICS = ML_ROOT / "reports" / "weather_metrics.json"
WEATHER_PREDICTIONS = ML_ROOT / "reports" / "weather_test_predictions.csv"

SENSORS = ("CAMERA", "LIDAR", "RADAR")
SENSOR_DISPLAY = {"CAMERA": "Camera", "LIDAR": "LiDAR", "RADAR": "Radar"}
SENSOR_COLORS = {"CAMERA": "#3569B0", "LIDAR": "#58A65C", "RADAR": "#E07A30"}
METHODS = ("Confidence only", "Direct formula", "Thresholds", "ML model")
WEATHER_ORDER = ("OVERALL", "DRY", "FOG", "RAIN", "SNOW")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty Experiment 6 table: {}".format(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def direct_reliability_formula(data: pd.DataFrame) -> np.ndarray:
    """Observable-feature heuristic fixed before evaluating the test split.

    The formula mirrors the conceptual label structure without using its
    oracle precision, recall, range-error, or velocity-error columns.
    """
    delivery = 1.0 - data["dropout_rate"].to_numpy(dtype=float)
    freshness = np.exp(-data["message_age_mean"].to_numpy(dtype=float) / 0.5)
    availability = 0.65 * delivery + 0.35 * freshness
    confidence = np.clip(data["confidence_mean"].to_numpy(dtype=float), 0.0, 1.0)
    continuity = np.clip(data["track_continuity"].to_numpy(dtype=float), 0.0, 1.0)
    innovation_quality = np.exp(-data["innovation_mean"].to_numpy(dtype=float) / 2.0)
    agreement_quality = np.exp(
        -data["cross_sensor_disagreement"].to_numpy(dtype=float) / 20.0
    )
    observation_quality = (
        0.45 * confidence
        + 0.25 * continuity
        + 0.15 * innovation_quality
        + 0.15 * agreement_quality
    )
    return np.clip(availability * observation_quality, 0.0, 1.0)


def threshold_reliability(data: pd.DataFrame) -> np.ndarray:
    """Simple, non-learned dropout/message-age reliability categories."""
    dropout = data["dropout_rate"].to_numpy(dtype=float)
    age_mean = data["message_age_mean"].to_numpy(dtype=float)
    age_max = data["message_age_max"].to_numpy(dtype=float)
    failed = (dropout >= 0.90) | (age_max >= 2.00)
    degraded = (dropout >= 0.40) | (age_mean >= 0.60) | (age_max >= 1.00)
    acceptable = (dropout >= 0.15) | (age_mean >= 0.25) | (age_max >= 0.50)
    return np.select(
        [failed, degraded, acceptable],
        [0.20, 0.45, 0.72],
        default=0.95,
    ).astype(float)


def _load_sensor_evidence() -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    data = pd.read_parquet(SENSOR_DATA)
    data = data[data["split"].astype(str).str.lower() == "test"].copy()
    predictions = pd.read_csv(SENSOR_PREDICTIONS)
    predictions = predictions[
        predictions["split"].astype(str).str.lower() == "test"
    ].copy()
    prediction_columns = ["scenario_id", "sensor_type", "prediction"]
    if predictions.duplicated(["scenario_id", "sensor_type"]).any():
        raise ValueError("Sensor test predictions contain duplicate scenario/sensor keys.")
    merged = data.merge(
        predictions[prediction_columns],
        on=["scenario_id", "sensor_type"],
        how="left",
        validate="one_to_one",
    )
    if merged["prediction"].isna().any():
        raise ValueError("Some sensor test rows do not have saved ML predictions.")
    metrics = json.loads(SENSOR_METRICS.read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []
    for sensor in SENSORS:
        selected = merged[merged["sensor_type"] == sensor]
        expected = selected["target_reliability"].to_numpy(dtype=float)
        learned = selected["prediction"].to_numpy(dtype=float)
        confidence = selected["confidence_mean"].to_numpy(dtype=float)
        saved = metrics["sensors"][sensor]["test"]
        observed_ml = float(mean_absolute_error(expected, learned))
        observed_confidence = float(mean_absolute_error(expected, confidence))
        checks.extend(
            [
                {
                    "evidence": "sensor",
                    "condition": sensor,
                    "check": "test_rows_match",
                    "passed": len(selected) == int(metrics["sensors"][sensor]["test_rows"]),
                    "observed": len(selected),
                    "expected": int(metrics["sensors"][sensor]["test_rows"]),
                },
                {
                    "evidence": "sensor",
                    "condition": sensor,
                    "check": "ml_mae_matches_saved_report",
                    "passed": math.isclose(observed_ml, float(saved["mae"]), abs_tol=1e-12),
                    "observed": observed_ml,
                    "expected": float(saved["mae"]),
                },
                {
                    "evidence": "sensor",
                    "condition": sensor,
                    "check": "confidence_mae_matches_saved_report",
                    "passed": math.isclose(
                        observed_confidence,
                        float(saved["confidence_baseline"]["mae"]),
                        abs_tol=1e-12,
                    ),
                    "observed": observed_confidence,
                    "expected": float(saved["confidence_baseline"]["mae"]),
                },
            ]
        )
    return merged, metrics, checks


def _graph21(data: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    direct = direct_reliability_formula(data)
    thresholds = threshold_reliability(data)
    work = data.copy()
    work["direct_prediction"] = direct
    work["threshold_prediction"] = thresholds
    columns = {
        "Confidence only": "confidence_mean",
        "Direct formula": "direct_prediction",
        "Thresholds": "threshold_prediction",
        "ML model": "prediction",
    }
    for method, column in columns.items():
        for sensor in SENSORS:
            selected = work[work["sensor_type"] == sensor]
            mae = float(
                mean_absolute_error(selected["target_reliability"], selected[column])
            )
            rows.append(
                {
                    "method": method,
                    "sensor": sensor,
                    "sensor_display": SENSOR_DISPLAY[sensor],
                    "test_rows": len(selected),
                    "mae": mae,
                }
            )

    x_values = np.arange(len(METHODS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.3, 5.6))
    for sensor_index, sensor in enumerate(SENSORS):
        values = [
            next(row["mae"] for row in rows if row["method"] == method and row["sensor"] == sensor)
            for method in METHODS
        ]
        offset = (sensor_index - 1) * width
        bars = ax.bar(
            x_values + offset,
            values,
            width,
            label=SENSOR_DISPLAY[sensor],
            color=SENSOR_COLORS[sensor],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.1,
                rotation=90 if value < 0.03 else 0,
            )
    ax.set_xticks(x_values, METHODS)
    ax.set_ylabel("Mean absolute error (reliability score)")
    ax.set_title("Graph 21. Sensor-reliability MAE by estimation method")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    _save_figure(fig, output_dir, "graph21_reliability_mae_by_method")
    return rows


def _graph22(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.0), sharex=True, sharey=True)
    for ax, sensor in zip(axes, SENSORS):
        selected = data[data["sensor_type"] == sensor]
        actual = selected["target_reliability"].to_numpy(dtype=float)
        predicted = selected["prediction"].to_numpy(dtype=float)
        mae = float(mean_absolute_error(actual, predicted))
        ax.scatter(
            actual,
            predicted,
            s=12,
            alpha=0.22,
            color=SENSOR_COLORS[sensor],
            linewidth=0,
            rasterized=True,
        )
        ax.plot([0, 1], [0, 1], color="#202020", linestyle="--", linewidth=1.2)
        ax.text(
            0.04,
            0.95,
            "MAE = {:.3f}\nn = {:,}".format(mae, len(selected)),
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"},
        )
        ax.set_title(SENSOR_DISPLAY[sensor])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True reliability")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("ML-predicted reliability")
    fig.suptitle("Graph 22. Learned reliability versus simulated truth (test set)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_dir, "graph22_reliability_vs_true_score")


def _load_weather_evidence() -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    predictions = pd.read_csv(WEATHER_PREDICTIONS)
    data = predictions[predictions["evaluation"] == "official_validation"].copy()
    metrics = json.loads(WEATHER_METRICS.read_text(encoding="utf-8"))
    expected = data["target_speed_factor"].to_numpy(dtype=float)
    learned = data["prediction"].to_numpy(dtype=float)
    fixed = data["fixed_rule_prediction"].to_numpy(dtype=float)
    report = metrics["official_validation"]
    checks = [
        {
            "evidence": "weather",
            "condition": "OVERALL",
            "check": "validation_rows_match",
            "passed": len(data) == int(report["rows"]),
            "observed": len(data),
            "expected": int(report["rows"]),
        },
        {
            "evidence": "weather",
            "condition": "OVERALL",
            "check": "ml_mae_matches_saved_report",
            "passed": math.isclose(
                float(mean_absolute_error(expected, learned)),
                float(report["mae"]),
                abs_tol=1e-12,
            ),
            "observed": float(mean_absolute_error(expected, learned)),
            "expected": float(report["mae"]),
        },
        {
            "evidence": "weather",
            "condition": "OVERALL",
            "check": "fixed_rule_mae_matches_saved_report",
            "passed": math.isclose(
                float(mean_absolute_error(expected, fixed)),
                float(report["fixed_rule_baseline"]["mae"]),
                abs_tol=1e-12,
            ),
            "observed": float(mean_absolute_error(expected, fixed)),
            "expected": float(report["fixed_rule_baseline"]["mae"]),
        },
    ]
    return data, metrics, checks


def _graph24(data: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for condition in WEATHER_ORDER:
        selected = data if condition == "OVERALL" else data[data["condition"] == condition]
        expected = selected["target_speed_factor"].to_numpy(dtype=float)
        learned = selected["prediction"].to_numpy(dtype=float)
        fixed = selected["fixed_rule_prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition.title(),
                "validation_rows": len(selected),
                "fixed_rule_mae": float(mean_absolute_error(expected, fixed)),
                "learned_model_mae": float(mean_absolute_error(expected, learned)),
            }
        )

    positions = np.arange(len(rows))
    width = 0.35
    fixed_values = [row["fixed_rule_mae"] for row in rows]
    learned_values = [row["learned_model_mae"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    fixed_bars = ax.bar(
        positions - width / 2,
        fixed_values,
        width,
        label="Fixed percentage rule",
        color="#8A8A8A",
    )
    learned_bars = ax.bar(
        positions + width / 2,
        learned_values,
        width,
        label="Learned weather model",
        color="#7B4FB3",
    )
    for bars, values in ((fixed_bars, fixed_values), (learned_bars, learned_values)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
            )
    labels = ["{}\nn={:,}".format(row["condition"], row["validation_rows"]) for row in rows]
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Speed-factor MAE (0–1 scale)")
    ax.set_title("Graph 24. Weather model versus fixed percentage rule")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    ax.text(
        0.01,
        -0.20,
        "Rain and snow have very small official-validation samples; do not infer strong generalization.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "graph24_weather_model_vs_fixed_rule")
    return rows


def _write_status_note(output_dir: Path) -> None:
    text = """# Experiment 6 status

Generated from existing frozen evidence:

- Graph 21: offline reliability MAE for confidence-only, direct observable-feature formula, simple dropout/message-age thresholds, and the learned model.
- Graph 22: learned reliability versus simulated truth for camera, LiDAR, and radar.
- Graph 24: learned weather-speed tendency model versus the fixed percentage fallback on the official validation split.

Not generated:

- Graph 23 requires new paired, closed-loop driving runs. The required treatments are no adaptation, threshold adaptation, learned adaptation, and perfect simulated reliability under identical fault scenarios and seeds. Offline prediction rows cannot establish collision rate, route completion, unnecessary stopping, or minimum TTC.
- The coordinate-accuracy error distribution requires known source poses to be passed through the canonical and Unity transforms and recorded. No such paired coordinate evidence currently exists.

Baseline definitions for Graph 21:

- Confidence only: the sensor window's mean detection confidence.
- Direct formula: observable confidence, track continuity, innovation and cross-sensor agreement multiplied by delivery/freshness availability. It does not use oracle label components.
- Thresholds: fixed reliability levels selected only from dropout and message-age limits.
- ML model: the frozen per-sensor HistGradientBoostingRegressor predictions.

Graph 24 estimates observed adverse-weather speed tendencies. It does not establish an optimal safe autonomous-driving speed. Rain and snow each have fewer than 20 official-validation rows.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def analyze(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_data, _, sensor_checks = _load_sensor_evidence()
    graph21_rows = _graph21(sensor_data, output_dir)
    _graph22(sensor_data, output_dir)
    weather_data, _, weather_checks = _load_weather_evidence()
    graph24_rows = _graph24(weather_data, output_dir)
    checks = sensor_checks + weather_checks
    failed = [check for check in checks if not check["passed"]]
    if failed:
        raise ValueError("Experiment 6 source validation failed: {}".format(failed))

    _write_csv(output_dir / "experiment6_reliability_method_mae.csv", graph21_rows)
    _write_csv(output_dir / "experiment6_weather_method_mae.csv", graph24_rows)
    _write_csv(output_dir / "experiment6_validation_checks.csv", checks)
    _write_status_note(output_dir)
    provenance = {
        "experiment": 6,
        "generated_graphs": [21, 22, 24],
        "pending_graphs": [23],
        "source_file_hashes_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                SENSOR_DATA,
                SENSOR_METRICS,
                SENSOR_PREDICTIONS,
                WEATHER_METRICS,
                WEATHER_PREDICTIONS,
            )
        },
        "all_source_validation_checks_passed": True,
    }
    (output_dir / "experiment6_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Experiment 6 available figures complete: {}".format(output_dir))
    print("Generated Graphs 21, 22, and 24. Graph 23 correctly remains pending closed-loop tests.")


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
