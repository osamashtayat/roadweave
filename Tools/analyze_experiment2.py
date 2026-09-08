#!/usr/bin/env python3
"""Analyze RoadWeave Experiment 2 update-rate CSV files.

The Unity recorder writes one summary, snapshot, and frame CSV for each run.
This script excludes invalid runs, aggregates the valid repetitions by input
rate, and creates the figures/table used by the experiment report.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_RATES = (2, 10, 30, 50, 100)
SUMMARY_SUFFIX = "_summary.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RoadWeave Experiment 2 tables and paper figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("ExperimentResults/experiment2"),
        help="Directory containing Unity's Experiment 2 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis directory (default: <results-dir>/analysis).",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="Expected valid repetitions per update rate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any rate has fewer valid runs than expected.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError):
        return math.nan


def as_int(value: str | None) -> int:
    number = as_float(value)
    return int(round(number)) if math.isfinite(number) else 0


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: Iterable[float]) -> float:
    clean = finite(values)
    return float(np.mean(clean)) if clean else math.nan


def sample_sd(values: Iterable[float]) -> float:
    clean = finite(values)
    return float(np.std(clean, ddof=1)) if len(clean) >= 2 else math.nan


def percentile(values: Iterable[float], percent: float) -> float:
    clean = finite(values)
    return float(np.percentile(clean, percent)) if clean else math.nan


def number(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6f}"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_runs(results_dir: Path) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    valid_runs: List[Dict[str, object]] = []
    invalid_runs: List[Dict[str, object]] = []

    for summary_path in sorted(results_dir.glob(f"*{SUMMARY_SUFFIX}")):
        rows = read_rows(summary_path)
        if len(rows) != 1:
            invalid_runs.append(
                {
                    "summary_file": summary_path.name,
                    "run_id": "",
                    "configured_rate_hz": "",
                    "completion_reason": "malformed_summary",
                    "collision_detected": "",
                }
            )
            continue

        row = rows[0]
        rate = as_int(row.get("configured_rate_hz"))
        stem = summary_path.name[: -len(SUMMARY_SUFFIX)]
        snapshot_path = results_dir / f"{stem}_snapshots.csv"
        frame_path = results_dir / f"{stem}_frames.csv"
        run = {
            "summary_path": summary_path,
            "snapshot_path": snapshot_path,
            "frame_path": frame_path,
            "summary": row,
            "rate": rate,
        }

        files_exist = snapshot_path.exists() and frame_path.exists()
        if as_bool(row.get("valid_for_analysis")) and files_exist and rate > 0:
            valid_runs.append(run)
        else:
            invalid_runs.append(
                {
                    "summary_file": summary_path.name,
                    "run_id": row.get("run_id", ""),
                    "configured_rate_hz": row.get("configured_rate_hz", ""),
                    "completion_reason": (
                        row.get("completion_reason", "")
                        if files_exist
                        else "missing_snapshot_or_frame_csv"
                    ),
                    "collision_detected": row.get("collision_detected", ""),
                }
            )

    return valid_runs, invalid_runs


def grouped_runs(valid_runs: Sequence[Dict[str, object]]) -> Dict[int, List[Dict[str, object]]]:
    groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for run in valid_runs:
        groups[int(run["rate"])].append(run)
    return groups


def summary_value(run: Dict[str, object], key: str) -> float:
    summary = run.get("cleaned_summary", run["summary"])
    assert isinstance(summary, dict)
    return as_float(summary.get(key))


def prepare_cleaned_metrics(valid_runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    audit_rows: List[Dict[str, object]] = []
    for run in valid_runs:
        raw_summary = run["summary"]
        assert isinstance(raw_summary, dict)
        cleaned = dict(raw_summary)
        unique_rows, duplicate_rows = load_unique_snapshot_rows(run["snapshot_path"])
        latencies = finite(as_float(row.get("latency_ms")) for row in unique_rows)
        applied = len(unique_rows)
        scheduled = as_int(raw_summary.get("scheduled_messages"))
        received = as_int(raw_summary.get("received_messages"))
        duration = as_float(raw_summary.get("actual_duration_s"))
        missing_scheduled = max(0, scheduled - applied)
        motion = load_motion_metrics(run["frame_path"])

        cleaned["applied_snapshots"] = str(applied)
        cleaned["total_dropped"] = str(missing_scheduled)
        cleaned["scheduled_sequence_coverage_percentage"] = number(
            100.0 * (scheduled - missing_scheduled) / scheduled if scheduled > 0 else math.nan
        )
        cleaned["transport_boundary_delta"] = str(received - scheduled)
        cleaned["applied_boundary_delta"] = str(applied - scheduled)
        cleaned["effective_applied_rate_hz"] = number(
            applied / duration if duration > 0 else math.nan
        )
        cleaned["mean_latency_ms"] = number(mean(latencies))
        cleaned["median_latency_ms"] = number(percentile(latencies, 50.0))
        cleaned["p95_latency_ms"] = number(percentile(latencies, 95.0))
        cleaned["p99_latency_ms"] = number(percentile(latencies, 99.0))
        cleaned["maximum_latency_ms"] = number(max(latencies) if latencies else math.nan)
        cleaned.update({key: number(value) for key, value in motion.items()})
        run["cleaned_summary"] = cleaned

        audit_rows.append(
            {
                "run_id": raw_summary.get("run_id", ""),
                "configured_rate_hz": raw_summary.get("configured_rate_hz", ""),
                "raw_snapshot_callback_rows": raw_summary.get("snapshot_rows", ""),
                "unique_source_sequences": applied,
                "freshness_callback_duplicates_removed": duplicate_rows,
                "scheduled_messages": scheduled,
                "raw_transport_received_messages": received,
                "unique_applied_sequences": applied,
                "transport_boundary_delta": received - scheduled,
                "applied_boundary_delta": applied - scheduled,
                "raw_mean_latency_ms": raw_summary.get("mean_latency_ms", ""),
                "cleaned_mean_latency_ms": cleaned["mean_latency_ms"],
                "raw_p99_latency_ms": raw_summary.get("p99_latency_ms", ""),
                "cleaned_p99_latency_ms": cleaned["p99_latency_ms"],
            }
        )
    return audit_rows


def write_run_summary(output_dir: Path, valid_runs: Sequence[Dict[str, object]]) -> Path:
    fields = [
        "run_id",
        "configured_rate_hz",
        "actual_duration_s",
        "scheduled_messages",
        "received_messages",
        "applied_snapshots",
        "total_dropped",
        "scheduled_sequence_coverage_percentage",
        "transport_boundary_delta",
        "applied_boundary_delta",
        "effective_applied_rate_hz",
        "median_latency_ms",
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "maximum_latency_ms",
        "mean_state_age_ms",
        "p95_state_age_ms",
        "p99_state_age_ms",
        "mean_fps",
        "mean_frame_time_ms",
        "p95_frame_time_ms",
        "allocated_memory_mean_mb",
        "position_step_mean_m",
        "position_step_sd_m",
        "position_step_cv",
        "zero_motion_frame_percentage",
        "displayed_speed_median_mps",
        "displayed_speed_p95_mps",
        "displayed_speed_cv",
    ]
    rows = []
    for run in sorted(valid_runs, key=lambda item: (int(item["rate"]), str(item["summary_path"]))):
        summary = run.get("cleaned_summary", run["summary"])
        assert isinstance(summary, dict)
        rows.append({field: summary.get(field, "") for field in fields})
    path = output_dir / "experiment2_valid_run_summary.csv"
    write_csv(path, fields, rows)
    return path


def write_rate_statistics(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> tuple[Path, Path]:
    metrics = [
        "effective_applied_rate_hz",
        "scheduled_sequence_coverage_percentage",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "maximum_latency_ms",
        "mean_state_age_ms",
        "p95_state_age_ms",
        "p99_state_age_ms",
        "mean_fps",
        "allocated_memory_mean_mb",
        "position_step_mean_m",
        "position_step_cv",
        "zero_motion_frame_percentage",
        "displayed_speed_median_mps",
        "displayed_speed_p95_mps",
        "displayed_speed_cv",
    ]
    fields = ["configured_rate_hz", "valid_runs"]
    for metric in metrics:
        fields.extend((f"{metric}_mean", f"{metric}_sd"))

    rows: List[Dict[str, object]] = []
    for rate in EXPECTED_RATES:
        runs = groups.get(rate, [])
        row: Dict[str, object] = {
            "configured_rate_hz": rate,
            "valid_runs": len(runs),
        }
        for metric in metrics:
            values = [summary_value(run, metric) for run in runs]
            row[f"{metric}_mean"] = number(mean(values))
            row[f"{metric}_sd"] = number(sample_sd(values))
        rows.append(row)

    statistics_path = output_dir / "experiment2_rate_statistics.csv"
    write_csv(statistics_path, fields, rows)

    fps_fields = [
        "input_rate_hz",
        "valid_runs_n",
        "mean_unity_fps",
        "sd_unity_fps",
        "mean_frame_time_ms",
        "sd_frame_time_ms",
    ]
    fps_rows: List[Dict[str, object]] = []
    for rate in EXPECTED_RATES:
        runs = groups.get(rate, [])
        fps_values = [summary_value(run, "mean_fps") for run in runs]
        frame_values = [summary_value(run, "mean_frame_time_ms") for run in runs]
        fps_rows.append(
            {
                "input_rate_hz": rate,
                "valid_runs_n": len(runs),
                "mean_unity_fps": number(mean(fps_values)),
                "sd_unity_fps": number(sample_sd(fps_values)),
                "mean_frame_time_ms": number(mean(frame_values)),
                "sd_frame_time_ms": number(sample_sd(frame_values)),
            }
        )
    fps_path = output_dir / "table_experiment2_unity_fps.csv"
    write_csv(fps_path, fps_fields, fps_rows)
    return statistics_path, fps_path


def write_graph_data(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    latency_rows: List[Dict[str, object]] = []
    message_rows: List[Dict[str, object]] = []
    state_age_by_rate: Dict[int, List[float]] = {}

    for rate in EXPECTED_RATES:
        runs = groups.get(rate, [])
        latencies: List[float] = []
        duplicate_callbacks = 0
        unique_applied: List[float] = []
        for run in runs:
            rows, duplicates = load_unique_snapshot_rows(run["snapshot_path"])
            duplicate_callbacks += duplicates
            unique_applied.append(float(len(rows)))
            latencies.extend(finite(as_float(row.get("latency_ms")) for row in rows))

        q1 = percentile(latencies, 25.0)
        q3 = percentile(latencies, 75.0)
        upper_fence = q3 + 1.5 * (q3 - q1)
        latency_rows.append(
            {
                "input_rate_hz": rate,
                "latency_observations_n": len(latencies),
                "minimum_ms": number(min(latencies) if latencies else math.nan),
                "q1_ms": number(q1),
                "median_ms": number(percentile(latencies, 50.0)),
                "q3_ms": number(q3),
                "p95_ms": number(percentile(latencies, 95.0)),
                "p99_ms": number(percentile(latencies, 99.0)),
                "maximum_ms": number(max(latencies) if latencies else math.nan),
                "upper_outlier_fence_ms": number(upper_fence),
                "observations_above_fence": sum(value > upper_fence for value in latencies),
                "freshness_callbacks_removed": duplicate_callbacks,
            }
        )

        sent = [summary_value(run, "scheduled_messages") for run in runs]
        received = [summary_value(run, "received_messages") for run in runs]
        dropped = [
            max(0.0, sent_count - applied_count)
            for sent_count, applied_count in zip(sent, unique_applied)
        ]
        message_rows.append(
            {
                "input_rate_hz": rate,
                "independent_runs_n": len(runs),
                "scheduled_mean": number(mean(sent)),
                "raw_transport_received_mean": number(mean(received)),
                "unique_applied_mean": number(mean(unique_applied)),
                "missing_scheduled_mean": number(mean(dropped)),
                "scheduled_sequence_coverage_percent": number(mean(
                    100.0 * (scheduled - missing) / scheduled if scheduled > 0 else math.nan
                    for scheduled, missing in zip(sent, dropped)
                )),
                "raw_transport_minus_scheduled_mean": number(mean(
                    raw_received - scheduled for raw_received, scheduled in zip(received, sent)
                )),
                "accounting_note": "Raw transport counters are sampled at Unity frame boundaries; unique sequence coverage is the primary outcome.",
            }
        )

        state_ages: List[float] = []
        for run in runs:
            state_ages.extend(
                value for value in load_column(run["frame_path"], "state_age_ms") if value > 0
            )
        state_age_by_rate[rate] = state_ages

    write_csv(
        output_dir / "graph4_latency_distribution_statistics.csv",
        [
            "input_rate_hz",
            "latency_observations_n",
            "minimum_ms",
            "q1_ms",
            "median_ms",
            "q3_ms",
            "p95_ms",
            "p99_ms",
            "maximum_ms",
            "upper_outlier_fence_ms",
            "observations_above_fence",
            "freshness_callbacks_removed",
        ],
        latency_rows,
    )
    write_csv(
        output_dir / "experiment2_pipeline_accounting.csv",
        [
            "input_rate_hz", "independent_runs_n", "scheduled_mean",
            "raw_transport_received_mean", "unique_applied_mean",
            "missing_scheduled_mean", "scheduled_sequence_coverage_percent",
            "raw_transport_minus_scheduled_mean", "accounting_note",
        ],
        message_rows,
    )

    cdf_rows: List[Dict[str, object]] = []
    for index in range(101):
        probability = index / 100.0
        row: Dict[str, object] = {"cumulative_probability": probability}
        for rate in EXPECTED_RATES:
            row[f"state_age_ms_{rate:03d}hz"] = number(
                percentile(state_age_by_rate[rate], probability * 100.0)
            )
        cdf_rows.append(row)
    write_csv(
        output_dir / "graph7_state_age_cdf_data.csv",
        ["cumulative_probability"] + [f"state_age_ms_{rate:03d}hz" for rate in EXPECTED_RATES],
        cdf_rows,
    )


def load_column(path: Path, column: str) -> List[float]:
    return finite(as_float(row.get(column)) for row in read_rows(path))


def load_motion_metrics(path: Path) -> Dict[str, float]:
    """Return frame-time-normalized continuity diagnostics for one run.

    Mean position step alone is not a smoothness measure because it shrinks as
    render FPS rises and because stationary/held frames dominate low-rate runs.
    These diagnostics expose held frames and normalize motion by elapsed time.
    """
    rows = read_rows(path)
    elapsed = np.asarray([as_float(row.get("experiment_elapsed_s")) for row in rows], dtype=float)
    steps = np.asarray([as_float(row.get("frame_position_step_m")) for row in rows], dtype=float)
    if len(elapsed) < 2 or len(steps) < 2:
        return {
            "zero_motion_frame_percentage": math.nan,
            "displayed_speed_median_mps": math.nan,
            "displayed_speed_p95_mps": math.nan,
            "displayed_speed_cv": math.nan,
        }
    dt = np.diff(elapsed)
    step = steps[1:]
    valid = np.isfinite(dt) & np.isfinite(step) & (dt > 0.00001) & (dt < 0.1)
    dt, step = dt[valid], step[valid]
    if len(step) == 0:
        return {
            "zero_motion_frame_percentage": math.nan,
            "displayed_speed_median_mps": math.nan,
            "displayed_speed_p95_mps": math.nan,
            "displayed_speed_cv": math.nan,
        }
    speed = step / dt
    speed_mean = float(np.mean(speed))
    return {
        "zero_motion_frame_percentage": float(100.0 * np.mean(step <= 0.00001)),
        "displayed_speed_median_mps": float(np.median(speed)),
        "displayed_speed_p95_mps": float(np.percentile(speed, 95.0)),
        "displayed_speed_cv": float(np.std(speed, ddof=1) / speed_mean) if speed_mean > 0 and len(speed) > 1 else math.nan,
    }


def load_unique_snapshot_rows(path: Path) -> tuple[List[Dict[str, str]], int]:
    """Return the first accepted callback for every source sequence.

    DigitalTwinStateManager also raises SnapshotUpdated when an old snapshot is
    relabelled stale. That event is useful to the UI but it is not another
    transport message. Keeping only the first row per sequence prevents those
    freshness callbacks from being counted as network latency observations.
    """
    unique: List[Dict[str, str]] = []
    seen = set()
    duplicates = 0
    for row in read_rows(path):
        sequence = row.get("sequence_number", "")
        if sequence in seen:
            duplicates += 1
            continue
        seen.add(sequence)
        unique.append(row)
    return unique, duplicates


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_latency_boxplot(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    data: List[List[float]] = []
    labels: List[str] = []
    for rate in EXPECTED_RATES:
        values: List[float] = []
        for run in groups.get(rate, []):
            rows, _ = load_unique_snapshot_rows(run["snapshot_path"])
            values.extend(finite(as_float(row.get("latency_ms")) for row in rows))
        if values:
            data.append(values)
            labels.append(str(rate))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.2, "alpha": 0.28},
        medianprops={"color": "#B91C1C", "linewidth": 1.5},
        boxprops={"color": "#1D4ED8"},
        whiskerprops={"color": "#475569"},
        capprops={"color": "#475569"},
    )
    ax.set_yscale("log")
    ax.set_xlabel("Input update rate (Hz)")
    ax.set_ylabel("Sender-to-Unity latency (ms)")
    ax.set_title("Graph 4. Latency versus input update rate")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        -0.23,
        "Logarithmic y-axis; dots are retained outliers. First callback per source sequence only.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    save_figure(fig, output_dir, "graph4_latency_vs_input_rate")


def plot_latency_by_run(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    """Plot run-level latency summaries without treating callbacks as replicates."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    colors = {"median_latency_ms": "#2563EB", "p95_latency_ms": "#F59E0B", "p99_latency_ms": "#DC2626"}
    labels = {"median_latency_ms": "Median", "p95_latency_ms": "P95", "p99_latency_ms": "P99"}
    offsets = {"median_latency_ms": -0.9, "p95_latency_ms": 0.0, "p99_latency_ms": 0.9}
    for metric in colors:
        for rate in EXPECTED_RATES:
            values = finite(summary_value(run, metric) for run in groups.get(rate, []))
            axes[0].scatter(np.full(len(values), rate + offsets[metric]), values, s=28, color=colors[metric], alpha=0.82, label=labels[metric] if rate == EXPECTED_RATES[0] else None)
            if len(values):
                axes[0].plot([rate - 1.6, rate + 1.6], [np.median(values)] * 2, color=colors[metric], linewidth=1.4)
    for rate in EXPECTED_RATES:
        values = finite(summary_value(run, "maximum_latency_ms") for run in groups.get(rate, []))
        axes[1].scatter(np.full(len(values), rate), values, s=32, color="#4F46E5", alpha=0.82)
        if len(values):
            axes[1].plot([rate - 1.8, rate + 1.8], [np.median(values)] * 2, color="#111827", linewidth=1.5)
    axes[0].set(xlabel="Input update rate (Hz)", ylabel="Run-level latency (ms)", title="Central and tail latency")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Input update rate (Hz)", ylabel="Maximum latency in run (ms)", title="Rare local stalls")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(EXPECTED_RATES)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 2 local sender-to-Unity latency", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent 300 s runs (n=3 per rate); short bars are medians. Maxima are descriptive and their causes are unresolved.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output_dir, "figure_experiment2_latency_by_run")


def plot_fps_vs_rate(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    means: List[float] = []
    errors: List[float] = []
    for rate in EXPECTED_RATES:
        values = [summary_value(run, "mean_fps") for run in groups.get(rate, [])]
        means.append(mean(values))
        sd = sample_sd(values)
        errors.append(0.0 if not math.isfinite(sd) else sd)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(
        EXPECTED_RATES,
        means,
        yerr=errors,
        color="#1D4ED8",
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=4,
    )
    ax.set_xticks(EXPECTED_RATES)
    ax.set_xlabel("Input update rate (Hz)")
    ax.set_ylabel("Mean Unity FPS")
    ax.set_title("Graph 5. Unity FPS versus input update rate")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        -0.20,
        "Points show the mean of three 300-second runs; error bars show ±1 SD.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    save_figure(fig, output_dir, "graph5_unity_fps_vs_input_rate")


def plot_frame_time_by_run(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for metric, color, label, offset in (
        ("mean_frame_time_ms", "#2563EB", "Mean frame time", -0.8),
        ("p95_frame_time_ms", "#F59E0B", "P95 frame time", 0.8),
    ):
        medians = []
        for rate in EXPECTED_RATES:
            values = finite(summary_value(run, metric) for run in groups.get(rate, []))
            ax.scatter(np.full(len(values), rate + offset), values, s=32, color=color, alpha=0.82, label=label if rate == EXPECTED_RATES[0] else None)
            medians.append(float(np.median(values)) if len(values) else math.nan)
        ax.plot(EXPECTED_RATES, medians, color=color, linewidth=1.5, alpha=0.9)
    ax.set_xticks(EXPECTED_RATES)
    ax.set_xlabel("Input update rate (Hz)")
    ax.set_ylabel("Unity frame time (ms)")
    ax.set_title("Experiment 2 frame cost under local ingestion")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.01, -0.20, "Dots are independent 300 s runs (n=3 per rate); lines connect rate medians. The 2-to-10 Hz change is observed, not causally assigned.", transform=ax.transAxes, fontsize=8, color="#475569")
    save_figure(fig, output_dir, "figure_experiment2_frame_time_by_run")


def plot_message_success(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    rates = np.asarray(EXPECTED_RATES, dtype=float)
    scheduled = []
    received = []
    applied = []
    dropped = []
    for rate in EXPECTED_RATES:
        runs = groups.get(rate, [])
        scheduled.append(mean(summary_value(run, "scheduled_messages") for run in runs))
        received.append(mean(summary_value(run, "received_messages") for run in runs))
        unique_applied = [
            float(len(load_unique_snapshot_rows(run["snapshot_path"])[0]))
            for run in runs
        ]
        applied.append(mean(unique_applied))
        dropped.append(mean(
            max(0.0, summary_value(run, "scheduled_messages") - applied_count)
            for run, applied_count in zip(runs, unique_applied)
        ))

    x = np.arange(len(rates))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - 1.5 * width, scheduled, width, label="Sent/scheduled")
    ax.bar(x - 0.5 * width, received, width, label="Received")
    ax.bar(x + 0.5 * width, applied, width, label="Applied")
    ax.bar(x + 1.5 * width, dropped, width, label="Dropped")
    ax.set_xticks(x, [str(int(rate)) for rate in rates])
    ax.set_xlabel("Input update rate (Hz)")
    ax.set_ylabel("Messages per 300-second run (mean)")
    ax.set_title("Graph 6. Message processing success")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        -0.20,
        "Received is the raw transport counter and may include one frame-boundary packet; applied counts unique source sequences.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    save_figure(fig, output_dir, "graph6_message_processing_success")


def plot_state_age_cdf(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax = axes[0]
    plotted = False
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(EXPECTED_RATES)))
    for rate, color in zip(EXPECTED_RATES, colors):
        values: List[float] = []
        for run in groups.get(rate, []):
            run_values = [value for value in load_column(run["frame_path"], "state_age_ms") if value > 0]
            if run_values:
                ordered_run = np.sort(np.asarray(run_values, dtype=float))
                cumulative_run = np.arange(1, len(ordered_run) + 1) / len(ordered_run)
                ax.plot(ordered_run, cumulative_run, color=color, linewidth=0.6, alpha=0.28)
                values.extend(run_values)
        if not values:
            continue
        ordered = np.sort(np.asarray(values, dtype=float))
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cumulative, label=f"{rate} Hz", linewidth=1.8, color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Displayed-state age (ms)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("Frame-level ECDF")
    ax.set_ylim(0, 1.01)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    for rate, color in zip(EXPECTED_RATES, colors):
        values = finite(summary_value(run, "p95_state_age_ms") for run in groups.get(rate, []))
        axes[1].scatter(np.full(len(values), rate), values, s=34, color=color, alpha=0.85)
        if len(values):
            axes[1].plot([rate - 1.8, rate + 1.8], [np.median(values)] * 2, color="#111827", linewidth=1.4)
    axes[1].set_xticks(EXPECTED_RATES)
    axes[1].set_xlabel("Input update rate (Hz)")
    axes[1].set_ylabel("Run P95 displayed-state age (ms)")
    axes[1].set_title("Independent run summaries")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 2 displayed-state age", fontsize=13)
    fig.text(0.01, 0.01, "Thin curves are individual runs; bold curves pool frames descriptively. Right-panel dots are the independent runs (n=3 per rate).", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output_dir, "figure_experiment2_state_age")


def plot_position_smoothness(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    means = []
    errors = []
    for rate in EXPECTED_RATES:
        run_means = [summary_value(run, "position_step_mean_m") for run in groups.get(rate, [])]
        means.append(mean(run_means))
        sd = sample_sd(run_means)
        errors.append(0.0 if not math.isfinite(sd) else sd)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(EXPECTED_RATES, means, yerr=errors, marker="o", capsize=4)
    ax.set_xlabel("Input update rate (Hz)")
    ax.set_ylabel("Mean displayed position step per frame (m)")
    ax.set_title("Experiment 2: Ego-position smoothness")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "figure_experiment2_position_smoothness")


def plot_motion_continuity(
    output_dir: Path, groups: Dict[int, List[Dict[str, object]]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for rate in EXPECTED_RATES:
        held = finite(summary_value(run, "zero_motion_frame_percentage") for run in groups.get(rate, []))
        speed = finite(summary_value(run, "displayed_speed_median_mps") for run in groups.get(rate, []))
        axes[0].scatter(np.full(len(held), rate), held, s=34, color="#2563EB", alpha=0.82)
        axes[1].scatter(np.full(len(speed), rate), speed, s=34, color="#059669", alpha=0.82)
        if len(held): axes[0].plot([rate - 1.8, rate + 1.8], [np.median(held)] * 2, color="#111827", linewidth=1.4)
        if len(speed): axes[1].plot([rate - 1.8, rate + 1.8], [np.median(speed)] * 2, color="#111827", linewidth=1.4)
    axes[0].set(xlabel="Input update rate (Hz)", ylabel="Held/unchanged rendered frames (%)", title="Displayed-position holds")
    axes[1].set(xlabel="Input update rate (Hz)", ylabel="Median displayed speed (m/s)", title="Frame-time-normalized motion")
    for ax in axes:
        ax.set_xticks(EXPECTED_RATES)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 2 motion-continuity diagnostics", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent runs. These diagnostics describe presentation continuity; they are not a perceptual smoothness score.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output_dir, "figure_experiment2_motion_continuity")


def main() -> int:
    arguments = parse_arguments()
    results_dir = arguments.results_dir.resolve()
    output_dir = (arguments.output_dir or results_dir / "analysis_publication").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_runs, invalid_runs = load_runs(results_dir)
    if not valid_runs:
        raise SystemExit(
            f"No valid Experiment 2 runs found in {results_dir}. "
            "Complete at least one 300-second run first."
        )

    groups = grouped_runs(valid_runs)
    cleaning_audit = prepare_cleaned_metrics(valid_runs)
    write_run_summary(output_dir, valid_runs)
    write_rate_statistics(output_dir, groups)
    write_graph_data(output_dir, groups)
    write_csv(
        output_dir / "experiment2_data_cleaning_audit.csv",
        [
            "run_id",
            "configured_rate_hz",
            "raw_snapshot_callback_rows",
            "unique_source_sequences",
            "freshness_callback_duplicates_removed",
            "scheduled_messages",
            "raw_transport_received_messages",
            "unique_applied_sequences",
            "transport_boundary_delta",
            "applied_boundary_delta",
            "raw_mean_latency_ms",
            "cleaned_mean_latency_ms",
            "raw_p99_latency_ms",
            "cleaned_p99_latency_ms",
        ],
        cleaning_audit,
    )
    invalid_fields = [
        "summary_file",
        "run_id",
        "configured_rate_hz",
        "completion_reason",
        "collision_detected",
    ]
    write_csv(output_dir / "experiment2_excluded_runs.csv", invalid_fields, invalid_runs)

    plot_latency_by_run(output_dir, groups)
    plot_frame_time_by_run(output_dir, groups)
    plot_state_age_cdf(output_dir, groups)
    plot_motion_continuity(output_dir, groups)

    missing = {
        rate: arguments.expected_runs - len(groups.get(rate, []))
        for rate in EXPECTED_RATES
        if len(groups.get(rate, [])) < arguments.expected_runs
    }
    print(f"Analyzed {len(valid_runs)} valid run(s); excluded {len(invalid_runs)} invalid run(s).")
    for rate in EXPECTED_RATES:
        print(f"  {rate:3d} Hz: {len(groups.get(rate, []))} valid run(s)")
    print(f"Tables and figures saved to: {output_dir}")
    if missing:
        print("More valid repetitions are still needed: " + ", ".join(
            f"{rate} Hz needs {count}" for rate, count in missing.items()
        ))
        return 2 if arguments.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
