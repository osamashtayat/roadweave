#!/usr/bin/env python3
"""Create a conservative, reproducible Experiment 1 integration analysis.

Experiment 1 validates two source adapters under their native test modes.  It
does not treat their deliberately different update rates or actor workloads as
a controlled performance comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = (
    (
        "nuscenes",
        "nuScenes replay",
        "Frame-driven replay; snapshots are canonical presentation samples, not independent sensor arrivals",
        math.nan,
    ),
    (
        "simulated-streaming",
        "Simulated stream",
        "Local 30 Hz simulated UDP stream",
        30.0,
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RoadWeave Experiment 1 source integration runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("ExperimentResults/experiment1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError):
        return math.nan


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def mean(values: Iterable[float]) -> float:
    clean = finite(values)
    return float(np.mean(clean)) if len(clean) else math.nan


def sd(values: Iterable[float]) -> float:
    clean = finite(values)
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    clean = finite(values)
    return float(np.percentile(clean, q)) if len(clean) else math.nan


def number(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6f}"


def read_single_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(results_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for folder, label, mode, nominal_rate in SOURCES:
        for path in sorted((results_dir / folder).glob("*_summary.csv")):
            row = read_single_row(path)
            row["source_label"] = label
            row["source_mode"] = mode
            row["nominal_rate_hz"] = nominal_rate
            row["summary_file"] = path.name
            runs.append(row)
    return runs


def write_run_table(output_dir: Path, runs: Sequence[Dict[str, object]]) -> None:
    fields = [
        "run_id", "source_label", "source_mode", "nominal_rate_hz", "summary_file",
        "completed_without_error", "actual_duration_s", "accepted_snapshots",
        "effective_update_rate_hz", "startup_time_ms", "drive_response_time_ms",
        "mean_inter_arrival_ms", "p95_inter_arrival_ms", "fresh_snapshot_percentage",
        "fresh_frame_percentage", "sequence_gap_count", "non_monotonic_accepted_count",
        "mean_actor_count", "maximum_actor_count", "mean_frame_time_ms",
        "p95_frame_time_ms", "mean_fps", "minimum_fps", "warning_count", "error_count",
    ]
    write_csv(output_dir / "experiment1_run_summary.csv", fields, [
        {field: row.get(field, "") for field in fields} for row in runs
    ])


def write_source_table(output_dir: Path, runs: Sequence[Dict[str, object]]) -> None:
    fields = [
        "source_label", "source_mode", "nominal_rate_hz", "independent_runs_n",
        "mean_measurement_duration_s", "accepted_snapshots_total",
        "completed_without_error_n", "zero_sequence_gap_runs_n",
        "weighted_fresh_snapshots_percent", "mean_run_fresh_frames_percent",
        "startup_time_median_ms", "startup_time_q1_ms", "startup_time_q3_ms",
        "drive_response_median_ms", "drive_response_q1_ms", "drive_response_q3_ms",
        "effective_update_rate_mean_hz", "effective_update_rate_sd_hz",
        "mean_actor_count_mean", "p95_frame_time_mean_ms", "p95_frame_time_sd_ms",
        "warnings_total", "errors_total",
    ]
    output: List[Dict[str, object]] = []
    for _, label, mode, nominal_rate in SOURCES:
        subset = [row for row in runs if row["source_label"] == label]
        accepted = [as_float(row.get("accepted_snapshots")) for row in subset]
        fresh = [as_float(row.get("fresh_snapshot_percentage")) for row in subset]
        fresh_count = sum(a * p / 100.0 for a, p in zip(accepted, fresh) if math.isfinite(a) and math.isfinite(p))
        accepted_total = sum(v for v in accepted if math.isfinite(v))
        startup = [as_float(row.get("startup_time_ms")) for row in subset]
        drive = [as_float(row.get("drive_response_time_ms")) for row in subset]
        rates = [as_float(row.get("effective_update_rate_hz")) for row in subset]
        p95_frame = [as_float(row.get("p95_frame_time_ms")) for row in subset]
        output.append({
            "source_label": label,
            "source_mode": mode,
            "nominal_rate_hz": number(nominal_rate),
            "independent_runs_n": len(subset),
            "mean_measurement_duration_s": number(mean(as_float(r.get("actual_duration_s")) for r in subset)),
            "accepted_snapshots_total": int(round(accepted_total)),
            "completed_without_error_n": sum(as_bool(r.get("completed_without_error")) for r in subset),
            "zero_sequence_gap_runs_n": sum(as_float(r.get("sequence_gap_count")) == 0 for r in subset),
            "weighted_fresh_snapshots_percent": number(100.0 * fresh_count / accepted_total if accepted_total else math.nan),
            "mean_run_fresh_frames_percent": number(mean(as_float(r.get("fresh_frame_percentage")) for r in subset)),
            "startup_time_median_ms": number(percentile(startup, 50)),
            "startup_time_q1_ms": number(percentile(startup, 25)),
            "startup_time_q3_ms": number(percentile(startup, 75)),
            "drive_response_median_ms": number(percentile(drive, 50)),
            "drive_response_q1_ms": number(percentile(drive, 25)),
            "drive_response_q3_ms": number(percentile(drive, 75)),
            "effective_update_rate_mean_hz": number(mean(rates)),
            "effective_update_rate_sd_hz": number(sd(rates)),
            "mean_actor_count_mean": number(mean(as_float(r.get("mean_actor_count")) for r in subset)),
            "p95_frame_time_mean_ms": number(mean(p95_frame)),
            "p95_frame_time_sd_ms": number(sd(p95_frame)),
            "warnings_total": int(round(sum(as_float(r.get("warning_count")) for r in subset))),
            "errors_total": int(round(sum(as_float(r.get("error_count")) for r in subset))),
        })
    write_csv(output_dir / "experiment1_source_summary.csv", fields, output)


def dot_panel(ax: plt.Axes, groups: Sequence[Sequence[float]], labels: Sequence[str], ylabel: str, log: bool = False) -> None:
    colors = ("#2563EB", "#F59E0B")
    for index, (values, color) in enumerate(zip(groups, colors), start=1):
        clean = finite(values)
        jitter = np.linspace(-0.08, 0.08, len(clean)) if len(clean) else np.asarray([])
        ax.scatter(index + jitter, clean, s=22, alpha=0.75, color=color, edgecolor="white", linewidth=0.35)
        if len(clean):
            med = float(np.median(clean))
            ax.plot([index - 0.18, index + 0.18], [med, med], color="#111827", linewidth=2)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.22)


def plot_integration_baseline(output_dir: Path, runs: Sequence[Dict[str, object]]) -> None:
    labels = [source[1] for source in SOURCES]
    grouped = [[row for row in runs if row["source_label"] == label] for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    dot_panel(axes[0], [[as_float(r.get("startup_time_ms")) for r in group] for group in grouped], labels, "Adapter-ready time (ms)", log=True)
    dot_panel(axes[1], [[as_float(r.get("drive_response_time_ms")) for r in group] for group in grouped], labels, "Drive to first accepted snapshot (ms)")
    dot_panel(axes[2], [[as_float(r.get("p95_frame_time_ms")) for r in group] for group in grouped], labels, "Run P95 frame time (ms)")
    axes[0].set_title("Startup distribution")
    axes[1].set_title("First-snapshot response")
    axes[2].set_title("Runtime frame cost")
    fig.suptitle("Experiment 1 source-integration baseline", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent runs; horizontal bars are medians. Sources retain native, unmatched workloads.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_dir / "figure_experiment1_integration_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure_experiment1_integration_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_arguments()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "analysis_publication").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No Experiment 1 summary files found in {results_dir}")
    write_run_table(output_dir, runs)
    write_source_table(output_dir, runs)
    plot_integration_baseline(output_dir, runs)
    counts = {label: sum(row["source_label"] == label for row in runs) for _, label, _, _ in SOURCES}
    print("Experiment 1 independent runs: " + ", ".join(f"{label}={count}" for label, count in counts.items()))
    print(f"Publication analysis saved to {output_dir}")
    if args.strict and any(count < args.expected_runs for count in counts.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
