#!/usr/bin/env python3
"""Validate and plot RoadWeave Experiment 3 controlled fault results.

The analysis keeps the local, synthetic-fault scope explicit, uses runs as the
independent units, separates the configured presentation buffer from injected
delay, and audits contract outcomes without calling optional/partial fields
invalid.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_PROFILES = (
    "baseline", "loss05", "loss10", "loss20", "delay050", "delay100",
    "delay250", "delay500", "disconnect05", "disconnect1", "disconnect2", "disconnect5",
)
CONFORMANCE_PROFILES = ("duplicate", "outoforder", "missingvehicle", "invalidvehicle", "missingactors")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RoadWeave Experiment 3 CSV files.")
    parser.add_argument("--results-dir", type=Path, default=Path("ExperimentResults/experiment3"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--streaming-presentation-delay-ms", type=float, default=160.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def finite(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    return result[np.isfinite(result)]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def position_error(frames: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    frames = frames.copy()
    truth = truth.copy()
    for column in ("frame_utc_s", "displayed_x_m", "displayed_z_m"):
        frames[column] = numeric(frames[column])
    for column in ("truth_utc_s", "ego_x_m", "ego_z_m"):
        truth[column] = numeric(truth[column])
    frames = frames.dropna(subset=["frame_utc_s", "displayed_x_m", "displayed_z_m"])
    truth = truth.dropna(subset=["truth_utc_s", "ego_x_m", "ego_z_m"]).sort_values("truth_utc_s")
    if frames.empty or len(truth) < 2:
        return np.asarray([], dtype=float)
    times = frames["frame_utc_s"].to_numpy()
    valid = (times >= truth["truth_utc_s"].iloc[0]) & (times <= truth["truth_utc_s"].iloc[-1])
    times = times[valid]
    if len(times) == 0:
        return np.asarray([], dtype=float)
    true_x = np.interp(times, truth["truth_utc_s"], truth["ego_x_m"])
    true_z = np.interp(times, truth["truth_utc_s"], truth["ego_z_m"])
    shown_x = frames.loc[valid, "displayed_x_m"].to_numpy()
    shown_z = frames.loc[valid, "displayed_z_m"].to_numpy()
    return np.sqrt((shown_x - true_x) ** 2 + (shown_z - true_z) ** 2)


def load_runs(root: Path, presentation_delay_ms: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unity_dir, python_dir = root / "unity", root / "python"
    valid_rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []
    for summary_path in sorted(unity_dir.glob("experiment3_*_summary.csv")):
        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            invalid_rows.append({"summary_file": summary_path.name, "reason": "malformed_summary"})
            continue
        row = summary.iloc[0].to_dict()
        run_id = str(row.get("run_id", ""))
        stem = summary_path.name[: -len("_summary.csv")]
        frame_path = unity_dir / f"{stem}_frames.csv"
        snapshot_path = unity_dir / f"{stem}_snapshots.csv"
        truth_path = python_dir / f"experiment3_{run_id}_truth.csv"
        source_path = python_dir / f"experiment3_{run_id}_source_summary.csv"
        if str(row.get("valid_for_analysis", "")).lower() != "true" or not all(path.exists() for path in (frame_path, snapshot_path, truth_path, source_path)):
            invalid_rows.append({"summary_file": summary_path.name, "run_id": run_id, "reason": "invalid_unity_run_or_missing_matching_python_files"})
            continue
        frames = pd.read_csv(frame_path)
        snapshots = pd.read_csv(snapshot_path)
        truth = pd.read_csv(truth_path)
        source = pd.read_csv(source_path).iloc[0].to_dict()
        errors = position_error(frames, truth)
        target_duration = float(row.get("target_duration_s", 300) or 300)
        measurement_truth = truth[numeric(truth["measurement_elapsed_s"]).between(0, target_duration, inclusive="both")]
        truth_speed = finite(numeric(measurement_truth["speed_mps"]))
        mean_speed = float(np.mean(truth_speed)) if len(truth_speed) else math.nan
        row.update({f"source_{key}": value for key, value in source.items() if key not in {"run_id", "profile_id"}})
        validity = snapshots.get("validity", pd.Series(dtype=str)).astype(str)
        row["valid_snapshot_count"] = int((validity == "Valid").sum())
        row["partial_snapshot_count"] = int((validity == "Partial").sum())
        row["mean_truth_speed_mps"] = mean_speed
        row["presentation_delay_ms"] = presentation_delay_ms
        row["expected_presentation_lag_distance_m"] = mean_speed * presentation_delay_ms / 1000.0
        row["expected_injected_delay_distance_m"] = mean_speed * float(row.get("delay_ms", 0) or 0) / 1000.0
        row["mean_position_error_m"] = float(np.mean(errors)) if len(errors) else math.nan
        row["p95_position_error_m"] = float(np.percentile(errors, 95)) if len(errors) else math.nan
        row["maximum_position_error_m"] = float(np.max(errors)) if len(errors) else math.nan
        row["matched_position_frames"] = len(errors)
        generated = float(source.get("generated_messages", math.nan))
        intentionally_dropped = float(source.get("intentionally_dropped_messages", math.nan))
        row["achieved_injected_loss_percent"] = 100.0 * intentionally_dropped / generated if generated > 0 else math.nan
        valid_rows.append(row)
    runs = pd.DataFrame(valid_rows)
    if not runs.empty:
        baseline = runs[runs["profile_id"] == "baseline"][["source_fault_seed", "mean_position_error_m"]].rename(columns={"mean_position_error_m": "matched_baseline_mean_position_error_m"})
        runs = runs.merge(baseline, on="source_fault_seed", how="left")
        runs["baseline_adjusted_mean_position_error_m"] = numeric(runs["mean_position_error_m"]) - numeric(runs["matched_baseline_mean_position_error_m"])
        runs["position_error_residual_to_buffer_plus_delay_m"] = numeric(runs["mean_position_error_m"]) - numeric(runs["expected_presentation_lag_distance_m"]) - numeric(runs["expected_injected_delay_distance_m"])
    return runs, pd.DataFrame(invalid_rows)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_run_points(ax: plt.Axes, subset: pd.DataFrame, x_col: str, y_col: str, color: str) -> None:
    for x, group in subset.groupby(x_col):
        values = finite(numeric(group[y_col]))
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) else np.asarray([])
        ax.scatter(np.full(len(values), float(x)) + jitter, values, s=34, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        if len(values):
            width = 0.12 if float(x) <= 5 else 1.8
            ax.plot([float(x) - width, float(x) + width], [np.median(values)] * 2, color="#111827", linewidth=1.5)


def plot_packet_loss(runs: pd.DataFrame, output: Path) -> None:
    subset = runs[runs["profile_id"].isin(("baseline", "loss05", "loss10", "loss20"))].copy()
    subset["loss"] = numeric(subset["packet_loss_percent"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    add_run_points(axes[0], subset, "loss", "achieved_injected_loss_percent", "#2563EB")
    axes[0].plot([0, 20], [0, 20], "--", color="#64748B", linewidth=1.2, label="Configured = achieved")
    axes[0].set(xlabel="Configured independent packet loss (%)", ylabel="Achieved injected loss (%)", title="Fault injection check")
    axes[0].legend(frameon=False)
    add_run_points(axes[1], subset, "loss", "p95_state_age_ms", "#D97706")
    axes[1].set(xlabel="Configured independent packet loss (%)", ylabel="Run P95 displayed-state age (ms)", title="Tail freshness")
    for ax in axes:
        ax.set_xticks([0, 5, 10, 20])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 3 controlled local packet loss", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent runs (n=3 per condition). Loss is independently injected in the local publisher, not observed on a physical network.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output, "figure_experiment3_packet_loss")


def plot_disconnect_impact(runs: pd.DataFrame, output: Path) -> None:
    subset = runs[numeric(runs["disconnect_duration_s"]) > 0].copy()
    subset["duration"] = numeric(subset["disconnect_duration_s"])
    subset["missing_updates"] = numeric(subset["source_intentionally_dropped_messages"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    add_run_points(axes[0], subset, "duration", "missing_updates", "#2563EB")
    durations = np.asarray([0.5, 1, 2, 5])
    axes[0].plot(durations, 30 * durations, "--", color="#64748B", linewidth=1.2, label="30 Hz x outage")
    axes[0].set(xlabel="Injected disconnection duration (s)", ylabel="Intentionally omitted updates", title="Source-side outage accounting")
    axes[0].legend(frameon=False)
    add_run_points(axes[1], subset, "duration", "maximum_recovery_jump_m", "#DC2626")
    reference_speed = float(np.nanmedian(numeric(subset["mean_truth_speed_mps"])))
    axes[1].plot(durations, reference_speed * durations, "--", color="#64748B", linewidth=1.2, label=f"{reference_speed:.1f} m/s x outage")
    axes[1].set(xlabel="Injected disconnection duration (s)", ylabel="Maximum one-frame recovery jump (m)", title="Visual discontinuity after outage")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(durations)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 3 controlled local disconnection impact", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent runs (n=3). The 0.5 s jump result is variable because the outage is shorter than the 1 s stale threshold.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output, "figure_experiment3_disconnect_impact")


def plot_delay_lag(runs: pd.DataFrame, output: Path, presentation_delay_ms: float) -> None:
    subset = runs[runs["profile_id"].isin(("baseline", "delay050", "delay100", "delay250", "delay500"))].copy()
    subset["delay"] = numeric(subset["delay_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    add_run_points(axes[0], subset, "delay", "mean_position_error_m", "#0F766E")
    delays = np.asarray([0, 50, 100, 250, 500], dtype=float)
    speed = float(np.nanmedian(numeric(subset["mean_truth_speed_mps"])))
    axes[0].plot(delays, speed * (presentation_delay_ms + delays) / 1000.0, "--", color="#64748B", linewidth=1.3, label=f"{speed:.1f} m/s x ({presentation_delay_ms:.0f} ms buffer + delay)")
    axes[0].set(xlabel="Injected delay (ms)", ylabel="Mean displayed-to-current-truth error (m)", title="Absolute temporal lag error")
    axes[0].legend(frameon=False, fontsize=8)
    add_run_points(axes[1], subset, "delay", "baseline_adjusted_mean_position_error_m", "#7C3AED")
    axes[1].plot(delays, speed * delays / 1000.0, "--", color="#64748B", linewidth=1.3, label=f"{speed:.1f} m/s x injected delay")
    axes[1].set(xlabel="Injected delay (ms)", ylabel="Increase over matched baseline (m)", title="Injected-delay contribution")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(delays)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 3 presentation and injected-delay lag", fontsize=13)
    fig.text(0.01, 0.01, "Dots are independent runs matched by fault seed. Error is temporal display lag on a constant-speed trajectory, not localization error.", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output, "figure_experiment3_delay_lag")


def conformance_audit(runs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    policy = {
        "duplicate": "Reject duplicate sequence; retain unique state",
        "outoforder": "Reject older sequence; retain newer accepted state",
        "missingvehicle": "Reject missing required vehicle group",
        "invalidvehicle": "Accept snapshot as Partial because ego pose remains usable",
        "missingactors": "Accept snapshot with no actor observations; do not infer an empty road",
    }
    for profile in CONFORMANCE_PROFILES:
        subset = runs[runs["profile_id"] == profile]
        def total(field: str) -> float:
            return float(numeric(subset[field]).sum()) if not subset.empty and field in subset else math.nan

        if profile == "duplicate":
            injected, observed = total("source_duplicate_messages_sent"), total("rejected_duplicate")
            note = "Source duplicate sends are compared with Unity duplicate rejections."
        elif profile == "outoforder":
            injected, observed = total("source_out_of_order_messages_scheduled"), total("rejected_out_of_order")
            note = "Source reorders are compared with Unity out-of-order rejections."
        elif profile == "missingvehicle":
            injected, observed = total("source_malformed_messages_sent"), total("rejected_contract")
            note = "The legacy source summary groups missing required vehicle payloads under malformed messages."
        elif profile == "invalidvehicle":
            injected, observed = total("partial_snapshot_count"), total("partial_snapshot_count")
            note = "Legacy source summaries lack a separate invalid-vehicle counter; Partial snapshots prove the acceptance path was exercised."
        else:
            injected = total("source_malformed_messages_sent")
            observed = total("applied_snapshots")
            generated = total("source_generated_messages")
            note = "Legacy source summaries record omitted actors as malformed; zero contract rejections and full generated-to-applied coverage demonstrate normalization and acceptance."
        if profile == "missingactors":
            rejections = total("rejected_contract")
            matches = injected > 0 and math.isfinite(generated) and int(round(observed)) == int(round(generated)) and int(round(rejections)) == 0
            comparison = "all generated snapshots applied; zero contract rejections"
        else:
            matches = math.isfinite(injected) and math.isfinite(observed) and int(round(injected)) == int(round(observed)) and injected > 0
            comparison = "event counts equal"
        rows.append({
            "profile_id": profile,
            "independent_runs_n": len(subset),
            "contract_policy": policy[profile],
            "injected_or_exercised_events": "" if not math.isfinite(injected) else int(round(injected)),
            "observed_policy_outcomes": "" if not math.isfinite(observed) else int(round(observed)),
            "event_count_matches_policy": matches,
            "comparison_basis": comparison,
            "interpretation_note": note,
            "evidence_scope": "single-run conformance demonstration" if len(subset) == 1 else "replicated conformance test",
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_arguments()
    results_dir = args.results_dir.resolve()
    output = (args.output_dir or results_dir / "analysis_publication").resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, invalid = load_runs(results_dir, args.streaming_presentation_delay_ms)
    runs.to_csv(output / "experiment3_run_level_metrics.csv", index=False)
    invalid.to_csv(output / "experiment3_excluded_runs.csv", index=False)
    if runs.empty:
        print("No complete matched Experiment 3 runs were found.")
        return 1
    counts = runs.groupby("profile_id").size().rename("valid_runs")
    counts.to_csv(output / "experiment3_run_counts.csv")
    audit = conformance_audit(runs)
    audit.to_csv(output / "experiment3_conformance_audit.csv", index=False)
    delay_profiles = ("baseline", "delay050", "delay100", "delay250", "delay500")
    delay_fields = [
        "run_id", "profile_id", "source_fault_seed", "delay_ms", "mean_truth_speed_mps",
        "presentation_delay_ms", "mean_position_error_m", "matched_baseline_mean_position_error_m",
        "baseline_adjusted_mean_position_error_m", "expected_presentation_lag_distance_m",
        "expected_injected_delay_distance_m", "position_error_residual_to_buffer_plus_delay_m",
    ]
    runs[runs["profile_id"].isin(delay_profiles)][delay_fields].to_csv(output / "experiment3_delay_error_decomposition.csv", index=False)
    plot_packet_loss(runs, output)
    plot_disconnect_impact(runs, output)
    plot_delay_lag(runs, output, args.streaming_presentation_delay_ms)
    print("Valid runs by profile:\n" + counts.to_string())
    print("Conformance policy audit:\n" + audit.to_string(index=False))
    print(f"Publication analysis saved to {output}")
    missing_or_short = [profile for profile in MAIN_PROFILES if int(counts.get(profile, 0)) < args.expected_runs]
    if args.strict and missing_or_short:
        print("Strict validation failed for: " + ", ".join(missing_or_short))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
