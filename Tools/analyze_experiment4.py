#!/usr/bin/env python3
"""Paired statistics and paper figures for RoadWeave Experiment 4."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


RULE = "Rule Controller"
HYBRID = "Hybrid ML Controller"
CONTROLLERS = (RULE, HYBRID)
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260903

METRICS: Dict[str, Tuple[str, str]] = {
    "route_progress_m": ("Route progress (m)", "higher"),
    "collision_count": ("Collision count", "lower"),
    "minimum_ttc_s": ("Minimum TTC (s)", "higher"),
    "minimum_clearance_m": ("Minimum clearance (m)", "higher"),
    "emergency_stops": ("Emergency stops", "lower"),
    "deadlock_duration_s": ("Deadlock duration (s)", "lower"),
    "overtake_attempts": ("Overtake attempts", "context"),
    "successful_overtakes": ("Successful overtakes", "higher"),
    "overtake_success_rate": ("Overtake success rate", "higher"),
    "mean_absolute_acceleration_mps2": ("Mean absolute acceleration (m/s²)", "lower"),
    "maximum_absolute_jerk_mps3": ("Maximum absolute jerk (m/s³)", "lower"),
    "mean_absolute_yaw_rate_deg_s": ("Mean absolute yaw rate (deg/s)", "lower"),
    "scenario_completion_rate": ("Scenario success rate", "higher"),
}

SCENARIO_ORDER = ("pedestrian", "stopped_car", "slow_car", "truck", "mixed")
SCENARIO_LABELS = {
    "pedestrian": "Pedestrian",
    "stopped_car": "Stopped car",
    "slow_car": "Slow car",
    "truck": "Truck",
    "mixed": "Mixed",
}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _numbers(values: Iterable[object]) -> np.ndarray:
    result = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return result[np.isfinite(result)]


def _bootstrap_mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    clean = _numbers(values)
    if len(clean) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_SAMPLES, 1000):
        count = min(1000, BOOTSTRAP_SAMPLES - start)
        samples = rng.choice(clean, size=(count, len(clean)), replace=True)
        means[start : start + count] = samples.mean(axis=1)
    return float(clean.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_test(rule: np.ndarray, hybrid: np.ndarray) -> Tuple[float, float, float, float]:
    difference = hybrid - rule
    mean, lower, upper = _bootstrap_mean_ci(difference)
    if len(difference) < 2 or np.allclose(difference, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(hybrid, rule, alternative="two-sided").pvalue)
        except ValueError:
            p_value = math.nan
    sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else math.nan
    effect = mean / sd if math.isfinite(sd) and sd > 1e-12 else (0.0 if abs(mean) <= 1e-12 else math.nan)
    return mean, lower, upper, p_value, effect


def _write_controller_summary(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric, (label, direction) in METRICS.items():
        for controller in CONTROLLERS:
            values = _numbers(runs.loc[runs["controller"] == controller, metric])
            mean, lower, upper = _bootstrap_mean_ci(values)
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "preferred_direction": direction,
                    "controller": controller,
                    "n": len(values),
                    "mean": mean,
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                    "median": float(np.median(values)) if len(values) else math.nan,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "experiment4_controller_summary.csv", index=False)
    return table


def _write_paired_statistics(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric, (label, direction) in METRICS.items():
        numeric = runs[["seed", "controller", metric]].copy()
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
        paired = numeric.pivot(index="seed", columns="controller", values=metric).dropna()
        if RULE not in paired or HYBRID not in paired:
            continue
        rule = paired[RULE].to_numpy(dtype=float)
        hybrid = paired[HYBRID].to_numpy(dtype=float)
        mean_difference, lower, upper, p_value, effect = _paired_test(rule, hybrid)
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "preferred_direction": direction,
                "paired_n": len(paired),
                "rule_mean": float(rule.mean()),
                "hybrid_mean": float(hybrid.mean()),
                "hybrid_minus_rule_mean_difference": mean_difference,
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
                "wilcoxon_two_sided_p": p_value,
                "paired_effect_size_dz": effect,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "experiment4_paired_statistics.csv", index=False)
    return table


def _common_scenario_rows(scenarios: pd.DataFrame) -> pd.DataFrame:
    data = scenarios.copy()
    data["evaluable_bool"] = data["evaluable"].map(_as_bool)
    data["success_bool"] = data["success"].map(_as_bool)
    evaluable = data[data["evaluable_bool"]]
    counts = evaluable.groupby(["seed", "scenario_id"])["controller"].nunique()
    common_index = set(counts[counts == 2].index)
    mask = [
        (seed, scenario_id) in common_index
        for seed, scenario_id in zip(evaluable["seed"], evaluable["scenario_id"])
    ]
    return evaluable.loc[mask].copy()


def _scenario_table(scenarios: pd.DataFrame, output: Path) -> pd.DataFrame:
    common = _common_scenario_rows(scenarios)
    rows: List[Dict[str, object]] = []
    for category in SCENARIO_ORDER:
        for controller in CONTROLLERS:
            selected = common[(common["scenario_category"] == category) & (common["controller"] == controller)]
            successes = int(selected["success_bool"].sum())
            total = len(selected)
            rows.append(
                {
                    "scenario_category": category,
                    "scenario_label": SCENARIO_LABELS[category],
                    "controller": controller,
                    "successes": successes,
                    "evaluable_common_scenarios": total,
                    "success_rate": successes / total if total else math.nan,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "experiment4_scenario_success.csv", index=False)
    return table


def _metric_summary(summary: pd.DataFrame, metric: str, controller: str) -> pd.Series:
    return summary[(summary["metric"] == metric) & (summary["controller"] == controller)].iloc[0]


def _graph11(summary: pd.DataFrame, output: Path) -> None:
    metrics = ("route_progress_m", "deadlock_duration_s", "emergency_stops", "successful_overtakes")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    colors = ("#3569B0", "#E07A30")
    for ax, metric in zip(axes.flat, metrics):
        rows = [_metric_summary(summary, metric, controller) for controller in CONTROLLERS]
        means = [float(row["mean"]) for row in rows]
        lower = [mean - float(row["ci95_lower"]) for mean, row in zip(means, rows)]
        upper = [float(row["ci95_upper"]) - mean for mean, row in zip(means, rows)]
        ax.bar(["Rule", "Hybrid ML"], means, color=colors, yerr=[lower, upper], capsize=6)
        ax.set_ylabel(METRICS[metric][0])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Graph 11. Rule versus Hybrid ML controller (mean and 95% bootstrap CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "graph11_rule_vs_hybrid.png", dpi=240)
    fig.savefig(output / "graph11_rule_vs_hybrid.pdf")
    plt.close(fig)


def _boxplot(runs: pd.DataFrame, metric: str, title: str, filename: str, output: Path) -> None:
    values = [_numbers(runs.loc[runs["controller"] == controller, metric]) for controller in CONTROLLERS]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    boxes = ax.boxplot(values, tick_labels=["Rule", "Hybrid ML"], patch_artist=True, showmeans=True)
    for patch, color in zip(boxes["boxes"], ("#3569B0", "#E07A30")):
        patch.set_facecolor(color); patch.set_alpha(0.72)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index, series in enumerate(values, 1):
        ax.scatter(rng.normal(index, 0.025, len(series)), series, s=17, color="#202020", alpha=0.48)
    ax.set_ylabel(METRICS[metric][0]); ax.set_title(title); ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(); fig.savefig(output / f"{filename}.png", dpi=240); fig.savefig(output / f"{filename}.pdf"); plt.close(fig)


def _graph14(runs: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    for controller, label, color, marker in (
        (RULE, "Rule", "#3569B0", "o"),
        (HYBRID, "Hybrid ML", "#E07A30", "s"),
    ):
        selected = runs[runs["controller"] == controller]
        ax.scatter(
            pd.to_numeric(selected["route_progress_m"], errors="coerce"),
            pd.to_numeric(selected["minimum_clearance_m"], errors="coerce"),
            label=label, color=color, marker=marker, alpha=0.78, edgecolor="white", linewidth=0.5,
        )
    ax.set(xlabel="Route progress (m)", ylabel="Minimum clearance (m)", title="Graph 14. Safety versus progress")
    ax.axhline(0.0, color="#B71C1C", linestyle="--", linewidth=1, label="Collision boundary")
    ax.legend(); ax.grid(alpha=0.22); fig.tight_layout()
    fig.savefig(output / "graph14_safety_vs_progress.png", dpi=240)
    fig.savefig(output / "graph14_safety_vs_progress.pdf")
    plt.close(fig)


def _graph15(table: pd.DataFrame, output: Path) -> None:
    matrix = np.full((len(SCENARIO_ORDER), 2), np.nan)
    annotations = np.empty((len(SCENARIO_ORDER), 2), dtype=object)
    for row_index, category in enumerate(SCENARIO_ORDER):
        for column_index, controller in enumerate(CONTROLLERS):
            row = table[(table["scenario_category"] == category) & (table["controller"] == controller)].iloc[0]
            rate = float(row["success_rate"]) if pd.notna(row["success_rate"]) else math.nan
            successes = int(row["successes"]); total = int(row["evaluable_common_scenarios"])
            matrix[row_index, column_index] = rate * 100.0 if math.isfinite(rate) else math.nan
            annotations[row_index, column_index] = "N/A" if total == 0 else f"{rate * 100:.1f}%\n({successes}/{total})"
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if math.isfinite(value) and (value < 28 or value > 78) else "black"
            ax.text(column, row, annotations[row, column], ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1], ["Rule", "Hybrid ML"])
    ax.set_yticks(range(len(SCENARIO_ORDER)), [SCENARIO_LABELS[item] for item in SCENARIO_ORDER])
    ax.set_title("Graph 15. Paired scenario success rate")
    colorbar = fig.colorbar(image, ax=ax); colorbar.set_label("Success rate (%)")
    fig.tight_layout(); fig.savefig(output / "graph15_scenario_success.png", dpi=240); fig.savefig(output / "graph15_scenario_success.pdf"); plt.close(fig)


def analyze_experiment(results_dir: Path, expected_pairs: int = 30, strict: bool = True) -> int:
    results_dir = results_dir.resolve()
    runs_path = results_dir / "experiment4_runs.csv"
    scenarios_path = results_dir / "experiment4_scenarios.csv"
    if not runs_path.exists() or not scenarios_path.exists():
        print("Experiment 4 raw CSV files are missing from {}".format(results_dir))
        return 1
    runs = pd.read_csv(runs_path)
    scenarios = pd.read_csv(scenarios_path)
    runs["run_completed_bool"] = runs["run_completed"].map(_as_bool)
    invalid = runs[~runs["run_completed_bool"]].copy()
    invalid.to_csv(results_dir / "experiment4_invalid_runs.csv", index=False)
    complete = runs[runs["run_completed_bool"]].copy()

    pair_counts = complete.groupby("seed")["controller"].nunique()
    paired_seeds = set(pair_counts[pair_counts == 2].index)
    paired = complete[complete["seed"].isin(paired_seeds)].copy()
    manifest_counts = paired.groupby("seed")["manifest_id"].nunique()
    mismatched = set(manifest_counts[manifest_counts != 1].index)
    if mismatched:
        paired = paired[~paired["seed"].isin(mismatched)]
        paired_seeds -= mismatched
    paired.to_csv(results_dir / "experiment4_valid_paired_runs.csv", index=False)

    if strict and len(paired_seeds) != expected_pairs:
        print("Expected {} complete seed pairs but found {}.".format(expected_pairs, len(paired_seeds)))
        return 2
    if not paired_seeds:
        print("No complete Rule/Hybrid seed pairs are available.")
        return 2

    paired_scenarios = scenarios[scenarios["seed"].isin(paired_seeds)].copy()
    summary = _write_controller_summary(paired, results_dir)
    _write_paired_statistics(paired, results_dir)
    scenario_summary = _scenario_table(paired_scenarios, results_dir)
    _graph11(summary, results_dir)
    _boxplot(paired, "minimum_ttc_s", "Graph 12. Minimum TTC distribution", "graph12_minimum_ttc", results_dir)
    _boxplot(paired, "minimum_clearance_m", "Graph 13. Minimum clearance distribution", "graph13_minimum_clearance", results_dir)
    _graph14(paired, results_dir)
    _graph15(scenario_summary, results_dir)
    print("Experiment 4 analysis complete: {} paired seeds.".format(len(paired_seeds)))
    print("Tables and Graphs 11–15: {}".format(results_dir))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("ExperimentResults/experiment4/final"))
    parser.add_argument("--expected-pairs", type=int, default=30)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return analyze_experiment(args.results_dir, args.expected_pairs, strict=not args.no_strict)


if __name__ == "__main__":
    raise SystemExit(main())
