#!/usr/bin/env python3
"""Generate reproducible object-level camera/LiDAR/radar reliability data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ML.src.sensor_reliability_features import (
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        SENSOR_TYPES,
        WEATHER_CONDITIONS,
        canonical_features,
        reliability_status,
    )
except ModuleNotFoundError:
    from sensor_reliability_features import (  # type: ignore
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        SENSOR_TYPES,
        WEATHER_CONDITIONS,
        canonical_features,
        reliability_status,
    )


ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ML_ROOT / "data" / "processed" / "sensor_reliability.parquet"
DEFAULT_REPORT = ML_ROOT / "reports" / "sensor_reliability_generation.json"
FAULT_TYPES = (
    "NONE",
    "DROPOUT",
    "RANGE_NOISE",
    "VELOCITY_NOISE",
    "BIAS",
    "LATENCY",
    "FALSE_POSITIVE",
    "CALIBRATION",
    "COMPLETE_FAILURE",
)

BASE = {
    "CAMERA": dict(dropout=0.02, range_noise=0.45, velocity_noise=0.80, latency=0.08, range=55.0),
    "LIDAR": dict(dropout=0.01, range_noise=0.12, velocity_noise=0.35, latency=0.10, range=70.0),
    "RADAR": dict(dropout=0.01, range_noise=0.30, velocity_noise=0.12, latency=0.05, range=100.0),
}


def parameters(sensor: str, weather: str, fault: str, severity: float) -> Dict[str, float]:
    result = dict(BASE[sensor])
    result.update(false_positive=0.0, range_bias=0.0, lateral_bias=0.0)
    if weather == "RAIN":
        if sensor == "CAMERA": result.update(dropout=result["dropout"] + 0.08, range_noise=result["range_noise"] * 1.6, range=result["range"] * 0.82)
        elif sensor == "LIDAR": result.update(dropout=result["dropout"] + 0.05, range_noise=result["range_noise"] * 1.8, range=result["range"] * 0.86)
        else: result.update(dropout=result["dropout"] + 0.015, range_noise=result["range_noise"] * 1.15)
    elif weather == "SNOW":
        if sensor == "CAMERA": result.update(dropout=result["dropout"] + 0.16, range_noise=result["range_noise"] * 2.2, range=result["range"] * 0.68)
        elif sensor == "LIDAR": result.update(dropout=result["dropout"] + 0.13, range_noise=result["range_noise"] * 2.8, range=result["range"] * 0.64)
        else: result.update(dropout=result["dropout"] + 0.04, range_noise=result["range_noise"] * 1.35)
    elif weather == "FOG":
        if sensor == "CAMERA": result.update(dropout=result["dropout"] + 0.28, range_noise=result["range_noise"] * 2.5, range=result["range"] * 0.45)
        elif sensor == "LIDAR": result.update(dropout=result["dropout"] + 0.14, range_noise=result["range_noise"] * 2.1, range=result["range"] * 0.70)
        else: result.update(dropout=result["dropout"] + 0.02, range_noise=result["range_noise"] * 1.1)

    if fault == "DROPOUT": result["dropout"] += 0.75 * severity
    elif fault == "RANGE_NOISE": result["range_noise"] *= 1.0 + 8.0 * severity
    elif fault == "VELOCITY_NOISE": result["velocity_noise"] *= 1.0 + 10.0 * severity
    elif fault == "BIAS": result["range_bias"] = 0.5 + 5.5 * severity
    elif fault == "LATENCY": result["latency"] += 0.1 + 1.1 * severity
    elif fault == "FALSE_POSITIVE": result["false_positive"] = 0.05 + 0.60 * severity
    elif fault == "CALIBRATION":
        result["range_bias"] = 0.2 + 1.8 * severity
        result["lateral_bias"] = 0.3 + 2.7 * severity
    elif fault == "COMPLETE_FAILURE":
        result["dropout"] = 1.0
        result["range"] = 0.0
    result["dropout"] = float(np.clip(result["dropout"], 0.0, 1.0))
    return result


def jaccard(first: set, second: set) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def simulate_channel(
    rng: np.random.Generator,
    sensor: str,
    weather: str,
    fault: str,
    severity: float,
    true_ranges: np.ndarray,
    true_velocities: np.ndarray,
    ego_speed: float,
    yaw_rate: float,
    samples: int,
    sample_period: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    config = parameters(sensor, weather, fault, severity)
    message_received: List[float] = []
    ages: List[float] = []
    detection_counts: List[float] = []
    confidences: List[float] = []
    measured_ranges: List[float] = []
    measured_velocities: List[float] = []
    innovations: List[float] = []
    track_scores: List[float] = []
    previous_tracks: set = set()
    previous_measurement: Dict[int, float] = {}
    matched = 0
    truth_opportunities = 0
    false_positives = 0
    range_errors: List[float] = []
    velocity_errors: List[float] = []
    last_success = -3.0

    for sample_index in range(samples):
        timestamp = sample_index * sample_period
        visible_truth = []
        for actor_index in range(len(true_ranges)):
            true_range = max(
                0.0,
                true_ranges[actor_index] - true_velocities[actor_index] * timestamp,
            )
            if true_range <= config["range"]:
                visible_truth.append((actor_index, true_range))
        truth_opportunities += len(visible_truth)

        received = rng.random() >= config["dropout"]
        message_received.append(float(received))
        if not received:
            ages.append(min(3.0, timestamp - last_success))
            detection_counts.append(0.0)
            confidences.append(0.0)
            track_scores.append(jaccard(previous_tracks, set()))
            previous_tracks = set()
            continue

        last_success = timestamp
        age = max(0.0, config["latency"] + rng.normal(0.0, 0.015))
        ages.append(age)
        current_tracks = set()
        frame_confidences: List[float] = []
        frame_count = 0
        for actor_index, true_range in visible_truth:
            weather_miss = {"DRY": 0.0, "RAIN": 0.04, "SNOW": 0.09, "FOG": 0.13}[weather]
            if sensor == "RADAR": weather_miss *= 0.2
            elif sensor == "LIDAR": weather_miss *= 0.65
            if rng.random() < min(0.95, config["dropout"] * 0.25 + weather_miss):
                continue
            measured_range = max(0.0, true_range + config["range_bias"] + rng.normal(0.0, config["range_noise"]))
            measured_velocity = true_velocities[actor_index] + rng.normal(0.0, config["velocity_noise"])
            normalized_error = abs(measured_range - true_range) / max(1.0, true_range)
            confidence = float(np.clip(0.98 - normalized_error - age * 0.15, 0.05, 1.0))
            matched += 1
            frame_count += 1
            current_tracks.add(actor_index)
            frame_confidences.append(confidence)
            measured_ranges.append(measured_range)
            measured_velocities.append(measured_velocity)
            range_errors.append(measured_range - true_range)
            velocity_errors.append(measured_velocity - true_velocities[actor_index])
            if actor_index in previous_measurement:
                predicted = previous_measurement[actor_index] - measured_velocity * sample_period
                innovations.append(abs(measured_range - predicted))
            previous_measurement[actor_index] = measured_range

        if rng.random() < config["false_positive"]:
            false_positives += 1
            frame_count += 1
            frame_confidences.append(float(rng.uniform(0.1, 0.45)))
        detection_counts.append(float(frame_count))
        confidences.append(float(np.mean(frame_confidences)) if frame_confidences else 1.0)
        track_scores.append(jaccard(previous_tracks, current_tracks))
        previous_tracks = current_tracks

    precision = (
        matched / (matched + false_positives)
        if matched + false_positives > 0
        else 1.0
    )
    recall = matched / truth_opportunities if truth_opportunities > 0 else 1.0
    no_measurement_expected = truth_opportunities == 0
    range_rmse = (
        math.sqrt(float(np.mean(np.square(range_errors))))
        if range_errors
        else 0.0 if no_measurement_expected else 20.0
    )
    velocity_rmse = (
        math.sqrt(float(np.mean(np.square(velocity_errors))))
        if velocity_errors
        else 0.0 if no_measurement_expected else 10.0
    )
    range_scale = {"CAMERA": 2.5, "LIDAR": 0.8, "RADAR": 1.2}[sensor]
    velocity_scale = {"CAMERA": 2.0, "LIDAR": 1.0, "RADAR": 0.5}[sensor]
    range_score = math.exp(-range_rmse / range_scale)
    velocity_score = math.exp(-velocity_rmse / velocity_scale)
    freshness = math.exp(-float(np.mean(ages)) / 0.5)
    continuity = float(np.mean(track_scores)) if track_scores else 0.0
    delivery = float(np.mean(message_received))
    observation_quality = (
        0.30 * recall +
        0.15 * precision +
        0.20 * range_score +
        0.15 * velocity_score +
        0.20 * continuity
    )
    availability = 0.65 * delivery + 0.35 * freshness
    target = float(np.clip(observation_quality * availability, 0.0, 1.0))

    observation: Dict[str, Any] = {
        "weather": weather,
        "dropout_rate": 1.0 - float(np.mean(message_received)),
        "message_age_mean": float(np.mean(ages)),
        "message_age_max": float(np.max(ages)),
        "detection_count_mean": float(np.mean(detection_counts)),
        "detection_count_std": float(np.std(detection_counts)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "track_continuity": continuity,
        "range_variance": float(np.var(measured_ranges)) if measured_ranges else 0.0,
        "velocity_variance": float(np.var(measured_velocities)) if measured_velocities else 0.0,
        "innovation_mean": float(np.mean(innovations)) if innovations else 0.0,
        "innovation_std": float(np.std(innovations)) if innovations else 0.0,
        "cross_sensor_disagreement": 0.0,
        "ego_speed_mean": ego_speed,
        "ego_speed_std": float(rng.uniform(0.0, 1.2)),
        "yaw_rate_mean": yaw_rate,
        "yaw_rate_std": float(rng.uniform(0.0, 4.0)),
    }
    diagnostics = {
        "target": target,
        "mean_range": float(np.mean(measured_ranges)) if measured_ranges else math.nan,
        "precision": precision,
        "recall": recall,
        "range_rmse": range_rmse,
        "velocity_rmse": velocity_rmse,
        "delivery": delivery,
    }
    return observation, diagnostics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-window", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.scenarios < 100:
        raise ValueError("At least 100 scenarios are required.")
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    split_choices = np.asarray(["train", "validation", "test"])
    split_probabilities = np.asarray([0.70, 0.15, 0.15])

    for scenario_index in range(args.scenarios):
        scenario_id = "sensor-scenario-{:06d}".format(scenario_index)
        split = str(rng.choice(split_choices, p=split_probabilities))
        weather = str(rng.choice(WEATHER_CONDITIONS, p=[0.36, 0.22, 0.18, 0.24]))
        actor_count = int(rng.integers(1, 6))
        true_ranges = np.sort(rng.uniform(6.0, 85.0, size=actor_count))
        true_velocities = rng.uniform(-3.0, 8.0, size=actor_count)
        ego_speed = float(rng.uniform(0.0, 22.0))
        yaw_rate = float(rng.normal(0.0, 12.0))
        channel_rows: List[Tuple[str, Dict[str, Any], Dict[str, float], str, float]] = []
        for sensor in SENSOR_TYPES:
            fault = str(rng.choice(FAULT_TYPES, p=[0.28, 0.11, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08, 0.06]))
            severity = 0.0 if fault == "NONE" else float(rng.uniform(0.15, 0.95))
            observation, diagnostics = simulate_channel(
                rng, sensor, weather, fault, severity, true_ranges,
                true_velocities, ego_speed, yaw_rate,
                args.samples_per_window, 0.1,
            )
            channel_rows.append((sensor, observation, diagnostics, fault, severity))

        mean_ranges = [item[2]["mean_range"] for item in channel_rows]
        for channel_index, (sensor, observation, diagnostics, fault, severity) in enumerate(channel_rows):
            own = mean_ranges[channel_index]
            peers = [value for index, value in enumerate(mean_ranges) if index != channel_index and math.isfinite(value)]
            observation["cross_sensor_disagreement"] = (
                float(np.mean([abs(own - peer) for peer in peers]))
                if math.isfinite(own) and peers else 0.0
            )
            row: Dict[str, Any] = {
                "source": "roadweave_virtual_sensor",
                "scenario_id": scenario_id,
                "split": split,
                "sensor_type": sensor,
                "weather": weather,
                "fault_type": fault,
                "fault_severity": severity,
                "target_reliability": diagnostics["target"],
                "target_status": reliability_status(diagnostics["target"]),
                "label_precision": diagnostics["precision"],
                "label_recall": diagnostics["recall"],
                "label_range_rmse": diagnostics["range_rmse"],
                "label_velocity_rmse": diagnostics["velocity_rmse"],
            }
            row.update(canonical_features(observation))
            rows.append(row)

        if scenario_index > 0 and scenario_index % 1000 == 0:
            print("Generated {:,}/{:,} scenarios".format(scenario_index, args.scenarios))

    data = pd.DataFrame(rows)
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "scenarios": args.scenarios,
        "rows": len(data),
        "feature_columns": list(FEATURE_COLUMNS),
        "sensor_rows": {str(k): int(v) for k, v in data["sensor_type"].value_counts().items()},
        "weather_rows": {str(k): int(v) for k, v in data["weather"].value_counts().items()},
        "fault_rows": {str(k): int(v) for k, v in data["fault_type"].value_counts().items()},
        "split_rows": {str(k): int(v) for k, v in data["split"].value_counts().items()},
        "output": str(output),
        "important_note": "Fault type and severity are metadata only and are excluded from model inputs.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote {:,} rows to {}".format(len(data), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
