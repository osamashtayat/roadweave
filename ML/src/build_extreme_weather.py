#!/usr/bin/env python3
"""Convert Extreme Driving episodes into RoadWeave weather-speed features.

The converter reads JSON members directly from each uncompressed episode tar;
camera, depth, and LiDAR payloads are never extracted.  Each row contains only
information available at decision time.  Future speed is used solely to create
the supervised weather speed-factor target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ML.src.weather_features import (
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        new_weather_state,
        summarize_weather_history,
    )
except ModuleNotFoundError:
    from weather_features import (  # type: ignore
        FEATURE_COLUMNS,
        SCHEMA_VERSION,
        new_weather_state,
        summarize_weather_history,
    )


DEFAULT_DATASET = Path(
    "/Users/asus/Downloads/RoadWeaveData/Extreme_Driving_Conditions_Dataset"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "weather_extreme.parquet"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "weather_conversion_report.json"

FOG_TOKENS = ("fog", "haze", "hazy", "mist", "大雾", "雾天")
WET_TOKENS = ("wet", "rain", "puddle", "post-rain", "积水", "潮湿")
SNOW_TOKENS = ("snow", "ice", "icy", "雪", "结冰")
NIGHT_TOKENS = ("night", "low light", "夜")


def _contains(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def classify_condition(category: str, weather: str, surface: str) -> Optional[str]:
    combined = "{} {}".format(weather, surface).strip()
    if category == "Snow_Driving" or _contains(combined, SNOW_TOKENS):
        return "SNOW"
    if category == "Rain_Driving" or _contains(weather, ("rain", "雨")):
        return "RAIN"
    if _contains(weather, FOG_TOKENS):
        return "FOG"
    if category != "Normal_Driving":
        return None
    if _contains(combined, WET_TOKENS) or _contains(combined, NIGHT_TOKENS):
        return None
    return "DRY"


def _read_json_members(path: Path) -> Tuple[bytes, Dict[str, Any]]:
    metadata = b""
    annotation: Dict[str, Any] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith("/metadata.jsonl"):
                handle = archive.extractfile(member)
                metadata = handle.read() if handle is not None else b""
            elif member.name.endswith("/episode_annotation.json"):
                handle = archive.extractfile(member)
                if handle is not None:
                    annotation = json.load(handle)
            if metadata and annotation:
                break
    if not metadata:
        raise ValueError("metadata.jsonl is missing")
    return metadata, annotation


def _valid_state(row: Dict[str, Any]) -> Optional[List[float]]:
    state = row.get("state")
    if not isinstance(state, list) or len(state) < 3:
        return None
    try:
        values = [float(value) for value in state]
    except (TypeError, ValueError):
        return None
    return values if all(np.isfinite(values[:3])) else None


def convert_episode(
    metadata: bytes,
    condition: str,
    source: str,
    group_id: str,
    split: str,
    category: str,
    subscene: str,
    weather_text: str,
    surface_text: str,
    history_seconds: float,
    future_seconds: float,
    reference_speed_kph: float,
    minimum_factor: float,
) -> List[Dict[str, Any]]:
    raw_rows = [
        json.loads(line)
        for line in metadata.decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    states: List[Tuple[int, List[float]]] = []
    for index, row in enumerate(raw_rows):
        state = _valid_state(row)
        if state is not None:
            states.append((index, state))
    if len(states) < 5:
        return []

    sample_period = 0.25
    history_frames = max(2, int(round(history_seconds / sample_period)))
    future_frames = max(1, int(round(future_seconds / sample_period)))
    converted: List[Dict[str, Any]] = []

    for state_index in range(2, len(states) - future_frames):
        history_start = max(0, state_index - history_frames + 1)
        history = []
        for local_index in range(history_start, state_index + 1):
            frame_index, state = states[local_index]
            history.append(
                new_weather_state(
                    frame_index * sample_period,
                    state[0] / 3.6,
                    state[2] * 9.80665,
                    state[1],
                )
            )

        future_speeds = [
            states[index][1][0]
            for index in range(state_index + 1, state_index + future_frames + 1)
        ]
        future_speed_kph = float(np.mean(future_speeds))
        target = float(np.clip(
            future_speed_kph / max(1.0, reference_speed_kph),
            minimum_factor,
            1.0,
        ))
        timestamp = states[state_index][0] * sample_period
        row: Dict[str, Any] = {
            "source": source,
            "group_id": group_id,
            "split": split,
            "category": category,
            "subscene": subscene,
            "condition": condition,
            "weather_text": weather_text,
            "surface_text": surface_text,
            "timestamp": timestamp,
            "future_speed_kph": future_speed_kph,
            "target_speed_factor": target,
        }
        row.update(summarize_weather_history(history, condition))
        converted.append(row)
    return converted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--history-seconds", type=float, default=3.0)
    parser.add_argument("--future-seconds", type=float, default=1.0)
    parser.add_argument("--reference-speed-kph", type=float, default=50.0)
    parser.add_argument("--minimum-factor", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    dataset_root = arguments.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError("Extreme Driving dataset not found: {}".format(dataset_root))
    if arguments.history_seconds <= 0 or arguments.future_seconds <= 0:
        raise ValueError("History and future durations must be positive.")
    if not 0.1 <= arguments.minimum_factor < 1.0:
        raise ValueError("minimum-factor must be in [0.1, 1.0).")

    archives = sorted(dataset_root.glob("train/*/*/*.tar")) + sorted(
        dataset_root.glob("val/*/*/*.tar")
    )
    if arguments.limit > 0:
        archives = archives[: arguments.limit]

    rows: List[Dict[str, Any]] = []
    hashes = set()
    selected_episodes: Dict[str, int] = {name: 0 for name in ("DRY", "RAIN", "SNOW", "FOG")}
    skipped_duplicates = 0
    skipped_conditions = 0
    errors: List[Dict[str, str]] = []

    for index, archive_path in enumerate(archives, 1):
        relative = archive_path.relative_to(dataset_root)
        split, category, subscene = relative.parts[:3]
        try:
            metadata, episode = _read_json_members(archive_path)
            digest = hashlib.sha256(metadata).hexdigest()
            if digest in hashes:
                skipped_duplicates += 1
                continue
            hashes.add(digest)

            annotation = episode.get("annotation", {}) if isinstance(episode, dict) else {}
            weather_text = str(annotation.get("weather", ""))
            surface_text = str(annotation.get("road_surface", ""))
            condition = classify_condition(category, weather_text, surface_text)
            if condition is None:
                skipped_conditions += 1
                continue

            group_id = str(relative.with_suffix(""))
            episode_rows = convert_episode(
                metadata,
                condition,
                "extreme_weather",
                group_id,
                split,
                category,
                subscene,
                weather_text,
                surface_text,
                arguments.history_seconds,
                arguments.future_seconds,
                arguments.reference_speed_kph,
                arguments.minimum_factor,
            )
            if episode_rows:
                rows.extend(episode_rows)
                selected_episodes[condition] += 1
        except Exception as error:
            errors.append({"path": str(archive_path), "error": str(error)})

        if index % 100 == 0:
            print("Processed {}/{} archives; rows={}".format(index, len(archives), len(rows)))

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No weather training rows were produced.")
    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise RuntimeError("Converter omitted weather feature(s): {}".format(", ".join(missing)))
    if data["group_id"].isna().any() or data["target_speed_factor"].isna().any():
        raise RuntimeError("Converted data contains missing group IDs or targets.")

    output = arguments.output.expanduser().resolve()
    report_path = arguments.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "output": str(output),
        "archives_scanned": len(archives),
        "rows": len(data),
        "groups": int(data["group_id"].nunique()),
        "selected_episodes": selected_episodes,
        "condition_rows": {
            str(key): int(value) for key, value in data["condition"].value_counts().items()
        },
        "split_rows": {
            str(key): int(value) for key, value in data["split"].value_counts().items()
        },
        "skipped_duplicates": skipped_duplicates,
        "skipped_conditions": skipped_conditions,
        "errors": errors,
        "history_seconds": arguments.history_seconds,
        "future_seconds": arguments.future_seconds,
        "reference_speed_kph": arguments.reference_speed_kph,
        "minimum_factor": arguments.minimum_factor,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote {:,} rows from {} episodes to {}".format(
        len(data), data["group_id"].nunique(), output
    ))
    print("Conditions: {}".format(report["condition_rows"]))
    print("Report: {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
