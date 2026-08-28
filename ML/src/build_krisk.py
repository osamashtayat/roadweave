#!/usr/bin/env python3

"""Convert selected K-Risk trajectory events into RoadWeave ML tables.

This converter intentionally exposes only the source-neutral physical features
defined by ``features.py``.  K-Risk risk scores, risk folders, behaviour flags,
collision labels, filenames, and GPT text are used only to select/label rows;
they never become model inputs.

Supported trajectory schemas:

* highD (moderate and high-risk event folders)
* CitySim ExpresswayA and FreewayB (moderate/high and extreme folders)

One event becomes one row summarising at most the three seconds ending at its
peak-risk frame.  Files are read one at a time so the 18 GB release is never
loaded into memory as a whole.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from common_labels import (  # noqa: E402
    policy_label_from_future_motion,
    risk_label_from_future,
)
from features import new_state, set_slot, summarize_history  # noqa: E402
from labels import ACTION_NAMES, KRISK_ACTION_MAP, DrivingAction  # noqa: E402


DEFAULT_DATA_ROOT = Path("/Users/asus/Downloads/32896772/K-Risk_data")
DEFAULT_OUTPUT_DIRECTORY = (
    Path("/Users/asus/Desktop/roadweave/ML/data/processed")
)
DEFAULT_REPORT_PATH = Path(
    "/Users/asus/Desktop/roadweave/ML/reports/krisk_conversion_report.json"
)

HIGHD_EGO_PATTERN = re.compile(
    r"^highd_(?P<track>\d+)_(?P<ego>\d+)_.*_frame_"
    r"(?P<start>\d+)_to_(?P<end>\d+)$",
    re.IGNORECASE,
)
CITYSIM_EGO_PATTERN = re.compile(
    r"^(?P<source>expresswayA|freewayB)_track_(?P<track>\d+)_"
    r"car_(?P<ego>\d+)_frame_(?P<start>\d+)_to_(?P<end>\d+)$",
    re.IGNORECASE,
)
LEVELX_EGO_PATTERN = re.compile(
    r"^recording_(?P<recording>\d+)_ego_(?P<ego>\d+)_frame_"
    r"(?P<start>-?\d+)_to_(?P<end>-?\d+)$",
    re.IGNORECASE,
)

ACTION_PATTERNS = (
    re.compile(
        r"final\s+decision\s*:\s*(?:action\s*)?[#*_\s]*([1-5])", re.I
    ),
    re.compile(r"action[_\s-]*id\s*[:=]\s*[#*_\s]*([1-5])", re.I),
    re.compile(
        r"recommended\s+action\s*:\s*(?:action\s*)?[#*_\s]*([1-5])",
        re.I,
    ),
)

SEVERITY_PRIORITY = {
    "MODERATE": 1,
    "HIGH": 2,
    "EXTREME": 3,
}

# Dataset-native sampling frequencies.  K-Risk retains the source frame IDs.
SOURCE_FPS = {
    "highd": 25.0,
    "expresswaya": 30.0,
    "freewayb": 30.0,
    "ind": 25.0,
    "round": 25.0,
}

FORBIDDEN_FEATURE_FRAGMENTS = (
    "risk",
    "collision",
    "acc_high",
    "brake_high",
    "turn_label",
    "lane_diff",
    "action_id",
    "filename",
    "dataset_source",
    "severity",
)


class ConversionError(RuntimeError):
    """An event cannot be converted safely."""


def finite_number(value: object, default: float = math.nan) -> float:
    """Return a finite float or ``default`` without raising."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def bounded_number(
    value: object,
    minimum: float,
    maximum: float,
    default: float = math.nan,
) -> float:
    number = finite_number(value, default)
    if not math.isfinite(number) or number < minimum or number > maximum:
        return default
    return number


def normalized_id(
    value: object,
    zero_is_missing: bool = True,
) -> Optional[str]:
    """Normalize numeric/string IDs using the source's zero-ID convention."""

    if value is None:
        return None

    number = finite_number(value)
    if math.isfinite(number):
        if number == 0.0 and zero_is_missing:
            return None
        if number.is_integer():
            return str(int(number))
        return str(number)

    text = str(value).strip()
    missing_text = {"none", "nan", "null"}
    if zero_is_missing:
        missing_text.add("0")
    if not text or text.lower() in missing_text:
        return None
    return text


def circular_difference(current: float, previous: float) -> float:
    """Smallest signed angular difference in radians."""

    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def canonical_response_stem(path: Path) -> str:
    """Map a GPT response filename to the corresponding trajectory stem."""

    stem = path.stem
    for prefix in (
        "response_extreme_desc_",
        "response_",
        "extreme_desc_",
    ):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def parse_gpt_action(path: Path) -> Optional[int]:
    """Extract the final K-Risk action ID from one GPT-4.1 response."""

    text = path.read_text(encoding="utf-8", errors="replace")

    for pattern in ACTION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return int(matches[-1])

    return None


def load_gpt_actions(gpt_directory: Path) -> Tuple[Dict[str, int], List[str]]:
    actions: Dict[str, int] = {}
    errors: List[str] = []

    if not gpt_directory.is_dir():
        return actions, ["GPT directory does not exist: {0}".format(gpt_directory)]

    for response_path in sorted(gpt_directory.glob("*.txt")):
        event_stem = canonical_response_stem(response_path)
        action_id = parse_gpt_action(response_path)

        if action_id is None:
            errors.append(
                "Could not parse an action from {0}".format(response_path.name)
            )
            continue

        actions[event_stem] = action_id

    return actions, errors


def event_directories(data_root: Path) -> Sequence[Tuple[str, str, str, Path, int]]:
    hv_root = data_root / "event_annotations" / "HV"

    return (
        (
            "highd",
            "highd",
            "MODERATE",
            hv_root / "highd" / "highd_normal_risk",
            0,
        ),
        (
            "highd",
            "highd",
            "HIGH",
            hv_root / "highd" / "highd_high_risk",
            0,
        ),
        (
            "expresswayA",
            "citysim",
            "MODERATE",
            hv_root / "expresswayA" / "expresswayA_normal_risk",
            0,
        ),
        (
            "expresswayA",
            "citysim",
            "HIGH",
            hv_root / "expresswayA" / "expresswayA_high_risk",
            0,
        ),
        (
            "freewayB",
            "citysim",
            "MODERATE",
            hv_root / "freewayB" / "freewayB_normal_risk",
            0,
        ),
        (
            "freewayB",
            "citysim",
            "HIGH",
            hv_root / "freewayB" / "freewayB_high_risk",
            0,
        ),
        (
            "ind",
            "ind",
            "MODERATE",
            hv_root / "ind" / "ind_normal_risk",
            0,
        ),
        (
            "ind",
            "ind",
            "HIGH",
            hv_root / "ind" / "ind_high_risk",
            0,
        ),
        (
            "round",
            "round",
            "MODERATE",
            hv_root / "round" / "round_normal_risk",
            0,
        ),
        (
            "round",
            "round",
            "HIGH",
            hv_root / "round" / "round_high_risk",
            0,
        ),
        # ttc_1s wins a same-severity tie with ttc_2s.
        (
            "extreme_ttc_2s",
            "citysim",
            "EXTREME",
            hv_root / "extreme" / "ttc_2s",
            1,
        ),
        (
            "extreme_ttc_1s",
            "citysim",
            "EXTREME",
            hv_root / "extreme" / "ttc_1s",
            2,
        ),
    )


def discover_events(data_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Find target events and deduplicate with Extreme > High > Moderate."""

    selected: Dict[str, Dict[str, object]] = {}
    discovered = Counter()

    for source, schema, severity, directory, tie_priority in event_directories(
        data_root
    ):
        if not directory.is_dir():
            raise FileNotFoundError(
                "Required K-Risk directory does not exist: {0}".format(directory)
            )

        for path in sorted(directory.glob("*.json")):
            discovered["{0}:{1}".format(source, severity)] += 1
            candidate = {
                "path": path,
                "stem": path.stem,
                "source": source,
                "schema": schema,
                "severity": severity,
                "priority": (SEVERITY_PRIORITY[severity], tie_priority),
            }

            previous = selected.get(path.stem)
            if previous is None or candidate["priority"] > previous["priority"]:
                selected[path.stem] = candidate

    candidates = sorted(
        selected.values(),
        key=lambda item: (str(item["source"]), str(item["stem"])),
    )

    summary = dict(discovered)
    summary["files_discovered"] = int(sum(discovered.values()))
    summary["unique_events_selected"] = len(candidates)
    summary["duplicates_removed"] = summary["files_discovered"] - len(candidates)
    return candidates, summary


def parse_event_identity(stem: str, schema: str) -> Tuple[str, str, int, int]:
    if schema == "highd":
        pattern = HIGHD_EGO_PATTERN
    elif schema == "citysim":
        pattern = CITYSIM_EGO_PATTERN
    else:
        pattern = LEVELX_EGO_PATTERN
    match = pattern.match(stem)
    if match is None:
        raise ConversionError("Unrecognized {0} filename: {1}".format(schema, stem))

    if schema == "highd":
        source = "highd"
    elif schema == "citysim":
        source = match.group("source")
    else:
        source = schema
    return (
        source,
        match.group("ego"),
        int(match.group("start")),
        int(match.group("end")),
    )


def _flatten_nested_frames(
    frames: Iterable[object],
    schema: str,
) -> List[Dict[str, object]]:
    frame_key = "frame" if schema == "highd" else "frame_id"
    flattened: List[Dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_id = frame.get("frame_id", frame.get("frame"))
        vehicles = frame.get("vehicles", [])
        if not isinstance(vehicles, list):
            continue
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            record = dict(vehicle)
            record.setdefault(frame_key, frame_id)
            flattened.append(record)
    return flattened


def flatten_records(payload: object, schema: str) -> List[Dict[str, object]]:
    """Accept flat vehicle lists and both documented nested frame shapes."""

    if isinstance(payload, list):
        # inD/rounD release a bare list of {"frame_id": ..., "vehicles": [...]}
        # frame objects, while highD/CitySim release a flat list of per-frame
        # vehicle records.
        if payload and isinstance(payload[0], dict) and "vehicles" in payload[0]:
            return _flatten_nested_frames(payload, schema)
        return [record for record in payload if isinstance(record, dict)]

    if not isinstance(payload, dict):
        raise ConversionError("Event JSON must contain a list or object.")

    frames = payload.get("frames")
    if not isinstance(frames, list):
        for key in ("records", "data", "trajectories"):
            records = payload.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        raise ConversionError("Event object does not contain frames/records/data.")

    return _flatten_nested_frames(frames, schema)


def record_fields(schema: str) -> Dict[str, object]:
    if schema == "highd":
        return {
            "id": "id",
            "frame": "frame",
            "x": "x",
            "y": "y",
            "length": "width",
            "speed": "speed",
            "vx": "xVelocity",
            "vy": "yVelocity",
            "ax": "xAcceleration",
            "ay": "yAcceleration",
            "accel": None,
            "heading": None,
            "risk": "total_risk",
            "ttc": "ttc",
            "front": ("precedingId",),
            "left_front": ("leftPrecedingId",),
            "left_rear": ("leftFollowingId",),
            "right_front": ("rightPrecedingId",),
            "right_rear": ("rightFollowingId",),
            "left_alongside": "leftAlongsideId",
            "right_alongside": "rightAlongsideId",
            "zero_relation_is_missing": True,
            "behaviour": ("acc_high", "brake_high", "yaw_left", "yaw_right"),
        }

    if schema in ("ind", "round"):
        return {
            "id": "trackId",
            "frame": "frame_id",
            "x": "xCenter",
            "y": "yCenter",
            "length": "length",
            "speed": "lonVelocity",
            "vx": "xVelocity",
            "vy": "yVelocity",
            "ax": "xAcceleration",
            "ay": "yAcceleration",
            "accel": ("lonAcceleration",),
            "heading": "heading",
            "yaw_rate": None,
            "risk": "risk_value",
            "ttc": None,
            "front": ("preceding_id",),
            "left_front": ("left_preceding_id",),
            "left_rear": ("left_following_id",),
            "right_front": ("right_preceding_id",),
            "right_rear": ("right_following_id",),
            "left_alongside": "left_alongside_id",
            "right_alongside": "right_alongside_id",
            # inD/rounD use NaN for a missing relation, so ID 0 is valid.
            "zero_relation_is_missing": False,
            "behaviour": (),
        }

    return {
        "id": "car_id",
        "frame": "frame_id",
        "x": "car_center_x",
        "y": "car_center_y",
        "length": "length",
        "speed": "speed",
        "vx": "vx",
        "vy": "vy",
        "ax": None,
        "ay": None,
        "accel": ("acceleration", "acc_rate"),
        "heading": "course",
        "yaw_rate": "yaw_rate",
        "risk": "risk_value",
        "ttc": "ttc",
        "front": ("preceding_id",),
        "left_front": ("left_preceding_id",),
        "left_rear": ("left_following_id",),
        "right_front": ("right_preceding_id",),
        "right_rear": ("right_following_id",),
        "left_alongside": "left_alongside_id",
        "right_alongside": "right_alongside_id",
        # CitySim uses null for a missing relation, so actor ID 0 is valid.
        "zero_relation_is_missing": False,
        "behaviour": (
            "acc_label",
            "brake_label",
            "left_turn_label",
            "right_turn_label",
        ),
    }


def actor_speed(record: Mapping[str, object], fields: Mapping[str, object]) -> float:
    speed = bounded_number(record.get(str(fields["speed"])), 0.0, 70.0)
    if math.isfinite(speed):
        return speed

    vx = finite_number(record.get(str(fields["vx"])))
    vy = finite_number(record.get(str(fields["vy"])))
    if math.isfinite(vx) and math.isfinite(vy):
        return bounded_number(math.hypot(vx, vy), 0.0, 70.0)

    return math.nan


def longitudinal_acceleration(
    record: Mapping[str, object],
    fields: Mapping[str, object],
    previous_speed: float,
    delta_time: float,
) -> float:
    accel_fields = fields["accel"]
    if isinstance(accel_fields, tuple):
        for name in accel_fields:
            value = bounded_number(record.get(name), -15.0, 15.0)
            if math.isfinite(value):
                return value

    ax_field = fields["ax"]
    ay_field = fields["ay"]
    if ax_field is not None and ay_field is not None:
        vx = finite_number(record.get(str(fields["vx"])))
        vy = finite_number(record.get(str(fields["vy"])))
        ax = finite_number(record.get(str(ax_field)))
        ay = finite_number(record.get(str(ay_field)))
        speed = math.hypot(vx, vy) if math.isfinite(vx) and math.isfinite(vy) else 0.0
        if speed > 0.05 and math.isfinite(ax) and math.isfinite(ay):
            projected = (vx * ax + vy * ay) / speed
            return bounded_number(projected, -15.0, 15.0)

    speed = actor_speed(record, fields)
    if (
        math.isfinite(speed)
        and math.isfinite(previous_speed)
        and delta_time > 0.0
    ):
        return bounded_number((speed - previous_speed) / delta_time, -15.0, 15.0)

    return math.nan


def heading_radians(
    record: Mapping[str, object], fields: Mapping[str, object]
) -> float:
    heading_field = fields["heading"]
    if heading_field is not None:
        heading_degrees = finite_number(record.get(str(heading_field)))
        if math.isfinite(heading_degrees):
            return math.radians(heading_degrees)

    vx = finite_number(record.get(str(fields["vx"])))
    vy = finite_number(record.get(str(fields["vy"])))
    if math.isfinite(vx) and math.isfinite(vy) and math.hypot(vx, vy) > 0.05:
        return math.atan2(vy, vx)

    return math.nan


def first_actor_id(
    ego: Mapping[str, object],
    relation_fields: Iterable[str],
    zero_is_missing: bool,
) -> Optional[str]:
    for field in relation_fields:
        actor_id = normalized_id(
            ego.get(field),
            zero_is_missing=zero_is_missing,
        )
        if actor_id is not None:
            return actor_id
    return None


def longitudinal_gap(
    ego: Mapping[str, object],
    actor: Mapping[str, object],
    fields: Mapping[str, object],
    slot: str,
) -> float:
    """Return bumper distance projected along the ego vehicle's heading.

    Euclidean distance incorrectly adds the lateral separation of an adjacent
    lane. K-Risk relation IDs already say whether an actor is ahead or behind,
    so the longitudinal projection is the useful gap for TTC and clearance.
    """

    ego_x = finite_number(ego.get(str(fields["x"])))
    ego_y = finite_number(ego.get(str(fields["y"])))
    actor_x = finite_number(actor.get(str(fields["x"])))
    actor_y = finite_number(actor.get(str(fields["y"])))

    if not all(math.isfinite(value) for value in (ego_x, ego_y, actor_x, actor_y)):
        return math.nan

    ego_length = bounded_number(ego.get(str(fields["length"])), 0.0, 30.0, 0.0)
    actor_length = bounded_number(
        actor.get(str(fields["length"])), 0.0, 30.0, 0.0
    )
    heading = heading_radians(ego, fields)
    if math.isfinite(heading):
        signed_distance = (
            (actor_x - ego_x) * math.cos(heading)
            + (actor_y - ego_y) * math.sin(heading)
        )
        centre_distance = (
            -signed_distance if slot.endswith("rear") else signed_distance
        )
        # Relation IDs are authoritative. Source coordinate systems can point
        # opposite to increasing X, and noisy records can cross zero slightly.
        if centre_distance < 0.0:
            centre_distance = abs(signed_distance)
    else:
        # Stationary records can lack a usable heading; retain their relation
        # with a conservative centre-distance fallback.
        centre_distance = math.hypot(actor_x - ego_x, actor_y - ego_y)

    gap = centre_distance - 0.5 * (ego_length + actor_length)
    return bounded_number(max(0.0, gap), 0.0, 150.0)


def add_actor_slot(
    state: Dict[str, float],
    slot: str,
    ego: Mapping[str, object],
    actor: Mapping[str, object],
    fields: Mapping[str, object],
) -> None:
    gap = longitudinal_gap(ego, actor, fields, slot)
    ego_speed = actor_speed(ego, fields)
    other_speed = actor_speed(actor, fields)

    if not math.isfinite(gap):
        return

    if not math.isfinite(other_speed):
        other_speed = math.nan

    if slot.endswith("rear"):
        closing_speed = other_speed - ego_speed
    else:
        closing_speed = ego_speed - other_speed

    closing_speed = bounded_number(closing_speed, -70.0, 70.0, 0.0)
    set_slot(state, slot, gap, closing_speed, other_speed)


def choose_peak_frame(
    ego_records: Sequence[Mapping[str, object]],
    fields: Mapping[str, object],
    severity: str,
) -> int:
    frame_field = str(fields["frame"])

    if severity == "EXTREME":
        conflict_candidates: List[Tuple[float, int]] = []
        ttc_candidates: List[Tuple[float, int]] = []

        for record in ego_records:
            frame = int(finite_number(record.get(frame_field)))
            conflict_ttc = finite_number(record.get("Time to Collision"))
            if math.isfinite(conflict_ttc) and conflict_ttc >= 0.0:
                conflict_candidates.append((conflict_ttc, frame))

            raw_ttc = finite_number(record.get(str(fields["ttc"])))
            if math.isfinite(raw_ttc) and abs(raw_ttc) > 0.001:
                ttc_candidates.append((abs(raw_ttc), frame))

        if conflict_candidates:
            return min(conflict_candidates)[1]
        if ttc_candidates:
            return min(ttc_candidates)[1]

    risk_field = str(fields["risk"])
    risk_candidates: List[Tuple[float, int]] = []
    for record in ego_records:
        risk = finite_number(record.get(risk_field))
        frame = int(finite_number(record.get(frame_field)))
        if math.isfinite(risk):
            risk_candidates.append((risk, frame))

    if risk_candidates:
        return max(risk_candidates)[1]

    behaviour_fields = fields["behaviour"]
    behaviour_frames = [
        int(finite_number(record.get(frame_field)))
        for record in ego_records
        if any(bool(record.get(name)) for name in behaviour_fields)
    ]
    if behaviour_frames:
        return behaviour_frames[len(behaviour_frames) // 2]

    # Last-resort event anchor: 60% through the segment leaves room for both a
    # lead-in and a resolution period without fabricating unavailable frames.
    fallback_index = int(round((len(ego_records) - 1) * 0.60))
    return int(finite_number(ego_records[fallback_index].get(frame_field)))


def detect_lane_change(
    ego_records: Sequence[Mapping[str, object]],
    fields: Mapping[str, object],
    peak_frame: int,
    schema: str,
) -> Optional[str]:
    """Return CHANGE_LEFT/CHANGE_RIGHT when the ego makes a lateral maneuver.

    highD carries explicit per-frame ``yaw_left``/``yaw_right`` signals and a
    clip-level ``lane_diff`` flag.  A yaw signal alone is only notable lateral
    movement, so require ``lane_diff`` before using its direction as a completed
    lane-change policy label. CitySim's threshold-based turn labels are
    intentionally ignored because the release documents them as coarse and, in
    FreewayB, permissive.
    """

    if schema != "highd":
        return None

    if not any(bool(record.get("lane_diff")) for record in ego_records):
        return None

    frame_field = str(fields["frame"])
    behaviour_fields = fields["behaviour"]  # (acc_high, brake_high, yaw_left, yaw_right)
    left_name = str(behaviour_fields[2])
    right_name = str(behaviour_fields[3])

    window = [
        record
        for record in ego_records
        if abs(int(finite_number(record.get(frame_field))) - peak_frame) <= 3
    ]
    if not window:
        window = ego_records

    left = any(bool(record.get(left_name)) for record in window)
    right = any(bool(record.get(right_name)) for record in window)
    if left and not right:
        return ACTION_NAMES[DrivingAction.CHANGE_LEFT]
    if right and not left:
        return ACTION_NAMES[DrivingAction.CHANGE_RIGHT]
    return None


def build_states(
    records: Sequence[Mapping[str, object]],
    ego_id: str,
    schema: str,
    source: str,
    anchor_frame: int,
    history_seconds: float,
    future_seconds: float = 0.0,
) -> List[Dict[str, float]]:
    fields = record_fields(schema)
    frame_field = str(fields["frame"])
    id_field = str(fields["id"])
    fps = SOURCE_FPS[source.lower()]
    first_allowed_frame = anchor_frame - int(round(history_seconds * fps))
    last_allowed_frame = anchor_frame + int(round(future_seconds * fps))

    by_frame: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        frame_number = finite_number(record.get(frame_field))
        if not math.isfinite(frame_number):
            continue
        frame = int(frame_number)
        if first_allowed_frame <= frame <= last_allowed_frame:
            by_frame[frame].append(record)

    states: List[Dict[str, float]] = []
    previous_heading = math.nan
    previous_speed = math.nan
    previous_timestamp = math.nan

    for frame in sorted(by_frame):
        frame_records = by_frame[frame]
        records_by_id = {
            actor_id: record
            for record in frame_records
            for actor_id in (
                normalized_id(record.get(id_field), zero_is_missing=False),
            )
            if actor_id is not None
        }
        ego = records_by_id.get(ego_id)
        if ego is None:
            continue

        timestamp = frame / fps
        speed = actor_speed(ego, fields)
        delta_time = (
            timestamp - previous_timestamp
            if math.isfinite(previous_timestamp)
            else 1.0 / fps
        )
        acceleration = longitudinal_acceleration(
            ego, fields, previous_speed, delta_time
        )
        heading = heading_radians(ego, fields)

        yaw_rate = math.nan
        yaw_rate_field = fields.get("yaw_rate")
        if yaw_rate_field is not None:
            # CitySim publishes this field in degrees per second.
            yaw_rate = bounded_number(
                ego.get(str(yaw_rate_field)),
                -180.0,
                180.0,
            )
        if (
            not math.isfinite(yaw_rate)
            and math.isfinite(heading)
            and math.isfinite(previous_heading)
            and delta_time > 0.0
        ):
            yaw_rate = bounded_number(
                math.degrees(
                    circular_difference(heading, previous_heading) / delta_time
                ),
                -180.0,
                180.0,
            )

        state = new_state(timestamp, speed, acceleration, yaw_rate)

        relation_ids: Dict[str, Optional[str]] = {}
        for slot in (
            "front",
            "left_front",
            "left_rear",
            "right_front",
            "right_rear",
        ):
            relation_ids[slot] = first_actor_id(
                ego,
                fields[slot],
                bool(fields["zero_relation_is_missing"]),
            )
            actor = records_by_id.get(relation_ids[slot])
            if actor is not None:
                add_actor_slot(state, slot, ego, actor, fields)

        zero_relation_is_missing = bool(fields["zero_relation_is_missing"])
        left_alongside = normalized_id(
            ego.get(str(fields["left_alongside"])),
            zero_is_missing=zero_relation_is_missing,
        )
        right_alongside = normalized_id(
            ego.get(str(fields["right_alongside"])),
            zero_is_missing=zero_relation_is_missing,
        )

        state["left_clear"] = float(
            left_alongside is None
            and (
                state["left_front_present"] == 0.0
                or state["left_front_gap"] >= 15.0
            )
            and (
                state["left_rear_present"] == 0.0
                or state["left_rear_gap"] >= 10.0
            )
        )
        state["right_clear"] = float(
            right_alongside is None
            and (
                state["right_front_present"] == 0.0
                or state["right_front_gap"] >= 15.0
            )
            and (
                state["right_rear_present"] == 0.0
                or state["right_rear_gap"] >= 10.0
            )
        )

        states.append(state)
        previous_heading = heading
        previous_speed = speed
        previous_timestamp = timestamp

    return states


def choose_observation_frame(
    ego_records: Sequence[Mapping[str, object]],
    fields: Mapping[str, object],
    fps: float,
    history_seconds: float,
    future_seconds: float,
    preferred_frame: Optional[int] = None,
) -> int:
    """Choose a real frame with enough measured past and future context."""

    frame_field = str(fields["frame"])
    frames = sorted(
        {
            int(finite_number(record.get(frame_field)))
            for record in ego_records
            if math.isfinite(finite_number(record.get(frame_field)))
        }
    )
    if not frames:
        raise ConversionError("Ego trajectory contains no finite frame IDs.")

    lower = frames[0] + int(math.ceil(history_seconds * fps))
    upper = frames[-1] - int(math.ceil(future_seconds * fps))
    candidates = [frame for frame in frames if lower <= frame <= upper]
    if not candidates:
        raise ConversionError(
            "Ego trajectory is too short for {:.1f}s history and {:.1f}s future."
            .format(history_seconds, future_seconds)
        )
    target = float(preferred_frame) if preferred_frame is not None else 0.5 * (lower + upper)
    return min(candidates, key=lambda frame: abs(frame - target))


def native_lane_change_anchor(
    ego_records: Sequence[Mapping[str, object]],
    fields: Mapping[str, object],
    schema: str,
    fps: float,
) -> Tuple[Optional[int], Optional[str]]:
    """Return a pre-maneuver anchor for highD's measured lane-change clips."""

    if schema != "highd" or not any(bool(record.get("lane_diff")) for record in ego_records):
        return None, None
    frame_field = str(fields["frame"])
    behaviour_fields = fields["behaviour"]
    left_name = str(behaviour_fields[2])
    right_name = str(behaviour_fields[3])
    for record in ego_records:
        left = bool(record.get(left_name))
        right = bool(record.get(right_name))
        if left == right:
            continue
        maneuver_frame = int(finite_number(record.get(frame_field)))
        anchor = maneuver_frame - int(round(0.25 * fps))
        direction = (
            ACTION_NAMES[DrivingAction.CHANGE_LEFT]
            if left
            else ACTION_NAMES[DrivingAction.CHANGE_RIGHT]
        )
        return anchor, direction
    return None, None


def future_motion(
    ego_records: Sequence[Mapping[str, object]],
    fields: Mapping[str, object],
    anchor_frame: int,
    fps: float,
    future_seconds: float,
) -> Tuple[float, float]:
    """Measure actual ego speed/lateral change after an observation frame."""

    frame_field = str(fields["frame"])
    target_frame = anchor_frame + int(round(future_seconds * fps))
    window = [
        record
        for record in ego_records
        if anchor_frame
        <= int(finite_number(record.get(frame_field)))
        <= target_frame
    ]
    if len(window) < 2:
        raise ConversionError("Insufficient future ego motion for a policy label.")
    window.sort(key=lambda record: finite_number(record.get(frame_field)))

    initial_speed = actor_speed(window[0], fields)
    final_speed = actor_speed(window[-1], fields)
    if not math.isfinite(initial_speed) or not math.isfinite(final_speed):
        raise ConversionError("Future ego motion has no finite speed.")

    lateral_displacement = 0.0
    x_field = str(fields["x"])
    y_field = str(fields["y"])
    for previous, current in zip(window, window[1:]):
        previous_x = finite_number(previous.get(x_field))
        previous_y = finite_number(previous.get(y_field))
        current_x = finite_number(current.get(x_field))
        current_y = finite_number(current.get(y_field))
        heading = heading_radians(previous, fields)
        if not all(
            math.isfinite(value)
            for value in (previous_x, previous_y, current_x, current_y, heading)
        ):
            continue
        delta_x = current_x - previous_x
        delta_y = current_y - previous_y
        # Unit vector to the driver's left in the ego heading frame.
        lateral_displacement += -math.sin(heading) * delta_x + math.cos(heading) * delta_y

    return final_speed - initial_speed, lateral_displacement


def convert_event(
    candidate: Mapping[str, object],
    history_seconds: float,
    future_seconds: float = 1.0,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    path = candidate["path"]
    if not isinstance(path, Path):
        raise ConversionError("Candidate path is invalid.")

    schema = str(candidate["schema"])
    filename_source, ego_id, _start, _end = parse_event_identity(path.stem, schema)
    source = "highd" if schema == "highd" else filename_source
    fields = record_fields(schema)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = flatten_records(payload, schema)
    if not records:
        raise ConversionError("Event has no trajectory records.")

    id_field = str(fields["id"])
    frame_field = str(fields["frame"])
    ego_records = [
        record
        for record in records
        if normalized_id(record.get(id_field), zero_is_missing=False) == ego_id
        and math.isfinite(finite_number(record.get(frame_field)))
    ]
    ego_records.sort(key=lambda record: finite_number(record.get(frame_field)))

    if not ego_records:
        raise ConversionError("Ego ID {0} is absent from event.".format(ego_id))

    fps = SOURCE_FPS[source.lower()]
    anchor_frame = choose_observation_frame(
        ego_records,
        fields,
        fps,
        history_seconds,
        future_seconds,
    )
    states = build_states(
        records,
        ego_id,
        schema,
        source,
        anchor_frame,
        history_seconds,
        future_seconds,
    )

    if not states:
        raise ConversionError("No ego states exist around the observation frame.")

    anchor_timestamp = anchor_frame / fps
    history_states = [state for state in states if state["timestamp"] <= anchor_timestamp + 1e-9]
    future_states = [state for state in states if state["timestamp"] >= anchor_timestamp - 1e-9]
    if len(history_states) < 2 or len(future_states) < 2:
        raise ConversionError("Observation does not contain both history and future states.")

    feature_values = summarize_history(history_states)
    speed_change, lateral_displacement = future_motion(
        ego_records,
        fields,
        anchor_frame,
        fps,
        future_seconds,
    )
    metadata = {
        "source": "krisk",
        "domain": source.lower(),
        "group_id": path.stem,
        "timestamp": float(anchor_timestamp),
        "label_origin": "observed_future",
        "history_seconds": float(history_seconds),
        "future_seconds": float(future_seconds),
        "sampling_hz": 2.0,
    }
    # Dataset-native flags are retained only as an audit comparison. They do
    # not select the observation time and never become the v2 policy target.
    native_lane_change = detect_lane_change(ego_records, fields, anchor_frame, schema)
    policy_target = policy_label_from_future_motion(
        speed_change,
        lateral_displacement,
    )
    audit = {
        "underlying_source": source.lower(),
        "history_frames": len(history_states),
        "history_duration_seconds": float(
            history_states[-1]["timestamp"] - history_states[0]["timestamp"]
        ),
        "future_frames": len(future_states),
        "anchor_frame": anchor_frame,
        "policy_target": policy_target,
        "native_lane_change": native_lane_change,
        "speed_change_mps": speed_change,
        "lateral_displacement_metres": lateral_displacement,
    }

    row: Dict[str, object] = dict(metadata)
    row.update(feature_values)
    row["target"] = risk_label_from_future(future_states)
    return row, audit


def expected_feature_columns() -> List[str]:
    example = new_state(0.0, 0.0, 0.0, 0.0)
    return sorted(summarize_history([example]).keys())


def select_smoke_candidates(
    candidates: Sequence[Dict[str, object]],
    gpt_actions: Mapping[str, int],
) -> List[Dict[str, object]]:
    """Select every schema/severity plus several action-labelled events."""

    selected: Dict[str, Dict[str, object]] = {}
    per_category = Counter()

    for candidate in candidates:
        key = "{0}:{1}".format(candidate["source"], candidate["severity"])
        if per_category[key] < 2:
            selected[str(candidate["stem"])] = candidate
            per_category[key] += 1

    # Exercise every GPT action class that exists, not merely the first five
    # response filenames (which are often all IDLE).
    action_ids_selected = set()
    for candidate in candidates:
        stem = str(candidate["stem"])
        action_id = gpt_actions.get(stem)
        if action_id is not None and action_id not in action_ids_selected:
            selected[stem] = candidate
            action_ids_selected.add(action_id)
        if len(action_ids_selected) >= len(KRISK_ACTION_MAP):
            break

    return sorted(selected.values(), key=lambda item: str(item["stem"]))


def validate_output(
    risk_data: pd.DataFrame,
    policy_data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Dict[str, object]:
    expected_columns = {
        "source",
        "domain",
        "group_id",
        "timestamp",
        "target",
        "label_origin",
        "history_seconds",
        "future_seconds",
        "sampling_hz",
    }.union(
        feature_columns
    )

    if set(risk_data.columns) != expected_columns:
        missing = sorted(expected_columns.difference(risk_data.columns))
        extra = sorted(set(risk_data.columns).difference(expected_columns))
        raise ConversionError(
            "Risk output schema mismatch. Missing={0}, extra={1}".format(
                missing, extra
            )
        )

    if set(policy_data.columns) != expected_columns:
        missing = sorted(expected_columns.difference(policy_data.columns))
        extra = sorted(set(policy_data.columns).difference(expected_columns))
        raise ConversionError(
            "Policy output schema mismatch. Missing={0}, extra={1}".format(
                missing, extra
            )
        )

    if risk_data["group_id"].duplicated().any():
        raise ConversionError("Risk output contains duplicate group IDs.")
    if policy_data["group_id"].duplicated().any():
        raise ConversionError("Policy output contains duplicate group IDs.")

    if not set(risk_data["target"]).issubset(
        {"LOW", "MODERATE", "HIGH", "EXTREME"}
    ):
        raise ConversionError("Risk output contains an unexpected target.")
    if not set(policy_data["target"]).issubset(
        {"KEEP", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"}
    ):
        raise ConversionError("Policy output contains an unexpected target.")

    forbidden_columns = [
        column
        for column in feature_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if forbidden_columns:
        raise ConversionError(
            "Label-leaking feature names detected: {0}".format(forbidden_columns)
        )

    return {
        "risk_rows": len(risk_data),
        "policy_rows": len(policy_data),
        "feature_count": len(feature_columns),
        "risk_targets": {
            str(key): int(value)
            for key, value in risk_data["target"].value_counts().items()
        },
        "policy_targets": {
            str(key): int(value)
            for key, value in policy_data["target"].value_counts().items()
        },
        "duplicate_risk_groups": int(risk_data["group_id"].duplicated().sum()),
        "duplicate_policy_groups": int(
            policy_data["group_id"].duplicated().sum()
        ),
    }


def run_conversion(arguments: argparse.Namespace) -> Dict[str, object]:
    data_root = arguments.data_root.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    report_path = arguments.report.expanduser().resolve()

    if arguments.history_seconds <= 0.0:
        raise ValueError("--history-seconds must be positive.")
    if arguments.future_seconds <= 0.0:
        raise ValueError("--future-seconds must be positive.")

    candidates, discovery = discover_events(data_root)
    gpt_actions, gpt_errors = load_gpt_actions(
        data_root / "llm_analysis" / "analysis_gpt4.1"
    )

    if arguments.smoke_test:
        candidates = select_smoke_candidates(candidates, gpt_actions)
    elif arguments.max_events is not None:
        candidates = candidates[: arguments.max_events]

    risk_rows: List[Dict[str, object]] = []
    policy_rows: List[Dict[str, object]] = []
    recommended_policy_rows: List[Dict[str, object]] = []
    native_lane_change_rows = 0
    native_lane_change_agreements = 0
    gpt_policy_rows_matched = 0
    errors: List[str] = list(gpt_errors)
    coverage_exclusions: List[str] = []
    source_counts = Counter()
    severity_counts = Counter()
    history_frames: List[int] = []
    history_durations: List[float] = []

    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        try:
            risk_row, audit = convert_event(
                candidate,
                arguments.history_seconds,
                arguments.future_seconds,
            )
            risk_rows.append(risk_row)
            source_counts[str(audit["underlying_source"])] += 1
            severity_counts[str(candidate["severity"])] += 1
            history_frames.append(int(audit["history_frames"]))
            history_durations.append(float(audit["history_duration_seconds"]))

            policy_row = dict(risk_row)
            policy_row["target"] = str(audit["policy_target"])
            policy_rows.append(policy_row)

            native_lane_change = audit.get("native_lane_change")
            if native_lane_change is not None:
                native_lane_change_rows += 1
                if native_lane_change == audit["policy_target"]:
                    native_lane_change_agreements += 1

            action_id = gpt_actions.get(str(candidate["stem"]))
            if action_id is not None:
                action = KRISK_ACTION_MAP.get(action_id)
                if action is None:
                    errors.append(
                        "Unknown K-Risk action {0} for {1}".format(
                            action_id, candidate["stem"]
                        )
                    )
                else:
                    recommended_row = dict(risk_row)
                    recommended_row["target"] = ACTION_NAMES[action]
                    recommended_row["label_origin"] = "recommended_gpt"
                    recommended_policy_rows.append(recommended_row)
                    gpt_policy_rows_matched += 1
        except ConversionError as error:
            message = str(error)
            if (
                "trajectory is too short" in message
                or "does not contain both history and future" in message
            ):
                coverage_exclusions.append(
                    "{0}: {1}".format(candidate["path"], message)
                )
            else:
                errors.append("{0}: {1}".format(candidate["path"], message))
        except Exception as error:  # Continue so one corrupt event is reportable.
            errors.append("{0}: {1}".format(candidate["path"], error))

        if arguments.progress_every and (
            index % arguments.progress_every == 0 or index == total
        ):
            print(
                "Processed {0:,}/{1:,} events ({2:,} errors, {3:,} coverage exclusions)".format(
                    index, total, len(errors), len(coverage_exclusions)
                ),
                flush=True,
            )

    feature_columns = expected_feature_columns()
    ordered_columns = [
        "source",
        "domain",
        "group_id",
        "timestamp",
        "target",
        "label_origin",
        "history_seconds",
        "future_seconds",
        "sampling_hz",
    ] + feature_columns

    risk_data = pd.DataFrame(risk_rows, columns=ordered_columns)
    policy_data = pd.DataFrame(policy_rows, columns=ordered_columns)
    recommended_policy_data = pd.DataFrame(
        recommended_policy_rows,
        columns=ordered_columns,
    )

    validation = validate_output(risk_data, policy_data, feature_columns)

    output_directory.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    risk_path = output_directory / "risk_krisk.parquet"
    policy_path = output_directory / "policy_krisk.parquet"
    recommended_policy_path = output_directory / "policy_krisk_recommended.parquet"

    risk_data.to_parquet(risk_path, index=False)
    policy_data.to_parquet(policy_path, index=False)
    recommended_policy_data.to_parquet(recommended_policy_path, index=False)

    report: Dict[str, object] = {
        "format": "roadweave.krisk-conversion-report/1.0",
        "data_root": str(data_root),
        "output_directory": str(output_directory),
        "history_seconds_requested": arguments.history_seconds,
        "future_seconds_requested": arguments.future_seconds,
        "smoke_test": bool(arguments.smoke_test),
        "max_events": arguments.max_events,
        "discovery": discovery,
        "selected_for_this_run": total,
        "converted_by_source": dict(source_counts),
        "converted_by_severity": dict(severity_counts),
        "gpt_responses_found": len(gpt_actions),
        "gpt_responses_matched": gpt_policy_rows_matched,
        "native_lane_change_rows": native_lane_change_rows,
        "native_lane_change_agreements": native_lane_change_agreements,
        "policy_rows_total": len(policy_rows),
        "recommended_policy_rows": len(recommended_policy_rows),
        "history": {
            "minimum_frames": min(history_frames) if history_frames else 0,
            "maximum_frames": max(history_frames) if history_frames else 0,
            "mean_frames": float(np.mean(history_frames)) if history_frames else 0.0,
            "minimum_duration_seconds": (
                min(history_durations) if history_durations else 0.0
            ),
            "maximum_duration_seconds": (
                max(history_durations) if history_durations else 0.0
            ),
            "mean_duration_seconds": (
                float(np.mean(history_durations)) if history_durations else 0.0
            ),
        },
        "validation": validation,
        "error_count": len(errors),
        "errors": errors[:100],
        "coverage_exclusion_count": len(coverage_exclusions),
        "coverage_exclusions": coverage_exclusions[:100],
        "outputs": {
            "risk": str(risk_path),
            "policy": str(policy_path),
            "recommended_policy_audit_only": str(recommended_policy_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors and arguments.fail_on_error:
        raise ConversionError(
            "Conversion completed with {0} errors; see {1}".format(
                len(errors), report_path
            )
        )

    print("Wrote {0:,} risk rows to {1}".format(len(risk_data), risk_path))
    print("Wrote {0:,} policy rows to {1}".format(len(policy_data), policy_path))
    print(
        "Wrote {0:,} recommendation-only rows to {1}".format(
            len(recommended_policy_data), recommended_policy_path
        )
    )
    print("Wrote conversion report to {0}".format(report_path))
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert K-Risk events into RoadWeave ML Parquet tables."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the extracted K-Risk_data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for risk_krisk.parquet and policy_krisk.parquet.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON conversion report path.",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=1.0,
        help="Past/current history before the observation frame (default: 1).",
    )
    parser.add_argument(
        "--future-seconds",
        type=float,
        default=1.0,
        help="Measured future used only for physical labels (default: 1).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Convert only the first N deduplicated events (development only).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Convert a balanced small sample including GPT-labelled events.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N events; use 0 to disable.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit unsuccessfully if any input event could not be converted.",
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    run_conversion(arguments)


if __name__ == "__main__":
    main()
