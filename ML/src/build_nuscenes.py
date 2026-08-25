#!/usr/bin/env python3
"""Build RoadWeave ML tables from the full nuScenes train/validation set.

The converter produces two Parquet files:

* ``risk_nuscenes.parquet`` contains LOW/MODERATE/HIGH/EXTREME windows graded
  by physical severity, so nuScenes is not a pure "safe" source shortcut.
* ``policy_nuscenes.parquet`` contains routine driving-policy examples whose
  target is derived from ego motion up to three seconds after the observation
  time (lateral motion is integrated per-frame to stay curvature-invariant).

Every input row summarizes exactly the current and previous three seconds.
The future ego pose is used only to construct the policy target and is never
included among the model features.  This prevents look-ahead leakage.

Example smoke test::

    python ML/src/build_nuscenes.py \
      --max-scenes 1 \
      --risk-output /private/tmp/risk_nuscenes_smoke.parquet \
      --policy-output /private/tmp/policy_nuscenes_smoke.parquet \
      --overwrite --validate

Run from the RoadWeave repository root with ``ML/.venv`` activated.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


# Import the canonical RoadWeave feature and label definitions without
# requiring ML/src to be installed as a Python package.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from features import BASE_SIGNALS, new_state, set_slot, summarize_history  # noqa: E402
from labels import ACTION_NAMES, DrivingAction, RISK_NAMES, RiskLevel  # noqa: E402


DEFAULT_DATAROOT = Path("/Users/asus/Downloads/v1.0-trainval")
DEFAULT_OUTPUT_DIRECTORY = Path("/Users/asus/Desktop/roadweave/ML/data/processed")
DEFAULT_REPORT_PATH = Path(
    "/Users/asus/Desktop/roadweave/ML/reports/nuscenes_conversion_report.json"
)

SUMMARY_SUFFIXES = ("now", "min", "max", "mean", "std", "trend")
FEATURE_COLUMNS = tuple(
    "{}_{}".format(signal, suffix)
    for signal in BASE_SIGNALS
    for suffix in SUMMARY_SUFFIXES
)
METADATA_COLUMNS = ("source", "group_id", "timestamp", "target")
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

# nuScenes uses x forward, y left, z up in the ego coordinate frame.
EGO_LENGTH_METRES = 4.7
EGO_WIDTH_METRES = 1.9
CURRENT_LANE_HALF_WIDTH_METRES = 2.0
ADJACENT_LANE_OUTER_EDGE_METRES = 6.0
MAX_ACTOR_LONGITUDINAL_RANGE_METRES = 120.0
MAX_PEDESTRIAN_LATERAL_RANGE_METRES = 10.0

# A neighboring lane is treated as clear only when both its front and rear
# safety envelopes are empty.  This describes occupancy; it does not claim
# that an adjacent lane physically exists on the map.
LANE_CLEAR_FRONT_METRES = 25.0
LANE_CLEAR_REAR_METRES = 15.0


@dataclass
class EgoFrame:
    """The data needed for one nuScenes keyframe."""

    sample: Mapping[str, Any]
    timestamp_seconds: float
    global_position: np.ndarray
    global_orientation: Quaternion
    speed: float
    acceleration: float
    yaw_rate: float
    state: Dict[str, float]
    has_overlap: bool


@dataclass
class BuildStatistics:
    """Counters reported after conversion."""

    scenes_requested: int = 0
    scenes_processed: int = 0
    scenes_skipped_can: int = 0
    scenes_failed: int = 0
    samples_seen: int = 0
    samples_missing_can: int = 0
    windows_incomplete: int = 0
    windows_without_future: int = 0
    policy_rows: int = 0
    risk_rows: int = 0


class IncrementalParquetWriter:
    """Write scene-sized row groups using one stable RoadWeave schema."""

    def __init__(self, path: Path, overwrite: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists() and not overwrite:
            raise FileExistsError(
                "Output already exists: {}. Pass --overwrite to replace it."
                .format(self.path)
            )

        self.schema = pa.schema(
            [
                pa.field("source", pa.string(), nullable=False),
                pa.field("group_id", pa.string(), nullable=False),
                pa.field("timestamp", pa.float64(), nullable=False),
                pa.field("target", pa.string(), nullable=False),
            ]
            + [pa.field(column, pa.float64()) for column in FEATURE_COLUMNS]
        )
        self.writer = pq.ParquetWriter(
            str(self.path),
            self.schema,
            compression="zstd",
            use_dictionary=("source", "group_id", "target"),
        )
        self.row_count = 0

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return

        columns: Dict[str, List[Any]] = {
            column: [] for column in OUTPUT_COLUMNS
        }
        for row in rows:
            for column in OUTPUT_COLUMNS:
                value = row.get(column)
                if column in FEATURE_COLUMNS:
                    value = _finite_or_nan(value)
                columns[column].append(value)

        table = pa.Table.from_pydict(columns, schema=self.schema)
        self.writer.write_table(table)
        self.row_count += len(rows)

    def close(self) -> None:
        self.writer.close()


def _finite_or_nan(value: Any) -> float:
    """Convert an optional numeric value to float while preserving missingness."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip_observation(value: Any, lower: float, upper: float) -> float:
    """Reject physically impossible observations instead of hiding them."""

    number = _finite_or_nan(value)
    if not math.isfinite(number) or number < lower or number > upper:
        return float("nan")
    return number


def _wrap_angle(angle_radians: float) -> float:
    return (angle_radians + math.pi) % (2.0 * math.pi) - math.pi


def _yaw(quaternion: Quaternion) -> float:
    """Return z-axis yaw from a nuScenes quaternion."""

    w, x, y, z = quaternion.elements
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _nearest_message(
    messages: Sequence[Mapping[str, Any]],
    timestamps: Sequence[int],
    target_timestamp: int,
    maximum_difference_seconds: float,
) -> Optional[Mapping[str, Any]]:
    """Find the nearest CAN message without looking beyond a tolerance."""

    if not messages:
        return None

    insertion = bisect.bisect_left(timestamps, target_timestamp)
    candidates = []
    if insertion < len(messages):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)

    closest_index = min(
        candidates,
        key=lambda index: abs(timestamps[index] - target_timestamp),
    )
    difference_seconds = (
        abs(timestamps[closest_index] - target_timestamp) / 1_000_000.0
    )
    if difference_seconds > maximum_difference_seconds:
        return None
    return messages[closest_index]


def _scene_samples(nusc: NuScenes, scene: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    samples: List[Mapping[str, Any]] = []
    token = str(scene["first_sample_token"])
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = str(sample["next"])
    return samples


def _ego_pose_for_sample(
    nusc: NuScenes,
    sample: Mapping[str, Any],
) -> Mapping[str, Any]:
    lidar_token = sample["data"]["LIDAR_TOP"]
    sample_data = nusc.get("sample_data", lidar_token)
    return nusc.get("ego_pose", sample_data["ego_pose_token"])


def _annotation_velocity_in_ego(
    nusc: NuScenes,
    annotation_token: str,
    inverse_ego_orientation: Quaternion,
) -> np.ndarray:
    global_velocity = np.asarray(
        nusc.box_velocity(annotation_token),
        dtype=np.float64,
    )
    if global_velocity.shape != (3,) or not np.all(np.isfinite(global_velocity)):
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
    return np.asarray(
        inverse_ego_orientation.rotate(global_velocity),
        dtype=np.float64,
    )


def _bumper_gap(
    longitudinal_centre: float,
    actor_length: float,
) -> float:
    raw_gap = (
        abs(longitudinal_centre)
        - 0.5 * EGO_LENGTH_METRES
        - 0.5 * actor_length
    )
    return max(0.0, raw_gap)


def _actor_overlaps_ego(
    relative_position: np.ndarray,
    actor_width: float,
    actor_length: float,
) -> bool:
    longitudinal_limit = 0.5 * (EGO_LENGTH_METRES + actor_length)
    lateral_limit = 0.5 * (EGO_WIDTH_METRES + actor_width)
    return (
        abs(float(relative_position[0])) < longitudinal_limit
        and abs(float(relative_position[1])) < lateral_limit
    )


def _assign_slot_if_closer(
    state: Dict[str, float],
    slot: str,
    gap: float,
    closing_speed: float,
    actor_speed: float,
) -> None:
    existing_gap = state.get("{}_gap".format(slot), float("nan"))
    if _finite(existing_gap) and float(existing_gap) <= gap:
        return

    if _finite(closing_speed) and _finite(actor_speed):
        set_slot(state, slot, gap, closing_speed, actor_speed)
        return

    # ``features.set_slot`` intentionally supplies defaults for missing
    # values.  Dataset conversion must retain missingness so that an unknown
    # actor velocity is not mistaken for a stationary actor.
    state["{}_present".format(slot)] = 1.0
    state["{}_gap".format(slot)] = gap
    state["{}_closing_speed".format(slot)] = float("nan")
    state["{}_ttc".format(slot)] = float("nan")
    state["{}_speed".format(slot)] = (
        float(actor_speed) if _finite(actor_speed) else float("nan")
    )


def _populate_surrounding_actors(
    nusc: NuScenes,
    sample: Mapping[str, Any],
    ego_position: np.ndarray,
    ego_orientation: Quaternion,
    ego_speed: float,
    state: Dict[str, float],
) -> bool:
    """Populate ego-relative actor slots and report physical overlap."""

    inverse_orientation = ego_orientation.inverse
    has_overlap = False

    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)
        category = str(annotation["category_name"])
        is_vehicle = category.startswith("vehicle.")
        is_pedestrian = category.startswith("human.pedestrian")
        if not is_vehicle and not is_pedestrian:
            continue

        relative = np.asarray(
            inverse_orientation.rotate(
                np.asarray(annotation["translation"], dtype=np.float64)
                - ego_position
            ),
            dtype=np.float64,
        )
        longitudinal = float(relative[0])
        lateral = float(relative[1])

        if abs(longitudinal) > MAX_ACTOR_LONGITUDINAL_RANGE_METRES:
            continue

        actor_width = max(0.1, float(annotation["size"][0]))
        actor_length = max(0.1, float(annotation["size"][1]))
        has_overlap = has_overlap or _actor_overlaps_ego(
            relative,
            actor_width,
            actor_length,
        )

        velocity_ego = _annotation_velocity_in_ego(
            nusc,
            annotation_token,
            inverse_orientation,
        )
        actor_speed = (
            float(np.linalg.norm(velocity_ego[:2]))
            if np.all(np.isfinite(velocity_ego[:2]))
            else float("nan")
        )

        if is_pedestrian:
            if longitudinal < -3.0 or abs(lateral) > MAX_PEDESTRIAN_LATERAL_RANGE_METRES:
                continue

            centre_distance = float(np.linalg.norm(relative[:2]))
            gap = max(
                0.0,
                centre_distance
                - 0.5 * EGO_LENGTH_METRES
                - 0.5 * max(actor_width, actor_length),
            )

            if np.all(np.isfinite(velocity_ego[:2])) and centre_distance > 0.01:
                ego_velocity = np.asarray([ego_speed, 0.0], dtype=np.float64)
                relative_velocity = velocity_ego[:2] - ego_velocity
                range_rate = float(
                    np.dot(relative[:2], relative_velocity) / centre_distance
                )
                closing_speed = -range_rate
            else:
                closing_speed = float("nan")

            _assign_slot_if_closer(
                state,
                "pedestrian",
                gap,
                closing_speed,
                actor_speed,
            )
            continue

        gap = _bumper_gap(longitudinal, actor_length)
        actor_longitudinal_speed = (
            float(velocity_ego[0]) if _finite(velocity_ego[0]) else float("nan")
        )
        if longitudinal >= 0.0:
            closing_speed = (
                ego_speed - actor_longitudinal_speed
                if _finite(actor_longitudinal_speed)
                else float("nan")
            )
        else:
            closing_speed = (
                actor_longitudinal_speed - ego_speed
                if _finite(actor_longitudinal_speed)
                else float("nan")
            )

        if abs(lateral) <= CURRENT_LANE_HALF_WIDTH_METRES:
            if longitudinal >= 0.0:
                _assign_slot_if_closer(
                    state,
                    "front",
                    gap,
                    closing_speed,
                    actor_speed,
                )
            continue

        if lateral > 0.0 and lateral <= ADJACENT_LANE_OUTER_EDGE_METRES:
            slot = "left_front" if longitudinal >= 0.0 else "left_rear"
        elif lateral < 0.0 and lateral >= -ADJACENT_LANE_OUTER_EDGE_METRES:
            slot = "right_front" if longitudinal >= 0.0 else "right_rear"
        else:
            continue

        _assign_slot_if_closer(
            state,
            slot,
            gap,
            closing_speed,
            actor_speed,
        )

    left_front_gap = state["left_front_gap"]
    left_rear_gap = state["left_rear_gap"]
    right_front_gap = state["right_front_gap"]
    right_rear_gap = state["right_rear_gap"]

    state["left_clear"] = float(
        (not _finite(left_front_gap) or left_front_gap >= LANE_CLEAR_FRONT_METRES)
        and (not _finite(left_rear_gap) or left_rear_gap >= LANE_CLEAR_REAR_METRES)
    )
    state["right_clear"] = float(
        (not _finite(right_front_gap) or right_front_gap >= LANE_CLEAR_FRONT_METRES)
        and (not _finite(right_rear_gap) or right_rear_gap >= LANE_CLEAR_REAR_METRES)
    )
    return has_overlap


def _build_ego_frame(
    nusc: NuScenes,
    sample: Mapping[str, Any],
    scene_start_timestamp: int,
    can_pose_messages: Sequence[Mapping[str, Any]],
    can_timestamps: Sequence[int],
    maximum_can_difference_seconds: float,
) -> Optional[EgoFrame]:
    can_pose = _nearest_message(
        can_pose_messages,
        can_timestamps,
        int(sample["timestamp"]),
        maximum_can_difference_seconds,
    )
    if can_pose is None:
        return None

    ego_pose = _ego_pose_for_sample(nusc, sample)
    ego_position = np.asarray(ego_pose["translation"], dtype=np.float64)
    ego_orientation = Quaternion(ego_pose["rotation"])

    can_velocity = np.asarray(can_pose.get("vel", []), dtype=np.float64)
    can_acceleration = np.asarray(can_pose.get("accel", []), dtype=np.float64)
    can_rotation_rate = np.asarray(
        can_pose.get("rotation_rate", []),
        dtype=np.float64,
    )

    speed = (
        float(can_velocity[0])
        if can_velocity.size >= 1
        else float("nan")
    )
    acceleration = (
        float(can_acceleration[0])
        if can_acceleration.size >= 1
        else float("nan")
    )
    yaw_rate = (
        float(can_rotation_rate[2])
        if can_rotation_rate.size >= 3
        else float("nan")
    )

    # Preserve missing values instead of clipping questionable measurements.
    # Linear quantities use SI units; RoadWeave's canonical yaw rate is
    # degrees/second because that is what Unity and the simulated stream use.
    speed = _clip_observation(speed, 0.0, 70.0)
    acceleration = _clip_observation(acceleration, -15.0, 15.0)
    yaw_rate = _clip_observation(
        math.degrees(yaw_rate) if _finite(yaw_rate) else yaw_rate,
        -180.0,
        180.0,
    )

    if not _finite(speed):
        return None

    timestamp_seconds = (
        int(sample["timestamp"]) - scene_start_timestamp
    ) / 1_000_000.0
    state = new_state(
        timestamp_seconds,
        speed,
        acceleration,
        yaw_rate,
    )
    has_overlap = _populate_surrounding_actors(
        nusc,
        sample,
        ego_position,
        ego_orientation,
        speed,
        state,
    )

    return EgoFrame(
        sample=sample,
        timestamp_seconds=timestamp_seconds,
        global_position=ego_position,
        global_orientation=ego_orientation,
        speed=speed,
        acceleration=acceleration,
        yaw_rate=yaw_rate,
        state=state,
        has_overlap=has_overlap,
    )


def _history_for_index(
    frames: Sequence[Optional[EgoFrame]],
    valid_timestamps: Sequence[float],
    index: int,
    history_seconds: float,
    timestamp_tolerance_seconds: float,
) -> Optional[List[Dict[str, float]]]:
    current = frames[index]
    if current is None:
        return None

    start_time = current.timestamp_seconds - history_seconds
    start_index = bisect.bisect_left(valid_timestamps, start_time)
    selected_frames = [
        frame
        for frame in frames[start_index : index + 1]
        if frame is not None
    ]
    if len(selected_frames) < 2:
        return None

    covered_seconds = (
        current.timestamp_seconds - selected_frames[0].timestamp_seconds
    )
    if covered_seconds < history_seconds - timestamp_tolerance_seconds:
        return None

    # Do not silently accept a hole of more than one expected 2 Hz keyframe.
    selected_times = [frame.timestamp_seconds for frame in selected_frames]
    if any(
        later - earlier > 1.0 + timestamp_tolerance_seconds
        for earlier, later in zip(selected_times, selected_times[1:])
    ):
        return None

    return [frame.state for frame in selected_frames]


def _future_index(
    frames: Sequence[Optional[EgoFrame]],
    valid_timestamps: Sequence[float],
    index: int,
    future_seconds: float,
    timestamp_tolerance_seconds: float,
) -> Optional[int]:
    current = frames[index]
    if current is None:
        return None

    target_time = current.timestamp_seconds + future_seconds
    insertion = bisect.bisect_left(valid_timestamps, target_time)
    candidates = []
    if insertion < len(frames):
        candidates.append(insertion)
    if insertion > index + 1:
        candidates.append(insertion - 1)

    candidates = [
        candidate
        for candidate in candidates
        if candidate > index and frames[candidate] is not None
    ]
    if not candidates:
        return None

    future_index = min(
        candidates,
        key=lambda candidate: abs(valid_timestamps[candidate] - target_time),
    )
    difference = abs(valid_timestamps[future_index] - target_time)
    if difference > timestamp_tolerance_seconds:
        return None
    return future_index


def _future_lateral_displacement(
    frames: Sequence[Optional[EgoFrame]],
    index: int,
    future_index: int,
) -> float:
    """Integrate lateral motion over the future window, per frame heading.

    Projecting the total future displacement onto the current heading would
    mistake road curvature for a lane change. Integrating each inter-frame
    displacement onto that frame's own heading removes the curvature signal and
    leaves the actual cross-lane motion, which is what a lane change produces.
    """

    current = frames[index]
    total = 0.0
    previous = current
    for later_index in range(index + 1, future_index + 1):
        frame = frames[later_index]
        if frame is None:
            continue
        displacement = frame.global_position - previous.global_position
        lateral = float(previous.global_orientation.inverse.rotate(displacement)[1])
        total += lateral
        previous = frame
    return total


def _policy_target(
    speed_change: float,
    lateral_displacement: float,
) -> str:
    """Derive a five-class action from future ego motion.

    Only this function sees future frames. Its output is a label string; no
    future measurement enters the feature dictionary.
    """

    if lateral_displacement > 1.4:
        action = DrivingAction.CHANGE_LEFT
    elif lateral_displacement < -1.4:
        action = DrivingAction.CHANGE_RIGHT
    elif speed_change > 1.0:
        action = DrivingAction.ACCELERATE
    elif speed_change < -1.0:
        action = DrivingAction.DECELERATE
    else:
        action = DrivingAction.KEEP
    return ACTION_NAMES[action]


def _minimum_present_value(
    states: Sequence[Mapping[str, float]],
    field: str,
) -> Optional[float]:
    values = [
        float(state[field])
        for state in states
        if field in state and _finite(state[field])
    ]
    return min(values) if values else None


def _risk_level(
    history_states: Sequence[Mapping[str, float]],
    history_frames: Sequence[EgoFrame],
) -> str:
    """Grade one nuScenes window by physical severity.

    nuScenes has no K-Risk driver-risk field, so severity is derived from the
    same physical signals the model is allowed to see: ego acceleration, front
    time-to-collision, pedestrian proximity, and physical overlap. Emitting
    non-LOW rows from nuScenes breaks the "LOW always means nuScenes" shortcut
    that otherwise lets the risk model learn dataset identity instead of risk.
    """

    if any(frame.has_overlap for frame in history_frames):
        return RISK_NAMES[RiskLevel.EXTREME]

    accelerations = [
        abs(float(state["ego_accel"]))
        for state in history_states
        if _finite(state.get("ego_accel"))
    ]
    max_abs_accel = max(accelerations) if accelerations else 0.0

    minimum_front_ttc = _minimum_present_value(history_states, "front_ttc")
    minimum_pedestrian_gap = _minimum_present_value(history_states, "pedestrian_gap")

    if minimum_front_ttc is not None and minimum_front_ttc <= 1.5:
        return RISK_NAMES[RiskLevel.EXTREME]
    if minimum_pedestrian_gap is not None and minimum_pedestrian_gap <= 3.0:
        return RISK_NAMES[RiskLevel.EXTREME]

    if max_abs_accel >= 4.0:
        return RISK_NAMES[RiskLevel.HIGH]
    if minimum_front_ttc is not None and minimum_front_ttc <= 3.0:
        return RISK_NAMES[RiskLevel.HIGH]
    if minimum_pedestrian_gap is not None and minimum_pedestrian_gap <= 8.0:
        return RISK_NAMES[RiskLevel.HIGH]

    if max_abs_accel >= 2.5:
        return RISK_NAMES[RiskLevel.MODERATE]
    if minimum_front_ttc is not None and minimum_front_ttc <= 5.0:
        return RISK_NAMES[RiskLevel.MODERATE]
    if minimum_pedestrian_gap is not None and minimum_pedestrian_gap <= 15.0:
        return RISK_NAMES[RiskLevel.MODERATE]

    # A present actor with unknown TTC is not confidently LOW-risk when it is
    # already close. Keep uncertainty out of the low-risk reference set.
    for state in history_states:
        if (
            state.get("front_present", 0.0) >= 0.5
            and _finite(state.get("front_gap"))
            and float(state["front_gap"]) < 20.0
            and not _finite(state.get("front_ttc"))
        ):
            return RISK_NAMES[RiskLevel.MODERATE]

    return RISK_NAMES[RiskLevel.LOW]


def _output_row(
    scene_name: str,
    timestamp_seconds: float,
    target: str,
    features: Mapping[str, float],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source": "nuscenes",
        "group_id": scene_name,
        "timestamp": float(timestamp_seconds),
        "target": target,
    }
    for column in FEATURE_COLUMNS:
        row[column] = features.get(column, float("nan"))
    return row


def _process_scene(
    nusc: NuScenes,
    can_bus: NuScenesCanBus,
    scene: Mapping[str, Any],
    history_seconds: float,
    future_seconds: float,
    timestamp_tolerance_seconds: float,
    maximum_can_difference_seconds: float,
    statistics: BuildStatistics,
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    scene_name = str(scene["name"])
    try:
        can_pose_messages = can_bus.get_messages(
            scene_name,
            "pose",
            print_warnings=False,
        )
    except (FileNotFoundError, ValueError, KeyError, AssertionError, Exception) as error:
        # NuScenesCanBus raises generic Exception for missing/blacklisted data.
        print("Skipping {}: CAN pose unavailable ({})".format(scene_name, error))
        statistics.scenes_skipped_can += 1
        return None

    if not isinstance(can_pose_messages, list) or not can_pose_messages:
        print("Skipping {}: CAN pose is empty".format(scene_name))
        statistics.scenes_skipped_can += 1
        return None

    can_pose_messages = sorted(
        can_pose_messages,
        key=lambda message: int(message["utime"]),
    )
    can_timestamps = [int(message["utime"]) for message in can_pose_messages]

    samples = _scene_samples(nusc, scene)
    statistics.samples_seen += len(samples)
    if not samples:
        statistics.scenes_failed += 1
        return None

    scene_start_timestamp = int(samples[0]["timestamp"])
    frames: List[Optional[EgoFrame]] = []
    for sample in samples:
        frame = _build_ego_frame(
            nusc,
            sample,
            scene_start_timestamp,
            can_pose_messages,
            can_timestamps,
            maximum_can_difference_seconds,
        )
        if frame is None:
            statistics.samples_missing_can += 1
        frames.append(frame)

    # Sample timestamps remain available even when one CAN match is missing;
    # this keeps list indices aligned with the original keyframes.
    relative_timestamps = [
        (int(sample["timestamp"]) - scene_start_timestamp) / 1_000_000.0
        for sample in samples
    ]

    risk_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []

    for index, current in enumerate(frames):
        if current is None:
            continue

        history_states = _history_for_index(
            frames,
            relative_timestamps,
            index,
            history_seconds,
            timestamp_tolerance_seconds,
        )
        if history_states is None:
            statistics.windows_incomplete += 1
            continue

        history_start = current.timestamp_seconds - history_seconds
        history_frames = [
            frame
            for frame in frames[: index + 1]
            if frame is not None and frame.timestamp_seconds >= history_start
        ]
        features = summarize_history(history_states)

        risk_rows.append(
            _output_row(
                scene_name,
                current.timestamp_seconds,
                _risk_level(history_states, history_frames),
                features,
            )
        )

        future_index = _future_index(
            frames,
            relative_timestamps,
            index,
            future_seconds,
            timestamp_tolerance_seconds,
        )
        if future_index is None:
            statistics.windows_without_future += 1
            continue

        future = frames[future_index]
        if future is None:
            statistics.windows_without_future += 1
            continue

        lateral_displacement = _future_lateral_displacement(
            frames,
            index,
            future_index,
        )

        # Speed change uses a shorter 1 s horizon so longitudinal labels stay
        # sharp; only the lateral (lane-change) signal integrates over the
        # longer future_seconds window.
        speed_index = _future_index(
            frames,
            relative_timestamps,
            index,
            1.0,
            timestamp_tolerance_seconds,
        )
        speed_future = frames[speed_index] if speed_index is not None else None
        if speed_future is None:
            speed_future = future
        speed_change = speed_future.speed - current.speed

        target = _policy_target(speed_change, lateral_displacement)
        policy_rows.append(
            _output_row(
                scene_name,
                current.timestamp_seconds,
                target,
                features,
            )
        )

    statistics.scenes_processed += 1
    statistics.policy_rows += len(policy_rows)
    statistics.risk_rows += len(risk_rows)
    return risk_rows, policy_rows


def _validate_output(
    path: Path,
    allowed_targets: Iterable[str],
    expected_source: str,
) -> Dict[str, Any]:
    parquet_file = pq.ParquetFile(str(path))
    actual_columns = tuple(parquet_file.schema_arrow.names)
    if actual_columns != OUTPUT_COLUMNS:
        raise ValueError(
            "Unexpected schema in {}.\nExpected: {}\nActual: {}"
            .format(path, OUTPUT_COLUMNS, actual_columns)
        )

    allowed_target_set = set(allowed_targets)
    target_counts: Counter = Counter()
    groups = set()
    seen_keys = set()
    row_count = 0

    for batch in parquet_file.iter_batches(
        columns=["source", "group_id", "timestamp", "target"],
        batch_size=8192,
    ):
        data = batch.to_pydict()
        for source, group_id, timestamp, target in zip(
            data["source"],
            data["group_id"],
            data["timestamp"],
            data["target"],
        ):
            row_count += 1
            if source != expected_source:
                raise ValueError(
                    "Unexpected source {!r} in {}".format(source, path)
                )
            if not group_id:
                raise ValueError("Missing group_id in {}".format(path))
            if not _finite(timestamp):
                raise ValueError("Invalid timestamp in {}".format(path))
            if target not in allowed_target_set:
                raise ValueError(
                    "Unexpected target {!r} in {}".format(target, path)
                )
            key = (group_id, float(timestamp))
            if key in seen_keys:
                raise ValueError(
                    "Duplicate group/timestamp {} in {}".format(key, path)
                )
            seen_keys.add(key)
            groups.add(group_id)
            target_counts[target] += 1

    return {
        "path": str(path),
        "rows": row_count,
        "row_groups": parquet_file.num_row_groups,
        "groups": len(groups),
        "targets": dict(sorted(target_counts.items())),
        "features": len(FEATURE_COLUMNS),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert full nuScenes train/validation scenes and CAN pose data "
            "into RoadWeave risk and policy Parquet tables."
        )
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=DEFAULT_DATAROOT,
        help="nuScenes root containing v1.0-trainval/ and can_bus/.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes table version (default: v1.0-trainval).",
    )
    parser.add_argument(
        "--risk-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "risk_nuscenes.parquet",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "policy_nuscenes.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON conversion report path.",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        default=[],
        help="Process only this scene name; repeat to select multiple scenes.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Process at most N selected scenes (use 1 for a smoke test).",
    )
    parser.add_argument("--history-seconds", type=float, default=3.0)
    parser.add_argument("--future-seconds", type=float, default=3.0)
    parser.add_argument(
        "--timestamp-tolerance-seconds",
        type=float,
        default=0.26,
        help="Tolerance for 2 Hz history/future keyframe alignment.",
    )
    parser.add_argument(
        "--max-can-time-difference-seconds",
        type=float,
        default=0.10,
        help="Maximum accepted distance from a keyframe to a CAN pose.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N requested scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output Parquet files if they already exist.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Read the completed outputs and validate schema/keys/targets.",
    )
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_scenes is not None and arguments.max_scenes <= 0:
        raise ValueError("--max-scenes must be greater than zero.")
    if arguments.history_seconds <= 0.0:
        raise ValueError("--history-seconds must be greater than zero.")
    if arguments.future_seconds <= 0.0:
        raise ValueError("--future-seconds must be greater than zero.")
    if arguments.timestamp_tolerance_seconds < 0.0:
        raise ValueError("--timestamp-tolerance-seconds cannot be negative.")
    if arguments.max_can_time_difference_seconds <= 0.0:
        raise ValueError(
            "--max-can-time-difference-seconds must be greater than zero."
        )
    if arguments.risk_output.resolve() == arguments.policy_output.resolve():
        raise ValueError("Risk and policy output paths must be different.")

    metadata_directory = arguments.dataroot / arguments.version
    can_directory = arguments.dataroot / "can_bus"
    if not metadata_directory.is_dir():
        raise FileNotFoundError(
            "nuScenes metadata directory not found: {}".format(metadata_directory)
        )
    if not can_directory.is_dir():
        raise FileNotFoundError(
            "nuScenes CAN directory not found: {}".format(can_directory)
        )


def main() -> None:
    arguments = _parse_arguments()
    _validate_arguments(arguments)

    print("Loading nuScenes metadata from {} ...".format(arguments.dataroot))
    load_started = time.monotonic()
    nusc = NuScenes(
        version=arguments.version,
        dataroot=str(arguments.dataroot),
        verbose=False,
    )
    can_bus = NuScenesCanBus(dataroot=str(arguments.dataroot))
    print(
        "Loaded {} scenes in {:.1f}s.".format(
            len(nusc.scene),
            time.monotonic() - load_started,
        )
    )

    scenes = list(nusc.scene)
    if arguments.scene_name:
        requested_names = set(arguments.scene_name)
        known_names = {str(scene["name"]) for scene in scenes}
        missing_names = sorted(requested_names - known_names)
        if missing_names:
            raise ValueError(
                "Unknown scene name(s): {}".format(", ".join(missing_names))
            )
        scenes = [
            scene for scene in scenes if str(scene["name"]) in requested_names
        ]
    if arguments.max_scenes is not None:
        scenes = scenes[: arguments.max_scenes]

    statistics = BuildStatistics(scenes_requested=len(scenes))
    risk_writer = IncrementalParquetWriter(
        arguments.risk_output,
        arguments.overwrite,
    )
    policy_writer = IncrementalParquetWriter(
        arguments.policy_output,
        arguments.overwrite,
    )

    conversion_started = time.monotonic()
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            scene_name = str(scene["name"])
            try:
                result = _process_scene(
                    nusc,
                    can_bus,
                    scene,
                    arguments.history_seconds,
                    arguments.future_seconds,
                    arguments.timestamp_tolerance_seconds,
                    arguments.max_can_time_difference_seconds,
                    statistics,
                )
            except Exception as error:
                statistics.scenes_failed += 1
                print("Failed {}: {}".format(scene_name, error))
                continue

            if result is not None:
                risk_rows, policy_rows = result
                risk_writer.write_rows(risk_rows)
                policy_writer.write_rows(policy_rows)

            if (
                scene_index == 1
                or scene_index == len(scenes)
                or scene_index % max(1, arguments.log_every) == 0
            ):
                print(
                    "[{}/{}] {} | risk rows={} policy rows={}".format(
                        scene_index,
                        len(scenes),
                        scene_name,
                        risk_writer.row_count,
                        policy_writer.row_count,
                    )
                )
    finally:
        risk_writer.close()
        policy_writer.close()

    elapsed = time.monotonic() - conversion_started
    print("\nConversion complete in {:.1f}s.".format(elapsed))
    print("Risk output: {} ({} rows)".format(arguments.risk_output, risk_writer.row_count))
    print(
        "Policy output: {} ({} rows)".format(
            arguments.policy_output,
            policy_writer.row_count,
        )
    )
    print("Statistics: {}".format(statistics))

    risk_validation: Optional[Dict[str, Any]] = None
    policy_validation: Optional[Dict[str, Any]] = None
    if arguments.validate:
        risk_validation = _validate_output(
            arguments.risk_output,
            list(RISK_NAMES.values()),
            "nuscenes",
        )
        policy_validation = _validate_output(
            arguments.policy_output,
            ACTION_NAMES.values(),
            "nuscenes",
        )
        print("\nValidation passed.")
        print("Risk: {}".format(risk_validation))
        print("Policy: {}".format(policy_validation))

    report = {
        "format": "roadweave.nuscenes-conversion-report/1.0",
        "dataroot": str(arguments.dataroot.expanduser().resolve()),
        "version": arguments.version,
        "parameters": {
            "history_seconds": arguments.history_seconds,
            "future_seconds": arguments.future_seconds,
            "timestamp_tolerance_seconds": arguments.timestamp_tolerance_seconds,
            "max_can_time_difference_seconds": (
                arguments.max_can_time_difference_seconds
            ),
        },
        "elapsed_seconds": elapsed,
        "statistics": asdict(statistics),
        "outputs": {
            "risk": str(arguments.risk_output.expanduser().resolve()),
            "policy": str(arguments.policy_output.expanduser().resolve()),
        },
        "validation": {
            "risk": risk_validation,
            "policy": policy_validation,
        },
    }
    report_path = arguments.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: {}".format(report_path))


if __name__ == "__main__":
    main()
