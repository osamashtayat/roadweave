#!/usr/bin/env python3
"""Prepare one nuScenes scene as a compact JSON replay for RoadWeave.

This is an offline preparation tool. Unity reads only the generated replay JSON;
it never needs direct access to the full nuScenes download.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


DEFAULT_DATASET_ROOT = Path("/Users/asus/Downloads/v1.0-trainval")
VERSION = "v1.0-trainval"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_with_header_comments(path: Path) -> Any:
    """Load a CAN file even if explanatory // lines were added at its top."""
    with path.open("r", encoding="utf-8") as file:
        clean_text = "\n".join(
            line for line in file if not line.lstrip().startswith("//")
        )
    return json.loads(clean_text)


def stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Read a large top-level JSON array without loading the entire file into RAM."""
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    reached_eof = False

    with path.open("r", encoding="utf-8") as file:
        while True:
            if not reached_eof:
                chunk = file.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True

            cursor = 0

            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1

                if not array_started:
                    if cursor >= len(buffer):
                        break
                    if buffer[cursor] != "[":
                        raise ValueError(f"Expected a JSON array in {path}")
                    array_started = True
                    cursor += 1
                    continue

                while cursor < len(buffer) and (
                    buffer[cursor].isspace() or buffer[cursor] == ","
                ):
                    cursor += 1

                if cursor >= len(buffer):
                    break

                if buffer[cursor] == "]":
                    return

                try:
                    item, item_end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break

                yield item
                cursor = item_end

            buffer = buffer[cursor:]

            if reached_eof:
                if buffer.strip():
                    raise ValueError(f"Incomplete or invalid JSON in {path}")
                return


def quaternion_yaw_degrees(rotation: List[float]) -> float:
    """Return world Z-axis yaw for a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = rotation
    yaw_radians = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return math.degrees(yaw_radians)


def wrapped_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def to_unity_position(
    world_position: List[float],
    origin_position: List[float],
    origin_yaw_degrees: float,
) -> Dict[str, float]:
    """Convert nuScenes global coordinates to a Unity-local right/up/forward frame."""
    delta_x = world_position[0] - origin_position[0]
    delta_y = world_position[1] - origin_position[1]
    delta_z = world_position[2] - origin_position[2]

    yaw = math.radians(origin_yaw_degrees)
    local_forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
    local_left = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y

    return {
        "x": round(-local_left, 5),
        "y": round(delta_z, 5),
        "z": round(local_forward, 5),
    }


def relative_unity_yaw(world_rotation: List[float], origin_yaw_degrees: float) -> float:
    relative_yaw = wrapped_degrees(
        quaternion_yaw_degrees(world_rotation) - origin_yaw_degrees
    )
    return round(-relative_yaw, 4)


def time_seconds(timestamp_microseconds: int, start_microseconds: int) -> float:
    return round((timestamp_microseconds - start_microseconds) / 1_000_000.0, 5)


def scene_samples(metadata_root: Path, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples_by_token = {
        sample["token"]: sample for sample in load_json(metadata_root / "sample.json")
    }

    ordered_samples: List[Dict[str, Any]] = []
    token = scene["first_sample_token"]

    while token:
        sample = samples_by_token[token]
        ordered_samples.append(sample)
        token = sample["next"]

    return ordered_samples


def read_can_messages(can_root: Path, scene_name: str, message_name: str) -> List[Dict[str, Any]]:
    path = can_root / f"{scene_name}_{message_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing CAN file: {path}")
    return load_json_with_header_comments(path)


def filter_time_range(
    messages: List[Dict[str, Any]],
    start_timestamp: int,
    end_timestamp: int,
) -> List[Dict[str, Any]]:
    return [
        message
        for message in messages
        if start_timestamp <= message["utime"] <= end_timestamp
    ]


def actor_prefab_type(category: str) -> str:
    if category.startswith("human.pedestrian"):
        return "pedestrian"
    if category == "movable_object.trafficcone":
        return "trafficCone"
    if category.startswith("vehicle.truck"):
        return "truck"
    if category.startswith("vehicle.bus"):
        return "bus"
    if category.startswith("vehicle.construction"):
        return "constructionVehicle"
    if category.startswith("vehicle"):
        return "car"
    return "genericObject"


def prepare_replay(dataset_root: Path, scene_name: str) -> Dict[str, Any]:
    metadata_root = dataset_root / VERSION
    can_root = dataset_root / "can_bus"

    if not metadata_root.is_dir():
        raise FileNotFoundError(f"nuScenes metadata folder was not found: {metadata_root}")
    if not can_root.is_dir():
        raise FileNotFoundError(f"nuScenes CAN folder was not found: {can_root}")

    scenes = load_json(metadata_root / "scene.json")
    scene = next((item for item in scenes if item["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found in {metadata_root / 'scene.json'}")

    ordered_samples = scene_samples(metadata_root, scene)
    start_timestamp = ordered_samples[0]["timestamp"]
    end_timestamp = ordered_samples[-1]["timestamp"]
    duration_seconds = (end_timestamp - start_timestamp) / 1_000_000.0
    sample_time_by_token = {
        sample["token"]: time_seconds(sample["timestamp"], start_timestamp)
        for sample in ordered_samples
    }

    print(f"Preparing {scene_name}: {len(ordered_samples)} samples, {duration_seconds:.1f} seconds")
    print("Reading ego vehicle and CAN data...")

    all_pose_messages = read_can_messages(can_root, scene_name, "pose")
    all_vehicle_messages = read_can_messages(can_root, scene_name, "vehicle_monitor")
    all_wheel_messages = read_can_messages(can_root, scene_name, "zoe_veh_info")

    pose_messages = filter_time_range(all_pose_messages, start_timestamp, end_timestamp)
    vehicle_messages = filter_time_range(all_vehicle_messages, start_timestamp, end_timestamp)
    wheel_messages = filter_time_range(all_wheel_messages, start_timestamp, end_timestamp)

    if not pose_messages:
        raise ValueError(f"No pose messages overlap {scene_name}")

    origin_position = pose_messages[0]["pos"]
    origin_yaw_degrees = quaternion_yaw_degrees(pose_messages[0]["orientation"])

    ego_frames = []
    for message in pose_messages:
        velocity = message.get("vel", [0.0, 0.0, 0.0])
        speed_mps = math.sqrt(sum(component * component for component in velocity))
        acceleration = message.get("accel", [0.0, 0.0, 0.0])
        ego_frames.append(
            {
                "time": time_seconds(message["utime"], start_timestamp),
                "position": to_unity_position(
                    message["pos"], origin_position, origin_yaw_degrees
                ),
                "yawDegrees": relative_unity_yaw(
                    message["orientation"], origin_yaw_degrees
                ),
                "speedMetersPerSecond": round(speed_mps, 4),
                "longitudinalAcceleration": round(float(acceleration[0]), 4),
            }
        )

    vehicle_frames = []
    for message in vehicle_messages:
        vehicle_frames.append(
            {
                "time": time_seconds(message["utime"], start_timestamp),
                "speedKilometersPerHour": float(message.get("vehicle_speed", 0.0)),
                "batteryPercent": float(message.get("battery_level", 0.0)),
                "availableDistanceKilometers": float(message.get("available_distance", 0.0)),
                "gearPosition": int(message.get("gear_position", 0)),
                "throttlePercent": float(message.get("throttle", 0.0)),
                "brake": float(message.get("brake", 0.0)),
                "brakeSwitch": int(message.get("brake_switch", 0)),
                "steeringDegrees": float(message.get("steering", 0.0)),
                "steeringSpeed": float(message.get("steering_speed", 0.0)),
                "yawRate": float(message.get("yaw_rate", 0.0)),
                "leftSignal": int(message.get("left_signal", 0)),
                "rightSignal": int(message.get("right_signal", 0)),
            }
        )

    wheel_frames = []
    for message in wheel_messages:
        wheel_frames.append(
            {
                "time": time_seconds(message["utime"], start_timestamp),
                "frontLeftRpm": float(message.get("FL_wheel_speed", 0.0)),
                "frontRightRpm": float(message.get("FR_wheel_speed", 0.0)),
                "rearLeftRpm": float(message.get("RL_wheel_speed", 0.0)),
                "rearRightRpm": float(message.get("RR_wheel_speed", 0.0)),
            }
        )

    print("Scanning annotations for vehicles, pedestrians, cones, and other actors...")

    instances = {
        item["token"]: item for item in load_json(metadata_root / "instance.json")
    }
    category_names = {
        item["token"]: item["name"] for item in load_json(metadata_root / "category.json")
    }
    attribute_names = {
        item["token"]: item["name"] for item in load_json(metadata_root / "attribute.json")
    }

    actor_annotations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned = 0

    for annotation in stream_json_array(metadata_root / "sample_annotation.json"):
        scanned += 1
        if scanned % 250_000 == 0:
            print(f"  checked {scanned:,} annotations...")

        sample_token = annotation["sample_token"]
        if sample_token in sample_time_by_token:
            actor_annotations[annotation["instance_token"]].append(annotation)

    actors = []
    for instance_token, annotations in actor_annotations.items():
        annotations.sort(key=lambda item: sample_time_by_token[item["sample_token"]])
        instance = instances[instance_token]
        category = category_names[instance["category_token"]]
        first_size = annotations[0]["size"]

        actor_frames = []
        for annotation in annotations:
            names = [
                attribute_names[token]
                for token in annotation.get("attribute_tokens", [])
                if token in attribute_names
            ]
            actor_frames.append(
                {
                    "time": sample_time_by_token[annotation["sample_token"]],
                    "position": to_unity_position(
                        annotation["translation"], origin_position, origin_yaw_degrees
                    ),
                    "yawDegrees": relative_unity_yaw(
                        annotation["rotation"], origin_yaw_degrees
                    ),
                    "attribute": ", ".join(names),
                    "visibility": annotation.get("visibility_token", ""),
                }
            )

        actors.append(
            {
                "id": instance_token,
                "category": category,
                "prefabType": actor_prefab_type(category),
                "size": {
                    "x": round(float(first_size[0]), 4),
                    "y": round(float(first_size[2]), 4),
                    "z": round(float(first_size[1]), 4),
                },
                "frames": actor_frames,
            }
        )

    actors.sort(key=lambda actor: actor["id"])

    return {
        "schemaVersion": 1,
        "source": "nuScenes",
        "sceneId": scene_name,
        "description": scene.get("description", ""),
        "durationSeconds": round(duration_seconds, 5),
        "sampleIntervalSeconds": 0.5,
        "coordinateSystem": "Unity local: x=right, y=up, z=forward; origin=first CAN pose",
        "hasTemperatureData": False,
        "egoFrames": ego_frames,
        "vehicleFrames": vehicle_frames,
        "wheelFrames": wheel_frames,
        "actors": actors,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one compact RoadWeave replay JSON from nuScenes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Folder containing {VERSION} and can_bus (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--scene",
        default="scene-0001",
        help="nuScenes scene name (default: scene-0001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. By default it is saved in Unity StreamingAssets/Replays.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    output_path: Optional[Path] = arguments.output

    if output_path is None:
        output_path = (
            project_root
            / "Assets"
            / "StreamingAssets"
            / "Replays"
            / f"{arguments.scene}-replay.json"
        )

    replay = prepare_replay(arguments.dataset_root.resolve(), arguments.scene)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(replay, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print()
    print(f"Replay created: {output_path}")
    print(f"  ego frames:     {len(replay['egoFrames']):,}")
    print(f"  vehicle frames: {len(replay['vehicleFrames']):,}")
    print(f"  wheel frames:   {len(replay['wheelFrames']):,}")
    print(f"  actors:         {len(replay['actors']):,}")
    print("Step 3 is complete. Unity can now load this one replay file.")


if __name__ == "__main__":
    main()
