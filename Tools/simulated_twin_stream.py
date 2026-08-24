#!/usr/bin/env python3

"""RoadWeave procedural real-time TwinSnapshot simulator.

The scenario planner creates a different sequence of simple road encounters on
every reset. The autonomous controller does not read that plan directly. It
receives observations from a small simulated sensor suite and decides whether
to cruise, follow, brake, change lanes, pass, or return to the right lane.

Data:     Python -> UDP 5055 -> Unity
Controls: Unity  -> UDP 5056 -> Python

Only Python standard-library modules are required.
"""

import argparse
import json
import math
import random
import secrets
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


VALID = 1
FRESH = 1

ACTOR_CAR = 1
ACTOR_TRUCK = 2
ACTOR_PEDESTRIAN = 4

MOTION_STOPPED = 1
MOTION_SLOW = 2
MOTION_MOVING = 3

RIGHT_LANE_X = 0.0
LEFT_LANE_X = -5.5
RIGHT_SIDEWALK_X = 7.2
LEFT_SIDEWALK_X = -7.2

EGO_WIDTH_METERS = 1.9
EGO_LENGTH_METERS = 4.6
CRUISE_SPEED_MPS = 13.9


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def move_towards(current: float, target: float, maximum_delta: float) -> float:
    if abs(target - current) <= maximum_delta:
        return target
    return current + math.copysign(maximum_delta, target - current)


@dataclass
class SimulatedActor:
    actor_id: str
    kind: str
    lane_x: float
    z: float
    speed_mps: float
    width: float
    height: float
    length: float
    source_class: str
    active: bool = False
    cleared: bool = False
    x: float = 0.0
    lateral_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        self.x = self.lane_x

    @property
    def is_pedestrian(self) -> bool:
        return self.kind == "pedestrian"

    @property
    def is_stopped(self) -> bool:
        return self.speed_mps <= 0.05 and not self.is_pedestrian

    @property
    def semantic_class(self) -> int:
        if self.kind == "truck":
            return ACTOR_TRUCK
        if self.kind == "pedestrian":
            return ACTOR_PEDESTRIAN
        return ACTOR_CAR

    @property
    def motion_state(self) -> int:
        if self.is_pedestrian:
            return MOTION_MOVING if self.active and not self.cleared else MOTION_STOPPED
        if self.is_stopped:
            return MOTION_STOPPED
        if self.speed_mps < 8.0:
            return MOTION_SLOW
        return MOTION_MOVING


@dataclass
class SensorHit:
    actor: SimulatedActor
    longitudinal_gap: float
    relative_speed_mps: float
    time_to_collision: float


@dataclass
class SensorFrame:
    current_front: Optional[SensorHit] = None
    left_front: Optional[SensorHit] = None
    left_rear: Optional[SensorHit] = None
    right_front: Optional[SensorHit] = None
    right_rear: Optional[SensorHit] = None
    pedestrian_hazard: Optional[SensorHit] = None


class ProceduralScenarioPlanner:
    """Builds random encounters without embedding driving decisions."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.random = random.Random(seed)

    def build(self, encounter_count: int) -> List[SimulatedActor]:
        encounter_count = max(4, encounter_count)
        kinds = ["slow_car", "stopped_car", "truck", "pedestrian"]
        while len(kinds) < encounter_count:
            kinds.append(
                self.random.choices(
                    ["slow_car", "stopped_car", "truck", "pedestrian"],
                    weights=[0.34, 0.25, 0.23, 0.18],
                    k=1,
                )[0]
            )
        self.random.shuffle(kinds)

        actors: List[SimulatedActor] = []
        event_z = self.random.uniform(58.0, 75.0)

        for index, kind in enumerate(kinds):
            actor = self._primary_actor(index, kind, event_z)
            actors.append(actor)

            # Some encounters include traffic in the other lane. The sensor
            # suite must notice it and delay or adapt the overtake.
            if kind != "pedestrian" and actor.lane_x == RIGHT_LANE_X:
                if self.random.random() < 0.38:
                    actors.append(self._left_lane_companion(index, event_z))

            event_z += self.random.uniform(68.0, 108.0)

        actors.sort(key=lambda candidate: candidate.z)
        return actors

    def _primary_actor(self, index: int, kind: str, z: float) -> SimulatedActor:
        if kind == "pedestrian":
            return SimulatedActor(
                actor_id=f"sim-pedestrian-{index:02d}",
                kind="pedestrian",
                lane_x=RIGHT_SIDEWALK_X,
                z=z,
                speed_mps=self.random.uniform(1.15, 1.65),
                width=0.65,
                height=1.78,
                length=0.65,
                source_class="human.pedestrian.adult",
            )

        lane_x = RIGHT_LANE_X if self.random.random() < 0.82 else LEFT_LANE_X
        if kind == "truck":
            return SimulatedActor(
                actor_id=f"sim-truck-{index:02d}",
                kind="truck",
                lane_x=lane_x,
                z=z,
                speed_mps=self.random.uniform(4.5, 7.0),
                width=2.5,
                height=3.2,
                length=8.0,
                source_class="vehicle.truck",
            )

        stopped = kind == "stopped_car"
        return SimulatedActor(
            actor_id=f"sim-{'stopped' if stopped else 'slow'}-car-{index:02d}",
            kind=kind,
            lane_x=lane_x,
            z=z,
            speed_mps=0.0 if stopped else self.random.uniform(3.8, 6.5),
            width=1.85,
            height=1.5,
            length=4.35,
            source_class="vehicle.car",
        )

    def _left_lane_companion(self, index: int, event_z: float) -> SimulatedActor:
        return SimulatedActor(
            actor_id=f"sim-left-traffic-{index:02d}",
            kind="traffic_car",
            lane_x=LEFT_LANE_X,
            z=event_z + self.random.uniform(-18.0, 28.0),
            speed_mps=self.random.uniform(8.0, 12.0),
            width=1.85,
            height=1.5,
            length=4.4,
            source_class="vehicle.car",
        )


class SimulatedSensorSuite:
    """Produces local detections; it has no access to the scenario plan."""

    SENSOR_RANGE_METERS = 75.0
    LANE_TOLERANCE_METERS = 1.65

    def observe(
        self,
        ego_x: float,
        ego_z: float,
        ego_speed_mps: float,
        actors: Iterable[SimulatedActor],
    ) -> SensorFrame:
        frame = SensorFrame()
        hits_by_lane: Dict[str, List[SensorHit]] = {"left": [], "right": []}

        for actor in actors:
            if not actor.active or actor.cleared:
                continue

            center_delta = actor.z - ego_z
            if abs(center_delta) > self.SENSOR_RANGE_METERS:
                continue

            if actor.is_pedestrian:
                crossing_zone = LEFT_SIDEWALK_X - 0.5 <= actor.x <= RIGHT_SIDEWALK_X + 0.5
                if crossing_zone and -3.0 <= center_delta <= 42.0:
                    gap = center_delta - EGO_LENGTH_METERS * 0.5
                    frame.pedestrian_hazard = self._hit(actor, gap, ego_speed_mps)
                continue

            lane_name = self._lane_name(actor.x)
            if lane_name is None:
                continue
            gap = self._bumper_gap(ego_z, actor)
            hits_by_lane[lane_name].append(self._hit(actor, gap, ego_speed_mps))

        frame.left_front, frame.left_rear = self._nearest(hits_by_lane["left"])
        frame.right_front, frame.right_rear = self._nearest(hits_by_lane["right"])
        current_lane = "left" if abs(ego_x - LEFT_LANE_X) < abs(ego_x - RIGHT_LANE_X) else "right"
        frame.current_front = frame.left_front if current_lane == "left" else frame.right_front
        return frame

    def _lane_name(self, x: float) -> Optional[str]:
        if abs(x - LEFT_LANE_X) <= self.LANE_TOLERANCE_METERS:
            return "left"
        if abs(x - RIGHT_LANE_X) <= self.LANE_TOLERANCE_METERS:
            return "right"
        return None

    @staticmethod
    def _bumper_gap(ego_z: float, actor: SimulatedActor) -> float:
        return actor.z - ego_z - (EGO_LENGTH_METERS + actor.length) * 0.5

    @staticmethod
    def _hit(actor: SimulatedActor, gap: float, ego_speed_mps: float) -> SensorHit:
        relative_speed = ego_speed_mps - actor.speed_mps
        time_to_collision = gap / relative_speed if gap > 0.0 and relative_speed > 0.05 else math.inf
        return SensorHit(actor, gap, relative_speed, time_to_collision)

    @staticmethod
    def _nearest(hits: List[SensorHit]) -> Tuple[Optional[SensorHit], Optional[SensorHit]]:
        front = [hit for hit in hits if hit.longitudinal_gap >= 0.0]
        rear = [hit for hit in hits if hit.longitudinal_gap < 0.0]
        nearest_front = min(front, key=lambda hit: hit.longitudinal_gap) if front else None
        nearest_rear = max(rear, key=lambda hit: hit.longitudinal_gap) if rear else None
        return nearest_front, nearest_rear


class AutonomousDrivingController:
    CRUISE = "CRUISE"
    FOLLOWING = "FOLLOWING"
    BRAKING_FOR_PEDESTRIAN = "BRAKING_FOR_PEDESTRIAN"
    CHANGE_LEFT = "CHANGE_LEFT"
    PASSING = "PASSING"
    CHANGE_RIGHT = "CHANGE_RIGHT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    def __init__(self) -> None:
        self.behavior = self.CRUISE
        self.target_lane_x = RIGHT_LANE_X
        self.overtake_actor_id: Optional[str] = None

    def decide(
        self,
        ego_x: float,
        ego_z: float,
        ego_speed_mps: float,
        sensors: SensorFrame,
        actors_by_id: Dict[str, SimulatedActor],
    ) -> Tuple[float, float, str]:
        pedestrian = sensors.pedestrian_hazard
        if pedestrian is not None:
            stop_gap = pedestrian.longitudinal_gap - 5.0
            if stop_gap <= 1.5:
                target_speed = 0.0
                self.behavior = self.EMERGENCY_STOP
            else:
                target_speed = min(CRUISE_SPEED_MPS, max(0.0, stop_gap * 0.42))
                self.behavior = self.BRAKING_FOR_PEDESTRIAN
            return target_speed, self.target_lane_x, self.behavior

        if self.behavior in (self.CHANGE_LEFT, self.PASSING, self.CHANGE_RIGHT):
            return self._continue_overtake(
                ego_x, ego_z, ego_speed_mps, sensors, actors_by_id
            )

        current_front = sensors.current_front
        if current_front is not None and current_front.longitudinal_gap < 50.0:
            slower = current_front.actor.speed_mps < CRUISE_SPEED_MPS - 1.0
            if current_front.longitudinal_gap <= 2.0 or current_front.time_to_collision < 1.1:
                # Braking has priority at speed. Once safely stopped, reassess
                # adjacent lanes so an emergency stop does not become a
                # permanent deadlock behind a stationary obstacle.
                on_right = abs(ego_x - RIGHT_LANE_X) <= 1.8
                on_left = abs(ego_x - LEFT_LANE_X) <= 1.8
                if ego_speed_mps <= 0.25 and on_right and self._lane_clear(
                    sensors.left_front, sensors.left_rear
                ):
                    self.behavior = self.CHANGE_LEFT
                    self.target_lane_x = LEFT_LANE_X
                    self.overtake_actor_id = current_front.actor.actor_id
                    return 0.0, self.target_lane_x, self.behavior
                if ego_speed_mps <= 0.25 and on_left and self._lane_clear(
                    sensors.right_front, sensors.right_rear
                ):
                    self.behavior = self.CHANGE_RIGHT
                    self.target_lane_x = RIGHT_LANE_X
                    self.overtake_actor_id = current_front.actor.actor_id
                    return 0.0, self.target_lane_x, self.behavior
                self.behavior = self.EMERGENCY_STOP
                return 0.0, self.target_lane_x, self.behavior

            if slower:
                on_right = abs(ego_x - RIGHT_LANE_X) <= 1.8
                on_left = abs(ego_x - LEFT_LANE_X) <= 1.8
                if on_right and self._lane_clear(sensors.left_front, sensors.left_rear):
                    self.behavior = self.CHANGE_LEFT
                    self.target_lane_x = LEFT_LANE_X
                    self.overtake_actor_id = current_front.actor.actor_id
                    return self._safe_change_speed(current_front, ego_speed_mps), self.target_lane_x, self.behavior
                if on_left and self._lane_clear(sensors.right_front, sensors.right_rear):
                    self.behavior = self.CHANGE_RIGHT
                    self.target_lane_x = RIGHT_LANE_X
                    self.overtake_actor_id = current_front.actor.actor_id
                    return self._safe_change_speed(current_front, ego_speed_mps), self.target_lane_x, self.behavior
                self.behavior = self.FOLLOWING
                return self._following_speed(current_front, ego_speed_mps), self.target_lane_x, self.behavior

        self.behavior = self.CRUISE
        self.target_lane_x = LEFT_LANE_X if abs(ego_x - LEFT_LANE_X) < 0.4 else RIGHT_LANE_X
        return CRUISE_SPEED_MPS, self.target_lane_x, self.behavior

    def _continue_overtake(
        self,
        ego_x: float,
        ego_z: float,
        ego_speed_mps: float,
        sensors: SensorFrame,
        actors_by_id: Dict[str, SimulatedActor],
    ) -> Tuple[float, float, str]:
        overtaken = actors_by_id.get(self.overtake_actor_id or "")

        if self.behavior == self.CHANGE_LEFT:
            self.target_lane_x = LEFT_LANE_X
            left_front = sensors.left_front
            left_lane_blocked = (
                left_front is not None
                and left_front.longitudinal_gap < 30.0
                and left_front.actor.speed_mps < CRUISE_SPEED_MPS - 1.0
            )
            if left_lane_blocked and self._lane_clear(
                sensors.right_front, sensors.right_rear, 20.0, 12.0
            ):
                self.behavior = self.CHANGE_RIGHT
                self.target_lane_x = RIGHT_LANE_X
                self.overtake_actor_id = None
                return self._safe_change_speed(left_front, ego_speed_mps), self.target_lane_x, self.behavior
            if abs(ego_x - LEFT_LANE_X) <= 0.18:
                self.behavior = self.PASSING
            front = sensors.right_front if abs(ego_x - RIGHT_LANE_X) < 2.0 else sensors.left_front
            target = CRUISE_SPEED_MPS
            if front is not None and front.longitudinal_gap < 16.0:
                target = self._following_speed(front, ego_speed_mps)
            return target, self.target_lane_x, self.behavior

        if self.behavior == self.PASSING:
            self.target_lane_x = LEFT_LANE_X
            left_front = sensors.left_front
            left_lane_blocked = (
                left_front is not None
                and left_front.longitudinal_gap < 30.0
                and left_front.actor.speed_mps < CRUISE_SPEED_MPS - 1.0
            )
            if left_lane_blocked and self._lane_clear(
                sensors.right_front, sensors.right_rear, 20.0, 12.0
            ):
                self.behavior = self.CHANGE_RIGHT
                self.target_lane_x = RIGHT_LANE_X
                self.overtake_actor_id = None
                return self._safe_change_speed(left_front, ego_speed_mps), self.target_lane_x, self.behavior
            passed = overtaken is None or ego_z > overtaken.z + overtaken.length * 0.5 + 10.0
            if passed and self._lane_clear(sensors.right_front, sensors.right_rear, 22.0, 14.0):
                self.behavior = self.CHANGE_RIGHT
                self.target_lane_x = RIGHT_LANE_X
            else:
                if left_front is not None and left_front.longitudinal_gap < 18.0:
                    return self._following_speed(left_front, ego_speed_mps), self.target_lane_x, self.behavior
            return CRUISE_SPEED_MPS, self.target_lane_x, self.behavior

        self.target_lane_x = RIGHT_LANE_X
        front = sensors.right_front
        target = CRUISE_SPEED_MPS
        if front is not None and front.longitudinal_gap < 16.0:
            target = self._following_speed(front, ego_speed_mps)
        if abs(ego_x - RIGHT_LANE_X) <= 0.18:
            self.behavior = self.CRUISE
            self.overtake_actor_id = None
        return target, self.target_lane_x, self.behavior

    @staticmethod
    def _lane_clear(
        front: Optional[SensorHit],
        rear: Optional[SensorHit],
        front_clearance: float = 34.0,
        rear_clearance: float = 15.0,
    ) -> bool:
        return (
            (front is None or front.longitudinal_gap > front_clearance)
            and (rear is None or abs(rear.longitudinal_gap) > rear_clearance)
        )

    @staticmethod
    def _following_speed(hit: SensorHit, ego_speed_mps: float) -> float:
        safe_gap = 7.0 + ego_speed_mps * 1.45
        if hit.longitudinal_gap <= 3.0:
            return 0.0
        if hit.time_to_collision < 1.8:
            return max(0.0, min(hit.actor.speed_mps, ego_speed_mps - 3.0))
        gap_error = hit.longitudinal_gap - safe_gap
        return clamp(hit.actor.speed_mps + gap_error * 0.32, 0.0, CRUISE_SPEED_MPS)

    def _safe_change_speed(self, hit: SensorHit, ego_speed_mps: float) -> float:
        if hit.longitudinal_gap < 13.0:
            return self._following_speed(hit, ego_speed_mps)
        return CRUISE_SPEED_MPS


class SimulatedWorld:
    def __init__(
        self,
        include_actors: bool,
        configured_seed: Optional[int],
        encounter_count: int,
        announce: bool = True,
    ) -> None:
        self.include_actors = include_actors
        self.configured_seed = configured_seed
        self.encounter_count = max(4, encounter_count)
        self.announce = announce
        self.reset_count = 0
        self.sensors = SimulatedSensorSuite()
        self.controller = AutonomousDrivingController()
        self.reset()

    def reset(self) -> None:
        self.scenario_seed = (
            secrets.randbits(32)
            if self.configured_seed is None
            else self.configured_seed + self.reset_count
        )
        self.reset_count += 1
        self.simulation_time = 0.0
        self.ego_x = RIGHT_LANE_X
        self.ego_y = 0.0
        self.ego_z = 0.0
        self.ego_yaw_degrees = 0.0
        self.ego_yaw_rate = 0.0
        self.speed_mps = 0.0
        self.acceleration_mps2 = 0.0
        self.lateral_speed_mps = 0.0
        self.battery_percent = 92.0
        self.controller = AutonomousDrivingController()
        self.latest_sensors = SensorFrame()
        self.minimum_observed_clearance = math.inf
        self.actors = (
            ProceduralScenarioPlanner(self.scenario_seed).build(self.encounter_count)
            if self.include_actors
            else []
        )
        self.actors_by_id = {actor.actor_id: actor for actor in self.actors}
        if self.announce:
            self._print_plan()

    def update(self, delta_seconds: float) -> None:
        delta_seconds = clamp(delta_seconds, 0.0, 0.1)
        if delta_seconds <= 0.0:
            return
        self.simulation_time += delta_seconds
        self._update_actors(delta_seconds)
        self.latest_sensors = self.sensors.observe(
            self.ego_x, self.ego_z, self.speed_mps, self.actors
        )
        target_speed, target_lane_x, _ = self.controller.decide(
            self.ego_x,
            self.ego_z,
            self.speed_mps,
            self.latest_sensors,
            self.actors_by_id,
        )
        self._update_ego(delta_seconds, target_speed, target_lane_x)
        self._enforce_non_penetration()
        self.battery_percent = max(0.0, self.battery_percent - 0.0003 * delta_seconds)

    def _update_actors(self, delta_seconds: float) -> None:
        for actor in self.actors:
            if actor.cleared:
                continue
            gap = actor.z - self.ego_z
            if not actor.active and gap <= 135.0:
                actor.active = True
            if not actor.active:
                continue

            if actor.is_pedestrian:
                if gap <= 38.0:
                    actor.lateral_speed_mps = -actor.speed_mps
                    actor.x += actor.lateral_speed_mps * delta_seconds
                    if actor.x <= LEFT_SIDEWALK_X:
                        actor.x = LEFT_SIDEWALK_X
                        actor.lateral_speed_mps = 0.0
                        actor.cleared = True
                continue

            actor.z += actor.speed_mps * delta_seconds
            if actor.z < self.ego_z - 55.0:
                actor.cleared = True

    def _update_ego(self, delta_seconds: float, target_speed: float, target_lane_x: float) -> None:
        previous_speed = self.speed_mps
        acceleration_limit = 2.0 if target_speed >= self.speed_mps else 4.8
        self.speed_mps = move_towards(
            self.speed_mps,
            max(0.0, target_speed),
            acceleration_limit * delta_seconds,
        )
        self.acceleration_mps2 = (self.speed_mps - previous_speed) / delta_seconds

        lane_error = target_lane_x - self.ego_x
        desired_lateral_speed = clamp(lane_error * 1.15, -2.15, 2.15)
        self.lateral_speed_mps = move_towards(
            self.lateral_speed_mps,
            desired_lateral_speed,
            2.8 * delta_seconds,
        )
        if abs(lane_error) < 0.035:
            self.ego_x = target_lane_x
            self.lateral_speed_mps = 0.0
        else:
            self.ego_x += self.lateral_speed_mps * delta_seconds

        self.ego_z += self.speed_mps * delta_seconds
        desired_yaw = math.degrees(
            math.atan2(self.lateral_speed_mps, max(self.speed_mps, 2.0))
        )
        previous_yaw = self.ego_yaw_degrees
        self.ego_yaw_degrees = move_towards(
            self.ego_yaw_degrees,
            desired_yaw,
            45.0 * delta_seconds,
        )
        self.ego_yaw_rate = (self.ego_yaw_degrees - previous_yaw) / delta_seconds

    def _enforce_non_penetration(self) -> None:
        for actor in self.actors:
            if not actor.active or actor.cleared:
                continue
            lateral_limit = (EGO_WIDTH_METERS + actor.width) * 0.5
            if abs(self.ego_x - actor.x) >= lateral_limit:
                continue
            required_gap = (
                EGO_LENGTH_METERS * 0.5 + 1.5
                if actor.is_pedestrian
                else (EGO_LENGTH_METERS + actor.length) * 0.5 + 1.2
            )
            center_gap = actor.z - self.ego_z
            if center_gap >= 0.0:
                self.minimum_observed_clearance = min(
                    self.minimum_observed_clearance,
                    center_gap - (EGO_LENGTH_METERS + actor.length) * 0.5,
                )
                maximum_ego_z = actor.z - required_gap
                if self.ego_z > maximum_ego_z:
                    self.ego_z = maximum_ego_z
                    self.speed_mps = min(
                        self.speed_mps,
                        actor.speed_mps if not actor.is_pedestrian else 0.0,
                    )
                    self.acceleration_mps2 = min(self.acceleration_mps2, -4.8)
            elif not actor.is_pedestrian:
                # Surrounding traffic also behaves safely. If the ego brakes
                # for a new hazard, a vehicle approaching from behind must not
                # continue blindly through it.
                maximum_actor_z = self.ego_z - required_gap
                if actor.z > maximum_actor_z:
                    actor.z = maximum_actor_z
                    actor.speed_mps = min(actor.speed_mps, self.speed_mps)

    def visible_actors(self) -> List[SimulatedActor]:
        return [
            actor
            for actor in self.actors
            if actor.active and not actor.cleared and -55.0 <= actor.z - self.ego_z <= 140.0
        ]

    def create_snapshot(self, sequence_number: int) -> Dict[str, Any]:
        wheel_radius = 0.34
        wheel_rpm = self.speed_mps / (2.0 * math.pi * wheel_radius) * 60.0
        speed_kmh = self.speed_mps * 3.6
        throttle = clamp(self.acceleration_mps2 / 2.0, 0.0, 1.0) * 100.0
        brake = clamp(-self.acceleration_mps2 / 4.8, 0.0, 1.0)

        return {
            "metadata": {
                "sequenceNumber": sequence_number,
                "sourceTimestampSeconds": round(self.simulation_time, 6),
                "receiptTimestampSeconds": 0.0,
                "timelineDurationSeconds": 0.0,
                "staleAfterSeconds": 1.0,
                "validity": VALID,
                "freshness": FRESH,
                "validityMessage": "",
                "coordinateFrame": {
                    "frameId": "roadweave.unity.local",
                    "parentFrameId": "",
                    "handedness": "left",
                    "horizontalUnit": "meter",
                    "angleUnit": "degree",
                    "xAxis": "right",
                    "yAxis": "up",
                    "zAxis": "forward",
                    "originDescription": f"procedural simulator seed {self.scenario_seed}",
                    "yawConvention": "rotation around +Y in Unity degrees",
                },
            },
            "ego": {
                "position": {"x": round(self.ego_x, 5), "y": 0.0, "z": round(self.ego_z, 5)},
                "yawDegrees": round(self.ego_yaw_degrees, 4),
                "speedMetersPerSecond": round(self.speed_mps, 4),
                "longitudinalAcceleration": round(self.acceleration_mps2, 4),
                "validity": VALID,
            },
            "vehicle": {
                "speedKilometersPerHour": round(speed_kmh, 4),
                "batteryPercent": round(self.battery_percent, 3),
                "availableDistanceKilometers": round(self.battery_percent * 4.0, 2),
                "gearPosition": 1 if self.speed_mps > 0.05 else 0,
                "throttlePercent": round(throttle, 3),
                "brake": round(brake, 4),
                "brakeSwitch": 1 if brake > 0.01 else 0,
                "steeringDegrees": round(self.ego_yaw_degrees * 1.8, 3),
                "steeringSpeed": round(abs(self.ego_yaw_rate), 3),
                "yawRate": round(self.ego_yaw_rate, 3),
                "leftSignal": 1 if self.controller.target_lane_x == LEFT_LANE_X and self.ego_x > LEFT_LANE_X + 0.2 else 0,
                "rightSignal": 1 if self.controller.target_lane_x == RIGHT_LANE_X and self.ego_x < RIGHT_LANE_X - 0.2 else 0,
                "temperatureCelsius": round(24.0 + min(8.0, self.speed_mps * 0.25), 3),
                "temperatureIsValid": True,
                "validity": VALID,
            },
            "wheels": {
                "frontLeftRpm": round(wheel_rpm, 4),
                "frontRightRpm": round(wheel_rpm, 4),
                "rearLeftRpm": round(wheel_rpm, 4),
                "rearRightRpm": round(wheel_rpm, 4),
                "validity": VALID,
            },
            "actors": [self._actor_state(actor, sequence_number) for actor in self.visible_actors()],
        }

    def _actor_state(self, actor: SimulatedActor, sequence_number: int) -> Dict[str, Any]:
        return {
            "id": actor.actor_id,
            "semanticClass": actor.semantic_class,
            "sourceClass": actor.source_class,
            "motionState": actor.motion_state,
            "observationTimestampSeconds": round(self.simulation_time, 6),
            "observationSequenceNumber": sequence_number,
            "freshness": FRESH,
            "position": {"x": round(actor.x, 5), "y": 0.0, "z": round(actor.z, 5)},
            "yawDegrees": -90.0 if actor.is_pedestrian else 0.0,
            "velocityMetersPerSecond": {
                "x": round(actor.lateral_speed_mps, 5),
                "y": 0.0,
                "z": 0.0 if actor.is_pedestrian else round(actor.speed_mps, 5),
            },
            "dimensionsMeters": {"x": actor.width, "y": actor.height, "z": actor.length},
            "confidence": 0.97,
            "visibility": "simulated-sensor-track",
            "validity": VALID,
        }

    def nearest_front_description(self) -> str:
        hit = self.latest_sensors.current_front
        return "clear" if hit is None else f"{hit.actor.actor_id}:{hit.longitudinal_gap:.1f}m"

    def _print_plan(self) -> None:
        print()
        print(f"Generated procedural scenario seed={self.scenario_seed}")
        if not self.actors:
            print("  clear road (actors disabled)")
            return
        for actor in self.actors:
            lane = "pedestrian crossing" if actor.is_pedestrian else (
                "left" if actor.lane_x == LEFT_LANE_X else "right"
            )
            print(
                f"  {actor.actor_id:25s} z={actor.z:6.1f}m "
                f"lane={lane:19s} speed={actor.speed_mps * 3.6:4.1f}km/h"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoadWeave procedural real-time simulator.")
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--data-port", type=int, default=5055)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5056)
    parser.add_argument("--rate", type=float, default=30.0, help="Snapshots per second. Default: 30")
    parser.add_argument("--encounters", type=int, default=7, help="Primary encounters per generated run. Default: 7")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible base seed.")
    parser.add_argument("--no-actors", action="store_true", help="Generate a clear road only.")
    parser.add_argument("--actors", action="store_true", help="Compatibility option; actors are enabled by default.")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic safety checks and exit.")
    return parser.parse_args()


def receive_control_commands(control_socket: socket.socket) -> List[str]:
    commands: List[str] = []
    while True:
        try:
            payload, _ = control_socket.recvfrom(1024)
        except BlockingIOError:
            break
        command = payload.decode("utf-8", errors="ignore").strip().upper()
        if command:
            commands.append(command)
    return commands


def run_self_test() -> None:
    world = SimulatedWorld(True, configured_seed=20260824, encounter_count=8, announce=False)
    observed_behaviors = set()
    for _ in range(3000):
        world.update(0.05)
        observed_behaviors.add(world.controller.behavior)
        assert math.isfinite(world.ego_x) and math.isfinite(world.ego_z)
        assert world.speed_mps >= -0.001
        for actor in world.visible_actors():
            lateral_overlap = abs(world.ego_x - actor.x) < (EGO_WIDTH_METERS + actor.width) * 0.5
            longitudinal_overlap = abs(world.ego_z - actor.z) < (EGO_LENGTH_METERS + actor.length) * 0.5
            assert not (lateral_overlap and longitudinal_overlap), f"collision with {actor.actor_id}"
    assert (
        AutonomousDrivingController.BRAKING_FOR_PEDESTRIAN in observed_behaviors
        or AutonomousDrivingController.EMERGENCY_STOP in observed_behaviors
    )
    assert any(
        behavior in observed_behaviors
        for behavior in (
            AutonomousDrivingController.FOLLOWING,
            AutonomousDrivingController.CHANGE_LEFT,
            AutonomousDrivingController.CHANGE_RIGHT,
        )
    )
    print("RoadWeave simulator self-test passed.")
    print("Observed behaviors: " + ", ".join(sorted(observed_behaviors)))


def main() -> None:
    arguments = parse_arguments()
    if arguments.self_test:
        run_self_test()
        return

    rate_hz = max(1.0, arguments.rate)
    interval_seconds = 1.0 / rate_hz
    data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.bind((arguments.control_host, arguments.control_port))
    control_socket.setblocking(False)
    unity_target = (arguments.unity_host, arguments.data_port)
    world = SimulatedWorld(
        include_actors=not arguments.no_actors,
        configured_seed=arguments.seed,
        encounter_count=arguments.encounters,
        announce=arguments.autostart,
    )
    # A non-autostart plan is not authoritative until Unity creates its source
    # session. Announce subsequent RESET-generated plans, not the unused draft.
    world.announce = True
    sequence_number = 0
    driving = arguments.autostart
    stream_enabled = arguments.autostart
    last_update_time = time.perf_counter()
    next_send_time = last_update_time
    last_status_print = last_update_time

    print()
    print("RoadWeave procedural simulated stream")
    print(f"Sending TwinSnapshots to {arguments.unity_host}:{arguments.data_port}")
    print(f"Listening for Unity controls on {arguments.control_host}:{arguments.control_port}")
    print(f"Rate: {rate_hz:.1f} Hz")
    print("Waiting for Unity RESET/START." if not stream_enabled else "Simulation started automatically.")

    try:
        while True:
            now = time.perf_counter()
            for command in receive_control_commands(control_socket):
                if command == "START":
                    stream_enabled = True
                    driving = True
                    last_update_time = now
                    next_send_time = now
                    print("Unity command: START")
                elif command == "PAUSE":
                    driving = False
                    print("Unity command: PAUSE")
                elif command == "RESET_START":
                    world.reset()
                    sequence_number = 0
                    stream_enabled = True
                    driving = True
                    last_update_time = now
                    next_send_time = now
                    print("Unity command: RESET_START")
                elif command == "RESET_PAUSE":
                    world.reset()
                    sequence_number = 0
                    stream_enabled = True
                    driving = False
                    last_update_time = now
                    next_send_time = now
                    print("Unity command: RESET_PAUSE")
                elif command == "STOP":
                    stream_enabled = False
                    driving = False
                    print("Unity command: STOP")
                else:
                    print(f"Unknown Unity command: {command}")

            if not stream_enabled:
                last_update_time = now
                next_send_time = now
                time.sleep(0.005)
                continue
            if now < next_send_time:
                time.sleep(0.001)
                continue

            delta_seconds = now - last_update_time
            last_update_time = now
            if driving:
                world.update(delta_seconds)

            snapshot = world.create_snapshot(sequence_number)
            json_bytes = json.dumps(
                snapshot, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            data_socket.sendto(json_bytes, unity_target)
            sequence_number += 1
            next_send_time += interval_seconds
            if next_send_time < now - interval_seconds:
                next_send_time = now + interval_seconds

            if now - last_status_print >= 1.0:
                print(
                    f"seq={sequence_number - 1:5d} "
                    f"time={world.simulation_time:6.1f}s "
                    f"speed={world.speed_mps * 3.6:5.1f}km/h "
                    f"lane={'L' if world.ego_x < -2.7 else 'R'} "
                    f"behavior={world.controller.behavior:24s} "
                    f"front={world.nearest_front_description()}"
                )
                last_status_print = now

    except KeyboardInterrupt:
        print("\nStopping RoadWeave simulated stream.")
    finally:
        data_socket.close()
        control_socket.close()


if __name__ == "__main__":
    main()
