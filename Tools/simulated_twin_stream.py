#!/usr/bin/env python3

"""RoadWeave procedural real-time TwinSnapshot simulator.

The scenario planner creates a different sequence of simple road encounters on
every reset. The autonomous controller does not read that plan directly. It
receives observations from a small simulated sensor suite and decides whether
to cruise, follow, brake, change lanes, pass, or return to the right lane.

Data:     Python -> UDP 5055 -> Unity
Controls: Unity  -> UDP 5056 -> Python

Rule-controller mode uses only Python standard-library modules. ML-controller
mode additionally uses the packages installed in ML/.venv.
"""

import argparse
import json
import math
import random
import secrets
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


VALID = 1
PARTIAL = 2
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
# Comfort-tuned longitudinal control. The controller may command a step target
# at its own decision rate (5 Hz for the ML controller), so the commanded target
# is slewed before the speed follows it, and braking is kept gentler than the
# original hard 4.8 m/s^2 emergency limit.
ACCELERATION_MPS2 = 2.0
BRAKING_MPS2 = 3.0
TARGET_SLEW_MPS2 = 3.0
ROUTE_SAMPLE_SPACING_METERS = 2.0
ROUTE_WINDOW_STEP_METERS = 200.0
ROUTE_LOOK_BEHIND_METERS = 180.0
ROUTE_LOOK_AHEAD_METERS = 900.0
ROUTE_OUTPUT_SPACING_METERS = 18.0
ROUTE_PUBLISH_RATE_HZ = 1.0
UDP_SAFE_PAYLOAD_BYTES = 8000


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def move_towards(current: float, target: float, maximum_delta: float) -> float:
    if abs(target - current) <= maximum_delta:
        return target
    return current + math.copysign(maximum_delta, target - current)


def shortest_angle_delta_degrees(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class RoutePose:
    x: float
    z: float
    yaw_degrees: float


class ProceduralRoad:
    """An unlimited, deterministic road centerline for one simulator seed.

    Distance along the route is kept separate from Unity world coordinates.
    This lets sensors and driving decisions use stable lane-relative distances
    while the published road turns left and right in world space.
    """

    def __init__(self, seed: int) -> None:
        route_random = random.Random(seed ^ 0x5A17C9E3)
        self.primary_amplitude = math.radians(route_random.uniform(18.0, 31.0))
        self.secondary_amplitude = math.radians(route_random.uniform(7.0, 15.0))
        self.primary_scale = route_random.uniform(105.0, 165.0)
        self.secondary_scale = route_random.uniform(42.0, 78.0)
        self.primary_phase = route_random.uniform(0.0, math.tau)
        self.secondary_phase = route_random.uniform(0.0, math.tau)
        self.samples: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

    def _heading_radians(self, route_distance: float) -> float:
        if route_distance <= 0.0:
            return 0.0
        primary = self.primary_amplitude * math.sin(
            route_distance / self.primary_scale + self.primary_phase
        )
        secondary = self.secondary_amplitude * math.sin(
            route_distance / self.secondary_scale + self.secondary_phase
        )
        ramp = clamp(route_distance / 90.0, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        return clamp(
            (primary + secondary) * ramp,
            math.radians(-68.0),
            math.radians(68.0),
        )

    def _ensure_distance(self, route_distance: float) -> None:
        target = max(0.0, route_distance)
        while self.samples[-1][0] + 0.001 < target + ROUTE_SAMPLE_SPACING_METERS:
            previous_s, previous_x, previous_z = self.samples[-1]
            next_s = previous_s + ROUTE_SAMPLE_SPACING_METERS
            midpoint = (previous_s + next_s) * 0.5
            heading = self._heading_radians(midpoint)
            next_x = previous_x + math.sin(heading) * ROUTE_SAMPLE_SPACING_METERS
            next_z = previous_z + math.cos(heading) * ROUTE_SAMPLE_SPACING_METERS
            self.samples.append((next_s, next_x, next_z))

    def pose(self, route_distance: float, lateral_offset: float = 0.0) -> RoutePose:
        if route_distance < 0.0:
            base_x = 0.0
            base_z = route_distance
            heading = 0.0
        else:
            self._ensure_distance(route_distance)
            lower_index = int(route_distance / ROUTE_SAMPLE_SPACING_METERS)
            upper_index = min(lower_index + 1, len(self.samples) - 1)
            lower_s, lower_x, lower_z = self.samples[lower_index]
            upper_s, upper_x, upper_z = self.samples[upper_index]
            span = max(0.001, upper_s - lower_s)
            blend = clamp((route_distance - lower_s) / span, 0.0, 1.0)
            base_x = lower_x + (upper_x - lower_x) * blend
            base_z = lower_z + (upper_z - lower_z) * blend
            heading = self._heading_radians(route_distance)

        # Positive local X is the route's right side, matching Unity's frame.
        base_x += math.cos(heading) * lateral_offset
        base_z -= math.sin(heading) * lateral_offset
        return RoutePose(base_x, base_z, math.degrees(heading))

    def velocity(
        self,
        route_distance: float,
        forward_speed_mps: float,
        lateral_speed_mps: float = 0.0,
    ) -> Tuple[float, float]:
        heading = self._heading_radians(route_distance)
        x = math.sin(heading) * forward_speed_mps + math.cos(heading) * lateral_speed_mps
        z = math.cos(heading) * forward_speed_mps - math.sin(heading) * lateral_speed_mps
        return x, z


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
    """Continuously builds random encounters without driving decisions."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.next_index = 0
        self.kind_bag: List[str] = []

    def build(self, encounter_count: int) -> List[SimulatedActor]:
        actors: List[SimulatedActor] = []
        event_z = self.random.uniform(58.0, 75.0)
        for _ in range(max(4, encounter_count)):
            actors.extend(self.build_next(event_z))
            event_z += self.next_spacing()
        actors.sort(key=lambda candidate: candidate.z)
        return actors

    def build_next(self, event_z: float) -> List[SimulatedActor]:
        index = self.next_index
        self.next_index += 1
        kind = self._next_kind()
        actor = self._primary_actor(index, kind, event_z)
        actors = [actor]

        # Some encounters include traffic in the other lane. The simulated
        # sensors must notice it and delay or adapt the overtake.
        if kind != "pedestrian" and actor.lane_x == RIGHT_LANE_X:
            if self.random.random() < 0.38:
                actors.append(self._left_lane_companion(index, event_z))
        return actors

    def next_spacing(self) -> float:
        return self.random.uniform(68.0, 108.0)

    def _next_kind(self) -> str:
        if not self.kind_bag:
            # Every bag contains all core hazards, plus two weighted random
            # additions. This keeps an infinite run varied without long gaps
            # where one important behavior is never exercised.
            self.kind_bag = ["slow_car", "stopped_car", "truck", "pedestrian"]
            self.kind_bag.extend(
                self.random.choices(
                    ["slow_car", "stopped_car", "truck", "pedestrian"],
                    weights=[0.34, 0.25, 0.23, 0.18],
                    k=2,
                )
            )
            self.random.shuffle(self.kind_bag)
        return self.kind_bag.pop()

    def _actor_id(self, label: str, index: int) -> str:
        return f"sim-{self.seed:08x}-{label}-{index:05d}"

    def _primary_actor(self, index: int, kind: str, z: float) -> SimulatedActor:
        if kind == "pedestrian":
            return SimulatedActor(
                actor_id=self._actor_id("pedestrian", index),
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
                actor_id=self._actor_id("truck", index),
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
            actor_id=self._actor_id("stopped-car" if stopped else "slow-car", index),
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
            actor_id=self._actor_id("left-traffic", index),
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
        simulation_time: float = 0.0,
        ego_acceleration: float = 0.0,
        ego_yaw_rate: float = 0.0,
    ) -> Tuple[float, float, str]:
        del simulation_time, ego_acceleration, ego_yaw_rate
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
        controller_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.include_actors = include_actors
        self.configured_seed = configured_seed
        self.encounter_count = max(4, encounter_count)
        self.announce = announce
        self.controller_factory = controller_factory
        self.reset_count = 0
        self.sensors = SimulatedSensorSuite()
        self.controller: Optional[Any] = None
        self.reset()

    def _new_controller(self) -> Any:
        if self.controller_factory is not None:
            return self.controller_factory()
        return AutonomousDrivingController()

    def reset(self) -> None:
        previous_controller = self.controller
        if previous_controller is not None:
            close_controller = getattr(previous_controller, "close", None)
            if callable(close_controller):
                close_controller()

        self.scenario_seed = (
            secrets.randbits(32)
            if self.configured_seed is None
            else self.configured_seed + self.reset_count
        )
        self.reset_count += 1
        self.simulation_time = 0.0
        self.road = ProceduralRoad(self.scenario_seed)
        self.ego_x = RIGHT_LANE_X
        self.ego_y = 0.0
        self.ego_z = 0.0
        self.ego_yaw_degrees = 0.0
        self.ego_heading_offset_degrees = 0.0
        self.ego_yaw_rate = 0.0
        self.speed_mps = 0.0
        self.acceleration_mps2 = 0.0
        self.smoothed_target_speed = 0.0
        self.lateral_speed_mps = 0.0
        self.battery_percent = 92.0
        self.controller = self._new_controller()
        self.latest_sensors = SensorFrame()
        self.minimum_observed_clearance = math.inf
        self.scenario_planner = ProceduralScenarioPlanner(self.scenario_seed)
        self.next_event_z = self.scenario_planner.random.uniform(58.0, 75.0)
        self.generation_horizon = max(650.0, self.encounter_count * 95.0)
        self.actors: List[SimulatedActor] = []
        self.actors_by_id: Dict[str, SimulatedActor] = {}
        self.generated_encounter_count = 0
        self.route_cache_revision = -1
        self.route_cache: Optional[Dict[str, Any]] = None
        self._ensure_future_scenarios()
        if self.announce:
            self._print_plan()

    def close(self) -> None:
        close_controller = getattr(self.controller, "close", None)
        if callable(close_controller):
            close_controller()

    def update(self, delta_seconds: float) -> None:
        delta_seconds = clamp(delta_seconds, 0.0, 0.1)
        if delta_seconds <= 0.0:
            return
        self.simulation_time += delta_seconds
        previous_ego_x = self.ego_x
        previous_ego_z = self.ego_z
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
            simulation_time=self.simulation_time,
            ego_acceleration=self.acceleration_mps2,
            ego_yaw_rate=self.ego_yaw_rate,
        )
        self._update_ego(delta_seconds, target_speed, target_lane_x)
        self._enforce_non_penetration(previous_ego_x, previous_ego_z)
        self._retire_passed_actors()
        self._ensure_future_scenarios()
        self.battery_percent = max(0.0, self.battery_percent - 0.0003 * delta_seconds)

    def _ensure_future_scenarios(self) -> None:
        if not self.include_actors:
            return
        horizon_end = self.ego_z + self.generation_horizon
        while self.next_event_z <= horizon_end:
            new_actors = self.scenario_planner.build_next(self.next_event_z)
            for actor in new_actors:
                self.actors.append(actor)
                self.actors_by_id[actor.actor_id] = actor
            self.generated_encounter_count += 1
            self.next_event_z += self.scenario_planner.next_spacing()

    def _retire_passed_actors(self) -> None:
        retained: List[SimulatedActor] = []
        for actor in self.actors:
            passed_and_expired = actor.cleared and actor.z < self.ego_z - 90.0
            left_sensor_region = (
                actor.active and not actor.is_pedestrian and actor.z > self.ego_z + 220.0
            )
            if passed_and_expired or left_sensor_region:
                self.actors_by_id.pop(actor.actor_id, None)
                continue
            retained.append(actor)
        self.actors = retained

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
        # Slew the commanded target before following it so a 5 Hz step decision
        # does not produce a 5 Hz throttle/brake sawtooth.
        self.smoothed_target_speed = move_towards(
            self.smoothed_target_speed,
            max(0.0, target_speed),
            TARGET_SLEW_MPS2 * delta_seconds,
        )
        acceleration_limit = (
            ACCELERATION_MPS2
            if self.smoothed_target_speed >= self.speed_mps
            else BRAKING_MPS2
        )
        self.speed_mps = move_towards(
            self.speed_mps,
            self.smoothed_target_speed,
            acceleration_limit * delta_seconds,
        )
        self.acceleration_mps2 = (self.speed_mps - previous_speed) / delta_seconds

        lane_error = target_lane_x - self.ego_x
        # A lane change should read as a deliberate steering maneuver, not a
        # sideways snap. Keep the lateral command and its acceleration below
        # normal comfort limits while the longitudinal controller continues at
        # 30 Hz.
        desired_lateral_speed = clamp(lane_error * 0.82, -1.45, 1.45)
        self.lateral_speed_mps = move_towards(
            self.lateral_speed_mps,
            desired_lateral_speed,
            1.8 * delta_seconds,
        )
        if abs(lane_error) < 0.035:
            self.ego_x = target_lane_x
            self.lateral_speed_mps = 0.0
        else:
            self.ego_x += self.lateral_speed_mps * delta_seconds

        self.ego_z += self.speed_mps * delta_seconds
        desired_heading_offset = math.degrees(
            math.atan2(self.lateral_speed_mps, max(self.speed_mps, 2.0))
        )
        self.ego_heading_offset_degrees = move_towards(
            self.ego_heading_offset_degrees,
            desired_heading_offset,
            20.0 * delta_seconds,
        )
        previous_yaw = self.ego_yaw_degrees
        route_yaw = self.road.pose(self.ego_z).yaw_degrees
        self.ego_yaw_degrees = route_yaw + self.ego_heading_offset_degrees
        self.ego_yaw_rate = shortest_angle_delta_degrees(
            previous_yaw, self.ego_yaw_degrees
        ) / delta_seconds

    def _enforce_non_penetration(
        self,
        previous_ego_x: float,
        previous_ego_z: float,
    ) -> None:
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
                    previously_separate_laterally = (
                        abs(previous_ego_x - actor.x) >= lateral_limit
                    )
                    if previously_separate_laterally and previous_ego_z <= maximum_ego_z:
                        # The longitudinal motion was safe in the old lane; the
                        # new lateral step alone entered an occupied envelope.
                        # Reject that lateral step instead of visibly rewinding
                        # the ego vehicle along the road.
                        self.ego_x = previous_ego_x
                        self.lateral_speed_mps = 0.0
                        self.ego_heading_offset_degrees = 0.0
                        self.ego_yaw_degrees = self.road.pose(self.ego_z).yaw_degrees
                        self.ego_yaw_rate = 0.0
                        continue

                    # A longitudinal clamp may stop progress, but it must never
                    # make an already-published streaming pose move backward.
                    self.ego_z = max(previous_ego_z, maximum_ego_z)
                    self.speed_mps = min(
                        self.speed_mps,
                        actor.speed_mps if not actor.is_pedestrian else 0.0,
                    )
                    self.acceleration_mps2 = min(self.acceleration_mps2, -BRAKING_MPS2)
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

    def create_snapshot(
        self,
        sequence_number: int,
        include_route: bool = True,
    ) -> Dict[str, Any]:
        wheel_radius = 0.34
        wheel_rpm = self.speed_mps / (2.0 * math.pi * wheel_radius) * 60.0
        speed_kmh = self.speed_mps * 3.6
        throttle = clamp(self.acceleration_mps2 / ACCELERATION_MPS2, 0.0, 1.0) * 100.0
        brake = clamp(-self.acceleration_mps2 / BRAKING_MPS2, 0.0, 1.0)
        ego_pose = self.road.pose(self.ego_z, self.ego_x)
        ego_yaw = ego_pose.yaw_degrees + self.ego_heading_offset_degrees

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
                "position": {"x": round(ego_pose.x, 5), "y": 0.0, "z": round(ego_pose.z, 5)},
                "yawDegrees": round(ego_yaw, 4),
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
                "steeringDegrees": round(
                    self.ego_heading_offset_degrees * 1.8 + self.ego_yaw_rate * 0.12,
                    3,
                ),
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
            # Route geometry changes far more slowly than vehicle telemetry.
            # Main sends it at 1 Hz; null means "keep the last route revision".
            "route": self._route_state() if include_route else None,
            "actors": [self._actor_state(actor, sequence_number) for actor in self.visible_actors()],
        }

    def _actor_state(self, actor: SimulatedActor, sequence_number: int) -> Dict[str, Any]:
        pose = self.road.pose(actor.z, actor.x)
        velocity_x, velocity_z = self.road.velocity(
            actor.z,
            0.0 if actor.is_pedestrian else actor.speed_mps,
            actor.lateral_speed_mps,
        )
        actor_yaw = pose.yaw_degrees
        if actor.is_pedestrian and actor.lateral_speed_mps < -0.01:
            actor_yaw -= 90.0
        return {
            "id": actor.actor_id,
            "semanticClass": actor.semantic_class,
            "sourceClass": actor.source_class,
            "motionState": actor.motion_state,
            "observationTimestampSeconds": round(self.simulation_time, 6),
            "observationSequenceNumber": sequence_number,
            "freshness": FRESH,
            "position": {"x": round(pose.x, 5), "y": 0.0, "z": round(pose.z, 5)},
            "yawDegrees": round(actor_yaw, 4),
            "velocityMetersPerSecond": {
                "x": round(velocity_x, 5),
                "y": 0.0,
                "z": round(velocity_z, 5),
            },
            "dimensionsMeters": {"x": actor.width, "y": actor.height, "z": actor.length},
            "confidence": 0.97,
            "visibility": "simulated-sensor-track",
            "validity": VALID,
        }

    def _route_state(self) -> Dict[str, Any]:
        revision = max(0, int(self.ego_z // ROUTE_WINDOW_STEP_METERS))
        if self.route_cache is not None and revision == self.route_cache_revision:
            return self.route_cache

        window_anchor = revision * ROUTE_WINDOW_STEP_METERS
        start = window_anchor - ROUTE_LOOK_BEHIND_METERS
        end = window_anchor + ROUTE_LOOK_AHEAD_METERS
        point_count = int(math.ceil((end - start) / ROUTE_OUTPUT_SPACING_METERS)) + 1
        points: List[Dict[str, float]] = []
        for index in range(point_count):
            route_distance = min(
                start + index * ROUTE_OUTPUT_SPACING_METERS,
                end,
            )
            pose = self.road.pose(route_distance, RIGHT_LANE_X)
            points.append(
                {
                    "x": round(pose.x, 4),
                    "y": 0.0,
                    "z": round(pose.z, 4),
                }
            )

        self.route_cache_revision = revision
        self.route_cache = {
            "routeId": f"procedural-route-{self.scenario_seed:08x}",
            "revision": revision,
            "laneWidthMeters": abs(LEFT_LANE_X - RIGHT_LANE_X),
            "points": points,
            "validity": VALID,
        }
        return self.route_cache

    def nearest_front_description(self) -> str:
        hit = self.latest_sensors.current_front
        return "clear" if hit is None else f"{hit.actor.actor_id}:{hit.longitudinal_gap:.1f}m"

    def _print_plan(self) -> None:
        print()
        print(f"Generated unlimited procedural world seed={self.scenario_seed}")
        if not self.actors:
            print("  clear road (actors disabled)")
            return
        print("  initial look-ahead queue (new encounters are added continuously):")
        for actor in self.actors:
            lane = "pedestrian crossing" if actor.is_pedestrian else (
                "left" if actor.lane_x == LEFT_LANE_X else "right"
            )
            print(
                f"  {actor.actor_id:38s} route={actor.z:6.1f}m "
                f"lane={lane:19s} speed={actor.speed_mps * 3.6:4.1f}km/h"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoadWeave procedural real-time simulator.")
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--data-port", type=int, default=5055)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5056)
    parser.add_argument("--rate", type=float, default=30.0, help="Snapshots per second. Default: 30")
    parser.add_argument(
        "--encounters",
        type=int,
        default=7,
        help="Approximate future encounters kept queued in the unlimited world. Default: 7",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible base seed.")
    parser.add_argument("--no-actors", action="store_true", help="Generate a clear road only.")
    parser.add_argument("--actors", action="store_true", help="Compatibility option; actors are enabled by default.")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument(
        "--controller",
        choices=("rule", "ml"),
        default="rule",
        help="Driving controller. 'rule' remains the safe default.",
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--risk-model",
        type=Path,
        default=project_root / "ML" / "models" / "risk_model.joblib",
        help="Risk artifact used only with --controller ml.",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=project_root / "ML" / "models" / "policy_model.joblib",
        help="Policy artifact used only with --controller ml.",
    )
    parser.add_argument(
        "--ml-decision-rate",
        type=float,
        default=5.0,
        help="ML decisions per second; physics and safety still run at --rate. Default: 5",
    )
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


def _compact_json_bytes(snapshot: Dict[str, Any]) -> bytes:
    return json.dumps(
        snapshot,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def encode_snapshot_for_udp(
    snapshot: Dict[str, Any],
    maximum_bytes: int = UDP_SAFE_PAYLOAD_BYTES,
) -> bytes:
    """Fits one canonical snapshot inside a conservative UDP datagram budget.

    Required ego telemetry is never reduced. Optional route points are thinned
    first. If an unusually crowded frame is still too large, the route is
    omitted for that frame and the farthest presentation-only actor tracks are
    removed while metadata is marked Partial.
    """

    candidate = snapshot
    payload = _compact_json_bytes(candidate)
    if len(payload) <= maximum_bytes:
        return payload

    route = snapshot.get("route")
    if isinstance(route, dict) and isinstance(route.get("points"), list):
        candidate = dict(snapshot)
        compact_route = dict(route)
        points = list(route["points"])
        compact_route["points"] = points
        candidate["route"] = compact_route

        while len(payload) > maximum_bytes and len(points) > 2:
            last_point = points[-1]
            points = points[::2]
            if points[-1] is not last_point:
                points.append(last_point)
            compact_route["points"] = points
            payload = _compact_json_bytes(candidate)

    if len(payload) > maximum_bytes and candidate.get("route") is not None:
        candidate = dict(candidate)
        candidate["route"] = None
        payload = _compact_json_bytes(candidate)

    if len(payload) > maximum_bytes:
        ego_position = snapshot.get("ego", {}).get("position", {})
        ego_x = float(ego_position.get("x", 0.0))
        ego_z = float(ego_position.get("z", 0.0))

        def distance_squared(actor_state: Dict[str, Any]) -> float:
            position = actor_state.get("position", {})
            delta_x = float(position.get("x", 0.0)) - ego_x
            delta_z = float(position.get("z", 0.0)) - ego_z
            return delta_x * delta_x + delta_z * delta_z

        actors = sorted(
            list(snapshot.get("actors") or []),
            key=distance_squared,
        )
        candidate = dict(candidate)
        metadata = dict(snapshot.get("metadata") or {})
        metadata["validity"] = PARTIAL
        metadata["validityMessage"] = (
            "UDP payload budget retained the nearest surrounding actors."
        )
        candidate["metadata"] = metadata
        candidate["actors"] = actors
        payload = _compact_json_bytes(candidate)
        while len(payload) > maximum_bytes and actors:
            actors.pop()
            payload = _compact_json_bytes(candidate)

    if len(payload) > maximum_bytes:
        raise ValueError(
            f"Canonical snapshot is {len(payload)} bytes after compaction; "
            f"the configured UDP budget is {maximum_bytes} bytes."
        )
    return payload


def run_self_test(
    controller_factory: Optional[Callable[[], Any]] = None,
    controller_name: str = "rule",
) -> None:
    world = SimulatedWorld(
        True,
        configured_seed=20260824,
        encounter_count=8,
        announce=False,
        controller_factory=controller_factory,
    )
    observed_behaviors = set()
    initial_generated_count = world.generated_encounter_count
    observed_actor_ids = set(world.actors_by_id)
    step_count = 7000 if controller_factory is None else 3500
    for _ in range(step_count):
        world.update(0.05)
        observed_behaviors.add(world.controller.behavior)
        observed_actor_ids.update(world.actors_by_id)
        assert math.isfinite(world.ego_x) and math.isfinite(world.ego_z)
        assert world.speed_mps >= -0.001
        for actor in world.visible_actors():
            lateral_overlap = abs(world.ego_x - actor.x) < (EGO_WIDTH_METERS + actor.width) * 0.5
            longitudinal_overlap = abs(world.ego_z - actor.z) < (EGO_LENGTH_METERS + actor.length) * 0.5
            assert not (lateral_overlap and longitudinal_overlap), f"collision with {actor.actor_id}"
    snapshot = world.create_snapshot(0)
    encoded_snapshot = encode_snapshot_for_udp(snapshot)
    route_points = snapshot["route"]["points"]
    route_headings = [world.road.pose(distance).yaw_degrees for distance in (100.0, 300.0, 500.0, 700.0, 900.0)]
    assert len(route_points) > 40
    assert len(encoded_snapshot) <= UDP_SAFE_PAYLOAD_BYTES
    assert max(route_headings) - min(route_headings) > 12.0
    if controller_factory is None:
        assert world.generated_encounter_count > initial_generated_count + 10
        assert len(observed_actor_ids) > initial_generated_count + 10
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
    else:
        assert world.ego_z > 50.0
        assert all(behavior.startswith("ML_") for behavior in observed_behaviors)
    print("RoadWeave {} controller self-test passed.".format(controller_name))
    print("Observed behaviors: " + ", ".join(sorted(observed_behaviors)))


def main() -> None:
    arguments = parse_arguments()
    controller_factory: Optional[Callable[[], Any]] = None
    if arguments.controller == "ml":
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            from ML.src.online_policy import OnlineModelBundle, RoadWeaveMLController
        except ModuleNotFoundError as exception:
            raise SystemExit(
                "Could not import RoadWeave ML dependencies ({}). Activate ML/.venv "
                "before using --controller ml.".format(exception)
            )
        try:
            model_bundle = OnlineModelBundle.load(
                arguments.risk_model,
                arguments.policy_model,
            )
        except (FileNotFoundError, ValueError) as exception:
            raise SystemExit(
                "Could not start the ML controller: {}\n"
                "Train both artifacts first, or run with --controller rule.".format(
                    exception
                )
            )

        def create_ml_controller() -> Any:
            return RoadWeaveMLController(
                model_bundle,
                right_lane_x=RIGHT_LANE_X,
                left_lane_x=LEFT_LANE_X,
                cruise_speed_mps=CRUISE_SPEED_MPS,
                decision_rate_hz=arguments.ml_decision_rate,
                # A live source must keep publishing at 30 Hz even when one
                # scikit-learn inference takes longer than a frame. Offline
                # self-tests remain synchronous and deterministic.
                asynchronous_inference=not arguments.self_test,
            )

        controller_factory = create_ml_controller

    if arguments.self_test:
        run_self_test(controller_factory, arguments.controller)
        return

    rate_hz = max(1.0, arguments.rate)
    interval_seconds = 1.0 / rate_hz
    route_publish_interval = max(
        1,
        int(round(rate_hz / ROUTE_PUBLISH_RATE_HZ)),
    )
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
        controller_factory=controller_factory,
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
    print("Controller: {}".format(arguments.controller))
    if arguments.controller == "ml":
        print("ML decision rate: {:.1f} Hz".format(arguments.ml_decision_rate))
    print(
        f"Route updates: {ROUTE_PUBLISH_RATE_HZ:.1f} Hz; "
        f"UDP payload budget: {UDP_SAFE_PAYLOAD_BYTES} bytes"
    )
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

            include_route = sequence_number % route_publish_interval == 0
            snapshot = world.create_snapshot(
                sequence_number,
                include_route=include_route,
            )
            try:
                json_bytes = encode_snapshot_for_udp(snapshot)
                data_socket.sendto(json_bytes, unity_target)
            except (OSError, ValueError) as exception:
                # A transient transport-size or socket failure must not end a
                # live simulation session. The next 30 Hz update can recover.
                if now - last_status_print >= 1.0:
                    print(f"Snapshot send skipped: {exception}")
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
        world.close()
        data_socket.close()
        control_socket.close()


if __name__ == "__main__":
    main()
