"""Online RoadWeave ML policy with deterministic safety supervision.

The learned models request a high-level maneuver at 5 Hz. The simulator keeps
updating physics and checking emergency conditions at its normal 30 Hz rate.
The model never writes vehicle transforms and never bypasses lane-clearance or
collision checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import math

try:
    from ML.src.features import new_state, set_slot, summarize_history
    from ML.src.model_support import load_artifact, predict_one
except ModuleNotFoundError:
    from features import new_state, set_slot, summarize_history  # type: ignore
    from model_support import load_artifact, predict_one  # type: ignore


@dataclass(frozen=True)
class OnlineModelBundle:
    risk: Mapping[str, Any]
    policy: Mapping[str, Any]

    @classmethod
    def load(cls, risk_path: Path, policy_path: Path) -> "OnlineModelBundle":
        return cls(
            risk=load_artifact(risk_path, expected_task="risk"),
            policy=load_artifact(policy_path, expected_task="policy"),
        )


class RoadWeaveMLController:
    """Drop-in controller for the procedural stream's SimulatedWorld."""

    KEEP = "KEEP"
    ACCELERATE = "ACCELERATE"
    DECELERATE = "DECELERATE"
    CHANGE_LEFT = "CHANGE_LEFT"
    CHANGE_RIGHT = "CHANGE_RIGHT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    def __init__(
        self,
        models: OnlineModelBundle,
        right_lane_x: float,
        left_lane_x: float,
        cruise_speed_mps: float,
        decision_rate_hz: float = 5.0,
        history_seconds: float = 3.0,
        confidence_threshold: float = 0.45,
    ) -> None:
        if decision_rate_hz <= 0.0:
            raise ValueError("decision_rate_hz must be greater than zero.")
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be greater than zero.")

        self.models = models
        self.right_lane_x = right_lane_x
        self.left_lane_x = left_lane_x
        self.cruise_speed_mps = cruise_speed_mps
        self.decision_interval = 1.0 / decision_rate_hz
        self.history_seconds = history_seconds
        self.confidence_threshold = confidence_threshold

        self.behavior = "ML_KEEP"
        self.target_lane_x = right_lane_x
        self.overtake_actor_id: Optional[str] = None
        self.history: Deque[Dict[str, float]] = deque()
        self.next_decision_time = 0.0
        self.requested_action = self.KEEP
        self.executed_action = self.KEEP
        self.action_confidence = 0.0
        self.risk_level = "LOW"
        self.risk_confidence = 0.0
        self.override_reason = ""
        self.last_target_speed = 0.0
        self.pending_lane_action: Optional[str] = None
        self.pending_lane_votes = 0

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "requestedAction": self.requested_action,
            "executedAction": self.executed_action,
            "actionConfidence": round(self.action_confidence, 4),
            "riskLevel": self.risk_level,
            "riskConfidence": round(self.risk_confidence, 4),
            "overrideReason": self.override_reason,
        }

    def decide(
        self,
        ego_x: float,
        ego_z: float,
        ego_speed_mps: float,
        sensors: Any,
        actors_by_id: Mapping[str, Any],
        simulation_time: float = 0.0,
        ego_acceleration: float = 0.0,
        ego_yaw_rate: float = 0.0,
    ) -> Tuple[float, float, str]:
        del ego_z, actors_by_id  # Available for interface compatibility.

        state = self._observation(
            simulation_time,
            ego_x,
            ego_speed_mps,
            ego_acceleration,
            ego_yaw_rate,
            sensors,
        )
        self.history.append(state)
        cutoff = simulation_time - self.history_seconds
        while len(self.history) > 1 and self.history[0]["timestamp"] < cutoff:
            self.history.popleft()

        emergency = self._emergency_override(ego_speed_mps, sensors)
        if emergency is not None:
            target_speed, reason = emergency
            self.executed_action = self.EMERGENCY_STOP
            self.override_reason = reason
            self.behavior = "ML_EMERGENCY_STOP"
            self.last_target_speed = target_speed
            return target_speed, self.target_lane_x, self.behavior

        # Finish an accepted lane change smoothly before accepting an opposing
        # request. Emergency checks above still run on every physics update.
        if abs(ego_x - self.target_lane_x) > 0.18:
            self.behavior = "ML_LANE_CHANGE"
            return self.last_target_speed, self.target_lane_x, self.behavior

        if simulation_time + 1e-9 >= self.next_decision_time:
            self._run_models()
            self.next_decision_time = simulation_time + self.decision_interval
            self.last_target_speed = self._execute_action(
                ego_x,
                ego_speed_mps,
                sensors,
            )

        return self.last_target_speed, self.target_lane_x, self.behavior

    def _observation(
        self,
        timestamp: float,
        ego_x: float,
        ego_speed: float,
        ego_acceleration: float,
        ego_yaw_rate: float,
        sensors: Any,
    ) -> Dict[str, float]:
        state = new_state(timestamp, ego_speed, ego_acceleration, ego_yaw_rate)
        on_left_lane = abs(ego_x - self.left_lane_x) < abs(ego_x - self.right_lane_x)
        # SensorFrame names the simulator's two absolute lanes. The learned
        # feature contract is ego-relative: front means the current lane and
        # left/right mean an adjacent lane. Remap them whenever ego changes
        # lane so the model sees the same semantics it saw during training.
        slot_hits = {
            "front": getattr(sensors, "current_front", None),
            "left_front": (
                None if on_left_lane else getattr(sensors, "left_front", None)
            ),
            "left_rear": (
                None if on_left_lane else getattr(sensors, "left_rear", None)
            ),
            "right_front": (
                getattr(sensors, "right_front", None) if on_left_lane else None
            ),
            "right_rear": (
                getattr(sensors, "right_rear", None) if on_left_lane else None
            ),
            "pedestrian": getattr(sensors, "pedestrian_hazard", None),
        }
        for slot, hit in slot_hits.items():
            if hit is None:
                continue
            gap = abs(float(hit.longitudinal_gap)) if slot.endswith("rear") else float(hit.longitudinal_gap)
            actor_speed = float(getattr(hit.actor, "speed_mps", 0.0))
            closing_speed = float(hit.relative_speed_mps)
            if slot.endswith("rear"):
                # Simulator hits use ego speed minus actor speed. For a rear
                # slot, invert it so positive consistently means the other
                # actor is closing on the ego vehicle.
                closing_speed = -closing_speed
            set_slot(
                state,
                slot,
                gap,
                closing_speed,
                actor_speed,
            )

        state["left_clear"] = float(
            not on_left_lane
            and self._lane_clear(slot_hits["left_front"], slot_hits["left_rear"])
        )
        state["right_clear"] = float(
            on_left_lane
            and self._lane_clear(slot_hits["right_front"], slot_hits["right_rear"])
        )
        return state

    def _run_models(self) -> None:
        summary = summarize_history(self.history)
        risk, risk_confidence, _ = predict_one(self.models.risk, summary)
        action, action_confidence, _ = predict_one(self.models.policy, summary)
        self.risk_level = risk
        self.risk_confidence = risk_confidence
        self.requested_action = action
        self.action_confidence = action_confidence
        self.override_reason = ""

        if action_confidence < self.confidence_threshold:
            candidate_action = self.DECELERATE
            self.override_reason = "policy confidence below {:.2f}".format(
                self.confidence_threshold
            )
        elif risk == "EXTREME" and action not in (self.DECELERATE,):
            candidate_action = self.DECELERATE
            self.override_reason = "extreme-risk conservative override"
        elif risk == "HIGH" and action == self.ACCELERATE:
            candidate_action = self.DECELERATE
            self.override_reason = "high-risk acceleration rejected"
        else:
            candidate_action = action

        if candidate_action in (self.CHANGE_LEFT, self.CHANGE_RIGHT):
            if self.pending_lane_action == candidate_action:
                self.pending_lane_votes += 1
            else:
                self.pending_lane_action = candidate_action
                self.pending_lane_votes = 1
            if self.pending_lane_votes < 3:
                self.executed_action = self.DECELERATE
                self.override_reason = "confirming {} ({}/3)".format(
                    candidate_action,
                    self.pending_lane_votes,
                )
            else:
                self.executed_action = candidate_action
                self.pending_lane_action = None
                self.pending_lane_votes = 0
        else:
            self.pending_lane_action = None
            self.pending_lane_votes = 0
            self.executed_action = candidate_action

    def _execute_action(
        self,
        ego_x: float,
        ego_speed: float,
        sensors: Any,
    ) -> float:
        action = self.executed_action
        self.behavior = "ML_{}".format(action)

        if action == self.ACCELERATE:
            return min(self.cruise_speed_mps, ego_speed + 2.0)
        if action == self.DECELERATE:
            return max(0.0, ego_speed - 3.0)
        if action == self.CHANGE_LEFT:
            if self._lane_clear(sensors.left_front, sensors.left_rear):
                self.target_lane_x = self.left_lane_x
                return min(self.cruise_speed_mps, max(ego_speed, 8.0))
            self.executed_action = self.DECELERATE
            self.override_reason = "left lane is not clear"
            self.behavior = "ML_LANE_CHANGE_REJECTED"
            return max(0.0, ego_speed - 3.0)
        if action == self.CHANGE_RIGHT:
            if self._lane_clear(sensors.right_front, sensors.right_rear):
                self.target_lane_x = self.right_lane_x
                return min(self.cruise_speed_mps, max(ego_speed, 8.0))
            self.executed_action = self.DECELERATE
            self.override_reason = "right lane is not clear"
            self.behavior = "ML_LANE_CHANGE_REJECTED"
            return max(0.0, ego_speed - 3.0)

        # KEEP means lane keeping at the normal route speed, not freezing the
        # current transform or forcing the speed to zero.
        self.target_lane_x = (
            self.left_lane_x
            if abs(ego_x - self.left_lane_x) < abs(ego_x - self.right_lane_x)
            else self.right_lane_x
        )
        return self.cruise_speed_mps

    def _emergency_override(
        self,
        ego_speed: float,
        sensors: Any,
    ) -> Optional[Tuple[float, str]]:
        pedestrian = getattr(sensors, "pedestrian_hazard", None)
        if pedestrian is not None:
            gap = max(0.0, float(pedestrian.longitudinal_gap))
            stopping_gap = ego_speed * ego_speed / (2.0 * 4.8) + 5.0
            if gap <= stopping_gap or pedestrian.time_to_collision < 2.0:
                return 0.0, "pedestrian emergency envelope"

        front = getattr(sensors, "current_front", None)
        if front is not None:
            closing_speed = max(0.0, float(front.relative_speed_mps))
            stopping_gap = closing_speed * closing_speed / (2.0 * 4.8) + 2.5
            if front.longitudinal_gap <= max(4.0, stopping_gap) or front.time_to_collision < 1.5:
                return 0.0, "front-object emergency envelope"
        return None

    @staticmethod
    def _lane_clear(
        front: Any,
        rear: Any,
        front_clearance: float = 34.0,
        rear_clearance: float = 15.0,
    ) -> bool:
        return (
            (front is None or float(front.longitudinal_gap) > front_clearance)
            and (rear is None or abs(float(rear.longitudinal_gap)) > rear_clearance)
        )
