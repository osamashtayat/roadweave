"""Online RoadWeave ML policy with deterministic safety supervision.

The learned models request a high-level maneuver at 5 Hz. The simulator keeps
updating physics and checking emergency conditions at its normal 30 Hz rate.
The model never writes vehicle transforms and never bypasses lane-clearance or
collision checks.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
        asynchronous_inference: bool = False,
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
        self.asynchronous_inference = asynchronous_inference
        self._inference_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="roadweave-ml")
            if asynchronous_inference
            else None
        )
        self._pending_prediction: Optional[Future] = None

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
        self.last_target_speed = cruise_speed_mps
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

        # A pedestrian hazard always wins, even over a lane change in progress.
        pedestrian_emergency = self._pedestrian_emergency(ego_speed_mps, sensors)
        if pedestrian_emergency is not None:
            target_speed, reason = pedestrian_emergency
            if target_speed <= 0.0:
                self.executed_action = self.EMERGENCY_STOP
                self.behavior = "ML_EMERGENCY_STOP"
            else:
                self.executed_action = self.DECELERATE
                self.behavior = "ML_BRAKING_FOR_PEDESTRIAN"
            self.override_reason = reason
            self.last_target_speed = target_speed
            return target_speed, self.target_lane_x, self.behavior

        # Finish an accepted lane change before re-evaluating the front envelope,
        # mirroring the rule controller's overtake state machine. Without this,
        # the vehicle being overtaken re-triggers the front brake and cancels the
        # maneuver.
        if abs(ego_x - self.target_lane_x) > 0.18:
            self.behavior = "ML_LANE_CHANGE"
            return self.last_target_speed, self.target_lane_x, self.behavior

        if self._update_model_decision(simulation_time):
            self.last_target_speed = self._execute_action(
                ego_x,
                ego_speed_mps,
                sensors,
            )

        # Smoothly follow a slower vehicle ahead instead of cruising at full
        # speed into the emergency envelope. The learned policy rarely requests
        # a follow, so this deterministic cap mirrors the rule controller's
        # FOLLOWING state and removes the cruise-then-hard-brake jerk.
        self.last_target_speed = self._cap_for_following(
            self.last_target_speed,
            ego_speed_mps,
            sensors,
        )

        # Deterministic overtake: a slower vehicle ahead with a clear adjacent
        # lane. Mirrors the rule controller, which changes lane while still
        # moving rather than parking behind the obstacle.
        overtake = self._overtake_request(ego_x, sensors)
        if overtake is not None:
            target_lane_x, overtake_reason = overtake
            self.target_lane_x = target_lane_x
            self.executed_action = (
                self.CHANGE_LEFT
                if target_lane_x == self.left_lane_x
                else self.CHANGE_RIGHT
            )
            self.override_reason = overtake_reason
            self.behavior = "ML_LANE_CHANGE"
            self.last_target_speed = min(
                self.cruise_speed_mps,
                max(ego_speed_mps, 8.0),
            )
            return self.last_target_speed, self.target_lane_x, self.behavior

        # Front emergency is the safety net for cases the following/overtake
        # path could not handle (a hard-braking or suddenly appearing actor).
        front_emergency = self._front_emergency(sensors)
        if front_emergency is not None:
            escape = self._deadlock_escape(ego_x, ego_speed_mps, sensors)
            if escape is not None:
                target_lane_x, escape_reason = escape
                self.target_lane_x = target_lane_x
                self.executed_action = (
                    self.CHANGE_LEFT
                    if target_lane_x == self.left_lane_x
                    else self.CHANGE_RIGHT
                )
                self.override_reason = escape_reason
                self.behavior = "ML_LANE_CHANGE"
                self.last_target_speed = min(
                    self.cruise_speed_mps,
                    max(ego_speed_mps, 8.0),
                )
                return self.last_target_speed, self.target_lane_x, self.behavior

            target_speed, reason = front_emergency
            self.executed_action = self.EMERGENCY_STOP
            self.override_reason = reason
            self.behavior = "ML_EMERGENCY_STOP"
            self.last_target_speed = target_speed
            return target_speed, self.target_lane_x, self.behavior

        return self.last_target_speed, self.target_lane_x, self.behavior

    def _cap_for_following(
        self,
        target_speed: float,
        ego_speed_mps: float,
        sensors: Any,
    ) -> float:
        front = getattr(sensors, "current_front", None)
        if front is None or front.longitudinal_gap >= 50.0:
            return target_speed
        actor_speed = float(getattr(front.actor, "speed_mps", 0.0))
        if actor_speed >= self.cruise_speed_mps - 0.5:
            return target_speed
        return min(target_speed, self._following_speed(front, ego_speed_mps))

    def _following_speed(self, front: Any, ego_speed_mps: float) -> float:
        actor_speed = float(getattr(front.actor, "speed_mps", 0.0))
        gap = float(front.longitudinal_gap)
        safe_gap = 7.0 + ego_speed_mps * 1.45
        if gap <= 3.0:
            return 0.0
        if front.time_to_collision < 1.8:
            return max(0.0, min(actor_speed, ego_speed_mps - 3.0))
        gap_error = gap - safe_gap
        return max(0.0, min(self.cruise_speed_mps, actor_speed + gap_error * 0.32))

    def _overtake_request(
        self,
        ego_x: float,
        sensors: Any,
    ) -> Optional[Tuple[float, str]]:
        """Return a clear adjacent lane when a slower vehicle blocks the current one.

        Mirrors the rule controller's overtake: it changes lane while still
        moving rather than following to a stop and then escaping. The learned
        policy rarely initiates a lane change, so this deterministic trigger is
        what actually performs the overtake.
        """

        front = getattr(sensors, "current_front", None)
        if front is None or front.longitudinal_gap >= 50.0:
            return None
        actor_speed = float(getattr(front.actor, "speed_mps", 0.0))
        if actor_speed >= self.cruise_speed_mps - 1.0:
            return None

        on_left = abs(ego_x - self.left_lane_x) < abs(ego_x - self.right_lane_x)
        if on_left and self._lane_clear(sensors.right_front, sensors.right_rear):
            return self.right_lane_x, "right lane is clear"
        if not on_left and self._lane_clear(sensors.left_front, sensors.left_rear):
            return self.left_lane_x, "left lane is clear"
        return None

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

    def close(self) -> None:
        """Release the optional background inference worker."""

        if self._pending_prediction is not None:
            self._pending_prediction.cancel()
            self._pending_prediction = None
        if self._inference_executor is not None:
            self._inference_executor.shutdown(wait=False, cancel_futures=True)
            self._inference_executor = None

    def _update_model_decision(self, simulation_time: float) -> bool:
        """Poll/launch inference without delaying the 30 Hz motion loop.

        Scikit-learn prediction takes about 80-110 ms on the development Mac.
        Running it inline five times per second therefore starved the UDP
        publisher and produced the visible hold/catch-up bounce in Unity.
        """

        updated = False
        if self._pending_prediction is not None and self._pending_prediction.done():
            try:
                self._apply_model_output(self._pending_prediction.result())
            except Exception as error:  # Keep deterministic safety available.
                self.requested_action = self.KEEP
                self.executed_action = self.DECELERATE
                self.override_reason = "ML inference failed: {}".format(error)
            self._pending_prediction = None
            updated = True

        if (
            simulation_time + 1e-9 >= self.next_decision_time
            and self._pending_prediction is None
        ):
            summary = summarize_history(self.history)
            self.next_decision_time = simulation_time + self.decision_interval
            if self._inference_executor is None:
                self._apply_model_output(self._predict_summary(summary))
                updated = True
            else:
                self._pending_prediction = self._inference_executor.submit(
                    self._predict_summary,
                    summary,
                )
        return updated

    def _predict_summary(self, summary: Mapping[str, float]) -> Tuple[str, float, str, float]:
        risk, risk_confidence, _ = predict_one(self.models.risk, summary)
        action, action_confidence, _ = predict_one(self.models.policy, summary)
        return risk, risk_confidence, action, action_confidence

    def _run_models(self) -> None:
        """Synchronous path retained for unit tests and offline benchmarks."""

        summary = summarize_history(self.history)
        self._apply_model_output(self._predict_summary(summary))

    def _apply_model_output(
        self,
        prediction: Tuple[str, float, str, float],
    ) -> None:
        risk, risk_confidence, action, action_confidence = prediction
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
            return self._cautious_speed(ego_speed)
        if action == self.CHANGE_LEFT:
            if self._lane_clear(sensors.left_front, sensors.left_rear):
                self.target_lane_x = self.left_lane_x
                return min(self.cruise_speed_mps, max(ego_speed, 8.0))
            self.executed_action = self.DECELERATE
            self.override_reason = "left lane is not clear"
            self.behavior = "ML_LANE_CHANGE_REJECTED"
            return self._cautious_speed(ego_speed)
        if action == self.CHANGE_RIGHT:
            if self._lane_clear(sensors.right_front, sensors.right_rear):
                self.target_lane_x = self.right_lane_x
                return min(self.cruise_speed_mps, max(ego_speed, 8.0))
            self.executed_action = self.DECELERATE
            self.override_reason = "right lane is not clear"
            self.behavior = "ML_LANE_CHANGE_REJECTED"
            return self._cautious_speed(ego_speed)

        # KEEP means lane keeping at the normal route speed, not freezing the
        # current transform or forcing the speed to zero.
        self.target_lane_x = (
            self.left_lane_x
            if abs(ego_x - self.left_lane_x) < abs(ego_x - self.right_lane_x)
            else self.right_lane_x
        )
        return self.cruise_speed_mps

    def _cautious_speed(self, ego_speed: float) -> float:
        """Reduce speed without turning a risk prediction into a full stop."""

        floor = min(self.cruise_speed_mps, max(2.0, self.cruise_speed_mps * 0.55))
        return min(self.cruise_speed_mps, max(floor, ego_speed - 1.0))

    def _pedestrian_emergency(
        self,
        ego_speed: float,
        sensors: Any,
    ) -> Optional[Tuple[float, str]]:
        pedestrian = getattr(sensors, "pedestrian_hazard", None)
        if pedestrian is None:
            return None
        gap = max(0.0, float(pedestrian.longitudinal_gap))
        # Mirror the rule controller: brake smoothly as the pedestrian gets
        # close, and hard-stop only when the remaining gap is nearly closed.
        stop_gap = gap - 5.0
        if stop_gap <= 1.5:
            return 0.0, "pedestrian emergency envelope"
        target = min(self.cruise_speed_mps, max(0.0, stop_gap * 0.42))
        if target < ego_speed - 0.1:
            return target, "braking for pedestrian"
        return None

    def _front_emergency(
        self,
        sensors: Any,
    ) -> Optional[Tuple[float, str]]:
        front = getattr(sensors, "current_front", None)
        if front is None:
            return None
        closing_speed = max(0.0, float(front.relative_speed_mps))
        stopping_gap = closing_speed * closing_speed / (2.0 * 4.8) + 2.5
        if (
            front.longitudinal_gap <= max(4.0, stopping_gap)
            or front.time_to_collision < 1.5
        ):
            return 0.0, "front-object emergency envelope"
        return None

    def _deadlock_escape(
        self,
        ego_x: float,
        ego_speed_mps: float,
        sensors: Any,
    ) -> Optional[Tuple[float, str]]:
        """Return a clear adjacent lane when fully stopped behind a stationary actor.

        The learned policy is effectively longitudinal: with only a handful of
        lane-change training labels it almost never requests CHANGE_LEFT or
        CHANGE_RIGHT. Without this deterministic escape the vehicle would brake
        to a stop and wait forever behind a parked or broken-down actor. It is
        gated on a full stop and a stationary obstacle so it never turns an
        active braking situation into an unsafe high-speed lane change.
        """
        if ego_speed_mps > 0.25:
            return None

        front = getattr(sensors, "current_front", None)
        if front is None:
            return None
        if float(getattr(front.actor, "speed_mps", 0.0)) > 1.0:
            return None

        on_left = abs(ego_x - self.left_lane_x) < abs(ego_x - self.right_lane_x)
        if on_left and self._lane_clear(sensors.right_front, sensors.right_rear):
            return self.right_lane_x, "right lane is clear"
        if not on_left and self._lane_clear(sensors.left_front, sensors.left_rear):
            return self.left_lane_x, "left lane is clear"
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
