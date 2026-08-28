"""Inference engine for the Unity RoadWeave Test Lab.

Unity sends source-neutral ego and simulated-sensor observations. This module
turns the latest three seconds into the same canonical features used during
training, runs the saved risk and policy models, and returns a high-level
decision. It deliberately does not know about Unity transforms, prefabs, or
nuScenes files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import math
import time

try:
    from ML.src.features import new_state, set_slot, summarize_history
    from ML.src.model_support import load_artifact, predict_one
    from ML.src.weather_features import normalize_condition, summarize_weather_history
    from ML.src.weather_model import load_weather_artifact, predict_weather_factor
    from ML.src.sensor_reliability_model import (
        load_sensor_reliability_artifact,
        predict_sensor_reliability,
    )
except ModuleNotFoundError:
    from features import new_state, set_slot, summarize_history  # type: ignore
    from model_support import load_artifact, predict_one  # type: ignore
    from weather_features import normalize_condition, summarize_weather_history  # type: ignore
    from weather_model import load_weather_artifact, predict_weather_factor  # type: ignore
    from sensor_reliability_model import (  # type: ignore
        load_sensor_reliability_artifact,
        predict_sensor_reliability,
    )


PROTOCOL_VERSION = "roadweave.testlab-ml/1.0"
OBSERVATION_MESSAGE = "observation"
RESET_MESSAGE = "reset"

KEEP = "KEEP"
ACCELERATE = "ACCELERATE"
DECELERATE = "DECELERATE"
CHANGE_LEFT = "CHANGE_LEFT"
CHANGE_RIGHT = "CHANGE_RIGHT"
EMERGENCY_STOP = "EMERGENCY_STOP"


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _present(slot: Any) -> bool:
    return isinstance(slot, Mapping) and bool(slot.get("present", False))


@dataclass
class TestLabModelBundle:
    risk: Mapping[str, Any]
    policy: Mapping[str, Any]
    weather: Optional[Mapping[str, Any]] = None
    sensor_reliability: Dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        risk_path: Path,
        policy_path: Path,
        weather_path: Optional[Path] = None,
        sensor_model_paths: Optional[Mapping[str, Path]] = None,
    ) -> "TestLabModelBundle":
        reliability_models: Dict[str, Mapping[str, Any]] = {}
        for sensor_type, model_path in (sensor_model_paths or {}).items():
            if model_path is not None and Path(model_path).is_file():
                normalized = str(sensor_type).strip().upper()
                reliability_models[normalized] = load_sensor_reliability_artifact(
                    Path(model_path), normalized
                )
        return cls(
            risk=load_artifact(risk_path, expected_task="risk"),
            policy=load_artifact(policy_path, expected_task="policy"),
            weather=(
                load_weather_artifact(weather_path)
                if weather_path is not None and Path(weather_path).is_file()
                else None
            ),
            sensor_reliability=reliability_models,
        )

    @property
    def version(self) -> str:
        risk_version = str(self.risk.get("model_version", "unknown"))
        policy_version = str(self.policy.get("model_version", "unknown"))
        weather_version = (
            str(self.weather.get("model_version", "unknown"))
            if self.weather is not None
            else "fallback"
        )
        sensor_versions = ",".join(
            "{}={}".format(name.lower(), model.get("model_version", "unknown"))
            for name, model in sorted(self.sensor_reliability.items())
        ) or "fallback"
        return "risk={};policy={};weather={};sensors={}".format(
            risk_version, policy_version, weather_version, sensor_versions
        )


@dataclass
class _Session:
    history: Deque[Dict[str, float]] = field(default_factory=deque)
    last_sequence: int = -1
    pending_lane_action: Optional[str] = None
    pending_lane_votes: int = 0
    last_seen_monotonic: float = field(default_factory=time.monotonic)


class TestLabInferenceEngine:
    """Stateful, transport-independent Test Lab prediction engine."""

    def __init__(
        self,
        models: TestLabModelBundle,
        history_seconds: float = 3.0,
        confidence_threshold: float = 0.45,
        lane_confirmation_votes: int = 3,
        maximum_sessions: int = 32,
    ) -> None:
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be greater than zero")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if lane_confirmation_votes < 1:
            raise ValueError("lane_confirmation_votes must be at least one")

        self.models = models
        self.history_seconds = history_seconds
        self.confidence_threshold = confidence_threshold
        self.lane_confirmation_votes = lane_confirmation_votes
        self.maximum_sessions = max(1, maximum_sessions)
        self.sessions: Dict[str, _Session] = {}

    def handle(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = str(message.get("sessionId", "")).strip()
        sequence = int(_finite_number(message.get("sequenceNumber"), -1.0))

        try:
            self._validate_envelope(message, session_id)
            message_type = str(message.get("messageType", "")).lower()
            if message_type == RESET_MESSAGE:
                self.reset(session_id)
                return self._response(
                    session_id,
                    sequence,
                    valid=True,
                    message_type="reset_ack",
                    process_started=started,
                )
            if message_type != OBSERVATION_MESSAGE:
                raise ValueError("unsupported messageType {!r}".format(message_type))
            return self._predict(message, session_id, sequence, started)
        except Exception as error:
            return self._response(
                session_id,
                sequence,
                valid=False,
                message_type="error",
                error=str(error),
                process_started=started,
            )

    def reset(self, session_id: str) -> None:
        if session_id:
            self.sessions[session_id] = _Session()
            self._trim_sessions()

    def _validate_envelope(self, message: Mapping[str, Any], session_id: str) -> None:
        if message.get("schemaVersion") != PROTOCOL_VERSION:
            raise ValueError(
                "unsupported schemaVersion {!r}; expected {!r}".format(
                    message.get("schemaVersion"),
                    PROTOCOL_VERSION,
                )
            )
        if not session_id:
            raise ValueError("sessionId is required")

    def _predict(
        self,
        message: Mapping[str, Any],
        session_id: str,
        sequence: int,
        process_started: float,
    ) -> Dict[str, Any]:
        if sequence < 0:
            raise ValueError("sequenceNumber must be zero or greater")

        session = self.sessions.setdefault(session_id, _Session())
        if sequence <= session.last_sequence:
            raise ValueError(
                "out-of-order sequence {}; last accepted sequence is {}".format(
                    sequence,
                    session.last_sequence,
                )
            )

        state = self._build_state(message)
        timestamp = state["timestamp"]
        session.history.append(state)
        cutoff = timestamp - self.history_seconds
        while len(session.history) > 1 and session.history[0]["timestamp"] < cutoff:
            session.history.popleft()
        session.last_sequence = sequence
        session.last_seen_monotonic = time.monotonic()

        summary = summarize_history(session.history)
        risk, risk_confidence, _ = predict_one(self.models.risk, summary)
        requested_action, action_confidence, _ = predict_one(self.models.policy, summary)

        weather_context = normalize_condition(message.get("weather", "Dry"))
        weather_model_used = (
            self.models.weather is not None and weather_context != "DRY"
        )
        weather_speed_factor = 1.0
        weather_target_speed_mps = max(
            0.0, _finite_number(message.get("cruiseSpeedMps"), 0.0)
        )
        if weather_model_used:
            assert self.models.weather is not None
            weather_summary = summarize_weather_history(
                session.history,
                weather_context,
            )
            weather_speed_factor = predict_weather_factor(
                self.models.weather,
                weather_summary,
                weather_context,
            )
            weather_reference_kph = max(
                1.0,
                _finite_number(self.models.weather.get("reference_speed_kph"), 50.0),
            )
            weather_target_speed_mps = (
                weather_speed_factor * weather_reference_kph / 3.6
            )

        executed_action, override_reason = self._supervise_model_decision(
            session,
            message,
            requested_action,
            action_confidence,
            risk,
        )
        target_speed = self._target_speed(message, executed_action)
        sensor_reliability, overall_sensor_reliability, sensor_safety_mode = (
            self._predict_sensor_reliability(message)
        )

        return self._response(
            session_id,
            sequence,
            valid=True,
            message_type="decision",
            process_started=process_started,
            observation_timestamp=_finite_number(message.get("timestampSeconds")),
            requested_action=requested_action,
            executed_action=executed_action,
            action_confidence=action_confidence,
            risk_level=risk,
            risk_confidence=risk_confidence,
            target_speed_mps=target_speed,
            override_reason=override_reason,
            history_samples=len(session.history),
            weather_context=weather_context.title(),
            weather_model_used=weather_model_used,
            weather_speed_factor=weather_speed_factor,
            weather_target_speed_mps=weather_target_speed_mps,
            sensor_reliability=sensor_reliability,
            overall_sensor_reliability=overall_sensor_reliability,
            sensor_safety_mode=sensor_safety_mode,
        )

    def _predict_sensor_reliability(
        self, message: Mapping[str, Any]
    ) -> Tuple[list, float, str]:
        raw_health = message.get("sensorHealth", [])
        if not isinstance(raw_health, list) or not self.models.sensor_reliability:
            return [], 1.0, "UNAVAILABLE"

        camel_to_snake = {
            "dropoutRate": "dropout_rate",
            "messageAgeMean": "message_age_mean",
            "messageAgeMax": "message_age_max",
            "detectionCountMean": "detection_count_mean",
            "detectionCountStd": "detection_count_std",
            "confidenceMean": "confidence_mean",
            "confidenceStd": "confidence_std",
            "trackContinuity": "track_continuity",
            "rangeVariance": "range_variance",
            "velocityVariance": "velocity_variance",
            "innovationMean": "innovation_mean",
            "innovationStd": "innovation_std",
            "crossSensorDisagreement": "cross_sensor_disagreement",
            "egoSpeedMean": "ego_speed_mean",
            "egoSpeedStd": "ego_speed_std",
            "yawRateMean": "yaw_rate_mean",
            "yawRateStd": "yaw_rate_std",
        }
        decisions = []
        values = []
        for item in raw_health:
            if not isinstance(item, Mapping):
                continue
            sensor_type = str(item.get("sensorType", "")).strip().upper()
            artifact = self.models.sensor_reliability.get(sensor_type)
            if artifact is None:
                continue
            observation = {
                snake: item.get(camel)
                for camel, snake in camel_to_snake.items()
            }
            observation["weather"] = item.get("weather", message.get("weather", "Dry"))
            result = predict_sensor_reliability(artifact, observation)
            reliability = float(result["reliability"])
            values.append(reliability)
            decisions.append(
                {
                    "sensorId": str(item.get("sensorId", sensor_type.lower())),
                    "sensorType": sensor_type,
                    "reliability": reliability,
                    "status": str(result["status"]),
                }
            )

        if not values:
            return [], 1.0, "UNAVAILABLE"

        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        weakest = ordered[0]
        # The median rewards sensor redundancy; the smaller weakest-sensor term
        # still makes a single failed channel visible as a cautious condition.
        overall = max(0.0, min(1.0, 0.75 * median + 0.25 * weakest))
        healthy = sum(value >= 0.70 for value in values)
        usable = sum(value >= 0.40 for value in values)
        if healthy == len(values):
            safety_mode = "NORMAL"
        elif healthy >= 2:
            safety_mode = "CAUTIOUS"
        elif usable >= 1:
            safety_mode = "RESTRICTED"
        else:
            safety_mode = "MINIMAL_RISK"
        return decisions, overall, safety_mode

    def _build_state(self, message: Mapping[str, Any]) -> Dict[str, float]:
        ego = message.get("ego")
        if not isinstance(ego, Mapping):
            raise ValueError("ego object is required")

        timestamp = _finite_number(message.get("timestampSeconds"), math.nan)
        if not math.isfinite(timestamp):
            raise ValueError("timestampSeconds must be finite")

        state = new_state(
            timestamp,
            _finite_number(ego.get("speedMps")),
            _finite_number(ego.get("accelerationMps2")),
            _finite_number(ego.get("yawRateDegreesPerSecond")),
        )

        current_lane = str(message.get("currentLane", "RIGHT")).upper()
        on_left = current_lane == "LEFT"
        right_front = message.get("rightFront")
        right_rear = message.get("rightRear")
        left_front = message.get("leftFront")
        left_rear = message.get("leftRear")

        # The trained contract is ego-relative. Unity reports two absolute road
        # lanes, so remap the current lane to `front` and only expose the lane
        # that is actually adjacent to the ego vehicle.
        slots = {
            "front": left_front if on_left else right_front,
            "left_front": None if on_left else left_front,
            "left_rear": None if on_left else left_rear,
            "right_front": right_front if on_left else None,
            "right_rear": right_rear if on_left else None,
            "pedestrian": message.get("pedestrian"),
        }
        for name, slot in slots.items():
            if not _present(slot):
                continue
            assert isinstance(slot, Mapping)
            set_slot(
                state,
                name,
                _finite_number(slot.get("gapMeters")),
                _finite_number(slot.get("closingSpeedMps")),
                _finite_number(slot.get("actorSpeedMps")),
            )

        state["left_clear"] = float(
            not on_left and bool(message.get("leftLaneClear", False))
        )
        state["right_clear"] = float(
            on_left and bool(message.get("rightLaneClear", False))
        )
        return state

    def _supervise_model_decision(
        self,
        session: _Session,
        message: Mapping[str, Any],
        requested_action: str,
        action_confidence: float,
        risk: str,
    ) -> Tuple[str, str]:
        emergency_reason = self._emergency_reason(message)
        if emergency_reason:
            session.pending_lane_action = None
            session.pending_lane_votes = 0
            return EMERGENCY_STOP, emergency_reason

        reason = ""
        if action_confidence < self.confidence_threshold:
            candidate = DECELERATE
            reason = "policy confidence below {:.2f}".format(self.confidence_threshold)
        elif risk == "EXTREME" and requested_action != DECELERATE:
            candidate = DECELERATE
            reason = "extreme-risk conservative override"
        elif risk == "HIGH" and requested_action == ACCELERATE:
            candidate = DECELERATE
            reason = "high-risk acceleration rejected"
        else:
            candidate = requested_action

        current_lane = str(message.get("currentLane", "RIGHT")).upper()
        if candidate == CHANGE_LEFT:
            if current_lane == "LEFT":
                return KEEP, "already in left lane"
            if not bool(message.get("leftLaneClear", False)):
                return DECELERATE, "left lane is not clear"
        elif candidate == CHANGE_RIGHT:
            if current_lane == "RIGHT":
                return KEEP, "already in right lane"
            if not bool(message.get("rightLaneClear", False)):
                return DECELERATE, "right lane is not clear"

        if candidate in (CHANGE_LEFT, CHANGE_RIGHT):
            if session.pending_lane_action == candidate:
                session.pending_lane_votes += 1
            else:
                session.pending_lane_action = candidate
                session.pending_lane_votes = 1
            if session.pending_lane_votes < self.lane_confirmation_votes:
                return DECELERATE, "confirming {} ({}/{})".format(
                    candidate,
                    session.pending_lane_votes,
                    self.lane_confirmation_votes,
                )
            session.pending_lane_action = None
            session.pending_lane_votes = 0
            return candidate, reason

        session.pending_lane_action = None
        session.pending_lane_votes = 0
        return candidate, reason

    @staticmethod
    def _emergency_reason(message: Mapping[str, Any]) -> str:
        ego = message.get("ego")
        ego_speed = _finite_number(ego.get("speedMps")) if isinstance(ego, Mapping) else 0.0
        pedestrian = message.get("pedestrian")
        if _present(pedestrian):
            assert isinstance(pedestrian, Mapping)
            gap = max(0.0, _finite_number(pedestrian.get("gapMeters")))
            ttc = _finite_number(pedestrian.get("timeToCollisionSeconds"), 20.0)
            stopping_gap = ego_speed * ego_speed / (2.0 * 4.8) + 5.0
            if gap <= stopping_gap or ttc < 2.0:
                return "pedestrian emergency envelope"

        current_lane = str(message.get("currentLane", "RIGHT")).upper()
        front = message.get("leftFront") if current_lane == "LEFT" else message.get("rightFront")
        if _present(front):
            assert isinstance(front, Mapping)
            gap = max(0.0, _finite_number(front.get("gapMeters")))
            closing = max(0.0, _finite_number(front.get("closingSpeedMps")))
            ttc = _finite_number(front.get("timeToCollisionSeconds"), 20.0)
            stopping_gap = closing * closing / (2.0 * 4.8) + 2.5
            if gap <= max(4.0, stopping_gap) or ttc < 1.5:
                return "front-object emergency envelope"
        return ""

    @staticmethod
    def _target_speed(message: Mapping[str, Any], action: str) -> float:
        ego = message.get("ego")
        speed = _finite_number(ego.get("speedMps")) if isinstance(ego, Mapping) else 0.0
        cruise = max(0.0, _finite_number(message.get("cruiseSpeedMps"), 8.33))
        if action == EMERGENCY_STOP:
            return 0.0
        if action == DECELERATE:
            # A learned risk classification is advisory, not proof that the
            # vehicle must stop. Repeated 5 Hz DECELERATE decisions used to
            # subtract 3 m/s each time until an EXTREME prediction parked the
            # car for several seconds. Keep a useful cautious-speed floor and
            # reserve zero speed for the deterministic emergency envelopes
            # above (pedestrian/front-object danger).
            cautious_speed = min(cruise, max(2.0, cruise * 0.55))
            return min(cruise, max(cautious_speed, speed - 1.0))
        if action == ACCELERATE:
            return min(cruise, speed + 2.0)
        if action in (CHANGE_LEFT, CHANGE_RIGHT):
            return min(cruise, max(speed, min(8.0, cruise)))
        return cruise

    def _trim_sessions(self) -> None:
        if len(self.sessions) <= self.maximum_sessions:
            return
        oldest = min(
            self.sessions.items(),
            key=lambda item: item[1].last_seen_monotonic,
        )[0]
        self.sessions.pop(oldest, None)

    def _response(
        self,
        session_id: str,
        sequence: int,
        valid: bool,
        message_type: str,
        process_started: float,
        **values: Any,
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "schemaVersion": PROTOCOL_VERSION,
            "messageType": message_type,
            "sessionId": session_id,
            "sequenceNumber": sequence,
            "valid": valid,
            "modelVersion": self.models.version,
            "processDurationMilliseconds": round(
                (time.perf_counter() - process_started) * 1000.0,
                3,
            ),
        }
        camel_names = {
            "observation_timestamp": "observationTimestampSeconds",
            "requested_action": "requestedAction",
            "executed_action": "executedAction",
            "action_confidence": "actionConfidence",
            "risk_level": "riskLevel",
            "risk_confidence": "riskConfidence",
            "target_speed_mps": "targetSpeedMps",
            "override_reason": "overrideReason",
            "history_samples": "historySamples",
            "weather_context": "weatherContext",
            "weather_model_used": "weatherModelUsed",
            "weather_speed_factor": "weatherSpeedFactor",
            "weather_target_speed_mps": "weatherTargetSpeedMps",
            "sensor_reliability": "sensorReliability",
            "overall_sensor_reliability": "overallSensorReliability",
            "sensor_safety_mode": "sensorSafetyMode",
        }
        for key, value in values.items():
            response[camel_names.get(key, key)] = value
        return response


def default_model_paths(project_root: Path) -> Tuple[Path, Path, Path]:
    root = Path(project_root).expanduser().resolve()
    return (
        root / "ML" / "models" / "risk_model.joblib",
        root / "ML" / "models" / "policy_model.joblib",
        root / "ML" / "models" / "weather_model.joblib",
    )


def default_sensor_model_paths(project_root: Path) -> Dict[str, Path]:
    root = Path(project_root).expanduser().resolve()
    model_root = root / "ML" / "models"
    return {
        "CAMERA": model_root / "sensor_camera_reliability_model.joblib",
        "LIDAR": model_root / "sensor_lidar_reliability_model.joblib",
        "RADAR": model_root / "sensor_radar_reliability_model.joblib",
    }
