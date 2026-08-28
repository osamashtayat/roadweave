"""Canonical, source-neutral labels for RoadWeave driving models.

Both raw-data adapters call this module.  The observation history is used as
model input; measurements after the observation time are used only to create
the supervised target.  This keeps nuScenes and K-Risk on the same physical
task and prevents look-ahead leakage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

try:
    from ML.src.labels import ACTION_NAMES, RISK_NAMES, DrivingAction, RiskLevel
except ModuleNotFoundError:
    from labels import ACTION_NAMES, RISK_NAMES, DrivingAction, RiskLevel  # type: ignore


POLICY_SPEED_CHANGE_THRESHOLD_MPS = 0.75
# A complete lane change is roughly 3.5 m over 4-6 seconds.  On the common
# one-second future horizon, about 0.45 m of curvature-normalized lateral
# travel is a meaningful maneuver (and is the time-scaled equivalent of the
# previous 1.4 m / 3 s threshold).
POLICY_LATERAL_DISPLACEMENT_THRESHOLD_METRES = 0.45


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _values(
    states: Iterable[Mapping[str, float]],
    field: str,
) -> list[float]:
    return [
        float(state[field])
        for state in states
        if field in state and _finite(state[field])
    ]


@dataclass(frozen=True)
class RiskEvidence:
    """Auditable physical evidence used to assign one risk target."""

    collision_or_overlap: bool
    minimum_ttc_seconds: Optional[float]
    minimum_vehicle_gap_metres: Optional[float]
    minimum_pedestrian_gap_metres: Optional[float]
    maximum_required_deceleration_mps2: float
    maximum_absolute_acceleration_mps2: float


def risk_evidence(
    future_states: Sequence[Mapping[str, float]],
    overlap_flags: Optional[Sequence[bool]] = None,
) -> RiskEvidence:
    """Summarize future physical outcomes without using dataset-native labels."""

    if not future_states:
        raise ValueError("At least one future/current state is required for a risk label.")

    vehicle_ttc: list[float] = []
    vehicle_gaps: list[float] = []
    pedestrian_ttc: list[float] = []
    pedestrian_gaps: list[float] = []
    required_decelerations: list[float] = []

    for state in future_states:
        # Risk is evaluated in the ego's current path. Adjacent-lane vehicles
        # remain policy inputs for lane-change decisions but are not immediate
        # collision threats merely because their longitudinal bumper gap is 0.
        for slot in ("front",):
            if float(state.get(f"{slot}_present", 0.0)) < 0.5:
                continue
            gap = state.get(f"{slot}_gap")
            closing = state.get(f"{slot}_closing_speed")
            ttc = state.get(f"{slot}_ttc")
            if _finite(gap):
                vehicle_gaps.append(max(0.0, float(gap)))
            if _finite(ttc) and _finite(closing) and float(closing) > 0.05:
                vehicle_ttc.append(max(0.0, float(ttc)))
            if _finite(gap) and _finite(closing):
                safe_gap = max(0.10, float(gap))
                closing_speed = max(0.0, float(closing))
                required_decelerations.append(
                    closing_speed * closing_speed / (2.0 * safe_gap)
                )

        if float(state.get("pedestrian_present", 0.0)) >= 0.5:
            gap = state.get("pedestrian_gap")
            closing = state.get("pedestrian_closing_speed")
            ttc = state.get("pedestrian_ttc")
            if _finite(gap):
                pedestrian_gaps.append(max(0.0, float(gap)))
            if _finite(ttc) and _finite(closing) and float(closing) > 0.05:
                pedestrian_ttc.append(max(0.0, float(ttc)))

    accelerations = [abs(value) for value in _values(future_states, "ego_accel")]
    all_ttc = vehicle_ttc + pedestrian_ttc
    return RiskEvidence(
        collision_or_overlap=(
            any(bool(value) for value in (overlap_flags or ()))
        ),
        minimum_ttc_seconds=min(all_ttc) if all_ttc else None,
        minimum_vehicle_gap_metres=min(vehicle_gaps) if vehicle_gaps else None,
        minimum_pedestrian_gap_metres=(
            min(pedestrian_gaps) if pedestrian_gaps else None
        ),
        maximum_required_deceleration_mps2=(
            max(required_decelerations) if required_decelerations else 0.0
        ),
        maximum_absolute_acceleration_mps2=max(accelerations) if accelerations else 0.0,
    )


def risk_label_from_evidence(evidence: RiskEvidence) -> str:
    """Map identical physical thresholds to the four RoadWeave risk classes."""

    ttc = evidence.minimum_ttc_seconds
    pedestrian_gap = evidence.minimum_pedestrian_gap_metres
    required_decel = evidence.maximum_required_deceleration_mps2
    absolute_accel = evidence.maximum_absolute_acceleration_mps2

    if evidence.collision_or_overlap:
        return RISK_NAMES[RiskLevel.EXTREME]
    if ttc is not None and ttc <= 1.0:
        return RISK_NAMES[RiskLevel.EXTREME]
    if pedestrian_gap is not None and pedestrian_gap <= 2.0:
        return RISK_NAMES[RiskLevel.EXTREME]
    if required_decel >= 6.0:
        return RISK_NAMES[RiskLevel.EXTREME]

    if ttc is not None and ttc <= 2.0:
        return RISK_NAMES[RiskLevel.HIGH]
    if pedestrian_gap is not None and pedestrian_gap <= 5.0:
        return RISK_NAMES[RiskLevel.HIGH]
    if required_decel >= 4.0 or absolute_accel >= 4.0:
        return RISK_NAMES[RiskLevel.HIGH]

    if ttc is not None and ttc <= 4.0:
        return RISK_NAMES[RiskLevel.MODERATE]
    if pedestrian_gap is not None and pedestrian_gap <= 12.0:
        return RISK_NAMES[RiskLevel.MODERATE]
    if required_decel >= 2.5 or absolute_accel >= 2.5:
        return RISK_NAMES[RiskLevel.MODERATE]

    return RISK_NAMES[RiskLevel.LOW]


def risk_label_from_future(
    future_states: Sequence[Mapping[str, float]],
    overlap_flags: Optional[Sequence[bool]] = None,
) -> str:
    return risk_label_from_evidence(risk_evidence(future_states, overlap_flags))


def policy_label_from_future_motion(
    speed_change_mps: float,
    lateral_displacement_metres: float,
) -> str:
    """Label the action the ego actually performs after the observation."""

    if lateral_displacement_metres > POLICY_LATERAL_DISPLACEMENT_THRESHOLD_METRES:
        return ACTION_NAMES[DrivingAction.CHANGE_LEFT]
    if lateral_displacement_metres < -POLICY_LATERAL_DISPLACEMENT_THRESHOLD_METRES:
        return ACTION_NAMES[DrivingAction.CHANGE_RIGHT]
    if speed_change_mps > POLICY_SPEED_CHANGE_THRESHOLD_MPS:
        return ACTION_NAMES[DrivingAction.ACCELERATE]
    if speed_change_mps < -POLICY_SPEED_CHANGE_THRESHOLD_MPS:
        return ACTION_NAMES[DrivingAction.DECELERATE]
    return ACTION_NAMES[DrivingAction.KEEP]
