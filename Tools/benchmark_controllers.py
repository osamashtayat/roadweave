#!/usr/bin/env python3
"""Compare the rule and learned controllers on the procedural simulator.

Runs the same scenario seeds through both controllers and reports safety and
comfort metrics so the learned controller can be judged against the
deterministic rule baseline on clearance, time-to-collision, interventions,
deadlocks, route progress, and comfort.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "Tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from simulated_twin_stream import (  # noqa: E402
    AutonomousDrivingController,
    CRUISE_SPEED_MPS,
    LEFT_LANE_X,
    RIGHT_LANE_X,
    SimulatedWorld,
)


def _finite_or_none(value: float) -> Any:
    return value if math.isfinite(value) else None


@dataclass
class StepStats:
    distance_m: float = 0.0
    min_clearance_m: float = math.inf
    min_ttc_s: float = math.inf
    emergency_stops: int = 0
    deadlock_seconds: float = 0.0
    abs_accel_sum: float = 0.0
    abs_accel_max: float = 0.0
    abs_yaw_sum: float = 0.0
    abs_yaw_max: float = 0.0
    samples: int = 0


def _front_ttc(world: SimulatedWorld) -> float:
    ttc = math.inf
    frame = world.latest_sensors
    for hit in (frame.current_front, frame.pedestrian_hazard):
        if hit is not None and math.isfinite(hit.time_to_collision):
            ttc = min(ttc, hit.time_to_collision)
    return ttc


def simulate_controller(
    world: SimulatedWorld,
    sim_seconds: float,
    dt: float,
) -> StepStats:
    stats = StepStats()
    previous_behavior = ""
    for _ in range(int(round(sim_seconds / dt))):
        world.update(dt)
        stats.samples += 1
        stats.distance_m += world.speed_mps * dt
        stats.min_clearance_m = min(
            stats.min_clearance_m,
            world.minimum_observed_clearance,
        )
        stats.min_ttc_s = min(stats.min_ttc_s, _front_ttc(world))

        behavior = getattr(world.controller, "behavior", "")
        if "EMERGENCY" in behavior and "EMERGENCY" not in previous_behavior:
            stats.emergency_stops += 1
        previous_behavior = behavior

        front = world.latest_sensors.current_front
        blocked = front is not None and front.longitudinal_gap < 15.0
        if world.speed_mps < 0.3 and blocked:
            stats.deadlock_seconds += dt

        accel = abs(world.acceleration_mps2)
        stats.abs_accel_sum += accel
        stats.abs_accel_max = max(stats.abs_accel_max, accel)
        yaw = abs(world.ego_yaw_rate)
        stats.abs_yaw_sum += yaw
        stats.abs_yaw_max = max(stats.abs_yaw_max, yaw)
    return stats


def summarize(name: str, runs: List[StepStats]) -> Dict[str, Any]:
    def mean(values: List[float]) -> float:
        return sum(values) / len(values)

    return {
        "controller": name,
        "runs": len(runs),
        "mean_distance_m": round(mean([r.distance_m for r in runs]), 2),
        "min_clearance_m": _finite_or_none(
            min([r.min_clearance_m for r in runs])
        ),
        "min_ttc_s": _finite_or_none(min([r.min_ttc_s for r in runs])),
        "mean_emergency_stops": round(
            mean([r.emergency_stops for r in runs]), 2
        ),
        "mean_deadlock_seconds": round(
            mean([r.deadlock_seconds for r in runs]), 2
        ),
        "mean_abs_accel_mps2": round(
            mean([r.abs_accel_sum / r.samples for r in runs]), 3
        ),
        "max_abs_accel_mps2": round(max([r.abs_accel_max for r in runs]), 3),
        "mean_abs_yaw_deg_s": round(
            mean([r.abs_yaw_sum / r.samples for r in runs]), 3
        ),
        "max_abs_yaw_deg_s": round(max([r.abs_yaw_max for r in runs]), 3),
    }


def build_ml_factory(model_root: Path) -> Callable[[], Any]:
    try:
        from ML.src.online_policy import OnlineModelBundle, RoadWeaveMLController
    except ModuleNotFoundError as error:
        raise SystemExit("ML dependencies unavailable: {}".format(error))

    bundle = OnlineModelBundle.load(
        model_root / "risk_model.joblib",
        model_root / "policy_model.joblib",
    )

    def factory() -> Any:
        return RoadWeaveMLController(
            bundle,
            right_lane_x=RIGHT_LANE_X,
            left_lane_x=LEFT_LANE_X,
            cruise_speed_mps=CRUISE_SPEED_MPS,
            decision_rate_hz=5.0,
        )

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "ML" / "models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ML" / "reports" / "controller_benchmark.json",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Run only the rule controller (useful before models exist).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(2026000, 2026000 + args.seeds))
    report: Dict[str, Any] = {
        "seconds": args.seconds,
        "dt": args.dt,
        "seeds": seeds,
        "controllers": {},
    }

    controllers: List[Tuple[str, Callable[[], Any]]] = [
        ("rule", lambda: AutonomousDrivingController()),
    ]
    if not args.rule_only:
        controllers.append(("ml", build_ml_factory(args.model_root)))

    for name, factory in controllers:
        runs: List[StepStats] = []
        for seed in seeds:
            world = SimulatedWorld(
                include_actors=True,
                configured_seed=seed,
                encounter_count=7,
                announce=False,
                controller_factory=factory,
            )
            runs.append(simulate_controller(world, args.seconds, args.dt))
        report["controllers"][name] = summarize(name, runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["controllers"], indent=2))
    print("Report: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
