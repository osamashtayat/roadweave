#!/usr/bin/env python3
"""Paired closed-loop benchmark for RoadWeave's rule and hybrid ML controllers.

Every seed creates one immutable scenario manifest. Both controllers receive a
deep copy of that manifest, the same procedural road, the same vehicle model,
and the same simulation duration. Raw evidence is saved per run and per
scenario before paper figures are produced.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "Tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from simulated_twin_stream import (  # noqa: E402
    AutonomousDrivingController,
    CRUISE_SPEED_MPS,
    EGO_LENGTH_METERS,
    EGO_WIDTH_METERS,
    LEFT_LANE_X,
    RIGHT_LANE_X,
    ProceduralScenarioPlanner,
    SimulatedActor,
    SimulatedWorld,
)

RULE = "Rule Controller"
HYBRID = "Hybrid ML Controller"
SCENARIO_SENSOR_RANGE_M = 75.0
SCENARIO_RESOLUTION_ALLOWANCE_S = 30.0
SCENARIO_DEADLOCK_LIMIT_S = 5.0
COMFORT_WARMUP_S = 2.0
BENCHMARK_VERSION = 3
REPRODUCIBILITY_FILES = (
    PROJECT_ROOT / "Tools" / "benchmark_controllers.py",
    PROJECT_ROOT / "Tools" / "analyze_experiment4.py",
    PROJECT_ROOT / "Tools" / "simulated_twin_stream.py",
    PROJECT_ROOT / "ML" / "src" / "online_policy.py",
)


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    scenario_type: str
    category: str
    primary_actor_id: str
    actor_ids: Tuple[str, ...]
    mixed: bool
    route_position_m: float


@dataclass(frozen=True)
class ScenarioManifest:
    seed: int
    manifest_id: str
    actors: Tuple[SimulatedActor, ...]
    scenarios: Tuple[ScenarioDefinition, ...]


@dataclass
class ScenarioState:
    definition: ScenarioDefinition
    encountered_at_s: float = math.nan
    resolved_at_s: float = math.nan
    minimum_ttc_s: float = math.inf
    minimum_clearance_m: float = math.inf
    deadlock_duration_s: float = 0.0
    collision: bool = False
    overtake_attempted: bool = False
    overtake_successful: bool = False


@dataclass
class RunResult:
    controller: str
    seed: int
    manifest_id: str
    target_duration_s: float
    actual_duration_s: float
    simulation_steps: int
    route_progress_m: float
    collision_count: int
    collision_intervention_frames: int
    minimum_ttc_s: float
    minimum_clearance_m: float
    emergency_stops: int
    deadlock_duration_s: float
    overtake_attempts: int
    successful_overtakes: int
    overtake_success_rate: float
    mean_absolute_acceleration_mps2: float
    maximum_absolute_jerk_mps3: float
    mean_absolute_yaw_rate_deg_s: float
    scenarios_encountered: int
    scenarios_evaluable: int
    scenarios_completed: int
    scenario_successes: int
    scenario_completion_rate: float
    run_completed: bool
    error: str = ""


@dataclass
class OvertakeTracker:
    attempted_actor_ids: Set[str] = field(default_factory=set)
    successful_actor_ids: Set[str] = field(default_factory=set)
    active_actor_id: Optional[str] = None
    previous_target_lane_x: float = RIGHT_LANE_X

    def update(self, world: SimulatedWorld) -> None:
        controller = world.controller
        target_lane = float(getattr(controller, "target_lane_x", self.previous_target_lane_x))
        lane_commitment = abs(target_lane - self.previous_target_lane_x) > 2.0
        explicit_actor = getattr(controller, "overtake_actor_id", None)
        front = world.latest_sensors.current_front
        candidate = explicit_actor
        if candidate is None and lane_commitment and front is not None:
            if front.actor.speed_mps < CRUISE_SPEED_MPS - 1.0 and front.longitudinal_gap < 50.0:
                candidate = front.actor.actor_id

        if candidate and candidate not in self.attempted_actor_ids:
            self.attempted_actor_ids.add(candidate)
            self.active_actor_id = candidate
        elif candidate:
            self.active_actor_id = candidate

        if self.active_actor_id:
            actor = _find_actor(world, self.active_actor_id)
            if actor is None:
                self.active_actor_id = None
            else:
                passed_by = world.ego_z - (actor.z + actor.length * 0.5)
                in_lane = min(abs(world.ego_x - RIGHT_LANE_X), abs(world.ego_x - LEFT_LANE_X)) <= 0.4
                if passed_by >= 10.0 and in_lane:
                    self.successful_actor_ids.add(actor.actor_id)
                    self.active_actor_id = None
        self.previous_target_lane_x = target_lane


def _find_actor(world: SimulatedWorld, actor_id: str) -> Optional[SimulatedActor]:
    actor = world.actors_by_id.get(actor_id)
    if actor is not None:
        return actor
    return getattr(world, "benchmark_all_actors_by_id", {}).get(actor_id)


def _safe_number(value: float) -> Any:
    return value if math.isfinite(value) else ""


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scenario_manifest(seed: int, sim_seconds: float) -> ScenarioManifest:
    """Pre-generate the complete road exposure before either controller runs."""
    planner = ProceduralScenarioPlanner(seed)
    next_position = planner.random.uniform(58.0, 75.0)
    maximum_position = CRUISE_SPEED_MPS * sim_seconds + 250.0
    actors: List[SimulatedActor] = []
    scenarios: List[ScenarioDefinition] = []
    while next_position <= maximum_position:
        built = planner.build_next(next_position)
        primary = built[0]
        mixed = len(built) > 1
        relevant = primary.is_pedestrian or abs(primary.lane_x - RIGHT_LANE_X) < 0.5
        if relevant:
            scenario_type = primary.kind
            category = "mixed" if mixed else scenario_type
            scenario_id = f"seed-{seed}-scenario-{planner.next_index - 1:05d}"
            scenarios.append(
                ScenarioDefinition(
                    scenario_id=scenario_id,
                    scenario_type=scenario_type,
                    category=category,
                    primary_actor_id=primary.actor_id,
                    actor_ids=tuple(actor.actor_id for actor in built),
                    mixed=mixed,
                    route_position_m=next_position,
                )
            )
        actors.extend(built)
        next_position += planner.next_spacing()

    canonical = {
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "type": scenario.scenario_type,
                "mixed": scenario.mixed,
                "position": round(scenario.route_position_m, 6),
                "actors": list(scenario.actor_ids),
            }
            for scenario in scenarios
        ],
        "all_actors": [
            {
                "id": actor.actor_id,
                "kind": actor.kind,
                "lane_x": round(actor.lane_x, 6),
                "z": round(actor.z, 6),
                "speed": round(actor.speed_mps, 6),
            }
            for actor in actors
        ],
    }
    manifest_id = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return ScenarioManifest(seed, manifest_id, tuple(actors), tuple(scenarios))


def install_manifest(world: SimulatedWorld, manifest: ScenarioManifest) -> Dict[str, SimulatedActor]:
    actors = copy.deepcopy(list(manifest.actors))
    actor_map = {actor.actor_id: actor for actor in actors}
    world.actors = actors
    world.actors_by_id = dict(actor_map)
    world.benchmark_all_actors_by_id = actor_map
    world.generated_encounter_count = len(manifest.scenarios)
    world.next_event_z = math.inf
    return actor_map


def _signed_clearance(world: SimulatedWorld, actor: SimulatedActor) -> float:
    lateral = abs(world.ego_x - actor.x) - (EGO_WIDTH_METERS + actor.width) * 0.5
    longitudinal = abs(world.ego_z - actor.z) - (EGO_LENGTH_METERS + actor.length) * 0.5
    if lateral <= 0.0 and longitudinal <= 0.0:
        return max(lateral, longitudinal)
    if lateral <= 0.0:
        return longitudinal
    if longitudinal <= 0.0:
        return lateral
    return math.hypot(lateral, longitudinal)


def _actor_ttc(world: SimulatedWorld, actor: SimulatedActor) -> float:
    if actor.cleared:
        return math.inf
    lateral_limit = (EGO_WIDTH_METERS + actor.width) * 0.5 + 0.5
    if abs(world.ego_x - actor.x) > lateral_limit:
        return math.inf
    center_gap = actor.z - world.ego_z
    if center_gap <= 0.0:
        return math.inf
    actor_forward_speed = 0.0 if actor.is_pedestrian else actor.speed_mps
    closing_speed = world.speed_mps - actor_forward_speed
    return center_gap / closing_speed if closing_speed > 0.05 else math.inf


def _scenario_resolved(world: SimulatedWorld, primary: SimulatedActor) -> bool:
    if primary.is_pedestrian:
        return primary.cleared and world.ego_z >= primary.z + 15.0
    return world.ego_z >= primary.z + primary.length * 0.5 + 10.0


def _scenario_deadlocked(world: SimulatedWorld, primary: SimulatedActor) -> bool:
    if world.speed_mps >= 0.3:
        return False
    gap = primary.z - world.ego_z
    if primary.is_pedestrian:
        return primary.cleared and -2.0 <= gap <= 20.0
    return 0.0 <= gap <= 15.0


def _front_ttc(world: SimulatedWorld) -> float:
    values = []
    for hit in (world.latest_sensors.current_front, world.latest_sensors.pedestrian_hazard):
        if hit is not None and math.isfinite(hit.time_to_collision):
            values.append(float(hit.time_to_collision))
    return min(values) if values else math.inf


def _global_clearance(world: SimulatedWorld) -> float:
    values = [
        _signed_clearance(world, actor)
        for actor in world.actors
        if actor.active and not actor.cleared and abs(actor.z - world.ego_z) <= 100.0
    ]
    return min(values) if values else math.inf


def simulate_controller(
    controller_name: str,
    controller_factory: Callable[[], Any],
    manifest: ScenarioManifest,
    sim_seconds: float,
    dt: float,
) -> Tuple[RunResult, List[Dict[str, Any]]]:
    world = SimulatedWorld(
        include_actors=True,
        configured_seed=manifest.seed,
        encounter_count=7,
        announce=False,
        controller_factory=controller_factory,
    )
    all_actors = install_manifest(world, manifest)
    states = {item.scenario_id: ScenarioState(item) for item in manifest.scenarios}
    overtakes = OvertakeTracker()
    emergency_stops = 0
    deadlock_seconds = 0.0
    minimum_ttc = math.inf
    minimum_clearance = math.inf
    absolute_accelerations: List[float] = []
    absolute_yaw_rates: List[float] = []
    maximum_jerk = 0.0
    previous_acceleration: Optional[float] = None
    previous_behavior = ""
    steps = int(round(sim_seconds / dt))
    completed_steps = 0
    error = ""

    try:
        for _ in range(steps):
            world.update(dt)
            completed_steps += 1
            elapsed = world.simulation_time
            overtakes.update(world)

            behavior = str(getattr(world.controller, "behavior", ""))
            if "EMERGENCY" in behavior and "EMERGENCY" not in previous_behavior:
                emergency_stops += 1
            previous_behavior = behavior

            front = world.latest_sensors.current_front
            if world.speed_mps < 0.3 and front is not None and 0.0 <= front.longitudinal_gap < 15.0:
                deadlock_seconds += dt

            minimum_ttc = min(minimum_ttc, _front_ttc(world))
            minimum_clearance = min(minimum_clearance, _global_clearance(world))

            if elapsed >= COMFORT_WARMUP_S:
                acceleration = float(world.acceleration_mps2)
                absolute_accelerations.append(abs(acceleration))
                absolute_yaw_rates.append(abs(float(world.ego_yaw_rate)))
                if previous_acceleration is not None:
                    maximum_jerk = max(maximum_jerk, abs(acceleration - previous_acceleration) / dt)
                previous_acceleration = acceleration

            for state in states.values():
                primary = all_actors[state.definition.primary_actor_id]
                gap = primary.z - world.ego_z
                if math.isnan(state.encountered_at_s) and 0.0 <= gap <= SCENARIO_SENSOR_RANGE_M:
                    state.encountered_at_s = elapsed
                if math.isnan(state.encountered_at_s):
                    continue

                for actor_id in state.definition.actor_ids:
                    actor = all_actors[actor_id]
                    state.minimum_clearance_m = min(state.minimum_clearance_m, _signed_clearance(world, actor))
                    state.minimum_ttc_s = min(state.minimum_ttc_s, _actor_ttc(world, actor))
                    if actor_id in world.collision_actor_ids:
                        state.collision = True
                    if actor_id in overtakes.attempted_actor_ids:
                        state.overtake_attempted = True
                    if actor_id in overtakes.successful_actor_ids:
                        state.overtake_successful = True
                if _scenario_deadlocked(world, primary):
                    state.deadlock_duration_s += dt
                if math.isnan(state.resolved_at_s) and _scenario_resolved(world, primary):
                    state.resolved_at_s = elapsed
    except Exception as exception:
        error = "{}: {}".format(type(exception).__name__, exception)
    finally:
        world.close()

    scenario_rows: List[Dict[str, Any]] = []
    scenarios_encountered = 0
    scenarios_evaluable = 0
    scenarios_completed = 0
    scenario_successes = 0
    for state in states.values():
        encountered = math.isfinite(state.encountered_at_s)
        evaluable = encountered and state.encountered_at_s <= sim_seconds - SCENARIO_RESOLUTION_ALLOWANCE_S
        completed = math.isfinite(state.resolved_at_s)
        success = evaluable and completed and not state.collision and state.deadlock_duration_s <= SCENARIO_DEADLOCK_LIMIT_S
        scenarios_encountered += int(encountered)
        scenarios_evaluable += int(evaluable)
        scenarios_completed += int(evaluable and completed)
        scenario_successes += int(success)
        scenario_rows.append(
            {
                "controller": controller_name,
                "seed": manifest.seed,
                "manifest_id": manifest.manifest_id,
                "scenario_id": state.definition.scenario_id,
                "scenario_type": state.definition.scenario_type,
                "scenario_category": state.definition.category,
                "primary_actor_id": state.definition.primary_actor_id,
                "actor_ids": "|".join(state.definition.actor_ids),
                "mixed": state.definition.mixed,
                "route_position_m": state.definition.route_position_m,
                "encountered": encountered,
                "evaluable": evaluable,
                "encountered_at_s": _safe_number(state.encountered_at_s),
                "resolved_at_s": _safe_number(state.resolved_at_s),
                "time_to_resolve_s": _safe_number(state.resolved_at_s - state.encountered_at_s) if completed and encountered else "",
                "completed": completed,
                "success": success,
                "collision": state.collision,
                "deadlock_duration_s": state.deadlock_duration_s,
                "overtake_attempted": state.overtake_attempted,
                "overtake_successful": state.overtake_successful,
                "minimum_ttc_s": _safe_number(state.minimum_ttc_s),
                "minimum_clearance_m": _safe_number(state.minimum_clearance_m),
            }
        )

    run_completed = not error and completed_steps == steps
    result = RunResult(
        controller=controller_name,
        seed=manifest.seed,
        manifest_id=manifest.manifest_id,
        target_duration_s=sim_seconds,
        actual_duration_s=completed_steps * dt,
        simulation_steps=completed_steps,
        route_progress_m=float(world.ego_z),
        collision_count=len(world.collision_actor_ids),
        collision_intervention_frames=int(world.collision_intervention_count),
        minimum_ttc_s=minimum_ttc,
        minimum_clearance_m=minimum_clearance,
        emergency_stops=emergency_stops,
        deadlock_duration_s=deadlock_seconds,
        overtake_attempts=len(overtakes.attempted_actor_ids),
        successful_overtakes=len(overtakes.successful_actor_ids),
        overtake_success_rate=(len(overtakes.successful_actor_ids) / len(overtakes.attempted_actor_ids) if overtakes.attempted_actor_ids else math.nan),
        mean_absolute_acceleration_mps2=_mean(absolute_accelerations),
        maximum_absolute_jerk_mps3=maximum_jerk,
        mean_absolute_yaw_rate_deg_s=_mean(absolute_yaw_rates),
        scenarios_encountered=scenarios_encountered,
        scenarios_evaluable=scenarios_evaluable,
        scenarios_completed=scenarios_completed,
        scenario_successes=scenario_successes,
        scenario_completion_rate=(scenario_successes / scenarios_evaluable if scenarios_evaluable else math.nan),
        run_completed=run_completed,
        error=error,
    )
    return result, scenario_rows


def build_ml_factory(model_root: Path) -> Callable[[], Any]:
    try:
        from ML.src.online_policy import OnlineModelBundle, RoadWeaveMLController
    except ModuleNotFoundError as error:
        raise SystemExit("ML dependencies unavailable: {}".format(error))
    bundle = OnlineModelBundle.load(model_root / "risk_model.joblib", model_root / "policy_model.joblib")

    def factory() -> Any:
        return RoadWeaveMLController(
            bundle,
            right_lane_x=RIGHT_LANE_X,
            left_lane_x=LEFT_LANE_X,
            cruise_speed_mps=CRUISE_SPEED_MPS,
            decision_rate_hz=5.0,
            asynchronous_inference=False,
        )

    return factory


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evidence file: {}".format(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=2026000)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "ML" / "models")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "ExperimentResults" / "experiment4" / "pilot")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Experiment 4 evidence in this directory.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run using completed checkpoints.")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds < 1 or args.seconds <= 0 or not 0.005 <= args.dt <= 0.1:
        raise SystemExit("Use at least one seed, positive seconds, and dt between 0.005 and 0.1.")
    if args.overwrite and args.resume:
        raise SystemExit("Choose either --overwrite or --resume, not both.")
    output_dir = args.output_dir.resolve()
    runs_path = output_dir / "experiment4_runs.csv"
    scenarios_path = output_dir / "experiment4_scenarios.csv"
    config_path = output_dir / "experiment4_config.json"
    evidence_paths = [runs_path, scenarios_path]
    if not args.overwrite and not args.resume and any(path.exists() for path in evidence_paths):
        raise SystemExit("Experiment evidence already exists in {}. Use --resume or another directory.".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    controllers: List[Tuple[str, Callable[[], Any]]] = [(RULE, lambda: AutonomousDrivingController())]
    model_hashes: Dict[str, str] = {}
    if not args.rule_only:
        controllers.append((HYBRID, build_ml_factory(args.model_root)))
        for name in ("risk_model.joblib", "policy_model.joblib"):
            path = args.model_root / name
            model_hashes[name] = _sha256(path)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    code_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in REPRODUCIBILITY_FILES
    }
    package_versions = {}
    for package_name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "not installed"
    config = {
        "benchmark_version": BENCHMARK_VERSION,
        "controllers": [name for name, _ in controllers],
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seeds": seeds,
        "duration_seconds": args.seconds,
        "dt_seconds": args.dt,
        "comfort_warmup_seconds": COMFORT_WARMUP_S,
        "scenario_resolution_allowance_seconds": SCENARIO_RESOLUTION_ALLOWANCE_S,
        "scenario_deadlock_limit_seconds": SCENARIO_DEADLOCK_LIMIT_S,
        "model_hashes_sha256": model_hashes,
        "code_hashes_sha256": code_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "wall_clock_runtime_seconds": 0.0,
    }
    if args.resume:
        if not config_path.exists():
            raise SystemExit("Cannot resume because experiment4_config.json is missing.")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark_version", "controllers", "seed_start", "seed_count",
            "duration_seconds", "dt_seconds", "model_hashes_sha256",
            "code_hashes_sha256", "environment",
        ):
            if existing_config.get(key) != config.get(key):
                raise SystemExit("Resume settings do not match the saved experiment: {} differs.".format(key))
        run_rows = _read_csv(runs_path)
        scenario_rows = _read_csv(scenarios_path)
    else:
        run_rows = []
        scenario_rows = []
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    completed_keys = {
        (int(float(row["seed"])), row["controller"])
        for row in run_rows
        if str(row.get("run_completed", "")).strip().lower() in {"true", "1", "yes"}
    }
    started = time.perf_counter()
    for seed_index, seed in enumerate(seeds, 1):
        manifest = build_scenario_manifest(seed, args.seconds)
        for controller_name, factory in controllers:
            key = (seed, controller_name)
            if key in completed_keys:
                print("[{}/{}] {} seed={} already complete; checkpoint reused.".format(seed_index, len(seeds), controller_name, seed))
                continue
            run_rows = [row for row in run_rows if not (int(float(row["seed"])) == seed and row["controller"] == controller_name)]
            scenario_rows = [row for row in scenario_rows if not (int(float(row["seed"])) == seed and row["controller"] == controller_name)]
            result, scenarios = simulate_controller(controller_name, factory, manifest, args.seconds, args.dt)
            row = asdict(result)
            for key, value in list(row.items()):
                if isinstance(value, float):
                    row[key] = _safe_number(value)
            run_rows.append(row)
            scenario_rows.extend(scenarios)
            _write_csv(runs_path, run_rows)
            _write_csv(scenarios_path, scenario_rows)
            print(
                "[{}/{}] {} seed={} progress={:.1f}m collisions={} success={}/{}".format(
                    seed_index, len(seeds), controller_name, seed, result.route_progress_m,
                    result.collision_count, result.scenario_successes, result.scenarios_evaluable,
                )
            )

    config["wall_clock_runtime_seconds"] = time.perf_counter() - started
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": config, "runs": run_rows}, indent=2), encoding="utf-8")

    print("Raw Experiment 4 evidence: {}".format(output_dir))
    if not args.skip_analysis and not args.rule_only:
        from analyze_experiment4 import analyze_experiment

        status = analyze_experiment(output_dir, expected_pairs=args.seeds, strict=True)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
