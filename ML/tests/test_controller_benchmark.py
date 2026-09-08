import unittest

from Tools.benchmark_controllers import (
    RULE,
    build_scenario_manifest,
    install_manifest,
    simulate_controller,
)
from Tools.simulated_twin_stream import (
    AutonomousDrivingController,
    SimulatedActor,
    SimulatedWorld,
)


class ControllerBenchmarkTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_contains_core_scenarios(self):
        first = build_scenario_manifest(2026000, 300.0)
        second = build_scenario_manifest(2026000, 300.0)
        self.assertEqual(first.manifest_id, second.manifest_id)
        self.assertEqual(first.scenarios, second.scenarios)
        kinds = {scenario.scenario_type for scenario in first.scenarios}
        self.assertTrue({"pedestrian", "stopped_car", "slow_car", "truck"}.issubset(kinds))

    def test_manifest_installation_creates_independent_actor_copies(self):
        manifest = build_scenario_manifest(2026001, 60.0)
        first_world = SimulatedWorld(True, 2026001, 7, announce=False)
        second_world = SimulatedWorld(True, 2026001, 7, announce=False)
        first = install_manifest(first_world, manifest)
        second = install_manifest(second_world, manifest)
        actor_id = next(iter(first))
        first[actor_id].z += 100.0
        self.assertNotEqual(first[actor_id].z, second[actor_id].z)
        first_world.close(); second_world.close()

    def test_collision_attempts_are_not_hidden_by_non_penetration(self):
        world = SimulatedWorld(False, 7, 4, announce=False)
        actor = SimulatedActor("collision-test", "stopped_car", 0.0, 1.0, 0.0, 1.9, 1.5, 4.35, "vehicle.car", active=True)
        world.actors = [actor]
        world.actors_by_id = {actor.actor_id: actor}
        world._enforce_non_penetration(world.ego_x, world.ego_z)
        world._enforce_non_penetration(world.ego_x, world.ego_z)
        self.assertEqual({"collision-test"}, world.collision_actor_ids)
        self.assertEqual(2, world.collision_intervention_count)
        world.close()

    def test_short_rule_run_completes_and_emits_scenario_rows(self):
        manifest = build_scenario_manifest(2026002, 5.0)
        result, scenarios = simulate_controller(
            RULE,
            lambda: AutonomousDrivingController(),
            manifest,
            sim_seconds=5.0,
            dt=0.05,
        )
        self.assertTrue(result.run_completed)
        self.assertEqual(100, result.simulation_steps)
        self.assertGreater(len(scenarios), 0)


if __name__ == "__main__":
    unittest.main()
