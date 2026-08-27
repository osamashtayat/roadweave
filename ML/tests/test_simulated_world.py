"""Regression tests for the procedural live-stream world."""

import math
import unittest

from Tools.simulated_twin_stream import SimulatedWorld


class SimulatedWorldMotionTests(unittest.TestCase):
    def test_ego_route_progress_never_rewinds_during_lane_changes(self):
        for seed in range(2026000, 2026003):
            world = SimulatedWorld(
                include_actors=True,
                configured_seed=seed,
                encounter_count=7,
                announce=False,
            )
            previous_z = world.ego_z
            for _ in range(int(120.0 / 0.05)):
                world.update(0.05)
                self.assertGreaterEqual(
                    world.ego_z + 1e-9,
                    previous_z,
                    "seed {} rewound the ego route position".format(seed),
                )
                previous_z = world.ego_z

    def test_published_ego_pose_has_bounded_frame_to_frame_motion(self):
        world = SimulatedWorld(
            include_actors=True,
            configured_seed=20260826,
            encounter_count=7,
            announce=False,
        )
        previous_pose = world.road.pose(world.ego_z, world.ego_x)
        previous_yaw = world.ego_yaw_degrees
        for _ in range(30 * 180):
            world.update(1.0 / 30.0)
            pose = world.road.pose(world.ego_z, world.ego_x)
            position_step = math.hypot(
                pose.x - previous_pose.x,
                pose.z - previous_pose.z,
            )
            yaw_step = abs((world.ego_yaw_degrees - previous_yaw + 180.0) % 360.0 - 180.0)
            self.assertLessEqual(position_step, 0.55)
            self.assertLessEqual(yaw_step, 1.25)
            previous_pose = pose
            previous_yaw = world.ego_yaw_degrees


if __name__ == "__main__":
    unittest.main()
