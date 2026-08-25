import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ML.src.model_support import normalize_target
from ML.src.online_policy import OnlineModelBundle, RoadWeaveMLController
from ML.src.train_model import best_group_holdout


class ConstantProbabilityModel:
    def __init__(self, labels, selected_label):
        self.classes_ = np.asarray(labels, dtype=object)
        self.selected_label = selected_label

    def predict_proba(self, frame):
        probabilities = np.full((len(frame), len(self.classes_)), 0.01)
        selected = list(self.classes_).index(self.selected_label)
        probabilities[:, selected] = 0.96
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def artifact(task, labels, selected_label):
    return {
        "task": task,
        "feature_columns": ["ego_speed_now"],
        "model": ConstantProbabilityModel(labels, selected_label),
    }


def empty_sensors():
    return SimpleNamespace(
        current_front=None,
        left_front=None,
        left_rear=None,
        right_front=None,
        right_rear=None,
        pedestrian_hazard=None,
    )


def hit(gap, relative_speed, actor_speed=0.0, ttc=10.0):
    return SimpleNamespace(
        longitudinal_gap=gap,
        relative_speed_mps=relative_speed,
        time_to_collision=ttc,
        actor=SimpleNamespace(speed_mps=actor_speed),
    )


class GroupSplitTests(unittest.TestCase):
    def test_group_holdout_has_no_overlap(self):
        labels = ["LOW", "MODERATE", "HIGH", "EXTREME"]
        rows = []
        for index in range(80):
            rows.append(
                {
                    "target": labels[index % len(labels)],
                    "_split_group": "event-{:03d}".format(index),
                }
            )
        data = pd.DataFrame(rows)
        train_indices, holdout_indices = best_group_holdout(data, 0.2, 2026)
        train_groups = set(data.iloc[train_indices]["_split_group"])
        holdout_groups = set(data.iloc[holdout_indices]["_split_group"])
        self.assertFalse(train_groups & holdout_groups)
        self.assertEqual(set(data.iloc[train_indices]["target"]), set(labels))

    def test_numeric_and_named_targets_normalize_identically(self):
        self.assertEqual(normalize_target(2, "policy"), "DECELERATE")
        self.assertEqual(normalize_target("DrivingAction.DECELERATE", "policy"), "DECELERATE")


class OnlineSafetyTests(unittest.TestCase):
    def controller(self, action):
        models = OnlineModelBundle(
            risk=artifact(
                "risk",
                ["LOW", "MODERATE", "HIGH", "EXTREME"],
                "LOW",
            ),
            policy=artifact(
                "policy",
                ["KEEP", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"],
                action,
            ),
        )
        return RoadWeaveMLController(
            models,
            right_lane_x=0.0,
            left_lane_x=-5.5,
            cruise_speed_mps=13.9,
        )

    def test_pedestrian_emergency_overrides_acceleration_model(self):
        controller = self.controller("ACCELERATE")
        sensors = empty_sensors()
        sensors.pedestrian_hazard = hit(12.0, 10.0, ttc=1.2)
        target_speed, _, behavior = controller.decide(
            0.0,
            0.0,
            10.0,
            sensors,
            {},
            simulation_time=1.0,
        )
        self.assertEqual(target_speed, 0.0)
        self.assertEqual(controller.executed_action, "EMERGENCY_STOP")
        self.assertIn("EMERGENCY", behavior)

    def test_blocked_left_lane_rejects_model_lane_change(self):
        controller = self.controller("CHANGE_LEFT")
        sensors = empty_sensors()
        sensors.left_front = hit(10.0, 2.0, actor_speed=8.0)
        target_speed = 10.0
        target_lane = 0.0
        behavior = ""
        for timestamp in (1.0, 1.21, 1.42):
            target_speed, target_lane, behavior = controller.decide(
                0.0,
                0.0,
                10.0,
                sensors,
                {},
                simulation_time=timestamp,
            )
        self.assertEqual(target_lane, 0.0)
        self.assertLess(target_speed, 10.0)
        self.assertEqual(controller.executed_action, "DECELERATE")
        self.assertEqual(behavior, "ML_LANE_CHANGE_REJECTED")


if __name__ == "__main__":
    unittest.main()
