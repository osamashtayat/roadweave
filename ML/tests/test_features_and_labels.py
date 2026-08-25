import math
import unittest

from ML.src.features import capped_ttc, new_state, set_slot, summarize_history
from ML.src.labels import DrivingAction, KRISK_ACTION_MAP, RiskLevel


class FeatureSchemaTests(unittest.TestCase):
    def test_ttc_is_capped_and_handles_non_closing_actor(self):
        self.assertAlmostEqual(capped_ttc(10.0, 5.0), 2.0)
        self.assertAlmostEqual(capped_ttc(100.0, 1.0), 20.0)
        self.assertAlmostEqual(capped_ttc(10.0, -1.0), 20.0)

    def test_history_uses_only_supplied_past_states(self):
        first = new_state(0.0, 5.0, 1.0, 0.0)
        second = new_state(1.0, 7.0, -1.0, 2.0)
        set_slot(second, "front", 12.0, 3.0, 4.0)

        summary = summarize_history([first, second])

        self.assertAlmostEqual(summary["ego_speed_now"], 7.0)
        self.assertAlmostEqual(summary["ego_speed_mean"], 6.0)
        self.assertAlmostEqual(summary["ego_speed_trend"], 2.0)
        self.assertAlmostEqual(summary["front_present_now"], 1.0)
        self.assertAlmostEqual(summary["front_ttc_now"], 4.0)

    def test_missing_slot_remains_missing_not_zero_distance(self):
        summary = summarize_history([new_state(0.0, 5.0, 0.0, 0.0)])
        self.assertEqual(summary["front_present_now"], 0.0)
        self.assertTrue(math.isnan(summary["front_gap_now"]))


class LabelSchemaTests(unittest.TestCase):
    def test_krisk_action_mapping(self):
        self.assertEqual(KRISK_ACTION_MAP[1], DrivingAction.KEEP)
        self.assertEqual(KRISK_ACTION_MAP[5], DrivingAction.DECELERATE)
        self.assertEqual(RiskLevel.EXTREME.value, 3)


if __name__ == "__main__":
    unittest.main()
