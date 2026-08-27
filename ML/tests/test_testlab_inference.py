import unittest

import numpy as np

from ML.src.testlab_inference import (
    PROTOCOL_VERSION,
    TestLabInferenceEngine,
    TestLabModelBundle,
)


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


class ConstantRegressionModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def artifact(labels, selected):
    return {
        "feature_columns": ["ego_speed_now"],
        "model": ConstantProbabilityModel(labels, selected),
        "model_version": "test",
    }


def weather_artifact(value=0.62):
    return {
        "feature_columns": ["weather_rain"],
        "model": ConstantRegressionModel(value),
        "model_version": "weather-test",
        "minimum_factor": 0.35,
        "maximum_factor": 1.0,
        "reference_speed_kph": 50.0,
    }


def message(sequence=1, action_lane="RIGHT"):
    return {
        "schemaVersion": PROTOCOL_VERSION,
        "messageType": "observation",
        "sessionId": "unit-test",
        "sequenceNumber": sequence,
        "timestampSeconds": sequence * 0.2,
        "currentLane": action_lane,
        "cruiseSpeedMps": 10.0,
        "leftLaneClear": True,
        "rightLaneClear": True,
        "weather": "Dry",
        "ego": {
            "speedMps": 8.0,
            "accelerationMps2": 0.0,
            "yawRateDegreesPerSecond": 0.0,
        },
        "rightFront": {"present": False},
        "rightRear": {"present": False},
        "leftFront": {"present": False},
        "leftRear": {"present": False},
        "pedestrian": {"present": False},
    }


class TestLabInferenceTests(unittest.TestCase):
    def engine(self, action="KEEP", risk="LOW"):
        return TestLabInferenceEngine(
            TestLabModelBundle(
                risk=artifact(["LOW", "MODERATE", "HIGH", "EXTREME"], risk),
                policy=artifact(
                    ["KEEP", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"],
                    action,
                ),
            )
        )

    def test_keep_requests_normal_cruise_speed(self):
        response = self.engine().handle(message())
        self.assertTrue(response["valid"])
        self.assertEqual(response["executedAction"], "KEEP")
        self.assertAlmostEqual(response["targetSpeedMps"], 10.0)

    def test_pedestrian_emergency_overrides_acceleration(self):
        request = message()
        request["pedestrian"] = {
            "present": True,
            "gapMeters": 4.0,
            "closingSpeedMps": 8.0,
            "actorSpeedMps": 0.0,
            "timeToCollisionSeconds": 0.5,
        }
        response = self.engine(action="ACCELERATE").handle(request)
        self.assertEqual(response["executedAction"], "EMERGENCY_STOP")
        self.assertEqual(response["targetSpeedMps"], 0.0)
        self.assertIn("pedestrian", response["overrideReason"])

    def test_extreme_risk_slows_to_cautious_speed_without_stopping(self):
        engine = self.engine(action="KEEP", risk="EXTREME")
        speeds = []
        for sequence in range(1, 8):
            request = message(sequence)
            request["ego"]["speedMps"] = speeds[-1] if speeds else 8.0
            response = engine.handle(request)
            self.assertEqual(response["executedAction"], "DECELERATE")
            self.assertIn("extreme-risk", response["overrideReason"])
            speeds.append(response["targetSpeedMps"])

        self.assertGreaterEqual(min(speeds), 5.5)
        self.assertLess(speeds[0], 8.0)

    def test_only_a_physical_emergency_commands_zero_speed(self):
        response = self.engine(action="DECELERATE", risk="EXTREME").handle(message())
        self.assertGreater(response["targetSpeedMps"], 0.0)

    def test_lane_change_needs_three_votes_and_a_clear_lane(self):
        engine = self.engine(action="CHANGE_LEFT")
        first = engine.handle(message(1))
        second = engine.handle(message(2))
        third = engine.handle(message(3))
        self.assertEqual(first["executedAction"], "DECELERATE")
        self.assertEqual(second["executedAction"], "DECELERATE")
        self.assertEqual(third["executedAction"], "CHANGE_LEFT")

        blocked_engine = self.engine(action="CHANGE_LEFT")
        blocked = message(1)
        blocked["leftLaneClear"] = False
        response = blocked_engine.handle(blocked)
        self.assertEqual(response["executedAction"], "DECELERATE")
        self.assertIn("not clear", response["overrideReason"])

    def test_duplicate_sequence_is_rejected(self):
        engine = self.engine()
        self.assertTrue(engine.handle(message(1))["valid"])
        duplicate = engine.handle(message(1))
        self.assertFalse(duplicate["valid"])
        self.assertIn("out-of-order", duplicate["error"])

    def test_adverse_weather_uses_the_weather_model(self):
        engine = self.engine()
        engine.models.weather = weather_artifact(0.62)
        request = message()
        request["weather"] = "Rain"
        response = engine.handle(request)
        self.assertTrue(response["weatherModelUsed"])
        self.assertEqual(response["weatherContext"], "Rain")
        self.assertAlmostEqual(response["weatherSpeedFactor"], 0.62)
        self.assertAlmostEqual(response["weatherTargetSpeedMps"], 31.0 / 3.6)

    def test_dry_weather_keeps_full_speed_even_when_model_is_loaded(self):
        engine = self.engine()
        engine.models.weather = weather_artifact(0.40)
        response = engine.handle(message())
        self.assertFalse(response["weatherModelUsed"])
        self.assertEqual(response["weatherContext"], "Dry")
        self.assertAlmostEqual(response["weatherSpeedFactor"], 1.0)
        self.assertAlmostEqual(response["weatherTargetSpeedMps"], 10.0)


if __name__ == "__main__":
    unittest.main()
