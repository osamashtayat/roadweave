import unittest

import numpy as np

from ML.src.sensor_reliability_features import (
    FEATURE_COLUMNS,
    canonical_features,
    reliability_status,
)
from ML.src.sensor_reliability_model import predict_sensor_reliability
from ML.src.generate_sensor_reliability import simulate_channel


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class SensorReliabilityTests(unittest.TestCase):
    def test_feature_contract_one_hot_encodes_weather(self):
        features = canonical_features(
            {
                "weather": "Fog",
                "dropout_rate": 0.25,
                "confidence_mean": 0.7,
            }
        )
        self.assertEqual(set(features), set(FEATURE_COLUMNS))
        self.assertEqual(features["weather_fog"], 1.0)
        self.assertEqual(features["weather_dry"], 0.0)
        self.assertAlmostEqual(features["dropout_rate"], 0.25)

    def test_status_thresholds_are_stable(self):
        self.assertEqual(reliability_status(0.95), "HEALTHY")
        self.assertEqual(reliability_status(0.75), "ACCEPTABLE")
        self.assertEqual(reliability_status(0.55), "DEGRADED")
        self.assertEqual(reliability_status(0.20), "FAILED")

    def test_prediction_is_clipped_to_valid_reliability_range(self):
        artifact = {
            "feature_columns": list(FEATURE_COLUMNS),
            "model": ConstantRegressor(1.4),
        }
        prediction = predict_sensor_reliability(artifact, {"weather": "Dry"})
        self.assertEqual(prediction["reliability"], 1.0)
        self.assertEqual(prediction["status"], "HEALTHY")

    def test_clear_empty_road_is_healthy_but_missing_messages_are_not(self):
        empty = np.asarray([], dtype=float)
        _, nominal = simulate_channel(
            np.random.default_rng(10), "RADAR", "DRY", "NONE", 0.0,
            empty, empty, 8.0, 0.0, 30, 0.1,
        )
        _, failed = simulate_channel(
            np.random.default_rng(10), "RADAR", "DRY", "COMPLETE_FAILURE", 1.0,
            empty, empty, 8.0, 0.0, 30, 0.1,
        )
        self.assertGreater(nominal["target"], 0.85)
        self.assertLess(failed["target"], 0.10)


if __name__ == "__main__":
    unittest.main()
