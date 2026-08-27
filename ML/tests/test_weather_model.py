import unittest

import numpy as np

from ML.src.build_extreme_weather import classify_condition
from ML.src.weather_features import normalize_condition, summarize_weather_history
from ML.src.weather_model import predict_weather_factor


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class WeatherFeatureTests(unittest.TestCase):
    def test_condition_classifier_includes_rain_snow_and_fog(self):
        self.assertEqual(classify_condition("Rain_Driving", "", "wet"), "RAIN")
        self.assertEqual(classify_condition("Snow_Driving", "", "icy"), "SNOW")
        self.assertEqual(classify_condition("Normal_Driving", "dense fog", "dry"), "FOG")
        self.assertEqual(classify_condition("Normal_Driving", "clear", "dry"), "DRY")
        self.assertIsNone(classify_condition("LowLight_Driving", "night", "dry"))

    def test_weather_aliases_are_stable(self):
        self.assertEqual(normalize_condition("rainy"), "RAIN")
        self.assertEqual(normalize_condition("haze"), "FOG")
        self.assertEqual(normalize_condition("unknown"), "DRY")

    def test_history_summary_uses_only_past_and_current_samples(self):
        states = [
            {"timestamp": 0.0, "ego_speed": 5.0, "ego_accel": 0.0, "ego_yaw_rate": 0.0},
            {"timestamp": 1.0, "ego_speed": 7.0, "ego_accel": 2.0, "ego_yaw_rate": 4.0},
        ]
        summary = summarize_weather_history(states, "Fog")
        self.assertEqual(summary["ego_speed_now"], 7.0)
        self.assertEqual(summary["ego_speed_trend"], 2.0)
        self.assertEqual(summary["weather_fog"], 1.0)
        self.assertEqual(summary["weather_rain"], 0.0)

    def test_prediction_is_clipped_to_artifact_safety_bounds(self):
        artifact = {
            "feature_columns": ["weather_snow"],
            "model": ConstantRegressor(0.10),
            "minimum_factor": 0.35,
            "maximum_factor": 1.0,
        }
        factor = predict_weather_factor(artifact, {}, "Snow")
        self.assertEqual(factor, 0.35)


if __name__ == "__main__":
    unittest.main()
