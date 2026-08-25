"""Focused unit tests for the K-Risk preparation pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from ML.src.build_krisk import (
    canonical_response_stem,
    convert_event,
    longitudinal_gap,
    normalized_id,
    parse_event_identity,
    parse_gpt_action,
    record_fields,
)


class KRiskConverterTests(unittest.TestCase):
    def test_citysim_actor_zero_is_valid_but_highd_zero_relation_is_missing(self):
        self.assertIsNone(normalized_id(0))
        self.assertEqual(normalized_id(0, zero_is_missing=False), "0")

    def test_adjacent_actor_gap_is_longitudinal_not_euclidean(self):
        fields = record_fields("citysim")
        ego = {
            "car_center_x": 0.0,
            "car_center_y": 0.0,
            "length": 4.0,
            "course": 0.0,
        }
        actor = {
            "car_center_x": 10.0,
            "car_center_y": 4.0,
            "length": 4.0,
        }
        self.assertAlmostEqual(
            longitudinal_gap(ego, actor, fields, "left_front"),
            6.0,
        )

    def test_identity_parsing_for_supported_schemas(self):
        self.assertEqual(
            parse_event_identity(
                "highd_44_1018_leftAlongsideId_brake_high_frame_11478_to_11603",
                "highd",
            ),
            ("highd", "1018", 11478, 11603),
        )
        self.assertEqual(
            parse_event_identity(
                "freewayB_track_4_car_895_frame_4419_to_4510", "citysim"
            ),
            ("freewayB", "895", 4419, 4510),
        )

    def test_gpt_action_parser_accepts_markdown_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (
                "response_extreme_desc_freewayB_track_4_car_895_"
                "frame_4419_to_4510.txt"
            )
            path.write_text(
                "Reasoning may mention action 1.\nFinal Decision: ####5",
                encoding="utf-8",
            )

            self.assertEqual(parse_gpt_action(path), 5)
            self.assertEqual(
                canonical_response_stem(path),
                "freewayB_track_4_car_895_frame_4419_to_4510",
            )

    def test_highd_conversion_excludes_risk_and_behaviour_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (
                "highd_1_10_preceding_brake_high_frame_100_to_103.json"
            )
            records = []
            for frame in range(100, 104):
                records.extend(
                    [
                        {
                            "frame": frame,
                            "id": 10,
                            "x": float(frame - 100),
                            "y": 0.0,
                            "width": 4.0,
                            "xVelocity": 20.0,
                            "yVelocity": 0.0,
                            "xAcceleration": -1.0,
                            "yAcceleration": 0.0,
                            "speed": 20.0,
                            "precedingId": 11,
                            "followingId": 0,
                            "leftPrecedingId": 0,
                            "leftAlongsideId": 0,
                            "leftFollowingId": 0,
                            "rightPrecedingId": 0,
                            "rightAlongsideId": 0,
                            "rightFollowingId": 0,
                            "total_risk": float(frame),
                            "acc_high": False,
                            "brake_high": frame == 103,
                            "yaw_left": False,
                            "yaw_right": False,
                        },
                        {
                            "frame": frame,
                            "id": 11,
                            "x": float(frame - 80),
                            "y": 0.0,
                            "width": 4.0,
                            "xVelocity": 15.0,
                            "yVelocity": 0.0,
                            "xAcceleration": 0.0,
                            "yAcceleration": 0.0,
                            "speed": 15.0,
                        },
                    ]
                )
            path.write_text(json.dumps(records), encoding="utf-8")

            row, audit = convert_event(
                {
                    "path": path,
                    "stem": path.stem,
                    "source": "highd",
                    "schema": "highd",
                    "severity": "HIGH",
                },
                history_seconds=3.0,
            )

            self.assertEqual(row["target"], "HIGH")
            self.assertEqual(row["source"], "krisk")
            self.assertEqual(audit["peak_frame"], 103)
            self.assertEqual(row["front_present_now"], 1.0)
            self.assertGreater(row["front_closing_speed_now"], 0.0)
            self.assertFalse(any("risk" in name.lower() for name in row))
            self.assertFalse(any("brake_high" in name.lower() for name in row))


if __name__ == "__main__":
    unittest.main()
