import json
import unittest

from Tools.simulated_faults import FaultConfig, FaultInjector


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def snapshot(sequence=1):
    return {
        "metadata": {"sequenceNumber": sequence, "validity": 1},
        "ego": {"position": {"x": 0, "y": 0, "z": 1}, "validity": 1},
        "vehicle": {"validity": 1},
        "wheels": {"validity": 1},
        "actors": [],
    }


class FaultInjectorTests(unittest.TestCase):
    def test_loss_drops_without_scheduling(self):
        injector = FaultInjector(FaultConfig(run_id="loss", packet_loss_percent=100))
        self.assertEqual("packet_loss", injector.submit(snapshot(), 10.0, 1.0, encode))
        self.assertEqual([], injector.pop_due(20.0))
        self.assertEqual(1, injector.intentionally_dropped)

    def test_delay_does_not_release_early(self):
        injector = FaultInjector(FaultConfig(run_id="delay", delay_ms=250))
        injector.submit(snapshot(), 10.0, 1.0, encode)
        self.assertEqual([], injector.pop_due(10.249))
        self.assertEqual(1, len(injector.pop_due(10.251)))

    def test_duplicate_schedules_two_identical_datagrams(self):
        injector = FaultInjector(FaultConfig(run_id="duplicate", duplicate_percent=100))
        injector.submit(snapshot(), 10.0, 1.0, encode)
        due = injector.pop_due(10.01)
        self.assertEqual(2, len(due))
        self.assertEqual(due[0], due[1])

    def test_missing_vehicle_and_canonical_invalid_vehicle_are_different(self):
        missing = FaultInjector(FaultConfig(run_id="missing", missing_vehicle_percent=100))
        missing.submit(snapshot(), 10.0, 1.0, encode)
        missing_payload = json.loads(missing.pop_due(10.01)[0])
        self.assertNotIn("vehicle", missing_payload)
        self.assertEqual(1, missing.missing_vehicle_sent)
        self.assertEqual(1, missing.malformed_sent)

        invalid = FaultInjector(FaultConfig(run_id="invalid", invalid_vehicle_percent=100))
        invalid.submit(snapshot(), 10.0, 1.0, encode)
        invalid_payload = json.loads(invalid.pop_due(10.01)[0])
        self.assertEqual(3, invalid_payload["vehicle"]["validity"])
        self.assertEqual(2, invalid_payload["metadata"]["validity"])
        self.assertEqual(1, invalid.invalid_vehicle_sent)
        self.assertEqual(0, invalid.malformed_sent)

    def test_missing_actors_is_counted_separately_from_missing_vehicle(self):
        injector = FaultInjector(FaultConfig(run_id="actors", missing_actors_percent=100))
        injector.submit(snapshot(), 10.0, 1.0, encode)
        payload = json.loads(injector.pop_due(10.01)[0])
        self.assertNotIn("actors", payload)
        self.assertEqual(1, injector.missing_actors_sent)
        self.assertEqual(0, injector.missing_vehicle_sent)
        self.assertEqual(1, injector.malformed_sent)

    def test_faults_are_not_applied_during_warmup(self):
        injector = FaultInjector(FaultConfig(run_id="warmup", packet_loss_percent=100))
        self.assertEqual("scheduled", injector.submit(snapshot(), 10.0, -0.5, encode))
        self.assertEqual(1, len(injector.pop_due(10.01)))


if __name__ == "__main__":
    unittest.main()
