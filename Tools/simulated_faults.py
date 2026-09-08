#!/usr/bin/env python3
"""Deterministic transport/data faults for RoadWeave Experiment 3."""

from __future__ import annotations

import copy
import csv
import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FaultConfig:
    run_id: str = ""
    profile_id: str = "baseline"
    rate_hz: float = 30.0
    seed: int = 901
    packet_loss_percent: float = 0.0
    delay_ms: float = 0.0
    disconnect_at_s: float = -1.0
    disconnect_duration_s: float = 0.0
    duplicate_percent: float = 0.0
    out_of_order_percent: float = 0.0
    missing_vehicle_percent: float = 0.0
    invalid_vehicle_percent: float = 0.0
    missing_actors_percent: float = 0.0
    warmup_s: float = 30.0
    duration_s: float = 300.0

    @property
    def enabled(self) -> bool:
        return bool(self.run_id)


@dataclass(order=True)
class ScheduledDatagram:
    release_time: float
    order: int
    payload: bytes = field(compare=False)


class FaultInjector:
    """Transforms and schedules datagrams without blocking the simulated world."""

    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self._queue: List[ScheduledDatagram] = []
        self._order = 0
        self.generated = 0
        self.intentionally_dropped = 0
        self.duplicates_sent = 0
        self.out_of_order_scheduled = 0
        self.malformed_sent = 0
        self.missing_vehicle_sent = 0
        self.invalid_vehicle_sent = 0
        self.missing_actors_sent = 0

    @staticmethod
    def _chance(random_source: random.Random, percent: float) -> bool:
        return percent > 0.0 and random_source.random() * 100.0 < percent

    def reset(self) -> None:
        self._queue.clear()
        self._order = 0
        self.generated = 0
        self.intentionally_dropped = 0
        self.duplicates_sent = 0
        self.out_of_order_scheduled = 0
        self.malformed_sent = 0
        self.missing_vehicle_sent = 0
        self.invalid_vehicle_sent = 0
        self.missing_actors_sent = 0
        self.random.seed(self.config.seed)

    def in_measurement(self, elapsed_s: float) -> bool:
        return 0.0 <= elapsed_s <= self.config.duration_s

    def disconnected(self, elapsed_s: float) -> bool:
        return (
            self.in_measurement(elapsed_s)
            and self.config.disconnect_at_s >= 0.0
            and self.config.disconnect_at_s
            <= elapsed_s
            < self.config.disconnect_at_s + self.config.disconnect_duration_s
        )

    def annotate(self, snapshot: Dict[str, Any], elapsed_s: float) -> None:
        metadata = snapshot.setdefault("metadata", {})
        active = self.in_measurement(elapsed_s) and (
            self.config.packet_loss_percent > 0.0
            or self.config.delay_ms > 0.0
            or self.config.duplicate_percent > 0.0
            or self.config.out_of_order_percent > 0.0
            or self.config.missing_vehicle_percent > 0.0
            or self.config.invalid_vehicle_percent > 0.0
            or self.config.missing_actors_percent > 0.0
        )
        metadata["transportDiagnostics"] = {
            "runId": self.config.run_id,
            "profileId": self.config.profile_id,
            "faultSeed": self.config.seed,
            "packetLossPercent": self.config.packet_loss_percent,
            "injectedDelayMilliseconds": self.config.delay_ms,
            "disconnectAtMeasurementSeconds": self.config.disconnect_at_s,
            "disconnectDurationSeconds": self.config.disconnect_duration_s,
            "duplicatePercent": self.config.duplicate_percent,
            "outOfOrderPercent": self.config.out_of_order_percent,
            "missingVehiclePercent": self.config.missing_vehicle_percent,
            "invalidVehiclePercent": self.config.invalid_vehicle_percent,
            "missingActorsPercent": self.config.missing_actors_percent,
            "experimentWarmupSeconds": self.config.warmup_s,
            "experimentDurationSeconds": self.config.duration_s,
            "measurementElapsedSeconds": elapsed_s,
            "faultActive": active,
            "generatedMessages": self.generated,
            "intentionallyDroppedMessages": self.intentionally_dropped,
            "duplicateMessagesSent": self.duplicates_sent,
            "outOfOrderMessagesScheduled": self.out_of_order_scheduled,
            "malformedMessagesSent": self.malformed_sent,
        }

    def submit(
        self,
        snapshot: Dict[str, Any],
        now: float,
        elapsed_s: float,
        encoder,
    ) -> str:
        candidate = copy.deepcopy(snapshot)
        within = self.in_measurement(elapsed_s)
        if within:
            self.generated += 1

        if self.disconnected(elapsed_s):
            self.intentionally_dropped += 1
            return "disconnect"
        if within and self._chance(self.random, self.config.packet_loss_percent):
            self.intentionally_dropped += 1
            return "packet_loss"

        malformed = False
        if within and self._chance(self.random, self.config.missing_vehicle_percent):
            candidate.pop("vehicle", None)
            malformed = True
            self.missing_vehicle_sent += 1
        elif within and self._chance(self.random, self.config.invalid_vehicle_percent):
            vehicle = candidate.setdefault("vehicle", {})
            vehicle["validity"] = 3
            metadata = candidate.setdefault("metadata", {})
            metadata["validity"] = 2
            metadata["validityMessage"] = "Vehicle telemetry unavailable; ego pose retained."
            self.invalid_vehicle_sent += 1
        if within and self._chance(self.random, self.config.missing_actors_percent):
            candidate.pop("actors", None)
            malformed = True
            self.missing_actors_sent += 1
        if malformed:
            self.malformed_sent += 1

        self.annotate(candidate, elapsed_s)
        diagnostics = candidate["metadata"]["transportDiagnostics"]
        diagnostics["generatedMessages"] = self.generated
        diagnostics["intentionallyDroppedMessages"] = self.intentionally_dropped
        diagnostics["malformedMessagesSent"] = self.malformed_sent
        payload = encoder(candidate)

        release = now + max(0.0, self.config.delay_ms) / 1000.0
        if within and self._chance(self.random, self.config.out_of_order_percent):
            release += max(0.1, 3.0 / max(1.0, self.config.rate_hz))
            self.out_of_order_scheduled += 1
        self._schedule(payload, release)

        if within and self._chance(self.random, self.config.duplicate_percent):
            self._schedule(payload, release + 0.002)
            self.duplicates_sent += 1
        return "scheduled"

    def _schedule(self, payload: bytes, release_time: float) -> None:
        self._order += 1
        heapq.heappush(
            self._queue,
            ScheduledDatagram(release_time, self._order, payload),
        )

    def pop_due(self, now: float) -> List[bytes]:
        due: List[bytes] = []
        while self._queue and self._queue[0].release_time <= now:
            due.append(heapq.heappop(self._queue).payload)
        return due

    def next_release_time(self) -> Optional[float]:
        return self._queue[0].release_time if self._queue else None


class FaultRunLogger:
    """Writes source truth continuously and one authoritative source summary."""

    def __init__(self, config: FaultConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.truth_path: Optional[Path] = None
        self.summary_path: Optional[Path] = None
        self._handle = None
        self._writer = None
        if not config.enabled:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in config.run_id)
        self.truth_path = output_dir / f"experiment3_{safe_id}_truth.csv"
        self.summary_path = output_dir / f"experiment3_{safe_id}_source_summary.csv"
        if self.truth_path.exists() or self.summary_path.exists():
            raise FileExistsError(
                f"Experiment 3 run ID '{config.run_id}' already exists. Use a new --run-id."
            )
        self._handle = self.truth_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            [
                "run_id", "profile_id", "truth_utc_s", "simulation_time_s",
                "measurement_elapsed_s", "sequence_number", "ego_x_m", "ego_y_m",
                "ego_z_m", "speed_mps", "fault_window_active",
            ]
        )

    def write_truth(
        self,
        snapshot: Dict[str, Any],
        utc_s: float,
        elapsed_s: float,
    ) -> None:
        if self._writer is None:
            return
        position = snapshot.get("ego", {}).get("position", {})
        self._writer.writerow(
            [
                self.config.run_id,
                self.config.profile_id,
                f"{utc_s:.9f}",
                snapshot.get("metadata", {}).get("sourceTimestampSeconds", ""),
                f"{elapsed_s:.6f}",
                snapshot.get("metadata", {}).get("sequenceNumber", ""),
                position.get("x", ""), position.get("y", ""), position.get("z", ""),
                snapshot.get("ego", {}).get("speedMetersPerSecond", ""),
                str(0.0 <= elapsed_s <= self.config.duration_s).lower(),
            ]
        )
        self._handle.flush()

    def close(self, injector: FaultInjector) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.summary_path is None:
            return
        with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "run_id", "profile_id", "rate_hz", "fault_seed", "packet_loss_percent",
                "delay_ms", "disconnect_at_s", "disconnect_duration_s", "duplicate_percent",
                "out_of_order_percent", "missing_vehicle_percent", "invalid_vehicle_percent",
                "missing_actors_percent", "generated_messages", "intentionally_dropped_messages",
                "duplicate_messages_sent", "out_of_order_messages_scheduled", "malformed_messages_sent",
                "missing_vehicle_messages_sent", "invalid_vehicle_messages_sent", "missing_actors_messages_sent",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": self.config.run_id,
                    "profile_id": self.config.profile_id,
                    "rate_hz": self.config.rate_hz,
                    "fault_seed": self.config.seed,
                    "packet_loss_percent": self.config.packet_loss_percent,
                    "delay_ms": self.config.delay_ms,
                    "disconnect_at_s": self.config.disconnect_at_s,
                    "disconnect_duration_s": self.config.disconnect_duration_s,
                    "duplicate_percent": self.config.duplicate_percent,
                    "out_of_order_percent": self.config.out_of_order_percent,
                    "missing_vehicle_percent": self.config.missing_vehicle_percent,
                    "invalid_vehicle_percent": self.config.invalid_vehicle_percent,
                    "missing_actors_percent": self.config.missing_actors_percent,
                    "generated_messages": injector.generated,
                    "intentionally_dropped_messages": injector.intentionally_dropped,
                    "duplicate_messages_sent": injector.duplicates_sent,
                    "out_of_order_messages_scheduled": injector.out_of_order_scheduled,
                    "malformed_messages_sent": injector.malformed_sent,
                    "missing_vehicle_messages_sent": injector.missing_vehicle_sent,
                    "invalid_vehicle_messages_sent": injector.invalid_vehicle_sent,
                    "missing_actors_messages_sent": injector.missing_actors_sent,
                }
            )
