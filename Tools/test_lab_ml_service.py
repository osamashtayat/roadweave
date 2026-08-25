#!/usr/bin/env python3
"""Local UDP inference service used by RoadWeave's Unity Test Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.src.testlab_inference import (  # noqa: E402
    PROTOCOL_VERSION,
    TestLabInferenceEngine,
    TestLabModelBundle,
    default_model_paths,
)


def arguments() -> argparse.Namespace:
    default_risk, default_policy = default_model_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Serve RoadWeave risk/policy predictions to Unity Test Lab."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5075)
    parser.add_argument("--risk-model", type=Path, default=default_risk)
    parser.add_argument("--policy-model", type=Path, default=default_policy)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def create_engine(args: argparse.Namespace) -> TestLabInferenceEngine:
    models = TestLabModelBundle.load(args.risk_model, args.policy_model)
    return TestLabInferenceEngine(
        models,
        confidence_threshold=args.confidence_threshold,
    )


def observation(sequence: int, pedestrian_gap: float = -1.0) -> Dict[str, Any]:
    pedestrian = {"present": False}
    if pedestrian_gap >= 0.0:
        pedestrian = {
            "present": True,
            "gapMeters": pedestrian_gap,
            "closingSpeedMps": 8.0,
            "actorSpeedMps": 0.0,
            "timeToCollisionSeconds": pedestrian_gap / 8.0,
        }
    return {
        "schemaVersion": PROTOCOL_VERSION,
        "messageType": "observation",
        "sessionId": "self-test",
        "sequenceNumber": sequence,
        "timestampSeconds": sequence * 0.2,
        "currentLane": "RIGHT",
        "cruiseSpeedMps": 8.33,
        "leftLaneClear": True,
        "rightLaneClear": False,
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
        "pedestrian": pedestrian,
    }


def run_self_test(engine: TestLabInferenceEngine) -> int:
    reset = engine.handle(
        {
            "schemaVersion": PROTOCOL_VERSION,
            "messageType": "reset",
            "sessionId": "self-test",
            "sequenceNumber": 0,
        }
    )
    normal = engine.handle(observation(1))
    emergency = engine.handle(observation(2, pedestrian_gap=5.0))
    if not reset.get("valid") or not normal.get("valid"):
        raise RuntimeError("normal Test Lab inference self-test failed")
    if emergency.get("executedAction") != "EMERGENCY_STOP":
        raise RuntimeError("pedestrian safety override self-test failed")
    print("Test Lab ML service self-test passed.")
    print(json.dumps(normal, indent=2, sort_keys=True))
    print(json.dumps(emergency, indent=2, sort_keys=True))
    return 0


def serve(engine: TestLabInferenceEngine, host: str, port: int, verbose: bool) -> int:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.bind((host, port))
    except OSError as error:
        udp.close()
        raise RuntimeError(
            "Could not bind {}:{} (is another Test Lab ML service running?): {}".format(
                host,
                port,
                error,
            )
        ) from error

    print(
        "RoadWeave Test Lab ML service listening on {}:{} using {}".format(
            host,
            port,
            engine.models.version,
        ),
        flush=True,
    )
    print("Press Control-C to stop it.", flush=True)

    try:
        while True:
            raw, sender = udp.recvfrom(65507)
            try:
                message = json.loads(raw.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("top-level JSON value must be an object")
                response = engine.handle(message)
                reply_port = int(message.get("replyPort", sender[1]))
                if not 1 <= reply_port <= 65535:
                    raise ValueError("replyPort must be between 1 and 65535")
                destination = (sender[0], reply_port)
            except Exception as error:
                response = {
                    "schemaVersion": PROTOCOL_VERSION,
                    "messageType": "error",
                    "sessionId": "",
                    "sequenceNumber": -1,
                    "valid": False,
                    "error": str(error),
                }
                destination = sender

            encoded = json.dumps(
                response,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            udp.sendto(encoded, destination)
            if verbose:
                print(
                    "{} seq={} action={} risk={} valid={}".format(
                        response.get("sessionId", ""),
                        response.get("sequenceNumber", -1),
                        response.get("executedAction", response.get("messageType", "")),
                        response.get("riskLevel", ""),
                        response.get("valid", False),
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nRoadWeave Test Lab ML service stopped.")
        return 0
    finally:
        udp.close()


def main() -> int:
    args = arguments()
    engine = create_engine(args)
    if args.self_test:
        return run_self_test(engine)
    return serve(engine, args.host, args.port, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
