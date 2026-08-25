from enum import IntEnum


class DrivingAction(IntEnum):
    KEEP = 0
    ACCELERATE = 1
    DECELERATE = 2
    CHANGE_LEFT = 3
    CHANGE_RIGHT = 4


class RiskLevel(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    EXTREME = 3


ACTION_NAMES = {
    DrivingAction.KEEP: "KEEP",
    DrivingAction.ACCELERATE: "ACCELERATE",
    DrivingAction.DECELERATE: "DECELERATE",
    DrivingAction.CHANGE_LEFT: "CHANGE_LEFT",
    DrivingAction.CHANGE_RIGHT: "CHANGE_RIGHT",
}


RISK_NAMES = {
    RiskLevel.LOW: "LOW",
    RiskLevel.MODERATE: "MODERATE",
    RiskLevel.HIGH: "HIGH",
    RiskLevel.EXTREME: "EXTREME",
}


# K-Risk uses these action IDs in its GPT-generated responses.
KRISK_ACTION_MAP = {
    1: DrivingAction.KEEP,
    2: DrivingAction.CHANGE_LEFT,
    3: DrivingAction.CHANGE_RIGHT,
    4: DrivingAction.ACCELERATE,
    5: DrivingAction.DECELERATE,
}