"""Runner state machine definitions (V4 Design Spec §24)."""

from enum import Enum


class RunnerState(str, Enum):
    """Explicit pipeline runner stage enum."""

    INITIALIZE = "INITIALIZE"
    BOOTSTRAP_GLOBAL = "BOOTSTRAP_GLOBAL"
    BOOTSTRAP_TILE_DECISION = "BOOTSTRAP_TILE_DECISION"
    BOOTSTRAP_TILED = "BOOTSTRAP_TILED"
    BUILD_QWEN_EVIDENCE = "BUILD_QWEN_EVIDENCE"
    PLAN = "PLAN"
    GLOBAL_SENSING = "GLOBAL_SENSING"
    REPLAN = "REPLAN"
    CLEANUP_DECISION = "CLEANUP_DECISION"
    CLEANUP = "CLEANUP"
    FINALIZE = "FINALIZE"
    DONE = "DONE"
