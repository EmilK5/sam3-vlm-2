"""Test clean imports across all architectural module boundaries."""

import pytest


def test_top_level_import():
    import sam3_vlm
    assert hasattr(sam3_vlm, "__version__")
    assert sam3_vlm.__version__ == "0.1.0"


def test_core_imports():
    from sam3_vlm.core import (
        NodeStatus,
        ObservationRelation,
        ActionFamily,
        SpatialMode,
        ActionSource,
        BudgetState,
        ClassBelief,
        RegistrationDiagnostics,
        NodeObservationRef,
        Detection,
        IDGenerator,
        Box,
        Geometry,
        BoxGeometry,
        PolygonGeometry,
        GeometryRef,
        V4Config,
        TilingConfig,
        BudgetConfig,
        StoppingConfig,
    )
    assert NodeStatus.ACTIVE == "ACTIVE"
    assert ActionFamily.DISCOVERY == "DISCOVERY"


def test_models_imports():
    from sam3_vlm.models import SAM3Sensor, DummySAM3Sensor, QwenPlanner, DummyQwenPlanner
    dummy_sam3 = DummySAM3Sensor()
    dummy_qwen = DummyQwenPlanner()
    assert isinstance(dummy_sam3, SAM3Sensor)
    assert isinstance(dummy_qwen, QwenPlanner)


def test_scene_imports():
    from sam3_vlm.scene import Node, SceneGraph, AssociationPolicy, SemanticRecord, SemanticMemory
    graph = SceneGraph()
    assert len(graph.nodes) == 0


def test_sensing_imports():
    from sam3_vlm.core import TilingConfig
    from sam3_vlm.sensing import SensingAction, SAM3Observation, EvidencePack, compute_tiles
    tiles = compute_tiles(1000, 1000, TilingConfig())
    assert len(tiles) == 4


def test_planning_imports():
    from sam3_vlm.planning import ActionBank, PlannerService, UtilityEvaluator, StoppingCondition
    bank = ActionBank()
    assert len(bank.actions) == 0


def test_pipeline_imports():
    from sam3_vlm.pipeline import BootstrapStage, RunnerState, ResidualCleanupStage
    assert RunnerState.INITIALIZE == "INITIALIZE"


def test_logging_imports():
    from sam3_vlm.logging import Event, ProvenanceRecord, ProvenanceTracker, RunArtifactPaths
    tracker = ProvenanceTracker()
    assert len(tracker.records) == 0


def test_datasets_imports():
    from sam3_vlm.datasets import CountingDataset, Sample, GroundTruth
    gt = GroundTruth(count=5)
    assert gt.count == 5


def test_evaluation_imports():
    from sam3_vlm.evaluation import CountingMetrics
    metrics = CountingMetrics(
        absolute_error=1.0, squared_error=1.0, relative_error=0.1, true_count=10, predicted_count=9
    )
    assert metrics.predicted_count == 9
