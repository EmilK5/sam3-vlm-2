import time
from dataclasses import replace

import pytest

from sam3_vlm.core.config import BootstrapConfig, BudgetConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import (
    ActionFamily,
    ActionSource,
    BudgetState,
    ClassBelief,
    Detection,
    NodeObservationRef,
    ObservationRelation,
    SpatialMode,
    StopReason,
)
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.planning.action_bank import (
    ActionBank,
    ActionBankGenerator,
    ActionRejectionReason,
)
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import DiscoveryState, SceneState
from sam3_vlm.sensing.action import SensingAction, validate_sam3_prompt_contract
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from sam3_vlm.sensing.observation import SAM3Observation


def _det(det_id, x1, y1, x2, y2, score=0.9):
    return Detection(
        detection_id=det_id,
        geometry=GeometryRef(Box(float(x1), float(y1), float(x2), float(y2))),
        score=score,
    )


def test_sam3_prompt_contract_is_strict_visual_noun_phrase():
    for prompt in ("green fruit", "round green fruit", "tree canopy", "flat leaf"):
        validate_sam3_prompt_contract(prompt)

    for prompt in (
        "fruit",
        "partially occluded round green fruit",
        "use green fruit",
        "spectral green fruit",
        "edge detected fruit",
        "green fruit in canopy",
    ):
        with pytest.raises(ValueError):
            validate_sam3_prompt_contract(prompt)


def test_belief_ontology_never_expands_from_qwen_aliases():
    node = Node(
        node_id="node_000001",
        geometry=BoxGeometry(Box(0, 0, 20, 20)),
        class_belief=ClassBelief(),
    )
    action = SensingAction(
        action_id="action_000001",
        semantic_key="green_fruit",
        prompt="green fruit",
        family=ActionFamily.DISCOVERY,
        semantic_prior={"target": 1.0},
    )
    obs = NodeObservationRef(
        observation_id="obs_000001",
        sam3_call_id="sam3_000001",
        action_id=action.action_id,
        semantic_key=action.semantic_key,
        relation=ObservationRelation.STRONG_MATCH,
        score=0.9,
    )
    updater = BeliefUpdater()
    updater.update_node_belief(
        node,
        action,
        obs,
        target_class="target",
        class_vocabulary=["target", "confounder1", "confounder2"],
    )
    assert list(node.class_belief.probabilities) == ["target", "confounder1", "confounder2"]

    bad_action = replace(action, semantic_prior={"Green Citrus": 0.9})
    with pytest.raises(ValueError, match="expand belief ontology"):
        updater.update_node_belief(
            node,
            bad_action,
            obs,
            target_class="target",
            class_vocabulary=["target", "confounder1", "confounder2"],
        )
    assert set(node.class_belief.probabilities) == {"target", "confounder1", "confounder2"}


def test_action_bank_rejects_bad_qwen_actions_and_inherits_tiling_and_search_region():
    cfg = V4Config()
    bank = ActionBank()
    generator = ActionBankGenerator()
    domain = BoxGeometry(Box(100, 120, 700, 800))
    output = PlannerOutput(
        proposed_actions=[
            ProposedAction(
                semantic_key="target",
                prompt="round green fruit",
                family=ActionFamily.DISCOVERY,
                suggested_spatial_mode=SpatialMode.TILED,
                semantic_prior={"target": 1.0},
            ),
            ProposedAction(
                semantic_key="fake_method",
                prompt="spectral green fruit",
                family=ActionFamily.DISCOVERY,
                semantic_prior={"target": 1.0},
            ),
            ProposedAction(
                semantic_key="bad_prior",
                prompt="flat green leaf",
                family=ActionFamily.CONFOUNDER,
                semantic_prior={"leaf": 0.9},
            ),
            ProposedAction(
                semantic_key="target",
                prompt="small green fruit",
                family=ActionFamily.DISCOVERY,
                suggested_spatial_mode=SpatialMode.ROI_BATCH,
                semantic_prior={"target": 1.0},
            ),
        ]
    )
    entries = generator.generate_entries(
        output,
        SemanticMemory(),
        bank,
        IDGenerator(),
        config=cfg,
        search_region=domain,
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )
    assert len(entries) == 1
    action = entries[0].action
    assert action.roi is None
    assert action.search_region.bbox().as_tuple() == domain.bbox().as_tuple()
    assert action.spatial_mode == SpatialMode.TILED
    assert action.tiling == cfg.tiling

    reasons = {r.reason for r in generator.last_rejections}
    assert ActionRejectionReason.INVALID_GROUNDING_PROMPT.value in reasons
    assert ActionRejectionReason.UNKNOWN_CLASS_PRIOR.value in reasons
    assert ActionRejectionReason.MISSING_ROI.value in reasons


def test_discovery_coverage_is_relative_to_locked_domain_and_zero_gains_age_plateau():
    state = DiscoveryState()
    domain = BoxGeometry(Box(100, 100, 500, 500))
    state.record_search([domain], domain)
    assert state.spatial_coverage.coverage_ratio == pytest.approx(1.0)
    assert state.spatial_coverage.total_area_searched == pytest.approx(160000.0)

    state.record_discovery_gain(1, ["node_000001"], plateau_window=3)
    state.record_discovery_gain(0, plateau_window=3)
    state.record_discovery_gain(0, plateau_window=3)
    assert state.recent_new_node_counts == [1.0, 0.0, 0.0]
    assert state.plateau_score == pytest.approx(2.0 / 3.0)
    assert not state.saturated
    state.record_discovery_gain(0, plateau_window=3)
    assert state.recent_new_node_counts[-3:] == [0.0, 0.0, 0.0]
    assert state.saturated


def test_mock_sam3_global_and_tiled_search_only_inside_locked_roi():
    sensor = MockSAM3Adapter()
    roi = BoxGeometry(Box(100, 100, 500, 500))
    global_action = SensingAction(
        action_id="action_global",
        semantic_key="green_fruit",
        prompt="green fruit",
        family=ActionFamily.DISCOVERY,
        search_region=roi,
        spatial_mode=SpatialMode.GLOBAL,
    )
    global_obs = sensor.observe((1000, 1000), global_action)
    assert [g.bbox().as_tuple() for g in global_obs.searched_regions] == [(100, 100, 500, 500)]

    tiled_action = replace(
        global_action,
        action_id="action_tiled",
        spatial_mode=SpatialMode.TILED,
        tiling=V4Config().tiling,
    )
    tiled_obs = sensor.observe((1000, 1000), tiled_action)
    assert tiled_obs.searched_regions
    for geom in tiled_obs.searched_regions:
        b = geom.bbox()
        assert 100 <= b.x1 < b.x2 <= 500
        assert 100 <= b.y1 < b.y2 <= 500

    coverage = DiscoveryState()
    coverage.record_search(tiled_obs.searched_regions, roi)
    assert coverage.spatial_coverage.coverage_ratio == pytest.approx(1.0)


class _CanopySensor:
    def __init__(self, canopy_detections):
        self.canopy_detections = canopy_detections
        self.actions = []
        self.call_count = 0
        self.model_id = "fake-sam3"

    def observe(self, image, action):
        action.validate()
        self.actions.append(action)
        self.call_count += 1
        if action.semantic_key == "locked_context_region":
            detections = list(self.canopy_detections)
            searched = [BoxGeometry(Box(0, 0, 1000, 1000))]
        else:
            detections = [_det(f"target_{self.call_count}", 200, 220, 240, 260)]
            domain = (
                action.search_region
                if action.spatial_mode in (SpatialMode.GLOBAL, SpatialMode.TILED)
                else action.roi
            )
            searched = [domain] if domain is not None else []
        return SAM3Observation(
            call_id=f"sam3_{self.call_count:06d}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=detections,
            searched_regions=searched,
            runtime_ms=3.0,
            model_metadata={"model_id": self.model_id},
        )


def test_citrus_bootstrap_locks_enclosing_canopy_before_target_and_excludes_context_nodes(monkeypatch):
    sensor = _CanopySensor([
        _det("canopy_a", 100, 120, 350, 600),
        _det("canopy_b", 300, 80, 700, 650),
    ])
    cfg = V4Config(
        bootstrap=BootstrapConfig(
            enable_tiled_bootstrap=False,
            locked_context_prompt="tree canopy",
            locked_context_threshold=0.4,
        )
    )
    monkeypatch.setattr(
        "sam3_vlm.pipeline.bootstrap.ContactSheetBuilder.build_contact_sheet",
        lambda self, **kwargs: ContactSheet(crops=[], total_candidates=len(kwargs["graph"].active_nodes())),
    )
    result = BootstrapPipeline(sensor=sensor, config=cfg).execute_bootstrap(
        image_id="img", image="dummy.jpg", user_prompt="green citrus"
    )
    state = result.state
    assert sensor.actions[0].prompt == "tree canopy"
    assert sensor.actions[0].roi is None
    assert sensor.actions[1].prompt == "green citrus"
    assert sensor.actions[1].roi is None
    assert sensor.actions[1].search_region.bbox().as_tuple() == (100, 80, 700, 650)
    assert state.search_region.bbox().as_tuple() == (100, 80, 700, 650) 
    assert state.search_region_locked
    assert not state.search_region_fallback_used
    assert len(state.graph.nodes) == 1  # canopy detections never become count nodes
    assert state.budget.sam3_calls == 2
    assert state.discovery_state.spatial_coverage.coverage_ratio == pytest.approx(1.0)


def test_citrus_bootstrap_canopy_failure_falls_back_to_full_image(monkeypatch):
    sensor = _CanopySensor([])
    cfg = V4Config(
        bootstrap=BootstrapConfig(
            enable_tiled_bootstrap=False,
            locked_context_prompt="tree canopy",
            locked_context_fallback_full_image=True,
        )
    )
    monkeypatch.setattr(
        "sam3_vlm.pipeline.bootstrap.ContactSheetBuilder.build_contact_sheet",
        lambda self, **kwargs: ContactSheet(crops=[], total_candidates=len(kwargs["graph"].active_nodes())),
    )
    state = BootstrapPipeline(sensor=sensor, config=cfg).execute_bootstrap(
        image_id="img", image="dummy.jpg", user_prompt="green citrus"
    ).state
    assert state.search_region.bbox().as_tuple() == (0, 0, 1000, 1000)
    assert state.search_region_locked
    assert state.search_region_fallback_used
    assert "FALLBACK" in state.search_region_source
    assert sensor.actions[1].roi is None
    assert sensor.actions[1].search_region.bbox().as_tuple() == (0, 0, 1000, 1000)


class _NoopPlanner:
    model = "fake-qwen"

    def plan_scene(self, evidence, budget, config):
        return PlannerOutput(scene_summary="none", proposed_actions=[])


def _empty_scene_state():
    return SceneState(
        image_id="img",
        user_prompt="green citrus",
        target_class="target",
        graph=SceneGraph(),
        semantic_memory=SemanticMemory(),
        belief_classes=["target", "confounder1", "confounder2"],
        search_region=BoxGeometry(Box(0, 0, 1000, 1000)),
        search_region_locked=True,
    )


def test_cleanup_disabled_does_not_misreport_cleanup_budget():
    cfg = V4Config(budget=BudgetConfig(max_cleanup_calls=0))
    runner = Runner(cfg, MockSAM3Adapter(), _NoopPlanner())
    runner.scene_state = _empty_scene_state()
    runner.user_prompt = "green citrus"
    runner.image = (1000, 1000)
    runner._run_start_perf = time.perf_counter()
    runner.state = RunnerState.CLEANUP
    runner._step()
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.NO_VALID_ACTIONS
    assert runner.scene_state.stop_reason != StopReason.CLEANUP_BUDGET


def test_empty_replan_without_new_evidence_does_not_burn_another_qwen_call():
    cfg = V4Config(budget=BudgetConfig(max_cleanup_calls=0))
    planner = _NoopPlanner()
    runner = Runner(cfg, MockSAM3Adapter(), planner)
    runner.scene_state = _empty_scene_state()
    runner.scene_state.action_bank = ActionBank()
    runner.scene_state.replans_executed = 1
    runner.scene_state.last_plan_accepted_actions = 0
    runner.scene_state.actions_since_replan = 0
    runner.evidence_pack = QwenEvidencePack(
        original_image_id="img",
        user_prompt="green citrus",
        target_class="target",
        contact_sheet=ContactSheet(),
    )
    runner._request_replan()
    assert runner.scene_state.budget.qwen_calls == 0
    assert runner.scene_state.stop_reason == StopReason.NO_VALID_ACTIONS
    assert runner.state == RunnerState.CLEANUP


class _SleepPlanner:
    def plan_scene(self, evidence, budget, config):
        time.sleep(0.002)
        return PlannerOutput(
            proposed_actions=[
                ProposedAction(
                    semantic_key="green_fruit",
                    prompt="green fruit",
                    family=ActionFamily.DISCOVERY,
                    semantic_prior={"target": 1.0},
                )
            ]
        )


def test_qwen_runtime_is_measured_separately_and_added_to_model_runtime():
    budget = BudgetState()
    evidence = QwenEvidencePack(
        original_image_id="img",
        user_prompt="green citrus",
        target_class="target",
        contact_sheet=ContactSheet(),
    )
    output = QwenPlannerService(_SleepPlanner()).plan_scene(evidence, budget, V4Config())
    assert output.proposed_actions
    assert budget.qwen_calls == 1
    assert budget.qwen_runtime_ms > 0.0
    assert budget.model_runtime_ms >= budget.qwen_runtime_ms
