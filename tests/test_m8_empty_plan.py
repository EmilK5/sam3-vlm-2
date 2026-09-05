"""Empty M8 plans remain visible contract failures without extra model calls."""

import json
from dataclasses import replace

import pytest

from sam3_vlm.core.config import BootstrapConfig, BudgetConfig, ReplanningConfig, V4Config
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, BudgetState, StopReason
from sam3_vlm.experiments.m8_smoke import _run_validator_and_replay, assemble_e2e_runner
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.models.sam3 import DummySAM3Sensor, MockSAM3Adapter
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack


@pytest.mark.parametrize("saturation", [False, None, True])
def test_empty_m8_plan_persists_diagnostic_without_sensor_action(tmp_path, saturation):
    class EmptyPlanner(MockQwenPlanner):
        def plan_scene(self, evidence, *args):
            if saturation is None:
                evidence.discovery_diagnostics.pop("discovery_saturated")
            else:
                assert evidence.discovery_diagnostics["discovery_saturated"] is saturation
            return super().plan_scene(evidence, *args)

    planner = EmptyPlanner(custom_output=PlannerOutput(scene_summary="Candidates look convincing."))
    sensor = DummySAM3Sensor() if saturation is True else MockSAM3Adapter()
    config = V4Config(
        bootstrap=BootstrapConfig(enable_tiled_bootstrap=False),
        budget=BudgetConfig(max_qwen_calls=2, max_cleanup_calls=0),
        replanning=ReplanningConfig(discovery_plateau_steps=1, max_replans=1),
        assets_dir=str(tmp_path / "assets"),
    )
    paths = RunArtifactPaths(tmp_path / "run")
    runner, _ = assemble_e2e_runner(paths, config, sensor, planner, "test", "green citrus", "target", "img")
    runner.run((100, 100), "green citrus", image_id="img")

    artifact_path, = paths.qwen_dir.glob("*.json")
    artifact = json.loads(artifact_path.read_text())
    metadata = artifact["metadata"]
    assert metadata["contract_diagnostic"] == (None if saturation is True else "EMPTY_UNSATURATED_PLAN")
    assert artifact["output"]["proposed_actions"] == []
    assert metadata["rejections"] == []
    assert metadata["repair_attempted"] is False
    assert metadata["fallback_used"] is False
    assert metadata["qwen_runtime_ms"] >= 0
    assert planner.call_count == runner.scene_state.budget.qwen_calls == 1
    assert sensor.call_count == runner.scene_state.budget.sam3_calls == 1  # Bootstrap only.
    assert runner.scene_state.action_bank.entries == []
    assert runner.scene_state.replans_executed == 0
    assert runner.scene_state.stop_reason == StopReason.NO_VALID_ACTIONS
    assert _run_validator_and_replay(paths, runner.scene_state)


@pytest.mark.parametrize("strict,classes", [(True, []), (False, ["target", "confounder1"])])
@pytest.mark.parametrize("format", ["object", "dict", "json", "fenced"])
def test_empty_plan_diagnostic_does_not_trigger_repair_and_resets(strict, classes, format):
    class Backend:
        strict_model_errors = strict
        call_count = 0

        def plan_scene(self, evidence, budget, config):
            self.call_count += 1
            output = PlannerOutput(scene_summary="No actions")
            return {
                "object": output,
                "dict": output.to_dict(),
                "json": output.to_json(),
                "fenced": "```json\n" + output.to_json() + "\n```yaml",
            }[format]

    backend = Backend()
    service = QwenPlannerService(backend)
    evidence = QwenEvidencePack("img", "green citrus", "target", ContactSheet(), belief_classes=classes)
    budget = BudgetState()
    assert service.plan_scene(evidence, budget).proposed_actions == []
    assert service.last_contract_diagnostic == "EMPTY_UNSATURATED_PLAN"
    evidence.discovery_diagnostics["discovery_saturated"] = True
    assert service.plan_scene(evidence, budget).proposed_actions == []
    assert service.last_contract_diagnostic is None
    assert not service.last_repair_attempted and not service.last_fallback_used
    assert backend.call_count == budget.qwen_calls == 2


def test_generic_empty_plan_and_prompt_are_unchanged():
    planner = MockQwenPlanner(custom_output=PlannerOutput(scene_summary="No actions"))
    service = QwenPlannerService(planner)
    evidence = QwenEvidencePack("img", "cars", "car", ContactSheet())
    output = service.plan_scene(evidence, BudgetState())
    assert output.to_dict() == planner.custom_output.to_dict()
    assert service.last_contract_diagnostic is None
    assert planner.call_count == 1
    assert "If no useful new target prompt remains, return no actions." in evidence.to_prompt_text()
    canonical = replace(evidence, target_class="target", belief_classes=["target", "confounder1"])
    assert "exactly one novel target DISCOVERY experiment" in canonical.to_prompt_text()
    assert "If no useful new target prompt remains" not in canonical.to_prompt_text()


def test_canonical_m8_service_caps_round_at_one_action():
    actions = [ProposedAction("target", prompt, ActionFamily.DISCOVERY, semantic_prior={"target": 1.0})
               for prompt in ("small fruit", "round fruit")]
    planner = MockQwenPlanner(custom_output=PlannerOutput(proposed_actions=actions))
    service = QwenPlannerService(planner)
    evidence = QwenEvidencePack("img", "green citrus", "target", ContactSheet(), belief_classes=["target"])
    assert len(service.plan_scene(evidence, BudgetState()).proposed_actions) == 1
    assert len(service.plan_scene(replace(evidence, belief_classes=[]), BudgetState()).proposed_actions) == 2


def test_strict_m8_rejects_nonunit_target_prior():
    generator = ActionBankGenerator()
    proposal = ProposedAction("target", "round fruit", ActionFamily.DISCOVERY, semantic_prior={"target": 0.9})
    entries = generator.generate_entries(
        PlannerOutput(proposed_actions=[proposal]), SemanticMemory(), ActionBank(), IDGenerator(),
        enforce_qwen_contract=True, config=V4Config(),
    )
    assert entries == []
    assert generator.last_rejections[0].reason == "NON_TARGET_ACTION"
