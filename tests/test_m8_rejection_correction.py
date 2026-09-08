import copy
import json
from dataclasses import replace

import pytest

from sam3_vlm.core.config import V4Config, BootstrapConfig, BudgetConfig, PlannerConfig, ReplanningConfig
from sam3_vlm.core.types import ActionFamily, StopReason
from sam3_vlm.experiments.m8_smoke import assemble_e2e_runner, _run_validator_and_replay, _pilot_variants
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction


def proposal(prompt, confounders=()):
    return PlannerOutput(proposed_actions=[ProposedAction(
        'target', prompt, ActionFamily.DISCOVERY, semantic_prior={'target': 1.0},
    )], likely_confounders=list(confounders))


class SequencePlanner:
    strict_model_errors = True
    model = 'sequence-test'

    def __init__(self, outputs):
        self.outputs = outputs
        self.evidence = []

    def plan_scene(self, evidence, budget, config):
        self.evidence.append(copy.deepcopy(evidence))
        return copy.deepcopy(self.outputs[min(len(self.evidence)-1, len(self.outputs)-1)])


def run_case(tmp_path, outputs, calls=2, **planner_fields):
    config = V4Config(
        bootstrap=BootstrapConfig(enable_tiled_bootstrap=False),
        budget=BudgetConfig(max_qwen_calls=calls, max_cleanup_calls=0),
        planner=PlannerConfig(enable_rejection_correction=True, **planner_fields),
        replanning=ReplanningConfig(max_replans=1), assets_dir=str(tmp_path / 'assets'),
    )
    sensor, planner = MockSAM3Adapter(), SequencePlanner(outputs)
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, config, sensor, planner, 'test', 'green fruit', 'target', 'img')
    runner.run((100, 100), 'green fruit', image_id='img')
    assert _run_validator_and_replay(paths, runner.scene_state)
    artifacts = [json.loads(p.read_text()) for p in sorted(paths.qwen_dir.glob('*.json'))]
    return runner, planner, artifacts


@pytest.mark.parametrize('initial', [proposal('green fruit in shadow'), proposal('green fruit'), PlannerOutput()])
def test_invalid_duplicate_and_empty_plans_get_one_audited_correction(tmp_path, initial):
    runner, planner, artifacts = run_case(tmp_path, [initial, proposal('dark green fruit')])
    assert runner.scene_state.budget.qwen_calls == 2
    assert runner.scene_state.budget.sam3_calls == 2  # bootstrap + valid corrected action
    assert runner.scene_state.replans_executed == 0
    assert len(artifacts) == 2
    assert artifacts[1]['metadata']['correction_of'] == artifacts[0]['qwen_call_id']
    assert artifacts[1]['metadata']['accepted_action_count'] == 1
    assert 'previous_plan_feedback' not in planner.evidence[0].discovery_diagnostics
    feedback = planner.evidence[1].discovery_diagnostics['previous_plan_feedback']
    assert feedback['rejections'] or feedback['contract_diagnostic'] == 'EMPTY_UNSATURATED_PLAN'
    assert all(e.user_prompt == 'green fruit' for e in planner.evidence)
    assert planner.evidence[0].contact_sheet.to_dict() == planner.evidence[1].contact_sheet.to_dict()
    assert planner.evidence[0].image_path == planner.evidence[1].image_path
    assert planner.evidence[0].scene_summary == planner.evidence[1].scene_summary


@pytest.mark.parametrize('calls,expected', [(1, 1), (2, 2), (100, 2)])
def test_failed_correction_is_bounded_even_with_large_budget(tmp_path, calls, expected):
    runner, planner, artifacts = run_case(tmp_path, [proposal('green fruit in shadow')], calls=calls)
    assert len(planner.evidence) == runner.scene_state.budget.qwen_calls == expected
    assert runner.scene_state.budget.sam3_calls == 1
    assert runner.scene_state.stop_reason == StopReason.NO_VALID_ACTIONS
    assert len(artifacts) == expected


def test_json_repair_does_not_trigger_a_third_attempt(tmp_path):
    runner, planner, artifacts = run_case(tmp_path, ['bad json', proposal('green fruit in shadow')], calls=100)
    assert runner.scene_state.budget.qwen_calls == len(planner.evidence) == 2
    assert len(artifacts) == 1 and artifacts[0]['metadata']['repair_attempted']
    assert planner.evidence[1].user_prompt == 'green fruit'
    assert planner.evidence[1].discovery_diagnostics['previous_plan_feedback']['rejections'][0]['reason'] == 'MALFORMED_JSON'


def test_partial_rejection_waits_for_normal_replan(tmp_path):
    runner, planner, artifacts = run_case(tmp_path, [
        proposal('dark green fruit', ['leaf cluster']), proposal('small green fruit', ['leaves']),
    ], execute_confounder_prompts=True)
    assert len(planner.evidence) == 2
    assert runner.scene_state.replans_executed == 1
    assert all(a['metadata']['correction_of'] is None for a in artifacts)
    feedback = planner.evidence[1].discovery_diagnostics['previous_plan_feedback']
    assert feedback['rejections'][0]['sam3_prompt'] == 'leaf cluster'
    assert 'dark green fruit' in planner.evidence[1].discovery_diagnostics['tried_sam3_prompts']


def test_runtime_budget_prevents_correction(tmp_path, monkeypatch):
    from sam3_vlm.pipeline.runner import Runner
    monkeypatch.setattr(Runner, '_elapsed_wall_ms', lambda self: 1_000_000.)
    runner, planner, artifacts = run_case(tmp_path, [proposal('green fruit in shadow')])
    assert len(planner.evidence) == runner.scene_state.budget.qwen_calls == 1


def test_recovery_suite_has_unchanged_reference_and_isolated_changes():
    from dataclasses import asdict
    configs = [v.config for v in _pilot_variants(V4Config(), 'recovery-ablation')]
    a, b, c = [asdict(v) for v in configs]
    assert not a['planner']['enable_rejection_correction']
    assert b['planner']['enable_rejection_correction']
    a['planner']['enable_rejection_correction'] = True
    assert a == b
    assert b['planner']['prompt_version'] == 'old' and c['planner']['prompt_version'] == 'v4'
    b['planner']['prompt_version'] = 'v4'
    assert b == c
    for cfg in configs:
        assert cfg.budget.max_qwen_calls == 2
        assert cfg.planner.execute_confounder_prompts
        assert cfg.belief.target_count_hard_threshold is None


def test_recovery_pilot_subset_and_compact_export(tmp_path, monkeypatch):
    from PIL import Image
    from zipfile import ZipFile
    from test_m8_orchestration import DummyArgs
    from sam3_vlm.experiments.m8_smoke import m8_4_and_5_pilot
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': str(i), 'image_path': str(image), 'gt_count': 2,
                                    'target': 'green fruit'} for i in range(3)]))
    planner = SequencePlanner([proposal('green fruit in shadow')])
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models', lambda args: (MockSAM3Adapter(), planner))
    args = DummyArgs(manifest=str(manifest), sample_ids=['2'], max_samples=1,
                     pilot_suite='recovery-ablation', output_dir=str(tmp_path / 'runs'))
    assert m8_4_and_5_pilot(args)
    report = json.loads((tmp_path / 'runs/pilot_report.json').read_text())
    assert len(report['samples']) == 3
    assert {r['sample_id'] for r in report['samples']} == {'2'}
    assert [r['qwen_calls'] for r in report['samples']] == [1, 2, 2]
    assert set(report['paired_comparisons']) == {'D_rejection_correction', 'D_evidence_prompt'}
    with ZipFile(tmp_path / 'runs/compact_review.zip') as z:
        d = json.loads(z.read('review_summary.json'))['diagnostics']
        assert d['D_ReferenceOld']['rejection_correction_calls'] == 0
        assert d['D_OldWithCorrection']['rejection_correction_calls'] == 1
        assert d['D_V4WithCorrection']['rejection_correction_calls'] == 1


def test_unknown_sample_id_is_rejected(tmp_path):
    from PIL import Image
    from test_m8_orchestration import DummyArgs
    from sam3_vlm.experiments.m8_smoke import _load_pilot_samples
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (10, 10)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': '1', 'image_path': str(image), 'gt_count': 2}]))
    with pytest.raises(ValueError, match='Unknown sample IDs'):
        _load_pilot_samples(DummyArgs(manifest=str(manifest), sample_ids=['wrong']), 1)
