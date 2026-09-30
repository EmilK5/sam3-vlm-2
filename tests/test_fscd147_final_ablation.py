from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from sam3_vlm.core.config import PlannerConfig, V4Config
from sam3_vlm.core.types import ActionFamily, BudgetState
from sam3_vlm.experiments.fscd147 import run_dataset
from sam3_vlm.experiments.fscd147_ablation import build_trials, main, run_suite, trial_manifest
from sam3_vlm.experiments.fscd147_report import write_summary
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from sam3_vlm.sensing.mask_validation import prepare_mask_observation
from sam3_vlm.sensing.observation import SAM3Observation
from test_fscd147_ae import dataset, deployment
from test_fscd147_ablation import NoopPlanner
from test_m8_qwen_payload import mock_openai_client
from test_mask_association import det


def test_final_suite_freezes_three_models_and_only_varies_stopping(dataset, deployment):
    reference, adaptive = build_trials(deployment, 'final')
    assert reference.name == 'final_reference' and reference.arms == 'DE'
    assert adaptive.name == 'final_adaptive' and adaptive.arms == 'E'
    assert adaptive.reference == reference.name
    config = reference.deployment.v4_config
    assert config.sam3.singularize_prompts
    assert not config.planner.validate_confounders and not config.belief.neutral_appearance_misses
    assert config.planner.temperature == 0 and config.planner.sampling_seed == deployment.seed
    assert config.planner.compact_json and config.planner.max_output_tokens == 1024
    assert reference.target_overrides == adaptive.target_overrides == {'donuts tray': 'donut'}
    assert adaptive.deployment.v4_config == replace(config,
        replanning=replace(config.replanning, adaptive_e_zero_gain_patience=3))
    assert trial_manifest(SimpleNamespace(split='val', image_ids=list(range(10))), [reference, adaptive])['expected_runs'] == 30
    with pytest.raises(ValueError):
        build_trials(deployment, 'unknown')


def test_final_mock_run_builds_complete_nine_file_bundle(dataset, deployment, tmp_path):
    output = tmp_path / 'final'
    result = run_suite(dataset, build_trials(deployment, 'final'), MockSAM3Adapter(), NoopPlanner(), output)
    assert result['complete']
    comparison = json.loads((output / 'comparison.json').read_text())
    assert comparison['expected_runs'] == 6
    assert comparison['cross_arm_comparisons']['reference_D_to_E']['paired_images'] == 2
    assert set(comparison['aggregates']) == {'final_reference', 'final_adaptive'}
    pair = next(iter(comparison['paired_comparisons']['final_adaptive']['arms'].values()))
    assert pair['complete'] and pair['paired_images'] == 2
    with ZipFile(result['archive']) as bundle:
        assert len(bundle.namelist()) == 9
        assert 'final_reference/visual_notes.md' in bundle.namelist()
        assert 'final_adaptive/summary.json' in bundle.namelist()


def test_final_cli_dry_run_and_ten_image_guard(dataset, tmp_path, capsys):
    config = Path(__file__).resolve().parents[1] / 'configs/fscd147_final.json'
    args = ['run', str(dataset.root), str(tmp_path / 'out'), '--suite', 'final', '--config', str(config), '--dry-run']
    with pytest.raises(SystemExit):
        main(args)
    assert main(args + ['--allow-full-split']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {'images': 2, 'expected_runs': 6, 'profiles': {'final_reference': 'DE', 'final_adaptive': 'E'}}
    assert not (tmp_path / 'out').exists()


def test_mask_filter_is_idempotent_and_preserves_coverage_runtime_and_pixels():
    pixels = np.eye(3, dtype=bool)
    good, empty = det('good', pixels), det('empty', np.zeros((3, 3), dtype=bool))
    obs = SAM3Observation('sam1', 'act1', 'target', [empty, good], runtime_ms=123)
    assert prepare_mask_observation(obs, mask_only=True) is obs
    assert obs.detections == [good] and np.array_equal(good.raw_metadata['mask'], pixels)
    assert obs.runtime_ms == 123
    audit = obs.model_metadata['mask_validation']
    assert audit['raw_detections'] == 2 and audit['accepted_detections'] == audit['empty_masks_skipped'] == 1
    prepare_mask_observation(obs, mask_only=True)
    assert obs.model_metadata['mask_validation'] == audit


@pytest.mark.parametrize('mask,metadata', [
    (None, {}), (np.empty((0, 3)), {}), (np.ones(3), {}),
    (np.array([[float('nan')]]), {}), (np.array([[.5]]), {}),
    (np.zeros((3, 3)), {'mask_offset_x': -1}),
    (np.zeros((3, 3)), {'crop_boundary_clipped': 'true'}),
])
def test_mask_filter_cannot_hide_malformed_masks(mask, metadata):
    detection = det('bad', mask)
    if mask is None:
        detection.raw_metadata.pop('mask')
    detection.raw_metadata.update(metadata)
    obs = SAM3Observation('sam1', 'a1', 'target', [detection])
    with pytest.raises(ValueError):
        prepare_mask_observation(obs, mask_only=True)
    assert obs.detections == [detection]


class EmptyProposalSensor(MockSAM3Adapter):
    def __init__(self, all_empty=False):
        super().__init__()
        self.all_empty = all_empty

    def observe(self, image, action):
        obs = super().observe(image, action)
        if self.all_empty:
            obs.detections = []
        obs.detections.append(det(f'empty_{self.call_count}', np.zeros((3, 3), dtype=bool)))
        return obs


def test_all_empty_bootstrap_remains_valid_zero_count_with_paid_call(dataset, deployment, tmp_path):
    sensor = EmptyProposalSensor(all_empty=True)
    path = run_dataset(dataset, deployment, sensor, None, tmp_path / 'empty', arm='A')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row['success'] and row['predicted_count'] == 0 and row['budget']['sam3_calls'] == 1 for row in rows)
    report = json.loads(Path(write_summary(dataset.root, path)['json']).read_text())
    aggregate = next(iter(report['aggregates'].values()))
    assert aggregate['empty_masks_skipped'] == 2
    assert aggregate['complete']


def test_mixed_empty_proposals_survive_bootstrap_and_qwen_sensing_with_valid_replay(dataset, deployment, tmp_path):
    class Discovery:
        def plan_scene(self, evidence, budget, config):
            prompt = 'shadowed ' + evidence.user_prompt
            if prompt in evidence.discovery_diagnostics.get('tried_sam3_prompts', []):
                return PlannerOutput()
            return PlannerOutput(proposed_actions=[ProposedAction('target', prompt,
                ActionFamily.DISCOVERY, semantic_prior={'target': 1.0})])
    sensor = EmptyProposalSensor()
    path = run_dataset(dataset, deployment, sensor, Discovery(), tmp_path / 'mixed', arms='DE')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row['success'] for row in rows), rows
    report = json.loads(Path(write_summary(dataset.root, path)['json']).read_text())
    assert sum(row['budget']['sam3_calls'] for row in rows) == sensor.call_count
    assert sum(arm['empty_masks_skipped'] for arm in report['aggregates'].values()) == sensor.call_count
    for image in report['per_image']:
        for row in image['variants'].values():
            assert row['diagnostics']['raw_detections'] > row['diagnostics']['association_new_nodes']


class BrokenJSON:
    strict_model_errors = True
    def plan_scene(self, evidence, budget, config):
        self.last_response_metadata = {'finish_reason': 'length', 'usage': {'completion_tokens': 1024}}
        return '{"scene_summary": "cut off'


def test_failed_qwen_calls_preserve_initial_repair_raw_responses_budget_and_zip(dataset, deployment, tmp_path):
    path = run_dataset(dataset, build_trials(deployment, 'final')[0].deployment,
                       MockSAM3Adapter(), BrokenJSON(), tmp_path / 'broken', arm='D')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(not row['success'] and row['predicted_count'] is None and row['budget']['qwen_calls'] == 2 for row in rows)
    files = list(path.parent.glob('runs/*/*/artifacts/qwen/*.json'))
    assert len(files) == 2
    for file in files:
        metadata = json.loads(file.read_text())['metadata']
        assert metadata['status'] == 'FAILED'
        assert [call['phase'] for call in metadata['call_attempts']] == ['initial', 'repair']
        assert all(call['finish_reason'] == 'length' and not call['parse_valid'] for call in metadata['call_attempts'])
    report_paths = write_summary(dataset.root, path)
    report = json.loads(Path(report_paths['json']).read_text())
    aggregate = next(iter(report['aggregates'].values()))
    assert aggregate['failed_runs'] == 2 and aggregate['mae'] is None and aggregate['qwen_malformed_attempts'] == 4
    for image in report['per_image']:
        row = next(iter(image['variants'].values()))
        diag = row['diagnostics']
        assert diag['qwen_finish_reasons'] == {'length': 2}
        assert diag['qwen_response_failures'][0]['raw_output'] == '{"scene_summary": "cut off'
        assert not diag['qwen_response_failures'][0]['raw_output_truncated']


def test_repaired_qwen_keeps_both_attempts_and_resets_next_call():
    class Repair(BrokenJSON):
        calls = 0
        def plan_scene(self, *args):
            self.calls += 1
            if self.calls == 1:
                return super().plan_scene(*args)
            self.last_response_metadata = {'finish_reason': 'stop', 'usage': {'completion_tokens': 12}}
            return '{"scene_summary":"valid","proposed_actions":[]}'
    pack = QwenEvidencePack('image', 'apple', 'target', ContactSheet([], 0))
    service = QwenPlannerService(Repair())
    budget = BudgetState()
    service.plan_scene(pack, budget)
    assert budget.qwen_calls == 2
    assert [call['parse_valid'] for call in service.last_call_attempts] == [False, True]
    service.plan_scene(pack, budget)
    assert len(service.last_call_attempts) == 1 and service.last_call_attempts[0]['phase'] == 'initial'
    assert budget.qwen_calls == 3


def test_transport_error_is_recorded_without_retrying(dataset, deployment, tmp_path):
    class Transport(BrokenJSON):
        def plan_scene(self, *args):
            raise RuntimeError('Endpoint unavailable')
    path = run_dataset(dataset, deployment, MockSAM3Adapter(), Transport(), tmp_path / 'transport', arm='D')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row['budget']['qwen_calls'] == 1 for row in rows)
    report = json.loads(Path(write_summary(dataset.root, path)['json']).read_text())
    for image in report['per_image']:
        diag = next(iter(image['variants'].values()))['diagnostics']
        assert diag['qwen_malformed_attempts'] == 0 and diag['qwen_failed_calls'] == 1
        assert diag['qwen_response_failures'][0]['error'] == 'Endpoint unavailable'


def test_qwen_final_payload_seed_compactness_and_response_usage(mock_openai_client, tmp_path):
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (10, 10)).save(image)
    response = mock_openai_client.chat.completions.create.return_value
    response.choices[0].finish_reason = 'stop'
    response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130)
    planner = RealQwenPlanner(base_url='http://fake', model='mock', strict_model_errors=True)
    pack = QwenEvidencePack('image', 'apple', 'target', ContactSheet([], 0), image_path=str(image))
    config = V4Config(planner=PlannerConfig(temperature=0, sampling_seed=42, compact_json=True, max_output_tokens=1024))
    planner.plan_scene(pack, BudgetState(), config)
    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs['seed'] == 42 and kwargs['temperature'] == 0 and kwargs['max_tokens'] == 1024
    assert 'scene_summary at most 20 words' in planner.last_request_text['user']
    assert planner.last_response_metadata == {'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130}}
    json.dumps(planner.last_response_metadata)
    planner.plan_scene(pack, BudgetState(), V4Config())
    assert 'seed' not in mock_openai_client.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize('kwargs', [{'sampling_seed': -1}, {'sampling_seed': True}, {'compact_json': 'yes'}])
def test_new_planner_settings_validate(kwargs):
    with pytest.raises(ValueError):
        PlannerConfig(**kwargs)


def test_cleanup_filters_empty_masks_logs_them_and_charges_the_call():
    import time
    from unittest.mock import MagicMock
    from PIL import Image
    from sam3_vlm.core.config import AssociationConfig
    from sam3_vlm.pipeline.runner import Runner, RunnerState
    from sam3_vlm.scene.state import SceneState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    from sam3_vlm.sensing.action import SensingAction
    recorder = MagicMock()
    runner = Runner(V4Config(association=AssociationConfig(mask_only=True)),
                    EmptyProposalSensor(all_empty=True), NoopPlanner(), recorder)
    runner.scene_state = SceneState('image', 'apple', 'target', SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    runner.state = RunnerState.CLEANUP
    runner.image = Image.new('RGB', (384, 384))
    runner._run_start_perf = time.perf_counter()
    action = SensingAction('cleanup', 'target', 'apple', ActionFamily.VERIFICATION)
    runner.cleanup_controller = SimpleNamespace(select_residual_nodes=lambda *args: [],
        generate_cleanup_action=lambda *args, **kwargs: SimpleNamespace(action=action))
    runner._step()
    assert runner.scene_state.budget.sam3_calls == runner.scene_state.budget.cleanup_calls == 1
    assert not runner.scene_state.graph.active_nodes()
    observation = recorder.record_sam3_observation.call_args.args[1]
    assert not observation.detections and observation.model_metadata['mask_validation']['empty_masks_skipped'] == 1
