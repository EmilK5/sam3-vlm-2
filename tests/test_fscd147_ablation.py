from dataclasses import asdict, replace
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from sam3_vlm.experiments.fscd147_ablation import build_trials, main, paired_result, report_suite, run_suite, trial_manifest
from sam3_vlm.experiments.fscd147 import run_dataset
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.planning.qwen_planner import PlannerOutput
from test_fscd147_ae import dataset, deployment


class NoopPlanner:
    model = 'mock'
    def plan_scene(self, evidence, budget, config):
        return PlannerOutput()


def test_suite_isolates_switches_and_has_270_runs(deployment, dataset):
    trials = {trial.name: trial for trial in build_trials(deployment)}
    assert sum(len(trial.arms) for trial in trials.values()) == 27
    control = trials['singular_control'].deployment.v4_config
    assert trials['plural_control'].deployment.v4_config == replace(control,
        sam3=replace(control.sam3, singularize_prompts=False))
    assert trials['safe_negatives'].deployment.v4_config == replace(control,
        planner=replace(control.planner, validate_confounders=True))
    assert trials['neutral_appearance_misses'].deployment.v4_config == replace(control,
        belief=replace(control.belief, neutral_appearance_misses=True))
    assert trials['adaptive_e'].deployment.v4_config == replace(control,
        replanning=replace(control.replanning, adaptive_e_zero_gain_patience=3))
    assert trials['counting_unit'].deployment.v4_config == control
    assert trials['counting_unit'].target_overrides == {'donuts tray': 'donut'}
    assert trials['adaptive_e'].arms == 'E'
    assert trial_manifest(dataset, list(trials.values()))['expected_runs'] == 54
    assert all(trial.deployment.seed == deployment.seed for trial in trials.values())


def test_counting_unit_changes_only_inference_and_freezes_original_label(dataset, deployment, tmp_path, monkeypatch):
    class CaptureSensor(MockSAM3Adapter):
        def observe(self, image, action):
            assert action.prompt == 'donut'
            return super().observe(image, action)
    classes = dataset.root / 'ImageClasses_FSC147.txt'
    classes.write_text('a.jpg donuts tray\nb.jpg donuts tray\n')
    from sam3_vlm.datasets.fscd147 import FSCD147
    dataset = FSCD147(dataset.root)
    source = classes.read_bytes()
    original = Path.read_text
    def guard(path, *args, **kwargs):
        if path.name == 'instances_val.json':
            pytest.fail('Inference read annotations')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', guard)
        path = run_dataset(dataset, deployment, CaptureSensor(), None, tmp_path / 'override',
                           arm='A', target_overrides={'donuts tray': 'donut'}, audit=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row['success'] and row['target'] == 'donuts tray' and row['inference_target'] == 'donut' for row in rows)
    assert all(row['mask_audit']['complete'] for row in rows)
    assert classes.read_bytes() == source
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    assert metadata['target_overrides'] == {'donuts tray': 'donut'}


def test_suite_executes_all_profiles_and_writes_one_portable_text_bundle(dataset, deployment, tmp_path):
    output = tmp_path / 'suite'
    result = run_suite(dataset, build_trials(deployment), MockSAM3Adapter(), NoopPlanner(), output)
    assert result['complete']
    comparison = json.loads((output / 'comparison.json').read_text())
    assert comparison['expected_runs'] == 54 and len(comparison['aggregates']) == 7
    for profile in comparison['paired_comparisons'].values():
        assert profile['available'] and all(arm['complete'] and arm['paired_images'] == 2 for arm in profile['arms'].values())
    with ZipFile(result['archive']) as bundle:
        assert len(bundle.namelist()) == 24
        assert all(Path(name).suffix in {'.md', '.json'} for name in bundle.namelist())
        assert 'combined/visual_notes.md' in bundle.namelist()
    # Rebuilding keeps manually entered notes and never requires inference models.
    notes = output / 'combined/report/visual_notes.md'
    notes.write_text('My visual observations: fragments remain.\n')
    report_suite(dataset.root, output)
    assert 'fragments remain' in notes.read_text()
    with pytest.raises(FileExistsError):
        run_suite(dataset, build_trials(deployment), MockSAM3Adapter(), NoopPlanner(), output)


def test_failed_runs_keep_incomplete_metrics_and_missing_reference(dataset, deployment, tmp_path):
    class Failure(MockSAM3Adapter):
        def observe(self, *args, **kwargs):
            raise RuntimeError('Model failed')
    trial = next(trial for trial in build_trials(deployment) if trial.name == 'adaptive_e')
    result = run_suite(dataset, [trial], Failure(), NoopPlanner(), tmp_path / 'failed')
    assert not result['complete']
    comparison = json.loads((tmp_path / 'failed/comparison.json').read_text())
    assert not comparison['paired_comparisons']['adaptive_e']['available']
    aggregate = next(iter(comparison['aggregates']['adaptive_e'].values()))
    assert aggregate['mae'] is None and aggregate['failed_runs'] == 2


def test_paired_metrics_are_unavailable_for_incomplete_pairs():
    current = {'i': {'success': True, 'predicted_count': 9, 'absolute_error': 1, 'runtime_seconds': 3}}
    reference = {'i': {'success': True, 'predicted_count': 7, 'absolute_error': 3, 'runtime_seconds': 5}}
    complete = paired_result(current, reference, ['i'])
    assert complete['mean_absolute_error_reduction'] == 2 and complete['mean_runtime_change_seconds'] == -2
    partial = paired_result(current, reference, ['i', 'missing'])
    assert not partial['complete'] and partial['mean_absolute_error_reduction'] is None


def test_dry_run_checks_split_size_and_never_reads_annotations(dataset, tmp_path, monkeypatch, capsys):
    config = Path(__file__).resolve().parents[1] / 'configs/fscd147.json'
    arguments = ['run', str(dataset.root), str(tmp_path / 'out'), '--config', str(config), '--dry-run']
    with pytest.raises(SystemExit):
        main(arguments)
    original = Path.read_text
    def guard(path, *args, **kwargs):
        if path.name == 'instances_val.json':
            pytest.fail('Dry run accessed annotations')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guard)
    assert main(arguments + ['--allow-full-split']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['expected_runs'] == 54
    assert not (tmp_path / 'out').exists()


def test_report_rejects_config_drift(dataset, deployment, tmp_path):
    trials = [build_trials(deployment)[0]]
    output = tmp_path / 'suite'
    run_suite(dataset, trials, MockSAM3Adapter(), NoopPlanner(), output)
    path = output / 'plural_control/metadata.json'
    metadata = json.loads(path.read_text())
    metadata['model_settings']['seed'] += 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Frozen profile'):
        report_suite(dataset.root, output)


def test_selected_profiles_have_their_own_frozen_counts(dataset, tmp_path, capsys):
    config = Path(__file__).resolve().parents[1] / 'configs/fscd147.json'
    assert main(['run', str(dataset.root), str(tmp_path / 'out'), '--config', str(config),
        '--dry-run', '--allow-full-split', '--profiles', 'plural_control', 'singular_control', 'combined']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['expected_runs'] == 30 and len(result['profiles']) == 3


def test_audit_failure_is_explicit_after_successful_inference(dataset, deployment, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('Audit failed')
    monkeypatch.setattr('sam3_vlm.experiments.mask_audit.audit_masks', broken)
    path = run_dataset(dataset, deployment, MockSAM3Adapter(), None, tmp_path / 'audit_failed', arm='A', audit=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(not row['success'] and row['predicted_count'] is None and row['error'] == 'Audit failed' for row in rows)


def test_run_cli_loads_models_once_for_multiple_profiles(dataset, tmp_path, monkeypatch):
    from unittest.mock import Mock
    sensor = Mock(return_value=MockSAM3Adapter())
    planner = Mock(return_value=NoopPlanner())
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('sam3_vlm.models.sam3.RealSAM3Sensor', sensor)
    monkeypatch.setattr('sam3_vlm.models.qwen.RealQwenPlanner', planner)
    config = Path(__file__).resolve().parents[1] / 'configs/fscd147.json'
    output = tmp_path / 'cli_suite'
    assert main(['run', str(dataset.root), str(output), '--config', str(config),
                 '--allow-full-split', '--profiles', 'plural_control', 'singular_control']) == 0
    assert sensor.call_count == planner.call_count == 1
    assert (output / 'summary.zip').exists()
