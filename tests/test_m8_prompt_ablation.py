import json
from dataclasses import asdict
from zipfile import ZipFile

import pytest
from PIL import Image

from sam3_vlm.core.config import BeliefConfig, PlannerConfig, V4Config
from sam3_vlm.experiments.m8_smoke import _pilot_variants, m8_4_and_5_pilot
from sam3_vlm.experiments.pilot_review import prompt_comparisons, write_compact_review
from test_m8_orchestration import DummyArgs, mock_models


def test_prompt_matrix_isolates_each_factor_and_forces_soft_count():
    base = V4Config(belief=BeliefConfig(target_count_hard_threshold=.5))
    variants = _pilot_variants(base, 'prompt-ablation')
    assert len(variants) == 8
    for variant in variants:
        assert variant.count_type == 'soft_posterior_count'
        assert variant.config.belief.target_count_hard_threshold is None
        assert variant.config.belief.target_count_commit_threshold is None
        assert variant.config.sam3.qwen_prompt_threshold == .5
    for left, right in ((0, 1), (2, 3), (4, 6), (5, 7)):
        a, b = asdict(variants[left].config), asdict(variants[right].config)
        assert a['planner']['prompt_version'] == 'old'
        assert b['planner']['prompt_version'] == 'v3'
        a['planner']['prompt_version'] = 'v3'
        assert a == b
    for left, right in ((4, 5), (6, 7)):
        a, b = asdict(variants[left].config), asdict(variants[right].config)
        assert a['planner']['execute_confounder_prompts'] is False
        assert b['planner']['execute_confounder_prompts'] is True
        a['planner']['execute_confounder_prompts'] = True
        assert a == b
    for i, v in enumerate(variants):
        assert v.config.budget.max_qwen_calls == (1 if i < 2 else 2 if i < 4 else 100)
        if i < 4:
            assert v.config.planner.execute_confounder_prompts
        else:
            assert v.config.budget.max_sam3_calls == 1000
            assert v.config.budget.max_sam3_tiles is None
            assert v.config.budget.max_runtime_seconds is None
            assert v.config.stopping.max_iterations is None
            assert v.config.replanning.continue_until_saturation


def test_prompt_version_validation():
    with pytest.raises(ValueError, match='prompt_version'):
        PlannerConfig(prompt_version='typo')


def test_family_cli_rejects_incompatible_suite(monkeypatch, capsys):
    from sam3_vlm.experiments.m8_smoke import main
    monkeypatch.setattr('sys.argv', ['m8_smoke', '--stage', 'pilot', '--pilot-family', 'C'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert '--pilot-family requires --pilot-suite prompt-ablation' in capsys.readouterr().err


@pytest.mark.parametrize('family,nvariants,ncomparisons', [(None, 8, 6), ('C', 2, 1), ('D', 2, 1), ('E', 4, 4)])
def test_prompt_ablation_pilot_and_review(mock_models, tmp_path, family, nvariants, ncomparisons):
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([dict(sample_id='img', image_path=str(image), target='green fruit', gt_count=5)]))
    mock_models[1].last_request_text = {'system': 'audited system', 'user': 'audited evidence'}
    args = DummyArgs(manifest=str(manifest), output_dir=str(tmp_path / 'runs'),
                     pilot_suite='prompt-ablation', pilot_family=family, max_samples=1)
    assert m8_4_and_5_pilot(args)
    report = json.loads((tmp_path / 'runs/pilot_report.json').read_text())
    assert len(report['samples']) == nvariants
    assert len(report['paired_comparisons']) == ncomparisons
    for row in report['samples']:
        assert row['success'] and row['validator_status'] == row['replay_status'] == 'PASS'
        assert row['predicted_count'] == pytest.approx(row['raw_soft_count'])
        assert row['count_type'] == 'soft_posterior_count'
        assert row['prompt_outcomes']
    with ZipFile(tmp_path / 'runs/compact_review.zip') as bundle:
        review = json.loads(bundle.read('review_summary.json'))
        assert len(review['paired_comparisons']) == ncomparisons
        assert len(review['selected_images']) == 1
        assert len(review['preview_files']) == 1
        examples = json.loads(bundle.read('prompt_examples.json'))
        assert all(e['request_text']['system'] == 'audited system' for e in examples.values())
        assert not review['export_warnings']


def test_compact_review_pairs_failures_rejections_and_case_limit(tmp_path):
    names = ['C_OldPrompt', 'C_NewPrompt']
    report = {'metadata': {'variants': names, 'sample_count': 5}, 'samples': [], 'aggregates': {}}
    for variant in names:
        for i in range(5):
            run = tmp_path / variant / str(i)
            qwen = run / 'artifacts/qwen'
            qwen.mkdir(parents=True)
            for call in range(4):
                (qwen / f'qwen_{call:06}.json').write_text(json.dumps({
                    'qwen_call_id': str(call),
                    'input': {'evidence_pack': {'large_field': 'x' * 10000},
                              'request_text': {'system': 'system', 'user': 'evidence'}},
                    'output': {'scene_summary': 'visible target'},
                    'metadata': {'rejections': [{'reason': 'CORRELATED_DUPLICATE'}],
                                 'contract_diagnostic': 'EMPTY_UNSATURATED_PLAN'},
                }))
            error = 5 if variant == names[0] else i * 3
            report['samples'].append(dict(
                variant=variant, sample_id=str(i), run_id=str(run), artifact_directory=str(run),
                success=(i != 4 or variant == names[0]), gt_count=20, predicted_count=20-error,
                absolute_error=error, runtime_ms=10, sam3_calls=2, qwen_calls=4,
                stop_reason='NO_VALID_ACTIONS',
                prompt_outcomes=[{'prompt': 'green fruit', 'family': 'DISCOVERY', 'new_nodes': 2}],
            ))
    pairs = prompt_comparisons(report)['C_prompt']
    assert pairs['n_paired'] == 4 and not pairs['complete']
    assert pairs['left_MAE'] == 5 and pairs['right_MAE'] == 4.5
    assert pairs['mean_absolute_error_reduction'] == .5
    assert pairs['right_better_images'] == pairs['right_worse_images'] == 2
    report['paired_comparisons'] = {'C_prompt': pairs}
    with ZipFile(write_compact_review(report, tmp_path)) as bundle:
        summary = json.loads(bundle.read('review_summary.json'))
        assert len(summary['selected_images']) == 3
        assert '4' in summary['selected_images']  # failure included
        assert summary['diagnostics'][names[0]]['rejection_reasons']['CORRELATED_DUPLICATE'] == 20
        cases = json.loads(bundle.read('selected_cases.json'))
        assert len(cases) == 6
        assert all(len(c['qwen_first_and_last']) == 2 for c in cases)
        assert all([q['qwen_call_id'] for q in c['qwen_first_and_last']] == ['0', '3'] for c in cases)
        assert b'large_field' not in bundle.read('selected_cases.json')
        yields = json.loads(bundle.read('prompt_yields.json'))
        assert all(r['new_nodes'] == 10 and r['executions'] == 5 for r in yields)


def test_comparison_with_no_valid_pairs_and_missing_artifacts(tmp_path):
    names = ['C_OldPrompt', 'C_NewPrompt']
    report = {'metadata': {'variants': names, 'sample_count': 1}, 'aggregates': {}, 'samples': [
        dict(variant=n, sample_id='i', success=False, gt_count=1, run_id=n,
             artifact_directory=str(tmp_path / n), qwen_calls=1) for n in names
    ]}
    pair = prompt_comparisons(report)['C_prompt']
    assert pair['n_paired'] == 0 and pair['right_MAE'] is None and not pair['complete']
    with ZipFile(write_compact_review(report, tmp_path)) as bundle:
        summary = json.loads(bundle.read('review_summary.json'))
        assert len(summary['export_warnings']) == 2
