"""Last comparison isolates negative non-retrieval, with the same prompt fixes."""
import json
from dataclasses import asdict, replace
from zipfile import ZipFile

import pytest
from PIL import Image

from sam3_vlm.core.config import BeliefConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, ClassBelief, NodeObservationRef, ObservationRelation, BudgetState
from sam3_vlm.experiments.m8_smoke import _pilot_variants, m8_4_and_5_pilot
from sam3_vlm.experiments.pilot_review import final_evaluation_summary
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.scene.belief import BeliefUpdater
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
from test_m8_orchestration import DummyArgs
from test_m8_qwen_payload import mock_openai_client
from test_m8_rejection_correction import SequencePlanner, proposal
from test_m8_low_discovery import production
from test_m8_negative_prompts_hard_count import run_precision, config as precision_config


def updated(neutral, family, relation, canonical=True):
    vocabulary = ['target', 'confounder1', 'confounder2']
    node = Node('n', BoxGeometry(Box(0, 0, 10, 10)),
        class_belief=ClassBelief(probabilities=dict(zip(vocabulary, [.6, .3, .1]))))
    key = 'confounder1' if family == ActionFamily.CONFOUNDER else 'target'
    action = SensingAction('a', key, 'leaf' if key == 'confounder1' else 'fruit', family,
                          semantic_prior={key: 1.0})
    obs = NodeObservationRef(observation_id='o', sam3_call_id='s', action_id='a',
                            semantic_key=key, relation=relation, score=.8)
    node.observations.append(obs)
    BeliefUpdater().update_node_belief(node, action, obs, target_class='target',
        config=BeliefConfig(neutral_confounder_misses=neutral),
        class_vocabulary=vocabulary if canonical else None)
    return node.class_belief.probabilities


@pytest.mark.parametrize('canonical', [False, True])
def test_missing_confounder_is_neutral_but_current_policy_boosts_target(canonical):
    current = updated(False, ActionFamily.CONFOUNDER, ObservationRelation.NOT_RETRIEVED, canonical)
    neutral = updated(True, ActionFamily.CONFOUNDER, ObservationRelation.NOT_RETRIEVED, canonical)
    assert current['target'] == pytest.approx(.6 / (1 - .15 * .3))
    assert neutral == pytest.approx({'target': .6, 'confounder1': .3, 'confounder2': .1})


@pytest.mark.parametrize('family', [ActionFamily.CONFOUNDER, ActionFamily.DISCOVERY])
@pytest.mark.parametrize('relation', list(ObservationRelation))
def test_only_confounder_nonretrieval_changes(family, relation):
    if family == ActionFamily.CONFOUNDER and relation == ObservationRelation.NOT_RETRIEVED:
        return
    a, b = updated(False, family, relation), updated(True, family, relation)
    assert a == b
    if family == ActionFamily.CONFOUNDER and relation in (ObservationRelation.STRONG_MATCH, ObservationRelation.WEAK_MATCH):
        assert b['target'] < .6
    if family == ActionFamily.DISCOVERY and relation == ObservationRelation.NOT_RETRIEVED:
        assert b['target'] < .6


def test_final_configs_differ_only_in_miss_policy():
    variants = _pilot_variants(production(), 'final-ablation')
    assert [v.name for v in variants] == ['D_CurrentNegativeEvidence', 'D_NeutralNegativeMisses']
    a, b = [asdict(v.config) for v in variants]
    assert a['belief']['neutral_confounder_misses'] is False
    assert b['belief']['neutral_confounder_misses'] is True
    a['belief']['neutral_confounder_misses'] = True
    assert a == b
    for v in variants:
        assert v.config.sam3.qwen_discovery_threshold == .2
        assert v.config.sam3.qwen_confounder_threshold == .5
        assert not v.config.sam3.qwen_discovery_use_exemplars
        assert v.config.belief.target_count_hard_threshold is None
        assert v.config.belief.target_count_commit_threshold is None
        assert v.config.budget.max_qwen_calls == 2
        assert v.config.planner.correct_missing_target
        assert v.config.planner.target_scope == production().planner.target_scope


@pytest.mark.parametrize('version', ['old', 'v3', 'v4'])
def test_scope_qualifiers_correction_and_images_reach_every_prompt(mock_openai_client, tmp_path, version):
    image = tmp_path / 'image.png'
    Image.new('RGB', (10, 10)).save(image)
    cfg = production()
    cfg = replace(cfg, planner=replace(cfg.planner, prompt_version=version))
    pack = QwenEvidencePack('i', 'green fruit', 'target', ContactSheet(contact_sheet_image_path=str(image)),
        image_path=str(image), scene_summary='complete scene history ' * 1000,
        discovery_diagnostics={'previous_plan_feedback': {'rejections': [{'sam3_prompt': 'green fruit in shade',
                              'reason': 'INVALID_GROUNDING_PROMPT'}]}})
    planner = RealQwenPlanner(base_url='http://fake', model='fake')
    planner.plan_scene(pack, BudgetState(), cfg)
    request = mock_openai_client.chat.completions.create.call_args.kwargs
    system, user = request['messages'][0]['content'], request['messages'][1]['content'][0]['text']
    assert cfg.planner.target_scope in system
    assert 'color, maturity and other eligibility qualifiers as constraints' in user
    assert 'object noun is LAST' in user
    assert 'repair the same intended target description' in user
    assert 'Vocabulary is open' in user
    assert pack.scene_summary in user
    assert len(request['messages'][1]['content']) == 3
    assert request['max_tokens'] == 512


def test_matched_negatives_still_lower_probability_and_replay(tmp_path):
    cfg = precision_config()
    cfg = replace(cfg, belief=replace(cfg.belief, neutral_confounder_misses=True,
                  target_count_hard_threshold=None))
    runner, sensor, paths = run_precision(tmp_path, cfg)
    assert runner.scene_state.graph.get_node('node_000001').class_belief.probabilities['target'] < .5
    assert len(runner.scene_state.graph.nodes) == 2
    trace = runner.confidence_trace[-1]
    assert trace['existing_node_observation_relations']['STRONG_MATCH'] == 1
    assert trace['existing_node_target_mass_change'] < 0
    assert trace['existing_node_observation_relations']['NOT_RETRIEVED'] == 1
    assert not trace['largest_gains']  # The unmatched candidate receives no confidence boost.


def test_pilot_compares_policies_and_exports_presentation_notes(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': 'img', 'image_path': str(image),
                                    'target': 'green fruit', 'gt_count': 2}]))
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models',
        lambda args: (MockSAM3Adapter(), SequencePlanner([proposal('dark green fruit', ['leaf'])])))
    args = DummyArgs(manifest=str(manifest), max_samples=1,
                     pilot_suite='final-ablation', output_dir=str(tmp_path / 'runs'))
    assert m8_4_and_5_pilot(args)  # Includes artifact validation and deterministic replay.
    with ZipFile(tmp_path / 'runs/compact_review.zip') as z:
        summary = json.loads(z.read('review_summary.json'))
        pair = summary['paired_comparisons']['D_negative_miss_policy']
        assert pair['n_paired'] == 1 and pair['complete']
        assert pair['left'] == 'D_CurrentNegativeEvidence'
        assert pair['right'] == 'D_NeutralNegativeMisses'
        assert len(json.loads(z.read('confidence_totals.json'))) == 2
        notes = z.read('mentor_summary.md').decode()
        assert 'Complete comparison: True' in notes
        assert 'not an independent held-out' in notes
        assert (tmp_path / 'runs/mentor_summary.md').read_text() == notes


def test_presentation_notes_do_not_hide_incomplete_pairs():
    report = {'metadata': {'sample_count': 34, 'variants': ['D_CurrentNegativeEvidence', 'D_NeutralNegativeMisses']},
              'aggregates': {}, 'paired_comparisons': {}}
    text = final_evaluation_summary(report)
    assert 'Paired successful images: 0 / 34' in text
    assert 'Complete comparison: False' in text
    assert 'unavailable' in text


def test_single_variant_selection_preserves_winning_configuration():
    from sam3_vlm.experiments.m8_smoke import _selected_pilot_variants
    base = production()
    all_variants = _pilot_variants(base, 'final-ablation')
    selected = _selected_pilot_variants(base, 'final-ablation', variant='D_CurrentNegativeEvidence')
    assert selected == [all_variants[0]]
    assert _selected_pilot_variants(base, 'final-ablation') == all_variants
    assert selected[0].config.belief.neutral_confounder_misses is False


def test_invalid_variant_fails_before_loading_models(tmp_path, monkeypatch):
    import sys
    from sam3_vlm.experiments.m8_smoke import main
    def unexpected_models(*args):
        pytest.fail('Models should not load for an invalid variant')
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models', unexpected_models)
    monkeypatch.setattr(sys, 'argv', ['m8_smoke', '--stage', 'pilot', '--allow-cpu',
        '--pilot-suite', 'final-ablation', '--pilot-variant', 'typo', '--output_dir', str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_single_variant_full_export_does_not_claim_paired_comparison(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': 'img', 'image_path': str(image),
                                    'target': 'green fruit', 'gt_count': 2}]))
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models',
        lambda args: (MockSAM3Adapter(), SequencePlanner([proposal('dark green fruit', ['leaf'])])))
    args = DummyArgs(manifest=str(manifest), max_samples=1, pilot_suite='final-ablation',
                     pilot_variant='D_CurrentNegativeEvidence', output_dir=str(tmp_path / 'runs'))
    assert m8_4_and_5_pilot(args)
    report = json.loads((tmp_path / 'runs/pilot_report.json').read_text())
    assert len(report['samples']) == 1
    assert report['metadata']['variants'] == ['D_CurrentNegativeEvidence']
    assert report['metadata']['pilot_variant'] == 'D_CurrentNegativeEvidence'
    assert report['samples'][0]['success']
    assert report['paired_comparisons'] == {}
    with ZipFile(tmp_path / 'runs/compact_review.zip') as z:
        summary = json.loads(z.read('review_summary.json'))
        assert summary['paired_comparisons'] == {}
        notes = z.read('mentor_summary.md').decode()
        assert 'Single-variant run; no within-run paired comparison.' in notes
        assert 'Paired successful images:' not in notes
        assert 'Neutral better / worse' not in notes
        assert len(json.loads(z.read('counts.json'))) == 1
        assert len(json.loads(z.read('confidence_totals.json'))) == 1


def test_family_and_single_variant_filter_are_consistent():
    from sam3_vlm.experiments.m8_smoke import _selected_pilot_variants
    with pytest.raises(ValueError, match='Unknown --pilot-variant'):
        _selected_pilot_variants(V4Config(), 'prompt-ablation', family='C', variant='D_OldPrompt')
    selected = _selected_pilot_variants(V4Config(), 'prompt-ablation', family='C', variant='C_OldPrompt')
    assert len(selected) == 1 and selected[0].name == 'C_OldPrompt'
