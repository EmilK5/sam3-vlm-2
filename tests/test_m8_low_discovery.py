"""Low sensor admission thresholds remain separate from posterior evidence fusion."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from sam3_vlm.core.config import SAM3Config, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, BudgetState, ClassBelief, NodeObservationRef, ObservationRelation
from sam3_vlm.experiments.m8_smoke import _pilot_variants, load_m8_config, m8_4_and_5_pilot
from sam3_vlm.logging.confidence import confidence_step
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import QwenPlannerService
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from test_m8_orchestration import DummyArgs
from test_m8_qwen_payload import mock_openai_client
from test_m8_rejection_correction import SequencePlanner, proposal, run_case


def production():
    return load_m8_config(DummyArgs(), config_path='configs/m8_real_smoke.json').v4_config


@pytest.mark.parametrize('field', ['qwen_discovery_threshold', 'qwen_confounder_threshold'])
@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_family_threshold(field, value):
    with pytest.raises(ValueError, match=field):
        SAM3Config(**{field: value})


def test_family_fallback_and_production_settings():
    cfg = production()
    assert cfg.sam3.threshold_for_family(ActionFamily.DISCOVERY) == .2
    assert cfg.sam3.threshold_for_family(ActionFamily.CONFOUNDER) == .5
    assert cfg.sam3.default_threshold == .25
    assert not cfg.sam3.qwen_discovery_use_exemplars
    assert cfg.planner.correct_missing_target
    assert cfg.bootstrap.enable_pseudoexemplar_refinement
    for family in ActionFamily:
        assert SAM3Config(qwen_prompt_threshold=.37).threshold_for_family(family) == .37
    assert SAM3Config(qwen_discovery_threshold=0).threshold_for_family('DISCOVERY') == 0


def test_controller_enforces_separate_thresholds_and_preserves_suggestions():
    cfg = production()
    output = proposal('shaded fruit', ['leaf'])
    output.proposed_actions[0].suggested_threshold = .99
    service = QwenPlannerService(SequencePlanner([output]))
    pack = QwenEvidencePack('i', 'green fruit', 'target', ContactSheet(),
                            belief_classes=['target', 'confounder1', 'confounder2'])
    normalized = service.plan_scene(pack, BudgetState(), cfg)
    entries = ActionBankGenerator().generate_entries(normalized, SemanticMemory(), ActionBank(),
                IDGenerator(), config=cfg, enforce_qwen_contract=True)
    assert [(e.action.family, e.action.threshold) for e in entries] == [
        (ActionFamily.DISCOVERY, .2), (ActionFamily.CONFOUNDER, .5)]
    assert normalized.proposed_actions[0].suggested_threshold == .99
    assert normalized.proposed_actions[1].suggested_threshold == .5


@pytest.mark.parametrize('enabled', [False, True])
def test_only_qwen_discovery_loses_exemplars(enabled):
    cfg = production()
    cfg = replace(cfg, bootstrap=replace(cfg.bootstrap, enable_pseudoexemplar_refinement=enabled))
    runner = Runner(cfg, MockSAM3Adapter(), SequencePlanner([]))
    runner.scene_state = SimpleNamespace(belief_classes=['target'], graph=SceneGraph(), uses_canonical_m8_policy=True)
    action = SensingAction(action_id='a', semantic_key='target', prompt='fruit',
        family=ActionFamily.DISCOVERY, source=ActionSource.QWEN,
        positive_exemplar_ids=('n',), positive_exemplar_boxes=((0, 0, 10, 10),))
    stripped = runner._attach_target_pseudoexemplars(action)
    assert not stripped.positive_exemplar_ids and not stripped.positive_exemplar_boxes
    for other in [replace(action, source=ActionSource.USER_BOOTSTRAP),
                  replace(action, family=ActionFamily.VERIFICATION)]:
        assert runner._attach_target_pseudoexemplars(other) == other
    assert action.positive_exemplar_ids == ('n',)


@pytest.mark.parametrize('corrected', ['dark green fruit', 'green fruit in shadow'])
def test_missing_target_correction_retains_negatives_and_replays(tmp_path, corrected):
    runner, planner, artifacts = run_case(tmp_path, [
        proposal('green fruit', ['leaves', 'branches']),
        proposal(corrected, ['stones', 'grass']),
    ], execute_confounder_prompts=True, correct_missing_target=True)
    valid = corrected == 'dark green fruit'
    assert runner.scene_state.budget.qwen_calls == 2
    assert runner.scene_state.budget.sam3_calls == 3 + valid
    assert artifacts[1]['metadata']['correction_of'] == artifacts[0]['qwen_call_id']
    assert artifacts[0]['metadata']['accepted_target_action_count'] == 0
    assert artifacts[1]['metadata']['accepted_target_action_count'] == int(valid)
    assert planner.evidence[1].confounder_labels == {'confounder1': 'leaves', 'confounder2': 'branches'}
    assert planner.evidence[1].discovery_diagnostics['pending_sam3_prompts'] == ['leaves', 'branches']
    assert 'leaves' not in planner.evidence[1].discovery_diagnostics.get('tried_sam3_prompts', [])
    assert runner.scene_state.last_plan_accepted_actions == 2 + valid
    assert len(runner.scene_state.last_plan_action_ids) == 2 + valid
    assert [r['prompt'] for r in runner.confidence_trace[1:]] == (
        [corrected] if valid else []) + ['leaves', 'branches']
    assert runner.scene_state.replans_executed == 0


def test_missing_target_correction_still_respects_one_call_cap(tmp_path):
    runner, planner, artifacts = run_case(tmp_path, [proposal('green fruit', ['leaves'])],
        calls=1, execute_confounder_prompts=True, correct_missing_target=True)
    assert len(planner.evidence) == 1
    assert runner.scene_state.budget.sam3_calls == 2
    assert artifacts[0]['metadata']['correction_of'] is None


def test_valid_target_does_not_spend_call_correcting_only_negative(tmp_path):
    runner, planner, artifacts = run_case(tmp_path, [proposal('dark green fruit', ['leaf cluster'])],
        execute_confounder_prompts=True, correct_missing_target=True)
    assert all(a['metadata']['correction_of'] is None for a in artifacts)


def test_discovery_suite_isolates_sensing_and_keeps_history_reproducible():
    base = production()
    variants = _pilot_variants(base, 'discovery-ablation')
    assert len(variants) == 4
    assert [v.config.sam3.qwen_discovery_threshold for v in variants] == [.5, .25, .2, .15]
    assert [v.config.sam3.qwen_discovery_use_exemplars for v in variants] == [True, False, False, False]
    dictionaries = []
    for v in variants:
        cfg = v.config
        assert cfg.planner.prompt_version == 'old'
        assert cfg.planner.execute_confounder_prompts and cfg.planner.correct_missing_target
        assert cfg.budget.max_qwen_calls == 2 and cfg.replanning.max_replans == 1
        assert cfg.association.enable_iom_dedup
        assert cfg.bootstrap == base.bootstrap
        assert cfg.belief.target_count_hard_threshold is None
        assert cfg.belief.target_count_commit_threshold is None
        d = asdict(cfg)
        d['sam3'].pop('qwen_discovery_threshold')
        d['sam3'].pop('qwen_discovery_use_exemplars')
        dictionaries.append(d)
    assert all(d == dictionaries[0] for d in dictionaries)
    for suite in ['recovery-ablation', 'prompt-ablation', 'negative-ablation', 'all']:
        for v in _pilot_variants(base, suite):
            assert v.config.sam3.threshold_for_family('DISCOVERY') == .5
            assert v.config.sam3.qwen_discovery_use_exemplars
            assert not v.config.planner.correct_missing_target


def test_payload_shows_both_thresholds_and_keeps_full_images(mock_openai_client, tmp_path):
    original, contact = tmp_path / 'image.png', tmp_path / 'sheet.png'
    for path in [original, contact]:
        Image.new('RGB', (10, 10)).save(path)
    pack = QwenEvidencePack('i', 'green fruit', 'target',
        ContactSheet(contact_sheet_image_path=str(contact)), image_path=str(original),
        scene_summary='full evidence ' * 1000,
        discovery_diagnostics={'pending_sam3_prompts': ['leaves']})
    RealQwenPlanner(base_url='http://fake', model='fake').plan_scene(pack, BudgetState(), production())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs['messages']
    text = messages[1]['content'][0]['text']
    assert 'SAM3 threshold is fixed by the controller at 0.2' in text
    assert 'negative SAM3 queries at fixed threshold 0.5' in text
    assert 'same fixed threshold' not in text
    assert pack.scene_summary in text and 'pending_sam3_prompts' in text
    assert len(messages[1]['content']) == 3


def test_probability_trace_accounts_for_new_existing_and_removed_nodes():
    graph = SceneGraph()
    for i, probability in enumerate([.2, .8, .3]):
        node = Node(str(i), BoxGeometry(Box(i*20, 0, i*20+10, 10)),
                    class_belief=ClassBelief(probabilities={'target': probability, 'confounder1': 1-probability}))
        node.observations.append(NodeObservationRef(observation_id=str(i), sam3_call_id='s',
            action_id='a', semantic_key='target', score=.1,
            relation=ObservationRelation.NOT_RETRIEVED))
        graph.add_node(node)
    action = SensingAction('a', 'target', 'fruit', ActionFamily.DISCOVERY)
    before = graph.to_dict()
    trace, current = confidence_step(graph, {'0': .7, '1': .4, 'removed': .6}, action=action)
    assert graph.to_dict() == before
    assert trace['new_nodes'] == 1
    assert trace['new_node_target_mass'] == .3
    assert trace['existing_node_target_mass_change'] == pytest.approx(-.1)
    assert trace['removed_node_target_mass'] == .6
    assert trace['raw_soft_count'] == pytest.approx(trace['previous_raw_soft_count']
        + trace['new_node_target_mass'] + trace['existing_node_target_mass_change']
        - trace['removed_node_target_mass'])
    assert trace['largest_losses'][0]['relation'] == 'NOT_RETRIEVED'
    assert trace['largest_losses'][0]['node_id'] == '0'
    assert trace['largest_gains'][0]['node_id'] == '1'
    assert len(current) == 3


def test_discovery_pilot_exports_compact_confidence_and_comparisons(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': 'img', 'image_path': str(image),
                                    'gt_count': 2, 'target': 'green fruit'}]))
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models',
        lambda args: (MockSAM3Adapter(), SequencePlanner([proposal('green fruit', ['leaves']),
                                                        proposal('dark green fruit')])))
    args = DummyArgs(manifest=str(manifest), max_samples=1,
                     pilot_suite='discovery-ablation', output_dir=str(tmp_path / 'runs'))
    assert m8_4_and_5_pilot(args)
    with ZipFile(tmp_path / 'runs/compact_review.zip') as z:
        summary = json.loads(z.read('review_summary.json'))
        assert len(summary['paired_comparisons']) == 4
        assert all(p['complete'] for p in summary['paired_comparisons'].values())
        totals = json.loads(z.read('confidence_totals.json'))
        assert len(totals) == 4
        for row in totals:
            assert row['final_target_mass'] == pytest.approx(row['bootstrap_target_mass']
                + row['new_node_target_mass_at_creation'] + row['existing_node_target_mass_change']
                - row['removed_node_target_mass'])
        assert summary['diagnostics']['D_Discovery_050_Exemplars']['corrections_with_accepted_target'] == 1
        cases = json.loads(z.read('selected_cases.json'))
        assert len(cases) == 4
        for case in cases:
            trace = case['confidence_trace']
            assert trace[0]['stage'] == 'bootstrap'
            assert trace[-1]['raw_soft_count'] == pytest.approx(case['predicted_count'])
            for row in trace:
                assert row['raw_soft_count'] == pytest.approx(row['previous_raw_soft_count']
                    + row['new_node_target_mass'] + row['existing_node_target_mass_change']
                    - row['removed_node_target_mass'])
