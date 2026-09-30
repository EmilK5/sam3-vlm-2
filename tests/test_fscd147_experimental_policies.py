from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sam3_vlm.core.config import BeliefConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry, MaskGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, BudgetState, ClassBelief, NodeObservationRef, ObservationRelation, StopReason
from sam3_vlm.models.qwen import MockQwenPlanner, RealQwenPlanner
from sam3_vlm.models.sam3 import DummySAM3Sensor
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.adaptive_search import zero_gain_streak
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService
from sam3_vlm.planning.semantic_guard import confounder_rejection
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from test_m8_qwen_payload import mock_openai_client


@pytest.mark.parametrize('label,target,relation', [
    ('paperback', 'books', 'target_subtype'), ('hardcover', 'books', 'target_subtype'),
    ('cap rim', 'bottle caps', 'distinct_object'), ('donut tray', 'donuts', 'distinct_object'),
    ('sphere', 'polka dots', 'target_container'), ('leaf', 'fruit', 'uncertain'),
    ('fruit', 'fruit', 'target_synonym'), ('leaf', 'fruit', None)])
def test_subtypes_parts_containers_and_unassessed_negatives_rejected(label, target, relation):
    assert confounder_rejection(label, target, relation, 'explanation')


def guarded_config():
    config = V4Config()
    return replace(config, planner=replace(config.planner, validate_confounders=True, execute_confounder_prompts=True))


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_adaptive_patience_rejects_invalid_values(value):
    config = V4Config()
    with pytest.raises(ValueError, match='positive integer'):
        replace(config.replanning, adaptive_e_zero_gain_patience=value)


@pytest.mark.parametrize('section,field', [('planner', 'validate_confounders'), ('belief', 'neutral_appearance_misses')])
def test_experimental_flags_require_booleans(section, field):
    with pytest.raises(ValueError, match='boolean'):
        replace(getattr(V4Config(), section), **{field: 'true'})


def test_gate_accepts_distinct_object_only_with_reason_and_does_not_mutate_proposal():
    assert confounder_rejection('wire rack', 'donut', 'distinct_object', 'Metal rack supports the pastries') is None
    assert confounder_rejection('wire rack', 'donut', 'distinct_object', '')
    proposal = ProposedAction('confounder1', 'paperback', ActionFamily.CONFOUNDER,
        semantic_prior={'confounder1': 1}, confounder_relation='target_subtype', confounder_reason='A type of book')
    generator = ActionBankGenerator()
    args = (PlannerOutput(proposed_actions=[proposal]), SemanticMemory(), ActionBank(), IDGenerator())
    assert not generator.generate_entries(*args, config=guarded_config(), target_prompt='books', enforce_qwen_contract=True)
    assert generator.last_rejections[0].reason == 'UNSAFE_CONFOUNDER'
    assert proposal.confounder_relation == 'target_subtype'
    config = guarded_config()
    assert generator.generate_entries(*args, config=replace(config, planner=replace(config.planner, validate_confounders=False)),
                                      target_prompt='books', enforce_qwen_contract=True)


@pytest.mark.parametrize('assessments,accepted', [([], False),
    ([{'label': 'wire rack', 'relationship': 'distinct_object', 'reason': 'Metal support'}], True),
    ([{'label': 'wire rack', 'relationship': 'distinct_object', 'reason': ''}], False),
    ([{'label': 'tray', 'relationship': 'distinct_object', 'reason': 'Different label'}], False),
    ([{'label': 'wire rack', 'relationship': 'distinct_object', 'reason': 'Metal support'}]*2, False)])
def test_assessment_must_match_exactly_one_negative(assessments, accepted):
    output = PlannerOutput(likely_confounders=['wire rack'], confounder_assessments=assessments)
    assert PlannerOutput.from_dict(output.to_dict()).confounder_assessments == assessments
    pack = QwenEvidencePack('i', 'donut', 'target', ContactSheet(), belief_classes=['target', 'confounder1', 'confounder2'])
    normalized = QwenPlannerService(MockQwenPlanner(custom_output=output)).plan_scene(pack, BudgetState(), guarded_config())
    entries = ActionBankGenerator().generate_entries(normalized, SemanticMemory(), ActionBank(), IDGenerator(),
        config=guarded_config(), target_prompt='donut', enforce_qwen_contract=True)
    assert bool(entries) == accepted


def test_qwen_assessment_instructions_are_conditional(mock_openai_client):
    pack = QwenEvidencePack('i', 'books', 'target', ContactSheet())
    planner = RealQwenPlanner(model='fake', base_url='http://fake')
    planner.plan_scene(pack, BudgetState(), guarded_config())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs['messages']
    assert 'target_subtype' in messages[0]['content']
    assert 'confounder_assessments' in messages[1]['content'][0]['text']
    planner.plan_scene(pack, BudgetState(), V4Config())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs['messages']
    assert 'confounder_assessments' not in messages[0]['content']


@pytest.mark.parametrize('canonical', [False, True])
def test_neutral_miss_preserves_probability_and_does_not_discount_later_positive(canonical):
    vocabulary = ['target', 'confounder1', 'confounder2'] if canonical else None
    probabilities = {'target': .6, 'confounder1': .2, 'confounder2': .2}
    node = Node('n', BoxGeometry(Box(0, 0, 10, 10)), ClassBelief(dict(probabilities)))
    control = Node('c', BoxGeometry(Box(0, 0, 10, 10)), ClassBelief(dict(probabilities)))
    action = SensingAction('a', 'target', 'red cap', ActionFamily.DISCOVERY, is_appearance_query=True)
    config = BeliefConfig(neutral_appearance_misses=True, discount_repeat_weight=.2)
    updater = BeliefUpdater()
    miss = NodeObservationRef('o1', 's1', 'a', 'target', relation=ObservationRelation.NOT_RETRIEVED)
    node.observations.append(miss)
    updater.update_node_belief(node, action, miss, target_class='target', config=config, class_vocabulary=vocabulary)
    assert node.class_belief.probabilities == pytest.approx(probabilities)
    assert miss.neutral_evidence and Node.from_dict(node.to_dict()).observations[0].neutral_evidence
    for current in (node, control):
        match = NodeObservationRef('o2', 's2', 'a', 'target', relation=ObservationRelation.STRONG_MATCH, score=.9)
        current.observations.append(match)
        updater.update_node_belief(current, action, match, target_class='target', config=config, class_vocabulary=vocabulary)
    assert node.class_belief.probabilities == pytest.approx(control.class_belief.probabilities)


@pytest.mark.parametrize('family,appearance,enabled', [
    (ActionFamily.DISCOVERY, False, True), (ActionFamily.DISCOVERY, True, False),
    (ActionFamily.VERIFICATION, True, True), (ActionFamily.CONFOUNDER, True, True)])
def test_other_misses_keep_existing_evidence(family, appearance, enabled):
    node = Node('n', BoxGeometry(Box(0, 0, 10, 10)), ClassBelief({'target': .6, 'confounder1': .4}))
    key = 'confounder1' if family == ActionFamily.CONFOUNDER else 'target'
    action = SensingAction('a', key, 'book', family, is_appearance_query=appearance)
    miss = NodeObservationRef('o', 's', 'a', key, relation=ObservationRelation.NOT_RETRIEVED)
    BeliefUpdater().update_node_belief(node, action, miss, target_class='target',
        config=BeliefConfig(neutral_appearance_misses=enabled))
    assert not miss.neutral_evidence
    assert node.class_belief.probabilities['target'] != pytest.approx(.6)


def test_canonical_and_narrow_queries_classified_after_inflection():
    config = V4Config()
    config = replace(config, belief=replace(config.belief, neutral_appearance_misses=True))
    output = PlannerOutput(proposed_actions=[ProposedAction('target', label, ActionFamily.DISCOVERY,
        semantic_prior={'target': 1}) for label in ('bottle caps', 'red caps')])
    entries = ActionBankGenerator().generate_entries(output, SemanticMemory(), ActionBank(), IDGenerator(),
        config=config, target_prompt='bottle cap', enforce_qwen_contract=True)
    assert [entry.action.is_appearance_query for entry in entries] == [False, True]


def test_e_streak_excludes_bootstrap_and_negatives_and_resets_on_recovery():
    trace = [{'action_id': None, 'new_nodes': 0}, {'action_id': 'a', 'new_nodes': 0},
        {'action_id': 'negative', 'new_nodes': 0}, {'action_id': 'b', 'new_nodes': 0}]
    assert zero_gain_streak(trace, {'a', 'b'}) == 2
    assert zero_gain_streak(trace + [{'action_id': 'c', 'new_nodes': 2}], {'a', 'b', 'c'}) == 0


@pytest.mark.parametrize('extended,patience,pending,stops', [(True, 3, False, True),
    (True, None, False, False), (False, 3, False, False), (True, 3, True, False)])
def test_adaptive_e_stops_only_when_enabled_and_pending_actions_drained(extended, patience, pending, stops):
    config = V4Config()
    config = replace(config, replanning=replace(config.replanning,
        continue_until_saturation=extended, adaptive_e_zero_gain_patience=patience))
    runner = Runner(config, DummySAM3Sensor(), MockQwenPlanner())
    runner.image = (384, 384)
    runner.scene_state = SceneState(image_id='i', user_prompt='book', target_class='target',
        graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    runner._uses_canonical_m8_policy = lambda: True
    runner.replan_evidence_builder = SimpleNamespace(build=lambda *args, **kwargs: None)
    runner._execute_plan = lambda **kwargs: setattr(runner.scene_state, 'last_plan_accepted_actions', 1)
    for index in range(3):
        action = SensingAction(str(index), 'target', f'book {index}', ActionFamily.DISCOVERY, source=ActionSource.QWEN)
        entry = runner.scene_state.action_bank.add_action(action)
        entry.executed = True
        runner.confidence_trace.append({'action_id': str(index), 'new_nodes': 0})
    if pending:
        runner.scene_state.action_bank.add_action(SensingAction('neg', 'confounder1', 'rack', ActionFamily.CONFOUNDER))
    runner._request_replan()
    assert (runner.scene_state.stop_reason == StopReason.LOW_MARGINAL_UTILITY) == stops


def test_adaptive_e_controller_executes_then_stops_and_replays(tmp_path):
    from sam3_vlm.core.config import BootstrapConfig
    from sam3_vlm.core.types import Detection
    from test_m8_saturation_experiment import experiment_e, run_e

    class BootstrapOnlySensor(DummySAM3Sensor):
        def observe(self, image, action):
            observation = super().observe(image, action)
            if action.source == ActionSource.USER_BOOTSTRAP:
                observation.detections = [Detection(f'd{index}', BoxGeometry(Box(index*10, 0, index*10+4, 4)), .8,
                    raw_metadata={'mask': np.ones((4, 4), dtype=bool), 'mask_offset_x': index*10, 'mask_offset_y': 0})
                    for index in range(8)]
            else:
                observation.detections = []
            return observation
    config = experiment_e()
    config = replace(config, belief=replace(config.belief, neutral_appearance_misses=True),
        replanning=replace(config.replanning, adaptive_e_zero_gain_patience=3))
    runner, planner = run_e(tmp_path, BootstrapOnlySensor(), config)
    assert runner.scene_state.stop_reason == StopReason.LOW_MARGINAL_UTILITY
    assert planner.call_count == 3
    assert runner.scene_state.budget.sam3_calls == 4
    assert all(step['existing_target_mass_lost'] == 0 for step in runner.confidence_trace[1:])


def test_rejected_negative_does_not_freeze_slot(tmp_path):
    from sam3_vlm.experiments.fscd147_report import _artifact_diagnostics
    from sam3_vlm.experiments.m8_smoke import _pilot_variants, assemble_e2e_runner, _run_validator_and_replay
    from sam3_vlm.logging.artifacts import RunArtifactPaths
    from sam3_vlm.models.sam3 import MockSAM3Adapter
    output = PlannerOutput(likely_confounders=['paperback', 'shelf'], confounder_assessments=[
        {'label': 'paperback', 'relationship': 'target_subtype', 'reason': 'A book subtype'},
        {'label': 'shelf', 'relationship': 'distinct_object', 'reason': 'Furniture holding books'}])
    config = _pilot_variants(guarded_config(), 'final-ae')[2].config
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, config, MockSAM3Adapter(), MockQwenPlanner(custom_output=output),
                                    'r', 'books', 'target', 'i')
    runner.run((384, 384), 'books', image_id='i')
    assert runner.scene_state.confounder_labels == {'confounder2': 'shelf'}
    assert _run_validator_and_replay(paths, runner.scene_state)
    diagnostic = _artifact_diagnostics(tmp_path / 'predictions.jsonl',
        {'image_id': 'i', 'variant': 'C', 'artifact_directory': str(paths.base_dir)})
    example = diagnostic['qwen_first_last'][0]
    assert example['confounder_assessments'][0]['relationship'] == 'target_subtype'
    assert example['unsafe_confounders'][0]['sam3_prompt'] == 'paperback'


def test_mask_audit_flags_pixels_without_mutating_nodes_or_using_boxes():
    from sam3_vlm.experiments.mask_audit import audit_masks
    a = np.ones((10, 10), dtype=bool)
    nodes = [Node('full', MaskGeometry(Box(0, 0, 100, 100), a, (0, 0), 100)),
             Node('same', MaskGeometry(Box(0, 0, 100, 100), a.copy(), (0, 0), 100)),
             Node('fragment', MaskGeometry(Box(0, 0, 100, 100), np.ones((2, 2), dtype=bool), (2, 2), 4)),
             Node('disjoint', MaskGeometry(Box(0, 0, 100, 100), a.copy(), (30, 30), 100))]
    before = [node.to_dict() for node in nodes]
    result = audit_masks(nodes)
    assert result['complete'] and result['high_iou_pairs'] == 1
    assert result['high_iom_pairs'] == 3 and result['large_area_ratio_containment_pairs'] == 2
    assert [node.to_dict() for node in nodes] == before
    nodes[0].geometry = replace(nodes[0].geometry, mask=None)
    missing = audit_masks(nodes)
    assert not missing['complete'] and missing['missing_mask_nodes'] == ['full']
