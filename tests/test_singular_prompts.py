from dataclasses import replace

import pytest

from sam3_vlm.core.config import V4Config, SAM3Config
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, BudgetState
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from sam3_vlm.sensing.prompts import singularize_prompt
from test_m8_qwen_payload import mock_openai_client


@pytest.mark.parametrize('original,expected', [
    ('polka dots', 'polka dot'), ('bottle caps', 'bottle cap'), ('chicken wings', 'chicken wing'),
    ('flamingos', 'flamingo'), ('books', 'book'), ('birds', 'bird'), ('skateboard', 'skateboard'),
    ('donuts tray', 'donut tray'), ('green apples', 'green apple'), ('wine glasses', 'wine glass'),
    ('green leaves', 'green leaf'), ('mice', 'mouse'), ('geese', 'goose'), ('children', 'child'),
    ('people', 'person'), ('teeth', 'tooth'), ('feet', 'foot'), ('red berries', 'red berry'),
    ('cookies', 'cookie'), ('glass', 'glass'), ('grass', 'grass'), ('citrus', 'citrus'),
    ('iris', 'iris'), ('lens', 'lens'), ('bus', 'bus'), ('canvas', 'canvas'), ('asparagus', 'asparagus'),
    ('scissors', 'scissors'), ('sunglasses', 'sunglasses'), ('reading glasses', 'reading glasses'),
    ('series', 'series'), ('species', 'species'), ('fish', 'fish'), ('sheep', 'sheep'),
    ('BOTTLE CAPS', 'BOTTLE CAP'), ('Polka Dots', 'Polka Dot'),
    ('shiny metal lids', 'shiny metal lid'), ('green citrus', 'green citrus')])
def test_inflection_preserves_identity_and_is_idempotent(original, expected):
    assert singularize_prompt(original) == expected
    assert singularize_prompt(expected) == expected


def cfg(enabled=True):
    config = V4Config(sam3=SAM3Config(singularize_prompts=enabled))
    return replace(config, planner=replace(config.planner, execute_confounder_prompts=True))


@pytest.mark.parametrize('enabled', [True, False])
def test_bootstrap_normalizes_every_sensor_pass_and_keeps_source_target(tmp_path, enabled):
    class CaptureSensor(MockSAM3Adapter):
        def __init__(self):
            super().__init__()
            self.prompts = []
        def observe(self, image, action):
            self.prompts.append(action.prompt)
            return super().observe(image, action)
    sensor = CaptureSensor()
    config = replace(cfg(enabled), assets_dir=str(tmp_path / 'assets'))
    result = BootstrapPipeline(sensor, config=config).execute_bootstrap('i', (384, 384), 'polka dots')
    assert sensor.prompts
    assert set(sensor.prompts) == {'polka dot' if enabled else 'polka dots'}
    assert result.state.user_prompt == result.qwen_evidence_pack.user_prompt == 'polka dots'
    assert {prompt for record in result.state.semantic_memory.records.values() for prompt in record.prompts} == set(sensor.prompts)


def test_qwen_positive_negative_prompts_and_duplicate_checks_share_normalization():
    proposals = [ProposedAction('target', prompt, ActionFamily.DISCOVERY, semantic_prior={'target': 1.0})
                 for prompt in ['red caps', 'red cap']]
    proposals.append(ProposedAction('confounder1', 'green leaves', ActionFamily.CONFOUNDER,
                                   semantic_prior={'confounder1': 1.0}))
    generator = ActionBankGenerator()
    entries = generator.generate_entries(PlannerOutput(proposed_actions=proposals), SemanticMemory(),
        ActionBank(), IDGenerator(), config=cfg(), enforce_qwen_contract=True)
    assert [entry.action.prompt for entry in entries] == ['red cap', 'green leaf']
    assert len(generator.last_rejections) == 1
    assert [proposal.prompt for proposal in proposals] == ['red caps', 'red cap', 'green leaves']


def test_plural_history_cannot_repeat_as_a_singular_proposal():
    memory = SemanticMemory()
    memory.record_execution(SensingAction('old', 'target', 'polka dots', ActionFamily.DISCOVERY), 'sam1')
    output = PlannerOutput(proposed_actions=[ProposedAction('target', 'polka dot', ActionFamily.DISCOVERY,
                                                           semantic_prior={'target': 1.0})])
    generator = ActionBankGenerator()
    assert not generator.generate_entries(output, memory, ActionBank(), IDGenerator(), config=cfg(),
                                          enforce_qwen_contract=True)
    assert generator.last_rejections[0].reason == 'DUPLICATE_SEMANTIC_KEY'


def test_singularization_does_not_bypass_invalid_prompt_contract():
    generator = ActionBankGenerator()
    output = PlannerOutput(proposed_actions=[ProposedAction('target', 'pink spheres with black dots',
        ActionFamily.DISCOVERY, semantic_prior={'target': 1.0})])
    assert not generator.generate_entries(output, SemanticMemory(), ActionBank(), IDGenerator(),
                                          config=cfg(), enforce_qwen_contract=True)
    assert generator.last_rejections[0].reason == 'INVALID_GROUNDING_PROMPT'


def test_qwen_receives_singular_query_guidance_and_original_target(mock_openai_client):
    pack = QwenEvidencePack('i', 'polka dots', 'target', ContactSheet())
    RealQwenPlanner(base_url='http://fake', model='fake').plan_scene(pack, BudgetState(), cfg())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs['messages']
    assert 'Use singular noun forms in all SAM3 queries' in messages[0]['content']
    text = messages[1]['content'][0]['text']
    assert "User Target Concept: 'polka dots'" in text
    assert "Singular SAM3 target phrase: 'polka dot'" in text


@pytest.mark.parametrize('value', ['true', 1, None])
def test_singularization_flag_requires_boolean(value):
    with pytest.raises(ValueError, match='singularize_prompts'):
        SAM3Config(singularize_prompts=value)
