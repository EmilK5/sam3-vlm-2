import base64

import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
from sam3_vlm.core.types import BudgetState
from sam3_vlm.core.config import PlannerConfig, V4Config

@pytest.fixture
def mock_openai_client(monkeypatch):
    # Unit tests must not inherit the live cluster credential exported by the
    # caller. Tests that exercise environment lookup set it explicitly.
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    import sys
    from unittest.mock import MagicMock
    mock_openai_module = MagicMock()
    mock_client_class = MagicMock()
    mock_openai_module.OpenAI = mock_client_class
    
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        client_instance = MagicMock()
        mock_client_class.return_value = client_instance
        client_instance.constructor_mock = mock_client_class
        
        # Mock response
        message_mock = MagicMock()
        message_mock.content = '{"scene_summary": "Test", "proposed_actions": [], "missing_appearance_modes": [], "likely_confounders": []}'
        choice_mock = MagicMock()
        choice_mock.message = message_mock
        response_mock = MagicMock()
        response_mock.choices = [choice_mock]
        
        client_instance.chat.completions.create.return_value = response_mock
        yield client_instance


def test_qwen_api_key_comes_from_environment(mock_openai_client, monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "ollama")

    RealQwenPlanner(
        base_url="http://fake",
        model="fake-model",
        strict_model_errors=True,
    )

    mock_openai_client.constructor_mock.assert_called_once_with(
        base_url="http://fake",
        api_key="ollama",
        max_retries=0,
    )

def test_qwen_payload_construction(mock_openai_client, tmp_path):
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    
    # Create fake images
    img1_path = str(tmp_path / "img1.png")
    Image.new("RGB", (10, 10)).save(img1_path)
    
    img2_path = str(tmp_path / "img2.webp")
    Image.new("RGB", (10, 10)).save(img2_path)
    
    pack = QwenEvidencePack(
        original_image_id="img_1",
        user_prompt="find target",
        target_class="target",
        image_path=img1_path,
        contact_sheet=ContactSheet(crops=[], total_candidates=0, contact_sheet_image_path=img2_path),
        discovery_diagnostics={
            "discovery_saturated": True,
            "tried_sam3_prompts": ["green fruit", "shadowed green fruit"],
        },
    )
    
    planner.plan_scene(pack, BudgetState(), V4Config())
    
    # Check that create was called with correct arguments
    mock_openai_client.chat.completions.create.assert_called_once()
    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    
    assert kwargs["model"] == "fake-model"
    assert kwargs["max_tokens"] == 512
    assert kwargs["timeout"] == 45.0
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}
    mock_openai_client.constructor_mock.assert_called_once_with(
        base_url="http://fake",
        api_key="EMPTY",
        max_retries=0,
    )
    messages = kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "semantic_key must be 'target'" in messages[0]["content"]
    assert "family must be 'DISCOVERY'" in messages[0]["content"]
    assert "never proposed as separate SAM3 actions" in messages[0]["content"]
    
    content = messages[1]["content"]
    assert isinstance(content, list)
    
    # Text prompt + 2 images
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert "find target" in content[0]["text"]
    assert "EXACT PROMPT BLACKLIST" in content[0]["text"]
    assert "shadowed green fruit" in content[0]["text"]
    assert "DISCOVERY IS SATURATED" in content[0]["text"]
    assert "Every action must use semantic_key='target'" in content[0]["text"]
    assert "An empty proposed_actions list is permitted only when discovery is explicitly saturated" in content[0]["text"]
    assert "otherwise return an empty proposed_actions list" in content[0]["text"]
    assert "DISCOVERY IS NOT SATURATED" not in content[0]["text"]
    
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].startswith("data:image/webp;base64,")


def test_qwen_preserves_full_evidence_and_original_image_bytes(mock_openai_client, tmp_path):
    original = tmp_path / "original.png"
    sheet = tmp_path / "contact_sheet.png"
    Image.new("RGB", (1600, 1200), "green").save(original)
    Image.new("RGB", (1024, 1280), "yellow").save(sheet)
    # Exceeds the former approximate byte guard. Transport must preserve the
    # evidence; only the serving model can enforce its actual token capacity.
    history = "semantic observation with complete provenance; " * 300 + "END_HISTORY"
    pack = QwenEvidencePack(
        original_image_id="full_context",
        user_prompt="green fruit",
        target_class="target",
        image_path=str(original),
        scene_summary=history,
        contact_sheet=ContactSheet(contact_sheet_image_path=str(sheet)),
    )
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    planner.plan_scene(pack, BudgetState(), V4Config())

    content = mock_openai_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert history in content[0]["text"]
    assert f"Image Path: {original}" in content[0]["text"]
    assert f"Contact Sheet Image: {sheet}" in content[0]["text"]
    for item, path in zip(content[1:], (original, sheet), strict=True):
        header, encoded = item["image_url"]["url"].split(",", 1)
        assert header == "data:image/png;base64"
        assert base64.b64decode(encoded) == path.read_bytes()


def test_qwen_language_rules_cover_targets_and_descriptive_labels(mock_openai_client, tmp_path):
    original = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(original)
    pack = QwenEvidencePack(
        "image", "green fruit", "target", ContactSheet(), image_path=str(original),
    )
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    planner.plan_scene(pack, BudgetState(), V4Config())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    for text in (messages[0]["content"], messages[1]["content"][0]["text"]):
        for field in ("sam3_prompt", "likely_confounders", "missing_appearance_modes"):
            assert field in text
        assert "1 to 3 words" in text
        assert "noun alone" in text
        assert "simple everyday object names" in text
        assert "Preserve the user's target object category" in text
        assert "different developmental stage" in text
        assert "Vocabulary is open" in text
        for example in ("small bud", "leaf shadow artifact", "unripe citrus bud", "partially shaded fruit"):
            assert example not in text
        assert "exactly 2 or 3 words" not in text
        assert "Allowed basic adjectives" not in text
        assert "Allowed basic object nouns" not in text
        assert "Unknown words are rejected" not in text
    assert "never proposed as separate SAM3 actions" in messages[0]["content"]


@pytest.mark.parametrize("target", ["green fruit", "red cars", "ripe pears"])
def test_qwen_preserves_current_target_and_actual_evidence(mock_openai_client, tmp_path, target):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(image_path)
    history = "Previous rejected proposal: small bud. Previous confounder: leaf shadow artifact."
    pack = QwenEvidencePack(
        "image", target, "target", ContactSheet(), image_path=str(image_path),
        scene_summary=history,
        discovery_diagnostics={"tried_sam3_prompts": ["small bud", target]},
    )
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    planner.plan_scene(pack, BudgetState(), V4Config())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    text = messages[1]["content"][0]["text"]
    assert f"TARGET TO PRESERVE: {target!r}" in text
    assert history in text
    assert f"EXACT PROMPT BLACKLIST: {['small bud', target]}" in text
    assert "never promote a confounder label into a target action" in text
    assert "same target objects" in text
    assert "prefer 'fruit' to 'citrus'" not in text
    assert "small bud" not in messages[0]["content"]


@pytest.mark.parametrize("diagnostics", [{}, {"discovery_saturated": False}])
def test_unsaturated_real_qwen_payload_requires_one_experiment(mock_openai_client, tmp_path, diagnostics):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(image_path)
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    pack = QwenEvidencePack(
        "img", "green citrus", "target", ContactSheet(),
        image_path=str(image_path), discovery_diagnostics=diagnostics,
    )
    planner.plan_scene(pack, BudgetState(), V4Config())
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    system = messages[0]["content"]
    dynamic = messages[1]["content"][0]["text"]
    for text in (system, dynamic):
        assert "Qwen never decides whether the pipeline should stop" in text
        assert "exactly one novel target DISCOVERY experiment" in text
        assert "even when current candidates look convincing" in text
        assert "Only the controller may stop after evaluating sensor evidence and budget" in text
        assert "An empty proposed_actions list is permitted only when discovery is explicitly saturated" in text
        assert "If no useful new target prompt remains, return no actions" not in text
    assert "DISCOVERY IS NOT SATURATED" in dynamic


def test_qwen_generation_limits_are_configurable(mock_openai_client, tmp_path):
    planner = RealQwenPlanner(
        base_url="http://fake",
        model="fake-model",
        strict_model_errors=True,
    )
    image_path = str(tmp_path / "image.png")
    Image.new("RGB", (10, 10)).save(image_path)
    pack = QwenEvidencePack(
        original_image_id="img_1",
        user_prompt="find target",
        target_class="target",
        image_path=image_path,
        contact_sheet=ContactSheet(crops=[], total_candidates=0),
    )
    config = V4Config(
        planner=PlannerConfig(
            max_output_tokens=256,
            request_timeout_seconds=12.5,
            reasoning_effort="low",
        )
    )

    planner.plan_scene(pack, BudgetState(), config)

    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout"] == 12.5
    assert kwargs["extra_body"] == {"reasoning_effort": "low"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"request_timeout_seconds": 0}, "request_timeout_seconds"),
        ({"reasoning_effort": "extreme"}, "reasoning_effort"),
    ],
)
def test_qwen_generation_limits_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PlannerConfig(**kwargs)

def test_qwen_strict_error_propagation(mock_openai_client, tmp_path):
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("Connection refused")
    
    img_path = str(tmp_path / "img1.png")
    Image.new("RGB", (10, 10)).save(img_path)
    
    pack = QwenEvidencePack(
        original_image_id="img_1",
        user_prompt="find target",
        target_class="target",
        image_path=img_path,
        contact_sheet=ContactSheet(crops=[], total_candidates=0)
    )
    
    with pytest.raises(RuntimeError, match="Strict Qwen execution failed: Connection refused"):
        planner.plan_scene(pack, BudgetState(), V4Config())

def test_qwen_payload_missing_image_strict(mock_openai_client, tmp_path):
    planner = RealQwenPlanner(base_url="http://fake", model="fake", strict_model_errors=True)
    
    # Missing path
    evidence = QwenEvidencePack(
        original_image_id="test",
        user_prompt="prompt",
        target_class="target",
        image_path=None,
        contact_sheet=ContactSheet(crops=[], total_candidates=0)
    )
    
    with pytest.raises(ValueError, match="Original image is strictly required"):
        planner.plan_scene(evidence, BudgetState(), V4Config())
        
    # Non-existent path
    evidence.image_path = str(tmp_path / "does_not_exist.jpg")
    with pytest.raises(ValueError, match="Original image not found at"):
        planner.plan_scene(evidence, BudgetState(), V4Config())

def test_qwen_payload_unsupported_mime(mock_openai_client, tmp_path):
    planner = RealQwenPlanner(base_url="http://fake", model="fake", strict_model_errors=True)
    
    evidence = QwenEvidencePack(
        original_image_id="test",
        user_prompt="prompt",
        target_class="target",
        image_path=str(tmp_path / "img.bmp"),
        contact_sheet=ContactSheet(crops=[], total_candidates=0)
    )
    
    Image.new("RGB", (10, 10)).save(evidence.image_path)
    
    with pytest.raises(ValueError, match="Unsupported image extension for Qwen: .bmp"):
        planner.plan_scene(evidence, BudgetState(), V4Config())


def test_e_prompt_requires_exploration_during_discovery_plateau(mock_openai_client, tmp_path):
    from sam3_vlm.experiments.m8_smoke import _pilot_variants

    config = _pilot_variants(V4Config())[-1].config
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(image_path)
    planner = RealQwenPlanner(base_url="http://fake", model="fake-model", strict_model_errors=True)
    pack = QwenEvidencePack(
        "img", "green fruit", "target", ContactSheet(), image_path=str(image_path),
        belief_classes=["target"],
        discovery_diagnostics={"discovery_saturated": True},
    )
    planner.plan_scene(pack, BudgetState(), config)
    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    text = kwargs["messages"][1]["content"][0]["text"]
    assert "CONTINUE UNTIL CONTROLLER SATURATION" in text
    assert "An empty proposed_actions list is permitted" not in text
    assert "otherwise return an empty" not in text
    assert "SAM3 threshold is fixed by the controller at 0.5" in text
    assert '"suggested_threshold": 0.5' in text
    assert kwargs["max_tokens"] == 512


def test_negative_prompt_instructions_keep_target_schema_and_full_evidence(mock_openai_client, tmp_path):
    from dataclasses import replace

    img = tmp_path / 'image.png'
    Image.new('RGB', (10, 10)).save(img)
    planner = RealQwenPlanner(base_url='http://fake', model='fake-model', strict_model_errors=True)
    config = V4Config(planner=replace(PlannerConfig(), execute_confounder_prompts=True))
    pack = QwenEvidencePack('img', 'green fruit', 'target', ContactSheet(), image_path=str(img),
                            scene_summary='complete scene evidence', belief_classes=['target', 'confounder1'])
    planner.plan_scene(pack, BudgetState(), config)
    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    system = kwargs['messages'][0]['content']
    text = kwargs['messages'][1]['content'][0]['text']
    assert 'executes likely_confounders as separate negative SAM3 queries' in system
    assert 'never proposed as separate SAM3 actions' not in system
    assert 'non-executable scene context' not in text
    assert 'separate SAM3 negative-evidence queries' in text
    assert 'complete scene evidence' in text
    assert 'Vocabulary is open' in text
    assert '0.5' in text
    assert len(kwargs['messages'][1]['content']) == 2


@pytest.mark.parametrize('negatives', [False, True])
@pytest.mark.parametrize('until_saturation', [False, True])
@pytest.mark.parametrize('version', ['v3', 'v4'])
def test_v3_prompt_grounding_scope_and_full_context(mock_openai_client, tmp_path, negatives, until_saturation, version):
    from dataclasses import replace
    original = tmp_path / 'image.png'
    Image.new('RGB', (10, 10)).save(original)
    pack = QwenEvidencePack('i', 'green fruit', 'target', ContactSheet(),
                            image_path=str(original), scene_summary='complete history ' * 1000,
                            discovery_diagnostics={'discovery_saturated': True})
    base = V4Config()
    config = replace(base,
                     planner=replace(base.planner, prompt_version=version,
                                     execute_confounder_prompts=negatives,
                                     target_scope='Only fruit on trees; exclude fallen fruit.'),
                     replanning=replace(base.replanning, continue_until_saturation=until_saturation))
    planner = RealQwenPlanner(base_url='http://fake', model='fake', strict_model_errors=True)
    planner.plan_scene(pack, BudgetState(), config)
    request = mock_openai_client.chat.completions.create.call_args.kwargs
    system = request['messages'][0]['content']
    user = request['messages'][1]['content'][0]['text']
    assert RealQwenPlanner.DISCOVERY_GUIDANCE_V3 in system
    assert (RealQwenPlanner.EVIDENCE_GUIDANCE_V4 in system) == (version == 'v4')
    assert 'Only fruit on trees; exclude fallen fruit.' in system
    assert 'Vocabulary remains open' in system
    assert pack.scene_summary in user
    assert request['max_tokens'] == 512
    assert request['messages'][1]['content'][1]['image_url']['url'].split(',')[1] == base64.b64encode(original.read_bytes()).decode()
    assert planner.last_request_text == {'system': system, 'user': user}
    assert ('controller executes likely_confounders' in system) == negatives
    if until_saturation:
        assert 'An empty proposed_actions list is permitted' not in system
        assert 'Whenever the controller requests a plan' in system
        assert 'Do not return an empty' in system
    else:
        assert 'An empty proposed_actions list is permitted' in system
    # Historical prompt arm remains byte-for-byte the original system, including
    # the historical E stopping wording, and does not inherit dataset scope.
    planner.plan_scene(pack, BudgetState(), replace(config, planner=replace(config.planner, prompt_version='old', execute_confounder_prompts=False)))
    assert planner.last_request_text['system'] == RealQwenPlanner.SYSTEM_PROMPT
    assert 'Only fruit on trees' not in planner.last_request_text['user']


def test_rejection_feedback_is_explicit_and_preserves_target_and_images(mock_openai_client, tmp_path):
    original = tmp_path / 'image.png'
    sheet = tmp_path / 'sheet.png'
    Image.new('RGB', (10, 10)).save(original)
    Image.new('RGB', (10, 10)).save(sheet)
    feedback = {'rejections': [{'sam3_prompt': 'green fruit in shadow', 'reason': 'INVALID_GROUNDING_PROMPT',
                                'detail': 'Use 1-3 words'}]}
    pack = QwenEvidencePack('i', 'green fruit', 'target', ContactSheet(contact_sheet_image_path=str(sheet)),
                            image_path=str(original), discovery_diagnostics={'previous_plan_feedback': feedback})
    planner = RealQwenPlanner(base_url='http://fake', model='fake', strict_model_errors=True)
    planner.plan_scene(pack, BudgetState(), V4Config())
    content = mock_openai_client.chat.completions.create.call_args.kwargs['messages'][1]['content']
    assert len(content) == 3
    assert 'CONTROLLER REJECTION FEEDBACK' in content[0]['text']
    assert 'INVALID_GROUNDING_PROMPT' in content[0]['text']
    assert "TARGET TO PRESERVE: 'green fruit'." in content[0]['text']
