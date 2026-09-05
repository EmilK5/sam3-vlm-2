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
