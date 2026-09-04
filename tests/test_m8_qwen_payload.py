import pytest
import os
import tempfile
import base64
from PIL import Image
from unittest.mock import patch, MagicMock

from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
from sam3_vlm.core.types import BudgetState
from sam3_vlm.core.config import V4Config

@pytest.fixture
def mock_openai_client():
    import sys
    from unittest.mock import MagicMock
    mock_openai_module = MagicMock()
    mock_client_class = MagicMock()
    mock_openai_module.OpenAI = mock_client_class
    
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        client_instance = MagicMock()
        mock_client_class.return_value = client_instance
        
        # Mock response
        message_mock = MagicMock()
        message_mock.content = '{"scene_summary": "Test", "proposed_actions": [], "missing_appearance_modes": [], "likely_confounders": []}'
        choice_mock = MagicMock()
        choice_mock.message = message_mock
        response_mock = MagicMock()
        response_mock.choices = [choice_mock]
        
        client_instance.chat.completions.create.return_value = response_mock
        yield client_instance

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
    messages = kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    content = messages[1]["content"]
    assert isinstance(content, list)
    
    # Text prompt + 2 images
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert "find target" in content[0]["text"]
    assert "EXACT PROMPT BLACKLIST" in content[0]["text"]
    assert "shadowed green fruit" in content[0]["text"]
    assert "DISCOVERY IS SATURATED" in content[0]["text"]
    
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].startswith("data:image/webp;base64,")

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

