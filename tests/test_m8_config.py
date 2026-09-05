import pytest
import json
from pathlib import Path
from sam3_vlm.experiments.m8_smoke import load_m8_config, M8DeploymentConfig

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_load_m8_config_precedence(tmp_path, monkeypatch):
    config_dict = {
        "sam3_model": "json_sam3",
        "qwen_model": "json_qwen",
        "require_cuda": True,
        "seed": 100,
        "pilot_sample_limit": 5,
        "budget": {"max_qwen_calls": 2},
        "tiling": {"grid_rows": 4},
        "planner": {"max_actions_per_prompt": 1},
    }

    monkeypatch.setenv("QWEN_MODEL", "env_qwen")
    monkeypatch.setenv("QWEN_BASE_URL", "env_url")
    
    p = tmp_path / "valid.json"
    with open(p, "w") as f:
        json.dump(config_dict, f)
    
    # CLI takes precedence over all
    args = DummyArgs(require_cuda=False, output_dir="cli_out", max_samples=3)
    cfg = load_m8_config(args, config_path=str(p))
    
    assert cfg.sam3_model == "json_sam3"
    assert cfg.qwen_model == "env_qwen" # Env over JSON
    assert cfg.qwen_base_url == "env_url"
    assert cfg.require_cuda is False # CLI over JSON
    assert cfg.seed == 100
    assert cfg.pilot_sample_limit == 3 # CLI over JSON
    assert cfg.output_root == str(Path("cli_out").resolve()) # CLI over Defaults
    assert cfg.v4_config.output_dir == cfg.output_root
    
    assert cfg.v4_config.budget.max_qwen_calls == 2
    assert cfg.v4_config.tiling.grid_rows == 4
    assert cfg.v4_config.planner.max_actions_per_prompt == 1
    assert cfg.v4_config.device == "cpu"
    
def test_load_m8_config_invalid_key(tmp_path):
    config_dict = {
        "budget": {"max_qwen_calls": 10},
        "unknown_key": "bad"
    }
    
    p = tmp_path / "invalid.json"
    with open(p, "w") as f:
        json.dump(config_dict, f)
        
    args = DummyArgs()
    with pytest.raises(ValueError, match="Unknown config key: unknown_key"):
        load_m8_config(args, config_path=str(p))

def test_load_m8_config_missing_file():
    args = DummyArgs(require_cuda=True)
    cfg = load_m8_config(args, config_path="does_not_exist.json")
    
    assert cfg.v4_config.device == "cuda"
    assert cfg.require_cuda is True
    assert cfg.sam3_model == "facebook/sam3"
    assert cfg.pilot_sample_limit == 10


@pytest.mark.parametrize("output_dir", ["", " ", "\t\n"])
@pytest.mark.parametrize("source", ["cli", "config"])
def test_empty_m8_output_is_rejected(tmp_path, output_dir, source):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_root": output_dir} if source == "config" else {}))
    args = DummyArgs(output_dir=output_dir) if source == "cli" else DummyArgs()
    with pytest.raises(ValueError, match="output directory.*non-empty"):
        load_m8_config(args, config_path=path)


def test_m8_output_paths_are_normalized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_root": "runs/../chosen"}))
    cfg = load_m8_config(DummyArgs(), config_path=path)
    assert cfg.output_root == str(tmp_path / "chosen")
    assert cfg.v4_config.output_dir == cfg.output_root
    assert not Path(cfg.output_root).exists()  # Loading config has no write effects.


def test_real_m8_config_uses_bounded_target_only_experiment(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    cfg = load_m8_config(DummyArgs(), config_path="configs/m8_real_smoke.json")
    assert cfg.qwen_model == "qwen3.5-9b-sam3"
    assert cfg.v4_config.belief.target_count_commit_threshold == pytest.approx(0.8)
    assert cfg.v4_config.planner.max_actions_per_prompt == 1
    assert cfg.v4_config.planner.max_output_tokens == 512
    assert cfg.v4_config.planner.request_timeout_seconds == pytest.approx(45.0)
    assert cfg.v4_config.planner.reasoning_effort == "none"
    assert cfg.v4_config.budget.max_qwen_calls == 2
    assert cfg.v4_config.replanning.max_replans == 1
