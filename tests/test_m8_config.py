import pytest
import json
import os
from sam3_vlm.experiments.m8_smoke import load_m8_config

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_load_m8_config_valid(tmp_path):
    config_dict = {
        "budgets": {"max_qwen_calls": 10},
        "tiling": {"grid_rows": 3},
        "seed": 123
    }
    
    p = tmp_path / "valid.json"
    with open(p, "w") as f:
        json.dump(config_dict, f)
        
    args = DummyArgs(require_cuda=False, output_dir="test_out")
    cfg = load_m8_config(args, config_path=str(p))
    
    assert cfg.budget.max_qwen_calls == 10
    assert cfg.tiling.grid_rows == 3
    assert cfg.device == "cpu"
    assert cfg.output_dir == "test_out"

def test_load_m8_config_invalid_key(tmp_path):
    config_dict = {
        "budgets": {"max_qwen_calls": 10},
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
    
    assert cfg.device == "cuda"
