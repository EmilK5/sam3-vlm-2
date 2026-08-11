import tempfile
import json
from pathlib import Path

from sam3_vlm.datasets.fsc147 import FSC147Dataset
from sam3_vlm.experiments.config import ExperimentConfig
from sam3_vlm.core.config import V4Config

def test_fsc147_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create mock annotations
        anno = {
            "img1.jpg": {"points": [[10, 10], [20, 20]]},
            "img2.jpg": {"points": [[5, 5]]}
        }
        with open(root / "annotation_FSC147_384.json", "w") as f:
            json.dump(anno, f)
            
        splits = {
            "val": ["img1.jpg", "img2.jpg"]
        }
        with open(root / "Train_Test_Val_FSC_147.json", "w") as f:
            json.dump(splits, f)
            
        # Do not create images (test lazy load failure)
        dataset = FSC147Dataset(data_root=str(root), split="val", max_samples=1)
        
        samples = list(dataset.samples())
        assert len(samples) == 1
        assert samples[0].sample_id == "img1.jpg"
        assert samples[0].ground_truth.count == 2
        # image should be None since we didn't mock PIL / the file doesn't exist
        assert samples[0].image is None
        
def test_experiment_config_override():
    base = V4Config()
    cfg = ExperimentConfig(
        experiment_name="test",
        dataset_name="fsc147",
        v4_config_overrides={
            "assets_dir": "/tmp/custom_assets"
        }
    )
    new_cfg = cfg.apply_to(base)
    assert new_cfg.assets_dir == "/tmp/custom_assets"
    # base config should be untouched
    assert base.assets_dir == "assets"
