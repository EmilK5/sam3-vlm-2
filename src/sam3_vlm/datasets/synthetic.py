"""Synthetic dataset for testing and validation (V4 Design Spec §17)."""

from typing import Iterable, Optional, Any
from sam3_vlm.datasets.base import CountingDataset, Sample, GroundTruth


class SyntheticDataset:
    """A deterministic synthetic dataset for regression tests."""

    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        
    def samples(self) -> Iterable[Sample]:
        for i in range(self.num_samples):
            count = i * 2
            boxes = [
                {"xmin": j * 10, "ymin": j * 10, "xmax": j * 10 + 5, "ymax": j * 10 + 5, "coordinate_space": "ABSOLUTE"}
                for j in range(count)
            ]
            
            yield Sample(
                sample_id=f"synth_{i}",
                image="mock_image_path.jpg",
                concept_name="target",
                ground_truth=GroundTruth(count=count, boxes=boxes)
            )

    def user_prompt(self, sample: Sample) -> str:
        return f"Count the {sample.concept_name}s in this image."

    def ground_truth(self, sample: Sample) -> Optional[GroundTruth]:
        return sample.ground_truth
