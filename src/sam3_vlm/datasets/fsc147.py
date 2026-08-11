"""FSC-147 dataset adapter for counting evaluation (V4 Design Spec §17.1)."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

from sam3_vlm.datasets.base import CountingDataset, GroundTruth, Sample


class FSC147Dataset(CountingDataset):
    """Adapter for FSC-147 benchmark. Uses lazy imports for image loading."""

    def __init__(self, data_root: str, split: str = "val", max_samples: Optional[int] = None):
        self.data_root = Path(data_root)
        self.split = split
        self.max_samples = max_samples
        
        # We only load annotations if the file exists (for smoke tests it might not)
        self.annotations = self._load_annotations()
        self.split_dict = self._load_split()
        
        self.image_ids = self.split_dict.get(self.split, [])
        if self.max_samples is not None:
            self.image_ids = self.image_ids[:self.max_samples]

    def _load_annotations(self) -> Dict[str, Any]:
        anno_path = self.data_root / "annotation_FSC147_384.json"
        if anno_path.exists():
            with open(anno_path, "r") as f:
                return json.load(f)
        return {}

    def _load_split(self) -> Dict[str, List[str]]:
        split_path = self.data_root / "Train_Test_Val_FSC_147.json"
        if split_path.exists():
            with open(split_path, "r") as f:
                return json.load(f)
        # Smoke test default
        return {"val": []}

    def _load_image(self, image_id: str) -> Any:
        """Lazy load image using PIL."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required to load FSC-147 images. Install Pillow.")
            
        img_path = self.data_root / "images_384_VarV2" / image_id
        if img_path.exists():
            return Image.open(img_path).convert("RGB")
        return None

    def samples(self) -> Iterable[Sample]:
        for img_id in self.image_ids:
            anno = self.annotations.get(img_id, {})
            concept = "object"
            # In FSC147, classes are given implicitly or we use 'object'
            
            gt = None
            if anno:
                pts = anno.get("points", [])
                gt = GroundTruth(
                    count=len(pts),
                    points=pts
                )
                
            img = self._load_image(img_id)
            
            yield Sample(
                sample_id=img_id,
                image=img,
                concept_name=concept,
                ground_truth=gt
            )

    def user_prompt(self, sample: Sample) -> str:
        return f"Find and count every {sample.concept_name} in this image."

    def ground_truth(self, sample: Sample) -> Optional[GroundTruth]:
        return sample.ground_truth
