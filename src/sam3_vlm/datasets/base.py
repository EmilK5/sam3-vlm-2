"""Dataset protocol and schema abstractions for counting benchmarks (V4 Design Spec §17)."""

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Protocol


@dataclass
class GroundTruth:
    """Evaluation ground truth annotations for a sample."""

    count: int
    boxes: Optional[List[Any]] = field(default_factory=list)
    points: Optional[List[Any]] = field(default_factory=list)


@dataclass
class Sample:
    """Dataset image sample container."""

    sample_id: str
    image: Any
    concept_name: str
    ground_truth: Optional[GroundTruth] = None


class CountingDataset(Protocol):
    """Dataset adapter interface."""

    def samples(self) -> Iterable[Sample]:
        ...

    def user_prompt(self, sample: Sample) -> str:
        ...

    def ground_truth(self, sample: Sample) -> Optional[GroundTruth]:
        ...
