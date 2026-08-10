"""SAM3 visual sensor interface protocol and adapter stubs (V4 Design Spec §4)."""

from typing import Any, Protocol, runtime_checkable
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation


@runtime_checkable
class SAM3Sensor(Protocol):
    """Clean SAM3 visual sensor interface contract."""

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        """Execute action on image and return raw SAM3 sensor observations."""
        ...


class DummySAM3Sensor:
    """Mock SAM3 sensor for testing and foundation verification."""

    def __init__(self, call_id_prefix: str = "sam3") -> None:
        self.call_count = 0
        self.call_id_prefix = call_id_prefix

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        self.call_count += 1
        return SAM3Observation(
            call_id=f"{self.call_id_prefix}_{self.call_count:06d}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[],
            searched_regions=[],
            runtime_ms=10.0,
            model_metadata={"mock": True},
        )
