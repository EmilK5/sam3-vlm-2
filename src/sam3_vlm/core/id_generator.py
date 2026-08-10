"""ID generator for stable, domain-prefixed entity identifiers in SAM3-VLM V4.

Invariants (V4 Design Spec §21.1):
- Every persistent entity has a stable string identifier.
- IDs are created once and never derived from transient array indices.
- Standard format: <prefix>_<index:06d> (e.g., node_000143, sam3_000009).
"""

from typing import Dict


class IDGenerator:
    """Thread-safe / deterministic sequential ID generator."""

    DEFAULT_PREFIXES = {
        "run": "run",
        "image": "img",
        "node": "node",
        "action": "action",
        "sam3_call": "sam3",
        "qwen_call": "qwen",
        "observation": "obs",
        "event": "evt",
        "detection": "det",
    }

    def __init__(self, seed_counters: Dict[str, int] | None = None) -> None:
        self._counters: Dict[str, int] = {}
        if seed_counters:
            self._counters.update(seed_counters)

    def next_id(self, domain_or_prefix: str) -> str:
        """Generate next ID for a given domain or prefix."""
        prefix = self.DEFAULT_PREFIXES.get(domain_or_prefix, domain_or_prefix)
        current = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = current
        return f"{prefix}_{current:06d}"

    def next_run_id(self) -> str:
        return self.next_id("run")

    def next_image_id(self) -> str:
        return self.next_id("image")

    def next_node_id(self) -> str:
        return self.next_id("node")

    def next_action_id(self) -> str:
        return self.next_id("action")

    def next_sam3_call_id(self) -> str:
        return self.next_id("sam3_call")

    def next_qwen_call_id(self) -> str:
        return self.next_id("qwen_call")

    def next_observation_id(self) -> str:
        return self.next_id("observation")

    def next_event_id(self) -> str:
        return self.next_id("event")

    def next_detection_id(self) -> str:
        return self.next_id("detection")

    def reset(self) -> None:
        """Reset all counters."""
        self._counters.clear()
