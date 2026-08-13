"""Deterministic replay engine for M7 evaluation (V4 Design Spec §16.5)."""

import dataclasses
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from sam3_vlm.core.geometry import Box, BoxGeometry, deserialize_geometry
from sam3_vlm.core.types import BudgetState, StopReason
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest, RunSummary
from sam3_vlm.scene.belief import SemanticMemory, SemanticRecord
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import CountEstimator, CoverageSummary, SceneState


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge a partial serialized object into an existing serialized
    object.

    Replay historically assumed NODE_UPDATED always contained a full Node
    snapshot. Keeping this merge makes replay robust to both full snapshots and
    compact/partial node updates.
    """
    result = deepcopy(base)

    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _deserialize_search_region(value: Any):
    """Deserialize the run-level locked search domain."""
    if value is None:
        return None

    # Normal geometry serialization.
    if isinstance(value, dict):
        return deserialize_geometry(value)

    # The current runner logs search_region as bbox().as_tuple().
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = value
        return BoxGeometry(
            Box(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )

    raise ValueError(f"Unsupported serialized search_region: {value!r}")


def _canonical_geometry(geometry: Any):
    """Stable canonical representation for geometry-like state fields."""
    if geometry is None:
        return None

    box = geometry.bbox()
    return [
        float(box.x1),
        float(box.y1),
        float(box.x2),
        float(box.y2),
    ]


class ReplayEngine:
    """Reconstructs state deterministically from an event log."""

    def __init__(self, paths: RunArtifactPaths):
        self.paths = paths
        self.manifest = self._load_manifest()
        self.summary = self._load_summary()

    def _load_manifest(self) -> RunManifest:
        with open(self.paths.run_json, "r") as f:
            data = json.load(f)
        return RunManifest(**data)

    def _load_summary(self) -> Optional[RunSummary]:
        if not self.paths.summary_json.exists():
            return None

        with open(self.paths.summary_json, "r") as f:
            data = json.load(f)

        return RunSummary(**data)

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        """Yield parsed events in logged order."""
        with open(self.paths.events_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

    def load_mask(self, relative_path: str) -> Any:
        path = self.paths.base_dir / relative_path
        with np.load(path) as data:
            return data["mask"]

    def load_qwen_artifact(self, relative_path: str) -> Dict[str, Any]:
        path = self.paths.base_dir / relative_path
        with open(path, "r") as f:
            return json.load(f)

    def _apply_controller_state(
        self,
        state: SceneState,
        data: Dict[str, Any],
    ) -> None:
        """
        Restore every persistent controller field represented in the event.

        Use hasattr so replay remains backward-compatible with old M7 logs and
        older SceneState schemas.
        """
        scalar_fields = (
            "iteration",
            "qwen_round",
            "replans_executed",
            "actions_since_replan",
            "last_plan_accepted_actions",
            "last_replan_evidence_iteration",
            "search_region_locked",
            "search_region_source",
            "search_region_fallback_used",
            "search_region_call_id",
        )

        for field_name in scalar_fields:
            if field_name in data and hasattr(state, field_name):
                setattr(state, field_name, data[field_name])

        if "belief_classes" in data and hasattr(state, "belief_classes"):
            state.belief_classes = list(data["belief_classes"] or [])

        if "confounder_labels" in data and hasattr(state, "confounder_labels"):
            state.confounder_labels = dict(data["confounder_labels"] or {})

        if "search_region" in data and hasattr(state, "search_region"):
            state.search_region = _deserialize_search_region(
                data["search_region"]
            )

    def _apply_budget_update(
        self,
        state: SceneState,
        data: Dict[str, Any],
    ) -> None:
        """
        Restore all BudgetState fields present in the event.

        This automatically supports old M7 budgets and the newer split runtime
        fields without duplicating schema knowledge here.
        """
        for key, value in data.items():
            if hasattr(state.budget, key):
                setattr(state.budget, key, value)

    def _apply_semantic_memory_update(
        self,
        state: SceneState,
        data: Dict[str, Any],
    ) -> None:
        from sam3_vlm.core.types import ActionFamily

        records = data.get("records", {})

        for semantic_key, raw_record in records.items():
            record_data = deepcopy(raw_record)

            if (
                "family" in record_data
                and isinstance(record_data["family"], str)
            ):
                record_data["family"] = ActionFamily(record_data["family"])

            state.semantic_memory.records[semantic_key] = SemanticRecord(
                **record_data
            )

    def _apply_discovery_state_update(
        self,
        state: SceneState,
        data: Dict[str, Any],
    ) -> None:
        discovery_data = deepcopy(data)

        if (
            "spatial_coverage" in discovery_data
            and isinstance(discovery_data["spatial_coverage"], dict)
        ):
            discovery_data["spatial_coverage"] = CoverageSummary(
                **discovery_data["spatial_coverage"]
            )

        if (
            "unresolved_regions" in discovery_data
            and isinstance(discovery_data["unresolved_regions"], list)
        ):
            discovery_data["unresolved_regions"] = [
                deserialize_geometry(geom)
                for geom in discovery_data["unresolved_regions"]
            ]

        for key, value in discovery_data.items():
            if hasattr(state.discovery_state, key):
                setattr(state.discovery_state, key, value)

    def _apply_stop_decided(
        self,
        state: SceneState,
        data: Dict[str, Any],
    ) -> None:
        reason_str = data.get("reason")
        if not reason_str:
            return

        try:
            reason = StopReason(reason_str)
        except ValueError:
            return

        # Use the same stop-reason precedence logic as the live runner.
        if hasattr(state, "set_stop_reason"):
            state.set_stop_reason(reason)
        else:
            state.stop_reason = reason

    def _apply_node_event(
        self,
        state: SceneState,
        event_type: str,
        data: Dict[str, Any],
        node_created_event: str,
    ) -> None:
        node_id = data.get("node_id")
        if not node_id:
            return

        if event_type == node_created_event:
            serialized_node = data.get("node_state", {})
            if not serialized_node:
                return

            reconstructed = Node.from_dict(serialized_node)

        else:
            node_update = data.get("node_update", {})
            if not node_update:
                return

            existing = state.graph.get_node(node_id)

            if existing is None:
                # A full NODE_UPDATED snapshot is sufficient to reconstruct an
                # absent node. A partial update is not.
                try:
                    reconstructed = Node.from_dict(node_update)
                except Exception as exc:
                    raise ValueError(
                        f"Replay encountered partial NODE_UPDATED for missing "
                        f"node {node_id}."
                    ) from exc
            else:
                # Works for BOTH full node snapshots and partial node updates.
                merged = _deep_merge(existing.to_dict(), node_update)
                reconstructed = Node.from_dict(merged)

        existing = state.graph.get_node(node_id)
        if existing is not None:
            del state.graph.nodes[node_id]

        state.graph.add_node(reconstructed)

    def replay_state(self) -> SceneState:
        """
        Reconstruct the final SceneState from events alone.

        No SAM3 or Qwen calls are performed during replay.
        """
        from sam3_vlm.logging.schema import EventKind

        state = SceneState(
            image_id=self.manifest.image_id,
            user_prompt=self.manifest.user_prompt or "<replay>",
            target_class=self.manifest.target_class,
            graph=SceneGraph(),
            semantic_memory=SemanticMemory(),
            budget=BudgetState(),
        )

        for event in self.iter_events():
            event_type = event["event_type"]
            data = event.get("data", {})

            if event_type == EventKind.CONTROLLER_STATE_UPDATED.value:
                self._apply_controller_state(state, data)

            elif event_type == EventKind.BUDGET_UPDATED.value:
                self._apply_budget_update(state, data)

            elif event_type == EventKind.SEMANTIC_MEMORY_UPDATED.value:
                self._apply_semantic_memory_update(state, data)

            elif event_type == EventKind.DISCOVERY_STATE_UPDATED.value:
                self._apply_discovery_state_update(state, data)

            elif event_type == EventKind.STOP_DECIDED.value:
                self._apply_stop_decided(state, data)

            elif event_type in (
                EventKind.NODE_CREATED.value,
                EventKind.NODE_UPDATED.value,
            ):
                self._apply_node_event(
                    state=state,
                    event_type=event_type,
                    data=data,
                    node_created_event=EventKind.NODE_CREATED.value,
                )

        state.count_estimate = CountEstimator.estimate(
            state.graph,
            state.target_class,
        )

        return state


def canonical_scene_state(state: SceneState) -> dict:
    """
    Extract a deterministic representation of scientifically relevant state.

    This intentionally excludes model objects, GPU tensors, transient action
    bank objects, and non-deterministic runtime-only objects.
    """
    sorted_node_dicts = [
        node.to_dict()
        for node in sorted(
            state.graph.active_nodes(),
            key=lambda node: node.node_id,
        )
    ]

    result = {
        "user_prompt": state.user_prompt,
        "target_class": state.target_class,
        "graph_nodes": sorted_node_dicts,
        "semantic_memory": state.semantic_memory.to_dict(),
        "discovery_state": dataclasses.asdict(state.discovery_state),
        "budget": dataclasses.asdict(state.budget),
        "iteration": state.iteration,
        "qwen_round": state.qwen_round,
        "replans_executed": state.replans_executed,
        "actions_since_replan": state.actions_since_replan,
        "count_mean": (
            state.count_estimate.mean_count
            if state.count_estimate
            else 0.0
        ),
        "count_variance": (
            state.count_estimate.variance
            if state.count_estimate
            else 0.0
        ),
        "stop_reason": (
            state.stop_reason.value
            if state.stop_reason
            else None
        ),
    }

    # These are now persistent scientific/controller state for M8, but older
    # states/logs may not define them.
    if hasattr(state, "belief_classes"):
        result["belief_classes"] = list(state.belief_classes or [])

    if hasattr(state, "confounder_labels"):
        result["confounder_labels"] = dict(
            sorted((state.confounder_labels or {}).items())
        )

    if hasattr(state, "last_plan_accepted_actions"):
        result["last_plan_accepted_actions"] = (
            state.last_plan_accepted_actions
        )

    if hasattr(state, "search_region"):
        result["search_region"] = _canonical_geometry(
            state.search_region
        )

    if hasattr(state, "search_region_locked"):
        result["search_region_locked"] = state.search_region_locked

    if hasattr(state, "search_region_source"):
        result["search_region_source"] = state.search_region_source

    if hasattr(state, "search_region_fallback_used"):
        result["search_region_fallback_used"] = (
            state.search_region_fallback_used
        )

    if hasattr(state, "search_region_call_id"):
        result["search_region_call_id"] = state.search_region_call_id

    return result