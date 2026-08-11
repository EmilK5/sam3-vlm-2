"""Deterministic replay engine for M7 evaluation (V4 Design Spec §16.5)."""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import numpy as np

from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest, RunSummary
from sam3_vlm.scene.state import SceneState
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.belief import SemanticRecord
from sam3_vlm.core.types import BudgetState, StopReason, NodeStatus, ClassBelief, ObservationRelation, NodeObservationRef
from sam3_vlm.core.geometry import BoxGeometry, Box


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
        """Yield parsed events."""
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

    def replay_state(self) -> SceneState:
        """Run through all events and reconstruct the final SceneState from events alone."""
        from sam3_vlm.scene.belief import SemanticMemory
        from sam3_vlm.logging.schema import EventKind
        state = SceneState(
            image_id=self.manifest.image_id,
            user_prompt="<replay>",
            target_class=self.manifest.target_class,
            graph=SceneGraph(),
            semantic_memory=SemanticMemory(),
            budget=BudgetState()
        )
        
        for event in self.iter_events():
            etype = event["event_type"]
            data = event.get("data", {})
            
            if etype == EventKind.QWEN_PLAN_STARTED.value:
                state.qwen_round = data.get("qwen_round", state.qwen_round)
                state.iteration += 1
                
            elif etype == EventKind.REPLAN_TRIGGERED.value:
                state.replans_executed += 1
                state.actions_since_replan = 0
                
            elif etype == EventKind.SAM3_ACTION_COMPLETED.value:
                state.actions_since_replan += 1
                
            elif etype == EventKind.BUDGET_UPDATED.value:
                state.budget.sam3_calls = data.get("sam3_calls", state.budget.sam3_calls)
                state.budget.sam3_tiles = data.get("sam3_tiles", state.budget.sam3_tiles)
                state.budget.cleanup_calls = data.get("cleanup_calls", state.budget.cleanup_calls)
                state.budget.qwen_calls = data.get("qwen_calls", state.budget.qwen_calls)
                state.budget.model_runtime_ms = data.get("model_runtime_ms", state.budget.model_runtime_ms)
                state.budget.total_runtime_ms = data.get("total_runtime_ms", state.budget.total_runtime_ms)
                
            elif etype == EventKind.SEMANTIC_MEMORY_UPDATED.value:
                records = data.get("records", {})
                for k, v in records.items():
                    # Just directly set via dictionary expansion
                    # Family is stored as str, needs conversion to Enum
                    from sam3_vlm.core.types import ActionFamily
                    if "family" in v and isinstance(v["family"], str):
                        v["family"] = ActionFamily(v["family"])
                    state.semantic_memory.records[k] = SemanticRecord(**v)
                
            elif etype == EventKind.DISCOVERY_STATE_UPDATED.value:
                from sam3_vlm.scene.state import CoverageSummary
                # Handle spatial_coverage nested object
                if "spatial_coverage" in data and isinstance(data["spatial_coverage"], dict):
                    data["spatial_coverage"] = CoverageSummary(**data["spatial_coverage"])
                    
                # Handle unresolved_regions
                from sam3_vlm.core.geometry import deserialize_geometry
                if "unresolved_regions" in data and isinstance(data["unresolved_regions"], list):
                    data["unresolved_regions"] = [deserialize_geometry(geom_dict) for geom_dict in data["unresolved_regions"]]
                    
                for k, v in data.items():
                    if hasattr(state.discovery_state, k):
                        setattr(state.discovery_state, k, v)
                
            elif etype == EventKind.STOP_DECIDED.value:
                reason_str = data.get("reason")
                if reason_str:
                    try:
                        state.stop_reason = StopReason(reason_str)
                    except ValueError:
                        pass
                        
            elif etype == EventKind.NODE_CREATED.value or etype == EventKind.NODE_UPDATED.value:
                node_id = data.get("node_id")
                if etype == EventKind.NODE_CREATED.value:
                    node_dict = data.get("node_state", {})
                else:
                    node_dict = data.get("node_update", {})
                
                if not node_id or not node_dict:
                    continue
                    
                # We expect the full dict representation of the Node
                # Node.from_dict will correctly deserialize geometry, beliefs, observations, and diagnostics.
                reconstructed_node = Node.from_dict(node_dict)
                
                existing_node = state.graph.get_node(node_id)
                if existing_node:
                    # Remove it and re-add the updated one
                    # This is slightly wasteful but ensures 100% fidelity without piecewise patching
                    del state.graph.nodes[node_id]
                    state.graph.add_node(reconstructed_node)
                else:
                    state.graph.add_node(reconstructed_node)
                        
        from sam3_vlm.scene.state import CountEstimator
        state.count_estimate = CountEstimator.estimate(state.graph, state.target_class)
        return state
