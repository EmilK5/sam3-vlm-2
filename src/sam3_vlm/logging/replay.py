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
            data = event["data"]
            
            if etype == EventKind.QWEN_PLAN_STARTED.value:
                state.qwen_round = data.get("qwen_round", state.qwen_round)
                
            elif etype == EventKind.BUDGET_UPDATED.value:
                state.budget.sam3_calls = data.get("sam3_calls", state.budget.sam3_calls)
                state.budget.sam3_tiles = data.get("sam3_tiles", state.budget.sam3_tiles)
                state.budget.cleanup_calls = data.get("cleanup_calls", state.budget.cleanup_calls)
                state.budget.qwen_calls = data.get("qwen_calls", state.budget.qwen_calls)
                state.budget.model_runtime_ms = data.get("model_runtime_ms", state.budget.model_runtime_ms)
                state.budget.total_runtime_ms = data.get("total_runtime_ms", state.budget.total_runtime_ms)
                
            elif etype == EventKind.DISCOVERY_STATE_UPDATED.value:
                state.discovery_state.recent_new_node_counts = data.get("recent_new_node_counts", [])
                state.discovery_state.saturated = data.get("saturated", False)
                
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
                    
                # Reconstruct node
                if "box" in node_dict:
                    box_coords = node_dict["box"]
                    geom = BoxGeometry(Box(
                        x1=box_coords[0],
                        y1=box_coords[1],
                        x2=box_coords[2],
                        y2=box_coords[3],
                        coordinate_space=node_dict.get("coordinate_space", "global")
                    ))
                    
                    status = NodeStatus(node_dict.get("status", "ACTIVE"))
                    created_by = node_dict.get("created_by_call_id", "")
                    
                    existing_node = state.graph.get_node(node_id)
                    cb_dict = node_dict.get("class_belief", {})
                    probs = cb_dict.get("probabilities", {})
                    if existing_node:
                        # Update existing
                        existing_node.geometry = geom
                        existing_node.status = status
                        existing_node.class_belief = ClassBelief(probabilities=probs if probs else existing_node.class_belief.probabilities)
                    else:
                        # Create new
                        node = Node(
                            node_id=node_id,
                            geometry=geom,
                            created_by_call_id=created_by,
                            status=status
                        )
                        node.class_belief = ClassBelief(probabilities=probs)
                        state.graph.add_node(node)
                        
        from sam3_vlm.scene.state import CountEstimator
        state.count_estimate = CountEstimator.estimate(state.graph, state.target_class)
        return state
