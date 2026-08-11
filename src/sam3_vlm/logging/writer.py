"""Run recorder for M7 logging (V4 Design Spec §16.4)."""

import json
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional, List
from pathlib import Path

from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest, RunSummary, EVENT_SCHEMA_VERSION
from sam3_vlm.logging.events import Event

logger = logging.getLogger(__name__)

class RunRecorder:
    """Records events, metadata, and artifacts failure-safely."""

    def __init__(self, paths: RunArtifactPaths, manifest: RunManifest):
        self.paths = paths
        self.manifest = manifest
        self.paths.ensure_directories()
        
        self.sequence_number = 0
        self._events_file = None
        
        # Write manifest atomically
        self._write_json_atomic(self.paths.run_json, self.manifest.__dict__)
        
        # Open events file for appending
        self._events_file = open(self.paths.events_jsonl, 'a')
        
    def _write_json_atomic(self, path: Path, data: dict):
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def record_event(self, event_type: str, data: Dict[str, Any], parent_event_id: Optional[str] = None) -> Event:
        """Record a structured event and flush to disk."""
        self.sequence_number += 1
        event = Event(
            event_id=f"evt_{self.manifest.run_id}_{self.sequence_number:06d}",
            event_type=event_type,
            timestamp_ms=time.time() * 1000,
            run_id=self.manifest.run_id,
            data=data,
            parent_event_id=parent_event_id,
        )
        
        # Add schema version when serialized
        row = {
            "event_id": event.event_id,
            "run_id": event.run_id,
            "sequence_number": self.sequence_number,
            "event_type": event.event_type,
            "timestamp_ms": event.timestamp_ms,
            "schema_version": EVENT_SCHEMA_VERSION,
            "parent_event_id": event.parent_event_id,
            "data": event.data
        }
        
        if self._events_file:
            self._events_file.write(json.dumps(row) + "\n")
            self._events_file.flush()
            
        return event

    def save_qwen_artifact(self, call_id: str, data: Dict[str, Any]) -> str:
        """Save Qwen input/output externally and return relative path."""
        file_name = f"{call_id}.json"
        path = self.paths.qwen_dir / file_name
        
        self._write_json_atomic(path, data)
        
        return str(path.relative_to(self.paths.base_dir))

    def save_mask_artifact(self, detection_id: str, mask_array: Any) -> str:
        """Save binary mask to npz and return relative path."""
        import numpy as np
        file_name = f"{detection_id}.npz"
        path = self.paths.masks_dir / file_name
        
        np.savez_compressed(path, mask=mask_array)
        return str(path.relative_to(self.paths.base_dir))
        
    def record_run_started(self):
        self.record_event("RUN_STARTED", {
            "user_prompt": self.manifest.user_prompt,
            "target_class": self.manifest.target_class
        })

    def record_bootstrap_started(self):
        self.record_event("BOOTSTRAP_STARTED", {})
        
    def record_bootstrap_completed(self, num_nodes: int):
        self.record_event("BOOTSTRAP_COMPLETED", {"num_nodes": num_nodes})
        
    def record_qwen_plan_started(self, round_num: int):
        self.record_event("QWEN_PLAN_STARTED", {"qwen_round": round_num})
        
    def record_qwen_plan_completed(self, qwen_artifact_path: str, action_ids: List[str]):
        self.record_event("QWEN_PLAN_COMPLETED", {
            "qwen_artifact": qwen_artifact_path,
            "action_ids": action_ids
        })

    def record_action_bank_refreshed(self, total_actions: int, invalid_actions: int):
        self.record_event("ACTION_BANK_REFRESHED", {
            "total_actions": total_actions,
            "invalid_actions": invalid_actions
        })

    def record_replan_triggered(self, reason: str):
        self.record_event("REPLAN_TRIGGERED", {"reason": reason})

    def record_sam3_action_selected(self, action_id: str, semantic_key: str):
        self.record_event("SAM3_ACTION_SELECTED", {
            "action_id": action_id,
            "semantic_key": semantic_key
        })

    def record_sam3_action_started(self, action_id: str):
        self.record_event("SAM3_ACTION_STARTED", {"action_id": action_id})

    def record_sam3_action_completed(
        self, 
        action_id: str, 
        call_id: str, 
        num_detections: int, 
        runtime_ms: float, 
        mask_artifacts: List[str]
    ):
        self.record_event("SAM3_ACTION_COMPLETED", {
            "action_id": action_id,
            "sam3_call_id": call_id,
            "num_detections": num_detections,
            "runtime_ms": runtime_ms,
            "mask_artifacts": mask_artifacts
        })

    def record_association_completed(self, action_id: str, matched_count: int, new_count: int):
        self.record_event("ASSOCIATION_COMPLETED", {
            "action_id": action_id,
            "matched_nodes": matched_count,
            "new_nodes": new_count
        })

    def record_node_created(self, node_id: str, action_id: str, sam3_call_id: str):
        self.record_event("NODE_CREATED", {
            "node_id": node_id,
            "action_id": action_id,
            "sam3_call_id": sam3_call_id
        })

    def record_node_updated(self, node_id: str, action_id: str, sam3_call_id: str, observation_type: str):
        self.record_event("NODE_UPDATED", {
            "node_id": node_id,
            "action_id": action_id,
            "sam3_call_id": sam3_call_id,
            "observation_type": observation_type
        })
        
    def record_belief_update_completed(self, node_count: int, total_entropy: float):
        self.record_event("BELIEF_UPDATE_COMPLETED", {
            "node_count": node_count,
            "total_entropy": total_entropy
        })

    def record_cleanup_started(self):
        self.record_event("CLEANUP_STARTED", {})
        
    def record_cleanup_action_completed(self, action_id: str):
        self.record_event("CLEANUP_ACTION_COMPLETED", {"action_id": action_id})

    def record_stop_decided(self, reason: str):
        self.record_event("STOP_DECIDED", {"reason": reason})

    def record_final_count(self, mean_count: float, variance: float):
        self.record_event("FINAL_COUNT", {
            "mean_count": mean_count,
            "variance": variance
        })

    def record_run_completed(self):
        self.record_event("RUN_COMPLETED", {})
        
    def close(self, summary: Optional[RunSummary] = None, final_graph_dict: Optional[Dict] = None):
        """Close recorder, write summary and graph atomically."""
        if self._events_file:
            self._events_file.close()
            self._events_file = None
            
        if summary:
            self._write_json_atomic(self.paths.summary_json, summary.__dict__)
            
        if final_graph_dict is not None:
            graph_path = self.paths.base_dir / "artifacts" / "graph" / "final_graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(graph_path, final_graph_dict)
            
    def record_run_failed(self, exception_msg: str):
        """Failure-safe recording of a crash."""
        self.record_event("RUN_FAILED", {"error": exception_msg})
        self.close()
