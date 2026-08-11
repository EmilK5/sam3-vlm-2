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
        from sam3_vlm.logging.schema import EventKind
        if event_type not in EventKind.__members__.values():
            raise ValueError(f"Unknown event_type: {event_type}")
        
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

    def save_qwen_artifact(self, call_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save Qwen input/output externally and return ArtifactRef dict."""
        import hashlib
        file_name = f"{call_id}.json"
        path = self.paths.qwen_dir / file_name
        
        # We save input, output and metadata
        self._write_json_atomic(path, data)
        
        # Calculate sha256
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
                
        return {
            "relative_path": str(path.relative_to(self.paths.base_dir)),
            "artifact_type": "qwen_json",
            "sha256": h.hexdigest(),
            "size_bytes": path.stat().st_size
        }

    def save_contact_sheet_artifact(self, qwen_call_id: str, image_bytes: bytes) -> Dict[str, Any]:
        """Save a contact sheet image and return ArtifactRef dict."""
        import hashlib
        file_name = f"{qwen_call_id}_contact_sheet.jpg"
        path = self.paths.contact_sheets_dir / file_name
        path.write_bytes(image_bytes)
        
        h = hashlib.sha256(image_bytes)
        return {
            "relative_path": str(path.relative_to(self.paths.base_dir)),
            "artifact_type": "contact_sheet_jpg",
            "sha256": h.hexdigest(),
            "size_bytes": len(image_bytes)
        }

    def save_mask_artifact(self, detection_id: str, mask_array: Any) -> Dict[str, Any]:
        """Save binary mask to npz and return ArtifactRef dict."""
        import numpy as np
        import hashlib
        file_name = f"{detection_id}.npz"
        path = self.paths.masks_dir / file_name
        
        np.savez_compressed(path, mask=mask_array)
        
        # Calculate sha256
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(path, 'rb', buffering=0) as f:
            while n := f.readinto(mv):
                h.update(mv[:n])
                
        return {
            "relative_path": str(path.relative_to(self.paths.base_dir)),
            "artifact_type": "mask_npz",
            "sha256": h.hexdigest(),
            "size_bytes": path.stat().st_size,
            "shape": list(mask_array.shape) if hasattr(mask_array, "shape") else None,
            "dtype": str(mask_array.dtype) if hasattr(mask_array, "dtype") else None
        }
        
    def record_run_started(self):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.RUN_STARTED.value, {
            "user_prompt": self.manifest.user_prompt,
            "target_class": self.manifest.target_class
        })

    def record_bootstrap_started(self):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.BOOTSTRAP_STARTED.value, {})
        
    def record_bootstrap_completed(self, num_nodes: int):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.BOOTSTRAP_COMPLETED.value, {"num_nodes": num_nodes})
        
    def record_qwen_plan_started(self, round_num: int):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.QWEN_PLAN_STARTED.value, {"qwen_round": round_num})
        
    def record_qwen_plan_completed(self, qwen_artifact: Dict[str, Any], action_ids: List[str]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.QWEN_PLAN_COMPLETED.value, {
            "qwen_artifact": qwen_artifact,
            "action_ids": action_ids
        })

    def record_action_bank_refreshed(self, total_actions: int, invalid_actions: int):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.ACTION_BANK_REFRESHED.value, {
            "total_actions": total_actions,
            "invalid_actions": invalid_actions
        })

    def record_replan_triggered(self, reason: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.REPLAN_TRIGGERED.value, {"reason": reason})

    def record_sam3_action_selected(self, action_id: str, semantic_key: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.SAM3_ACTION_SELECTED.value, {
            "action_id": action_id,
            "semantic_key": semantic_key
        })

    def record_sam3_action_started(self, action_id: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.SAM3_ACTION_STARTED.value, {"action_id": action_id})

    def record_sam3_action_completed(
        self, 
        action_id: str, 
        call_id: str, 
        observation_dict: Dict[str, Any]
    ):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.SAM3_ACTION_COMPLETED.value, {
            "action_id": action_id,
            "sam3_call_id": call_id,
            "observation": observation_dict
        })

    def record_association_completed(self, action_id: str, matched_count: int, new_count: int):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.ASSOCIATION_COMPLETED.value, {
            "action_id": action_id,
            "matched_nodes": matched_count,
            "new_nodes": new_count
        })

    def record_node_created(self, node_id: str, node_dict: Dict[str, Any], provenance: Dict[str, Any]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.NODE_CREATED.value, {
            "node_id": node_id,
            "node_state": node_dict,
            "provenance": provenance
        })

    def record_node_updated(self, node_id: str, updated_dict: Dict[str, Any], provenance: Dict[str, Any]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.NODE_UPDATED.value, {
            "node_id": node_id,
            "node_update": updated_dict,
            "provenance": provenance
        })
        
    def record_semantic_memory_updated(self, record_dict: Dict[str, Any]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.SEMANTIC_MEMORY_UPDATED.value, record_dict)
        
    def record_discovery_state_updated(self, discovery_dict: Dict[str, Any]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.DISCOVERY_STATE_UPDATED.value, discovery_dict)

    def record_budget_updated(self, budget_dict: Dict[str, Any]):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.BUDGET_UPDATED.value, budget_dict)
        
    def record_belief_update_completed(self, node_count: int, total_entropy: float):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.BELIEF_UPDATE_COMPLETED.value, {
            "node_count": node_count,
            "total_entropy": total_entropy
        })

    def record_cleanup_started(self):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.CLEANUP_STARTED.value, {})
        
    def record_cleanup_action_completed(self, action_id: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.CLEANUP_ACTION_COMPLETED.value, {"action_id": action_id})

    def record_stop_decided(self, reason: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.STOP_DECIDED.value, {"reason": reason})

    def record_final_count(self, mean_count: float, variance: float):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.FINAL_COUNT.value, {
            "mean_count": mean_count,
            "variance": variance
        })

    def record_run_completed(self):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.RUN_COMPLETED.value, {})
        
    def record_run_failed(self, error_message: str):
        from sam3_vlm.logging.schema import EventKind
        self.record_event(EventKind.RUN_FAILED.value, {"error": error_message})
        
    def finalize_success(self, summary: RunSummary, final_graph_dict: Dict):
        """Failure-safe finalization of a successful run."""
        try:
            # Write final graph and summary atomically
            graph_path = self.paths.base_dir / "artifacts" / "graph" / "final_graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(graph_path, final_graph_dict)
            self._write_json_atomic(self.paths.summary_json, summary.__dict__)
            
            # Verify required artifacts
            if not graph_path.exists() or not self.paths.summary_json.exists():
                raise RuntimeError("Finalization failed to write required artifacts")
        except Exception as e:
            self.record_run_failed(str(e))
            raise
            
        self.record_final_count(summary.final_soft_count, summary.count_variance)
        self.record_run_completed()
        
        if self._events_file:
            self._events_file.close()
            self._events_file = None
            
    def record_run_failed(self, exception_msg: str):
        """Failure-safe recording of a crash."""
        self.record_event("RUN_FAILED", {"error": exception_msg})
        if self._events_file:
            self._events_file.close()
            self._events_file = None
