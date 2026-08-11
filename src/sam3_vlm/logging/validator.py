"""Validator for M7 execution logs (V4 Design Spec §16.6)."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path

from sam3_vlm.logging.artifacts import RunArtifactPaths


@dataclass
class ValidatorResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class RunValidator:
    """Validates structural integrity, provenance, and schemas of an M7 run."""

    def __init__(self, paths: RunArtifactPaths):
        self.paths = paths

    def validate(self) -> ValidatorResult:
        errors = []
        warnings = []
        
        # 1. Check manifests and summary
        if not self.paths.run_json.exists():
            errors.append("run.json is missing.")
            return ValidatorResult(valid=False, errors=errors, warnings=warnings)
            
        with open(self.paths.run_json, "r") as f:
            try:
                manifest = json.load(f)
                if manifest.get("schema_version") != "1.0":
                    errors.append(f"run.json invalid schema version: {manifest.get('schema_version')}")
            except Exception as e:
                errors.append(f"run.json parsing failed: {e}")
                
        if not self.paths.summary_json.exists():
            errors.append("summary.json is missing. Run may have failed.")
        else:
            with open(self.paths.summary_json, "r") as f:
                summary = json.load(f)
                if summary.get("schema_version") != "1.0":
                    errors.append(f"summary.json invalid schema version: {summary.get('schema_version')}")

        if not self.paths.events_jsonl.exists():
            errors.append("events.jsonl is missing.")
            return ValidatorResult(valid=len(errors)==0, errors=errors, warnings=warnings)
            
        # 2. Check event continuity and structure
        event_ids = set()
        last_seq = 0
        qwen_events = 0
        
        with open(self.paths.events_jsonl, "r") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except Exception as e:
                    errors.append(f"Line {line_idx+1}: invalid JSON: {e}")
                    continue
                        
                seq = event.get("sequence_number", 0)
                if seq != last_seq + 1:
                    errors.append(f"Sequence discontinuity at line {line_idx+1}: expected {last_seq+1}, got {seq}")
                last_seq = seq
                
                eid = event.get("event_id")
                if eid in event_ids:
                    errors.append(f"Duplicate event_id: {eid}")
                event_ids.add(eid)
                
                if event.get("event_type") == "QWEN_PLAN_COMPLETED":
                    qwen_events += 1
                    path_str = event["data"].get("qwen_artifact")
                    if path_str:
                        full_path = self.paths.base_dir / path_str
                        if not full_path.exists():
                            errors.append(f"Qwen artifact missing: {path_str}")
                            
                if event.get("event_type") == "SAM3_ACTION_COMPLETED":
                    for mask_path in event["data"].get("mask_artifacts", []):
                        full_path = self.paths.base_dir / mask_path
                        if not full_path.exists():
                            errors.append(f"Mask artifact missing: {mask_path}")

        if qwen_events == 0:
            warnings.append("No Qwen planning events found (0 calls).")
            
        # 3. Check final graph
        graph_path = self.paths.base_dir / "artifacts" / "graph" / "final_graph.json"
        if not graph_path.exists():
            errors.append("final_graph.json is missing.")
            
        return ValidatorResult(valid=len(errors)==0, errors=errors, warnings=warnings)
