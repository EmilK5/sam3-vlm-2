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
        
        import hashlib
        
        def verify_artifact(art_dict: Dict[str, Any], path_prefix: str) -> bool:
            if not isinstance(art_dict, dict) or "relative_path" not in art_dict:
                errors.append(f"{path_prefix} is not a valid ArtifactRef dict: {art_dict}")
                return False
            path_str = art_dict["relative_path"]
            full_path = self.paths.base_dir / path_str
            if not full_path.exists():
                errors.append(f"Artifact missing: {path_str}")
                return False
            if "sha256" in art_dict:
                h = hashlib.sha256()
                with open(full_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        h.update(chunk)
                if h.hexdigest() != art_dict["sha256"]:
                    errors.append(f"Hash mismatch for {path_str}: expected {art_dict['sha256']}, got {h.hexdigest()}")
                    return False
            return True

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
                    art_dict = event["data"].get("qwen_artifact")
                    if art_dict:
                        verify_artifact(art_dict, "Qwen artifact")
                            
                if event.get("event_type") == "SAM3_ACTION_COMPLETED":
                    for art_dict in event["data"].get("mask_artifacts", []):
                        if isinstance(art_dict, str):
                            # Fallback if old format
                            full_path = self.paths.base_dir / art_dict
                            if not full_path.exists():
                                errors.append(f"Mask artifact missing: {art_dict}")
                        else:
                            verify_artifact(art_dict, "Mask artifact")

        if qwen_events == 0:
            warnings.append("No Qwen planning events found (0 calls).")
            
        # Referential integrity structures
        actions = set()
        sam3_calls = {} # call_id -> action_id
        detections = set() # (call_id, detection_id)
        node_observations = [] # list of dicts
        nodes_created_by = {}
        stop_events = 0
        run_completed = False
        
        with open(self.paths.events_jsonl, "r") as f:
            for line in f:
                if not line.strip(): continue
                event = json.loads(line)
                etype = event.get("event_type")
                data = event.get("data", {})
                
                if etype == "SAM3_ACTION_SELECTED":
                    actions.add(data["action_id"])
                elif etype == "SAM3_ACTION_COMPLETED":
                    call_id = data["sam3_call_id"]
                    sam3_calls[call_id] = data["action_id"]
                    obs = data.get("observation", {})
                    if "detections" in obs:
                        for d in obs["detections"]:
                            detections.add((call_id, d["detection_id"]))
                elif etype == "NODE_CREATED":
                    nstate = data.get("node_state", {})
                    nid = nstate.get("node_id")
                    if nid:
                        created_by = nstate.get("created_by_call_id", "")
                        nodes_created_by[nid] = created_by
                        if created_by.startswith("qwen") or not created_by:
                            errors.append(f"Node {nid} created by non-sensor or missing call: {created_by}")
                    
                    for obs in nstate.get("observations", []):
                        node_observations.append({
                            "node_id": nid,
                            "sam3_call_id": obs.get("sam3_call_id"),
                            "action_id": obs.get("action_id"),
                            "detection_id": obs.get("detection_id")
                        })
                elif etype == "NODE_UPDATED":
                    nup = data.get("node_update", {})
                    nid = nup.get("node_id")
                    for obs in nup.get("observations", []):
                        node_observations.append({
                            "node_id": nid,
                            "sam3_call_id": obs.get("sam3_call_id"),
                            "action_id": obs.get("action_id"),
                            "detection_id": obs.get("detection_id")
                        })
                elif etype == "STOP_DECIDED":
                    stop_events += 1
                elif etype == "RUN_COMPLETED":
                    run_completed = True
                    
        # Check integrity
        for obs in node_observations:
            cid = obs["sam3_call_id"]
            aid = obs["action_id"]
            did = obs["detection_id"]
            if cid not in sam3_calls:
                errors.append(f"Node Observation references unknown sam3_call_id: {cid}")
            elif sam3_calls[cid] != aid:
                errors.append(f"Observation mismatch: call {cid} belongs to action {sam3_calls[cid]}, but observation claims action {aid}")
            
            if (cid, did) not in detections:
                errors.append(f"Node Observation references unknown detection {did} for call {cid}")
                
        if not run_completed:
            errors.append("RUN_COMPLETED event is missing (strict boundary validation failed).")
        if stop_events == 0:
            warnings.append("No STOP_DECIDED event recorded.")
            
        # 3. Check final graph
        graph_path = self.paths.base_dir / "artifacts" / "graph" / "final_graph.json"
        if not graph_path.exists():
            errors.append("final_graph.json is missing.")
        else:
            # Replay equivalence
            try:
                from sam3_vlm.logging.replay import ReplayEngine
                engine = ReplayEngine(self.paths)
                replayed_state = engine.replay_state()
                
                with open(graph_path, "r") as f:
                    oracle_graph_dict = json.load(f)
                    
                replayed_graph_dict = replayed_state.graph.to_dict()
                
                oracle_nodes = oracle_graph_dict.get("nodes", {})
                replayed_nodes = replayed_graph_dict.get("nodes", {})
                
                if len(oracle_nodes) != len(replayed_nodes):
                    errors.append(f"Replay mismatch: Oracle graph has {len(oracle_nodes)} nodes, replay produced {len(replayed_nodes)}")
                else:
                    for nid in oracle_nodes:
                        if nid not in replayed_nodes:
                            errors.append(f"Replay mismatch: Node {nid} in Oracle but not in replay")
                            
                # Check hard budgets
                from sam3_vlm.core.config import V4Config
                cfg = V4Config().budget
                b = replayed_state.budget
                if b.sam3_calls > cfg.max_sam3_calls:
                    errors.append(f"Hard limit exceeded: sam3_calls {b.sam3_calls} > {cfg.max_sam3_calls}")
                if b.sam3_tiles > cfg.max_sam3_tiles:
                    errors.append(f"Hard limit exceeded: sam3_tiles {b.sam3_tiles} > {cfg.max_sam3_tiles}")
                    
            except Exception as e:
                errors.append(f"Replay equivalence failed: {e}")
            
        return ValidatorResult(valid=len(errors)==0, errors=errors, warnings=warnings)
