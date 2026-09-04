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
        
        last_budget = {
            "sam3_calls": 0, "sam3_tiles": 0, "cleanup_calls": 0, "qwen_calls": 0, 
            "model_runtime_ms": 0.0, "total_runtime_ms": 0.0
        }
        
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
            if "size_bytes" in art_dict and full_path.stat().st_size != art_dict["size_bytes"]:
                errors.append(f"Size mismatch for {path_str}: expected {art_dict['size_bytes']}, got {full_path.stat().st_size}")
                return False
            if art_dict.get("artifact_type") == "mask_npz":
                try:
                    import numpy as np
                    with np.load(full_path) as data:
                        mask = data["mask"]
                        expected_shape = tuple(art_dict.get("shape", [])) if art_dict.get("shape") else None
                        expected_dtype = art_dict.get("dtype")
                        if expected_shape and mask.shape != expected_shape:
                            errors.append(f"Mask shape mismatch: expected {expected_shape}, got {mask.shape}")
                            return False
                        if expected_dtype and str(mask.dtype) != expected_dtype:
                            errors.append(f"Mask dtype mismatch: expected {expected_dtype}, got {mask.dtype}")
                            return False
                except Exception as e:
                    errors.append(f"Failed to load mask NPZ {path_str}: {e}")
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
                
                # Check run_id and schema
                if event.get("run_id") != manifest.get("run_id"):
                    errors.append(f"Event run_id {event.get('run_id')} does not match manifest run_id {manifest.get('run_id')}")
                if event.get("schema_version") != "1.0":
                    errors.append(f"Event invalid schema version: {event.get('schema_version')}")
                
                etype = event.get("event_type")
                from sam3_vlm.logging.schema import EventKind
                if etype not in EventKind.__members__.values():
                    errors.append(f"Unknown event type: {etype}")
                
                if etype == "BUDGET_UPDATED":
                    bdata = event.get("data", {})
                    for k in last_budget.keys():
                        if k in bdata:
                            if bdata[k] < last_budget[k]:
                                errors.append(f"Budget {k} decreased from {last_budget[k]} to {bdata[k]}")
                            last_budget[k] = bdata[k]
                            
                if etype == "QWEN_PLAN_COMPLETED":
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
        run_started_events = 0
        run_completed_events = 0
        run_failed_events = 0
        final_count_events = 0
        
        with open(self.paths.events_jsonl, "r") as f:
            for line in f:
                if not line.strip(): continue
                event = json.loads(line)
                etype = event.get("event_type")
                data = event.get("data", {})
                
                if etype == "RUN_STARTED":
                    run_started_events += 1
                elif etype == "SAM3_ACTION_SELECTED":
                    actions.add(data["action_id"])
                elif etype == "SAM3_ACTION_COMPLETED":
                    call_id = data["sam3_call_id"]
                    aid = data["action_id"]
                    if aid not in actions:
                        errors.append(f"SAM3 call {call_id} references unknown action {aid}")
                    sam3_calls[call_id] = aid
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
                            "detection_id": obs.get("detection_id"),
                            "relation": obs.get("relation")
                        })
                elif etype == "NODE_UPDATED":
                    nup = data.get("node_update", {})
                    nid = nup.get("node_id")
                    if nid not in nodes_created_by:
                        errors.append(f"NODE_UPDATED references node {nid} which was not previously created.")
                    for obs in nup.get("observations", []):
                        node_observations.append({
                            "node_id": nid,
                            "sam3_call_id": obs.get("sam3_call_id"),
                            "action_id": obs.get("action_id"),
                            "detection_id": obs.get("detection_id"),
                            "relation": obs.get("relation")
                        })
                elif etype == "STOP_DECIDED":
                    stop_events += 1
                elif etype == "FINAL_COUNT":
                    final_count_events += 1
                elif etype == "RUN_COMPLETED":
                    run_completed_events += 1
                elif etype == "RUN_FAILED":
                    run_failed_events += 1
                    
        # Check integrity
        for obs in node_observations:
            cid = obs["sam3_call_id"]
            aid = obs["action_id"]
            did = obs["detection_id"]
            rel = obs["relation"]
            
            if rel in ("NOT_RETRIEVED", "NOT_OBSERVABLE"):
                if did is not None:
                    errors.append(f"Node Observation has detection_id {did} but relation is {rel}")
            else:
                if did is None:
                    errors.append(f"Node Observation is missing detection_id for positive relation {rel}")
                    
            if cid not in sam3_calls:
                errors.append(f"Node Observation references unknown sam3_call_id: {cid}")
            elif sam3_calls[cid] != aid:
                errors.append(f"Observation mismatch: call {cid} belongs to action {sam3_calls[cid]}, but observation claims action {aid}")
            
            if did is not None and (cid, did) not in detections:
                errors.append(f"Node Observation references unknown detection {did} for call {cid}")
                
        if run_started_events != 1:
            errors.append(f"Expected exactly 1 RUN_STARTED event, got {run_started_events}")
        if run_completed_events != 1 and run_failed_events == 0:
            errors.append(f"Expected exactly 1 RUN_COMPLETED event, got {run_completed_events}")
        if final_count_events != 1 and run_failed_events == 0:
            errors.append(f"Expected exactly 1 FINAL_COUNT event, got {final_count_events}")
        if stop_events == 0 and run_failed_events == 0:
            errors.append("No STOP_DECIDED event recorded.")
            
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
                    
                replayed_graph_dict_raw = replayed_state.graph.to_dict()
                replayed_graph_dict = json.loads(json.dumps(replayed_graph_dict_raw))
                
                oracle_nodes = oracle_graph_dict.get("nodes", {})
                replayed_nodes = replayed_graph_dict.get("nodes", {})
                
                if len(oracle_nodes) != len(replayed_nodes):
                    errors.append(f"Replay mismatch: Oracle graph has {len(oracle_nodes)} nodes, replay produced {len(replayed_nodes)}")
                else:
                    for nid, onode in oracle_nodes.items():
                        if nid not in replayed_nodes:
                            errors.append(f"Replay mismatch: Node {nid} in Oracle but not in replay")
                        elif onode != replayed_nodes[nid]:
                            errors.append(f"Replay mismatch: Node {nid} state differs between Oracle and replay")
                            
                # Check hard budgets
                from sam3_vlm.core.config import V4Config
                
                # Try to use persisted config, else default
                cfg_dict = {}
                try:
                    with open(self.paths.run_json, "r") as f:
                        run_man = json.load(f)
                        cfg_dict = run_man.get("v4_config", {})
                except Exception:
                    pass
                
                cfg = V4Config().budget
                if cfg_dict and "budget" in cfg_dict:
                    cfg.__dict__.update(cfg_dict["budget"])
                
                b = replayed_state.budget
                if b.sam3_calls > cfg.max_sam3_calls:
                    errors.append(f"Hard limit exceeded: sam3_calls {b.sam3_calls} > {cfg.max_sam3_calls}")
                if b.sam3_tiles > cfg.max_sam3_tiles:
                    errors.append(f"Hard limit exceeded: sam3_tiles {b.sam3_tiles} > {cfg.max_sam3_tiles}")
                if b.qwen_calls > cfg.max_qwen_calls:
                    errors.append(f"Hard limit exceeded: qwen_calls {b.qwen_calls} > {cfg.max_qwen_calls}")
                if b.cleanup_calls > getattr(cfg, 'max_cleanup_calls', 5):
                    errors.append(f"Hard limit exceeded: cleanup_calls {b.cleanup_calls} > {getattr(cfg, 'max_cleanup_calls', 5)}")
                    
                # Compare replayed state with summary oracle
                if 'summary' in locals():
                    s = summary
                    if s.get("sam3_calls") != b.sam3_calls:
                        errors.append(f"Summary mismatch: sam3_calls {s.get('sam3_calls')} != replayed {b.sam3_calls}")
                    if s.get("qwen_calls") != b.qwen_calls:
                        errors.append(f"Summary mismatch: qwen_calls {s.get('qwen_calls')} != replayed {b.qwen_calls}")
                    if s.get("sam3_tiles") != b.sam3_tiles:
                        errors.append(f"Summary mismatch: sam3_tiles {s.get('sam3_tiles')} != replayed {b.sam3_tiles}")
                    if s.get("cleanup_calls") != b.cleanup_calls:
                        errors.append(f"Summary mismatch: cleanup_calls {s.get('cleanup_calls')} != replayed {b.cleanup_calls}")
                    
                    rep_mean = replayed_state.count_estimate.mean_count if replayed_state.count_estimate else 0.0
                    rep_var = replayed_state.count_estimate.variance if replayed_state.count_estimate else 0.0
                    if abs(s.get("final_soft_count", 0.0) - rep_mean) > 1e-4:
                        errors.append(f"Summary mismatch: count {s.get('final_soft_count')} != replayed {rep_mean}")
                    if abs(s.get("count_variance", 0.0) - rep_var) > 1e-4:
                        errors.append(f"Summary mismatch: variance {s.get('count_variance')} != replayed {rep_var}")
                    
                    rep_stop = replayed_state.stop_reason.value if replayed_state.stop_reason else None
                    if s.get("final_stop_reason") != rep_stop:
                        errors.append(f"Summary mismatch: stop_reason {s.get('final_stop_reason')} != replayed {rep_stop}")
                    
            except Exception as e:
                errors.append(f"Replay equivalence failed: {e}")
            
        return ValidatorResult(valid=len(errors)==0, errors=errors, warnings=warnings)
