import pytest
import os
import json
import tempfile
from pathlib import Path

from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.validator import RunValidator

def test_validator_detects_corruptions():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "run_artifacts"
        paths = RunArtifactPaths(base_dir=base_dir)
        paths.ensure_directories()
        
        # 1. Empty dir: everything missing
        validator = RunValidator(paths)
        res = validator.validate()
        assert not res.valid
        assert any("run.json is missing" in e for e in res.errors)
        
        # 2. Write valid run.json and summary.json, and events.jsonl
        with open(paths.run_json, "w") as f:
            json.dump({
                "schema_version": "1.0",
                "run_id": "test_run",
                "image_id": "test.jpg",
                "target_class": "target"
            }, f)
        with open(paths.summary_json, "w") as f:
            json.dump({
                "schema_version": "1.0",
                "run_id": "test_run",
                "final_soft_count": 0.0,
                "node_count": 0,
                "qwen_calls": 0,
                "sam3_calls": 0,
                "sam3_tiles": 0,
                "cleanup_calls": 0,
                "runtime_ms": 0.0,
                "number_of_replans": 0,
                "discovery_statistics": {"coverage_ratio": 0.0, "saturated": False}
            }, f)
            
        # final graph
        graph_path = paths.base_dir / "artifacts" / "graph" / "final_graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w") as f:
            json.dump({"nodes": {}}, f)
            
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"schema_version": "1.0", "run_id": "test_run", "sequence_number": 1, "event_id": "e1", "event_type": "RUN_STARTED"}) + "\n")
            f.write(json.dumps({"schema_version": "1.0", "run_id": "test_run", "sequence_number": 2, "event_id": "e2", "event_type": "RUN_COMPLETED", "data": {}}) + "\n")
            
        res = validator.validate()
        assert res.valid
        
        # 3. Inject sequence discontinuity
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "TEST"}) + "\n")
            f.write(json.dumps({"sequence_number": 3, "event_id": "e2", "event_type": "RUN_COMPLETED", "data": {}}) + "\n")
            
        res = validator.validate()
        assert not res.valid
        assert any("Sequence discontinuity" in e for e in res.errors)
        
        # 4. Missing Qwen artifact
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "QWEN_PLAN_COMPLETED", "data": {"qwen_artifact": {"relative_path": "artifacts/qwen/missing.json"}}}) + "\n")
        res = validator.validate()
        assert not res.valid
        assert any("Artifact missing" in e for e in res.errors)
        
        # 5. Referential integrity corruption (mismatched detection)
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "SAM3_ACTION_COMPLETED", "data": {"sam3_call_id": "c1", "action_id": "a1", "observation": {"detections": [{"detection_id": "d1"}]}}}) + "\n")
            f.write(json.dumps({"sequence_number": 2, "event_id": "e2", "event_type": "NODE_CREATED", "data": {"node_state": {"node_id": "n1", "observations": [{"sam3_call_id": "c1", "action_id": "a1", "detection_id": "d2"}]}}}) + "\n")
            f.write(json.dumps({"sequence_number": 3, "event_id": "e3", "event_type": "RUN_COMPLETED", "data": {}}) + "\n")
        res = validator.validate()
        assert not res.valid
        assert any("references unknown detection d2" in e for e in res.errors)
