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
            json.dump({"schema_version": "1.0"}, f)
        with open(paths.summary_json, "w") as f:
            json.dump({"schema_version": "1.0"}, f)
            
        # final graph
        graph_path = paths.base_dir / "artifacts" / "graph" / "final_graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w") as f:
            json.dump({"nodes": {}}, f)
            
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "TEST"}) + "\n")
            f.write(json.dumps({"sequence_number": 2, "event_id": "e2", "event_type": "TEST"}) + "\n")
            
        res = validator.validate()
        assert res.valid
        
        # 3. Inject sequence discontinuity
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "TEST"}) + "\n")
            f.write(json.dumps({"sequence_number": 3, "event_id": "e2", "event_type": "TEST"}) + "\n")
            
        res = validator.validate()
        assert not res.valid
        assert any("Sequence discontinuity" in e for e in res.errors)
        
        # 4. Missing Qwen artifact
        with open(paths.events_jsonl, "w") as f:
            f.write(json.dumps({"sequence_number": 1, "event_id": "e1", "event_type": "QWEN_PLAN_COMPLETED", "data": {"qwen_artifact": {"relative_path": "artifacts/qwen/missing.json"}}}) + "\n")
        res = validator.validate()
        assert not res.valid
        assert any("Artifact missing" in e for e in res.errors)
