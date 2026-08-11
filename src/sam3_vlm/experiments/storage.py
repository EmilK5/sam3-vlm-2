"""Storage accounting for M7 evaluation (V4 Design Spec §17.3)."""

from pathlib import Path
from typing import Dict, Any

def compute_run_storage(run_dir: str) -> Dict[str, Any]:
    """Compute the actual disk footprint of a run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return {}
        
    def get_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        
    events_bytes = get_size(run_path / "events.jsonl") if (run_path / "events.jsonl").exists() else 0
    mask_bytes = get_size(run_path / "artifacts" / "masks") if (run_path / "artifacts" / "masks").exists() else 0
    contact_sheet_bytes = get_size(run_path / "artifacts" / "contact_sheets") if (run_path / "artifacts" / "contact_sheets").exists() else 0
    qwen_bytes = get_size(run_path / "artifacts" / "qwen") if (run_path / "artifacts" / "qwen").exists() else 0
    total_bytes = get_size(run_path)
    
    return {
        "total_run_bytes": total_bytes,
        "events_json_bytes": events_bytes,
        "mask_bytes": mask_bytes,
        "contact_sheet_bytes": contact_sheet_bytes,
        "qwen_bytes": qwen_bytes
    }
