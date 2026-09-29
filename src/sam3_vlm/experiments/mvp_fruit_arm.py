"""Bridge the VLM-first MVP into the existing fruit pilot matrix."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from sam3_vlm.mvp.adapters import RealSAM3, RealVLM
from sam3_vlm.mvp.cli import result_dict
from sam3_vlm.mvp.core import Config, Controller, Result


FRUIT_ARM = "F_VLMFirst_AdaptiveROI"
FRUIT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "fscd147.json"


def fruit_config() -> Config:
    """Use the same frozen VLM-first policy as the FSCD-147 adapter."""
    return Config(**json.loads(FRUIT_CONFIG.read_text()))


def make_adapters(legacy_sam3, deployment, config: Config | None = None):
    """Reuse the already-loaded SAM3 weights; Qwen uses the fruit target scope."""
    config = config or fruit_config()
    sam3 = RealSAM3(sensor=legacy_sam3)
    vlm = RealVLM(base_url=deployment.qwen_base_url, model=deployment.qwen_model,
                  scope=deployment.v4_config.planner.target_scope,
                  coordinate_mode=config.vlm_coordinate_mode)
    return sam3, vlm


def run_fruit_arm(image, target: str, config: Config, sam3, vlm, artifact_dir: Path) -> Result:
    """Persist the MVP trace independently of the historical M8 replay format."""
    result = Controller(sam3, vlm, config).run(image, target)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mvp_result.json").write_text(json.dumps(result_dict(result), indent=2) + "\n")
    qwen_dir = artifact_dir / "artifacts" / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    requests = getattr(vlm, "request_log", [])
    for proposal in result.proposals:
        index = proposal["call"] - 1
        request = requests[index] if index < len(requests) else {}
        artifact = {
            "qwen_call_id": f"mvp_qwen_{proposal['call']:03d}",
            "input": {"request_text": request.get("system", "") + "\n" + request.get("user_text", "")},
            "output": proposal.get("raw"),
            "metadata": {"rejections": proposal.get("rejected", []),
                         "accepted_action_count": len(proposal.get("accepted", [])),
                         "roi": proposal.get("roi"), "tile_mode": proposal.get("tile_mode")},
        }
        (qwen_dir / f"call_{proposal['call']:03d}.json").write_text(json.dumps(artifact, indent=2) + "\n")
    summary = {
        "engine": "mvp_vlm_first",
        "final_stop_reason": result.stop_reason,
        "final_soft_count": result.counts["soft"],
        "final_hard_count": result.counts["hard"],
        "node_count": len(result.nodes),
        "sam3_calls": result.calls["sam3_attempted"],
        "sam3_tiles": sum(action.get("source") == "adaptive_tile" and
                          action.get("status") == "completed" for action in result.actions),
        "qwen_calls": result.calls["vlm_attempted"],
        "cleanup_calls": 0,
        "number_of_replans": max(0, result.calls["vlm_attempted"] - 1),
        "runtime_ms": result.elapsed_seconds * 1000,
        "partial": result.partial,
        "roi": result.roi,
        "discovery_statistics": {"raw_soft_count": result.counts["soft"]},
        "config": asdict(config),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result
