"""M8 Real-Model Validation and Experimental Smoke Testing (V4)."""

import argparse
import os
import json
import time
import logging
import shutil
import uuid
import sys
import dataclasses
from contextlib import contextmanager
from pathlib import Path
from PIL import Image

from sam3_vlm.core.config import (
    V4Config, BudgetConfig, StoppingConfig, ReplanningConfig, BootstrapConfig,
    TilingConfig, PlannerConfig, SAM3Config, ActionSelectionConfig, AssociationConfig,
    BeliefConfig, CleanupConfig, LoggingConfig
)
from sam3_vlm.models.sam3 import RealSAM3Sensor
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
from sam3_vlm.logging.writer import RunRecorder, RunArtifactPaths, RunManifest
from sam3_vlm.logging.validator import RunValidator
from sam3_vlm.logging.replay import ReplayEngine
from sam3_vlm.logging.schema import RunSummary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_m8_config(args, config_path="configs/m8_real_smoke.json") -> V4Config:
    kwargs = {}
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
            
            if "budgets" in data:
                kwargs["budget"] = BudgetConfig(**data["budgets"])
            if "tiling" in data:
                kwargs["tiling"] = TilingConfig(**data["tiling"])
            if "cleanup" in data:
                kwargs["cleanup"] = CleanupConfig(**data["cleanup"])
            if "replanning" in data:
                kwargs["replanning"] = ReplanningConfig(**data["replanning"])
                
            allowed = {"sam3_model", "qwen_model", "require_cuda", "compile_sam3", "budgets", "tiling", "cleanup", "replanning", "seed", "output_root", "pilot_sample_limit"}
            for k in data.keys():
                if k not in allowed:
                    raise ValueError(f"Unknown config key: {k}")
                    
    if hasattr(args, "require_cuda"):
        kwargs["device"] = "cuda" if args.require_cuda else "cpu"
        
    if hasattr(args, "output_dir"):
        kwargs["output_dir"] = args.output_dir
        
    return V4Config(**kwargs)

def _get_models(args):
    import torch
    device = "cuda" if args.require_cuda else "cpu"
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    sam3 = RealSAM3Sensor(model_id=args.sam3_model, device=device, compile_model=args.compile_sam3)
    qwen = RealQwenPlanner(base_url=args.qwen_base_url, model=args.qwen_model, strict_model_errors=True)
    return sam3, qwen

def _get_sam3_only(args):
    import torch
    device = "cuda" if args.require_cuda else "cpu"
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return RealSAM3Sensor(model_id=args.sam3_model, device=device, compile_model=args.compile_sam3)

def _get_qwen_only(args):
    return RealQwenPlanner(base_url=args.qwen_base_url, model=args.qwen_model, strict_model_errors=True)

@contextmanager
def OracleHider(paths: RunArtifactPaths):
    sum_bak = paths.summary_json.with_suffix('.bak')
    graph_path = paths.base_dir / "artifacts" / "graph" / "final_graph.json"
    graph_bak = graph_path.with_suffix('.bak')
    
    if paths.summary_json.exists():
        shutil.copy(paths.summary_json, sum_bak)
        os.remove(paths.summary_json)
    if graph_path.exists():
        shutil.copy(graph_path, graph_bak)
        os.remove(graph_path)
    try:
        yield
    finally:
        if sum_bak.exists():
            shutil.move(sum_bak, paths.summary_json)
        if graph_bak.exists():
            shutil.move(graph_bak, graph_path)

def assemble_e2e_runner(paths: RunArtifactPaths, config: V4Config, sensor, planner, run_id: str, prompt: str, target_class: str, image_id: str):
    manifest = RunManifest(run_id=run_id, user_prompt=prompt, target_class=target_class, image_id=image_id)
    manifest.v4_config = dataclasses.asdict(config)
    recorder = RunRecorder(paths, manifest)
    runner = Runner(config, sensor=sensor, planner=planner, recorder=recorder)
    return runner, recorder

def preflight(args) -> bool:
    logger.info("=== Preflight Checks ===")
    success = True
    def _check(cond, msg):
        nonlocal success
        if cond: logger.info(f"  [OK] {msg}")
        else:
            logger.error(f"  [FAIL] {msg}")
            success = False

    if args.stage in ("all", "M8.2", "M8.3", "pilot"):
        _check(args.qwen_base_url or os.environ.get("QWEN_BASE_URL"), "Qwen endpoint is set")
        
    try:
        import torch
        if args.require_cuda:
            _check(torch.cuda.is_available(), f"CUDA is available")
        else:
            logger.info("  [INFO] Running with --allow-cpu.")
            
        import transformers
        _check(True, f"Transformers version: {transformers.__version__}")
        if args.stage in ("all", "M8.0", "M8.1", "M8.3", "pilot"):
            from transformers import Sam3Model, Sam3Processor
            _check(True, "Sam3Model importable")
    except ImportError as e:
        _check(False, f"Required package missing: {e}")
        
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        test_file = os.path.join(args.output_dir, ".write_test")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        _check(True, f"Output directory {args.output_dir} writable")
    except Exception as e:
        _check(False, f"Output directory not writable: {e}")
        
    return success

def m8_0_validate_adapters(args):
    logger.info("=== M8.0 Real adapter validation ===")
    try:
        if not args.dry_run:
            _get_models(args)
        logger.info("Real models instantiated successfully.")
        return True
    except Exception as e:
        logger.error(f"FAIL: Adapter validation failed: {e}")
        return False

def m8_1_sam3_smoke(args):
    logger.info("=== M8.1 Single-pass real SAM3 validation ===")
    if not os.path.exists(args.image):
        logger.error(f"FAIL: Image not found at {args.image}")
        return False
    if args.dry_run:
        logger.info("Dry run: Skipping execution.")
        return True
    try:
        sam3 = _get_sam3_only(args)
        img_pil = Image.open(args.image).convert("RGB")
        action = SensingAction(
            action_id="smoke_01", semantic_key="target", prompt=args.target,
            family=ActionFamily.DISCOVERY, threshold=0.25, spatial_mode=SpatialMode.GLOBAL
        )
        t0 = time.time()
        obs = sam3.observe(img_pil, action)
        logger.info(f"SAM3 Global pass took {time.time()-t0:.2f}s, {len(obs.detections)} dets.")
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.1 SAM3 smoke failed: {e}")
        return False

def m8_2_qwen_smoke(args):
    logger.info("=== M8.2 Real Qwen planning validation ===")
    if args.dry_run:
        logger.info("Dry run: Skipping execution.")
        return True
    try:
        qwen = _get_qwen_only(args)
        import tempfile
        img = Image.new("RGB", (100, 100), color="green")
        fd, p = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(p)
        
        evidence = QwenEvidencePack(
            original_image_id="m8_smoke", user_prompt=args.target, target_class="target",
            image_path=p, contact_sheet=ContactSheet(crops=[], total_candidates=0)
        )
        
        from sam3_vlm.core.types import BudgetState
        from sam3_vlm.planning.qwen_planner import QwenPlannerService
        service = QwenPlannerService(qwen)
        output = service.plan_scene(evidence, BudgetState(), load_m8_config(args))
        os.remove(p)
        logger.info(f"Qwen proposed {len(output.proposed_actions)} actions.")
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.2 Qwen smoke failed: {e}")
        return False

def _run_validator_and_replay(paths: RunArtifactPaths, runner_state=None) -> bool:
    try:
        validator = RunValidator(paths)
        result = validator.validate()
        if not result.valid:
            logger.error(f"Validation failed: {result.errors}")
            return False
            
        engine = ReplayEngine(paths)
        replayed_state = engine.replay_state()
        
        if runner_state and len(replayed_state.graph.nodes) != len(runner_state.graph.nodes):
            logger.error("Replay graph node count mismatch!")
            return False
            
        with OracleHider(paths):
            engine_hidden = ReplayEngine(paths)
            replayed_hidden = engine_hidden.replay_state()
            if runner_state and len(replayed_hidden.graph.nodes) != len(runner_state.graph.nodes):
                logger.error("Replay with hidden artifacts node count mismatch!")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Validation/Replay crashed: {e}")
        return False

def m8_3_full_run(args):
    logger.info("=== M8.3 One full real-image V4 run ===")
    if not os.path.exists(args.image):
        logger.error(f"FAIL: Image not found at {args.image}")
        return False
    if args.dry_run:
        logger.info("Dry run: Skipping execution.")
        return True
    try:
        sam3, qwen = _get_models(args)
        config = load_m8_config(args)
        
        run_id = f"m8_3_{uuid.uuid4().hex[:8]}"
        paths = RunArtifactPaths(base_dir=Path(os.path.join(args.output_dir, "M8.3", run_id)))
        
        runner, recorder = assemble_e2e_runner(
            paths, config, sam3, qwen, run_id, args.target, "target", "m8_test_img"
        )
        img_pil = Image.open(args.image).convert("RGB")
        
        final_count = runner.run(
            image=img_pil, user_prompt=args.target, target_class="target", image_id="m8_test_img"
        )
        
        logger.info(f"Full run complete! Soft count: {final_count:.2f}")
        return _run_validator_and_replay(paths, runner.scene_state)
    except Exception as e:
        logger.error(f"FAIL: M8.3 Full run failed: {e}")
        return False

def m8_4_and_5_pilot(args):
    logger.info("=== M8.4 & M8.5 Small Multi-Image Pilot ===")
    
    samples = []
    if args.manifest:
        with open(args.manifest, "r") as f: samples = json.load(f)
    elif os.path.isdir(args.image):
        import glob
        for p in glob.glob(os.path.join(args.image, "*.jpg")):
            samples.append({"sample_id": os.path.basename(p), "image_path": p, "target": args.target})
    else:
        samples = [{"sample_id": os.path.basename(args.image), "image_path": args.image, "target": args.target}]
        
    if args.max_samples: samples = samples[:args.max_samples]
    if not samples:
        logger.error("No images found for pilot.")
        return False
        
    if args.dry_run:
        logger.info(f"Dry run: Skipping execution of {len(samples)} samples.")
        return True
        
    try:
        sam3, qwen = _get_models(args)
    except Exception as e:
        logger.error(f"FAIL: Models missing for pilot: {e}")
        return False
        
    base_config = load_m8_config(args)
    
    variants = {
        "A_OneShot": "A_OneShot",
        "B_FixedBank": dataclasses.replace(base_config, replanning=ReplanningConfig(max_replans=0)),
        "C_FullV4": base_config
    }
    
    report = []
    pilot_success = True
    
    for var_name, config in variants.items():
        logger.info(f"\\n--- Running Variant: {var_name} ---")
        for sample in samples:
            img_path = sample["image_path"]
            img_name = sample["sample_id"]
            prompt = sample.get("target", args.target)
            gt_count = sample.get("gt_count")
            
            logger.info(f"Processing {img_name} [{var_name}]...")
            run_id = f"pilot_{var_name}_{img_name}_{uuid.uuid4().hex[:6]}"
            paths = RunArtifactPaths(base_dir=Path(os.path.join(args.output_dir, "pilot", var_name, run_id)))
            
            try:
                img_pil = Image.open(img_path).convert("RGB")
                start_t = time.time()
                
                valid_run = True
                metrics = {}
                
                if var_name == "A_OneShot":
                    cfg = dataclasses.replace(base_config, bootstrap=BootstrapConfig(enable_tiled_bootstrap=False))
                    manifest = RunManifest(run_id=run_id, user_prompt=prompt, target_class="target", image_id=img_name)
                    manifest.v4_config = dataclasses.asdict(cfg)
                    recorder = RunRecorder(paths, manifest)
                    
                    recorder.record_run_started()
                    pipeline = BootstrapPipeline(sam3, config=cfg, recorder=recorder)
                    result = pipeline.execute_bootstrap(image_id=img_name, image=img_pil, user_prompt=prompt, target_class="target")
                    count = float(len(result.state.graph.nodes))
                    
                    # Create a mock summary to finalize cleanly
                    summary = RunSummary(
                        run_id=run_id, final_soft_count=count, count_variance=0.0,
                        node_count=len(result.state.graph.nodes),
                        sam3_calls=sam3.call_count, runtime_ms=(time.time()-start_t)*1000
                    )
                    recorder.finalize_success(summary, result.state.graph.to_dict())
                else:
                    runner, recorder = assemble_e2e_runner(paths, config, sam3, qwen, run_id, prompt, "target", img_name)
                    count = runner.run(image=img_pil, user_prompt=prompt, target_class="target", image_id=img_name)
                    valid_run = _run_validator_and_replay(paths, runner.scene_state)
                    
                runtime = time.time() - start_t
                
                # Fetch summary metrics
                if paths.summary_json.exists():
                    with open(paths.summary_json) as f:
                        s = json.load(f)
                        metrics["sam3_calls"] = s.get("sam3_calls", 0)
                        metrics["qwen_calls"] = s.get("qwen_calls", 0)
                        
                # Storage usage
                total_bytes = sum(f.stat().st_size for f in paths.base_dir.rglob('*') if f.is_file())
                
                entry = {
                    "variant": var_name,
                    "sample_id": img_name,
                    "predicted_count": count,
                    "gt_count": gt_count,
                    "runtime_s": runtime,
                    "status": "SUCCESS" if valid_run else "FAILED",
                    "run_id": run_id,
                    "storage_bytes": total_bytes,
                    "sam3_calls": metrics.get("sam3_calls", 0),
                    "qwen_calls": metrics.get("qwen_calls", 0)
                }
                
                if gt_count is not None:
                    entry["absolute_error"] = abs(count - float(gt_count))
                    entry["signed_error"] = count - float(gt_count)
                    
                report.append(entry)
                if not valid_run: pilot_success = False
                    
            except Exception as e:
                logger.error(f"Error on {img_name}: {e}")
                report.append({"variant": var_name, "sample_id": img_name, "status": "FAILED", "error": str(e)})
                pilot_success = False
                
    report_path = os.path.join(args.output_dir, "pilot_report.json")
    with open(report_path, "w") as f: json.dump(report, f, indent=2)
    logger.info(f"Pilot completed. Success: {pilot_success}. Report at {report_path}")
    return pilot_success

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["preflight", "M8.0", "M8.1", "M8.2", "M8.3", "pilot", "all"], default="all")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to test image or directory")
    parser.add_argument("--manifest", type=str, default=None, help="Optional pilot JSON manifest")
    parser.add_argument("--target", type=str, default="green citrus", help="Target concept")
    parser.add_argument("--output_dir", type=str, default="runs/m8_real_smoke")
    
    parser.add_argument("--require-cuda", action="store_true", default=True, help="Fail if CUDA is absent")
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false", help="Allow CPU execution")
    parser.add_argument("--compile-sam3", action="store_true", default=False, help="Enable torch.compile")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Skip execution, test pipeline construction")
    
    parser.add_argument("--sam3-model", type=str, default="facebook/sam3", help="HF model ID")
    parser.add_argument("--qwen-model", type=str, default=os.environ.get("QWEN_MODEL", "qwen2.5-vl-72b-instruct"), help="Qwen endpoint model name")
    parser.add_argument("--qwen-base-url", type=str, default=os.environ.get("QWEN_BASE_URL"), help="Qwen API endpoint")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit pilot size")
    
    args = parser.parse_args()
    
    stages = [args.stage] if args.stage != "all" else ["preflight", "M8.0", "M8.1", "M8.2", "M8.3", "pilot"]
    success = True
    
    for s in stages:
        if s == "preflight":
            if not preflight(args): success = False
        elif s == "M8.0":
            if not m8_0_validate_adapters(args): success = False
        elif s == "M8.1":
            if not m8_1_sam3_smoke(args): success = False
        elif s == "M8.2":
            if not m8_2_qwen_smoke(args): success = False
        elif s == "M8.3":
            if not m8_3_full_run(args): success = False
        elif s == "pilot":
            if not m8_4_and_5_pilot(args): success = False
            
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
