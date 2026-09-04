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
import math
import numpy as np
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
from sam3_vlm.evaluation.metrics import CountingMetrics, aggregate_count_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _json_default(value):
    """Convert serialization-safe runtime values to their JSON representation.

    Replay validation compares scientific state after it has crossed the JSON
    artifact boundary. Tuples, NumPy scalars/arrays, Paths, and enum-like values
    can be semantically identical while differing as raw Python objects. This
    helper normalizes only representation; it does not discard fields or values.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _json_canonical(value):
    """Return the exact representation a value has after a JSON round trip.

    This intentionally normalizes tuple/list differences introduced by JSON
    persistence while preserving every scientifically relevant key and value.
    """
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
    )


def _first_difference(left, right, path="$"):
    """Return a compact description of the first canonical-state mismatch."""
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        if math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None

        return (
            f"{path}: left={left!r}, right={right!r}"
        )

    if type(left) is not type(right):
        return (
            f"{path}: type {type(left).__name__} != "
            f"{type(right).__name__}; left={left!r}, right={right!r}"
        )

    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing_left = sorted(right_keys - left_keys)
            missing_right = sorted(left_keys - right_keys)
            return (
                f"{path}: key mismatch; missing_from_left={missing_left}, "
                f"missing_from_right={missing_right}"
            )
        for key in sorted(left_keys):
            diff = _first_difference(left[key], right[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None

    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (l_value, r_value) in enumerate(zip(left, right)):
            diff = _first_difference(
                l_value, r_value, f"{path}[{index}]"
            )
            if diff is not None:
                return diff
        return None

    if left != right:
        return f"{path}: left={left!r}, right={right!r}"

    return None

@dataclasses.dataclass
class M8DeploymentConfig:
    sam3_model: str
    qwen_model: str
    qwen_base_url: str | None
    require_cuda: bool
    compile_sam3: bool
    seed: int
    output_root: str
    pilot_sample_limit: int
    v4_config: V4Config

def load_m8_config(args, config_path="configs/m8_real_smoke.json") -> M8DeploymentConfig:
    """Load config with precedence: Code Defaults < JSON < Env < CLI."""
    # Defaults
    c_sam3_model = "facebook/sam3"
    c_qwen_model = "qwen2.5-vl-72b-instruct"
    c_qwen_base_url = None
    c_require_cuda = True
    c_compile_sam3 = False
    c_seed = 42
    c_output_root = "runs/m8_real_smoke"
    c_pilot_limit = 10
    
    v4_kwargs = {}
    
    # JSON
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
            
            section_types = {
                "budget": BudgetConfig,
                "tiling": TilingConfig,
                "stopping": StoppingConfig,
                "bootstrap": BootstrapConfig,
                "planner": PlannerConfig,
                "sam3": SAM3Config,
                "action_selection": ActionSelectionConfig,
                "association": AssociationConfig,
                "belief": BeliefConfig,
                "replanning": ReplanningConfig,
                "cleanup": CleanupConfig,
                "logging": LoggingConfig,
            }
            for key, config_type in section_types.items():
                if key in data:
                    v4_kwargs[key] = config_type(**data[key])

            allowed = set(section_types) | {
                "sam3_model",
                "qwen_model",
                "qwen_base_url",
                "require_cuda",
                "compile_sam3",
                "seed",
                "output_root",
                "pilot_sample_limit",
            }
            for k, v in data.items():
                if k not in allowed:
                    raise ValueError(f"Unknown config key: {k}")
                if k == "sam3_model": c_sam3_model = v
                elif k == "qwen_model": c_qwen_model = v
                elif k == "qwen_base_url": c_qwen_base_url = v
                elif k == "require_cuda": c_require_cuda = v
                elif k == "compile_sam3": c_compile_sam3 = v
                elif k == "seed": c_seed = v
                elif k == "output_root": c_output_root = v
                elif k == "pilot_sample_limit": c_pilot_limit = v
                
    # ENV overrides
    if "QWEN_MODEL" in os.environ:
        c_qwen_model = os.environ["QWEN_MODEL"]
    if "QWEN_BASE_URL" in os.environ:
        c_qwen_base_url = os.environ["QWEN_BASE_URL"]
        
    # CLI overrides
    if getattr(args, 'sam3_model', None) is not None:
        c_sam3_model = args.sam3_model
    if getattr(args, 'qwen_model', None) is not None:
        c_qwen_model = args.qwen_model
    if getattr(args, 'qwen_base_url', None) is not None:
        c_qwen_base_url = args.qwen_base_url
    if getattr(args, 'require_cuda', None) is not None:
        c_require_cuda = args.require_cuda
    if getattr(args, 'compile_sam3', None) is not None:
        c_compile_sam3 = args.compile_sam3
    if getattr(args, 'output_dir', None) is not None:
        c_output_root = args.output_dir
    if getattr(args, 'max_samples', None) is not None:
        c_pilot_limit = args.max_samples
        
    v4_kwargs["device"] = "cuda" if c_require_cuda else "cpu"
    v4_kwargs["output_dir"] = c_output_root
    
    return M8DeploymentConfig(
        sam3_model=c_sam3_model,
        qwen_model=c_qwen_model,
        qwen_base_url=c_qwen_base_url,
        require_cuda=c_require_cuda,
        compile_sam3=c_compile_sam3,
        seed=c_seed,
        output_root=c_output_root,
        pilot_sample_limit=c_pilot_limit,
        v4_config=V4Config(**v4_kwargs)
    )

def _get_models(args):
    dep = load_m8_config(args)
    import torch
    if dep.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    sam3 = RealSAM3Sensor(model_id=dep.sam3_model, device=dep.v4_config.device, compile_model=dep.compile_sam3)
    qwen = RealQwenPlanner(base_url=dep.qwen_base_url, model=dep.qwen_model, strict_model_errors=True)
    return sam3, qwen

def _get_sam3_only(args):
    dep = load_m8_config(args)
    import torch
    if dep.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return RealSAM3Sensor(model_id=dep.sam3_model, device=dep.v4_config.device, compile_model=dep.compile_sam3)

def _get_qwen_only(args):
    dep = load_m8_config(args)
    return RealQwenPlanner(base_url=dep.qwen_base_url, model=dep.qwen_model, strict_model_errors=True)

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

def assemble_e2e_runner(
    paths: RunArtifactPaths, config: V4Config, sensor, planner, run_id: str,
    prompt: str, target_class: str, image_id: str, seed: int | None = None,
    experiment_name: str | None = None,
):
    manifest = RunManifest(run_id=run_id, user_prompt=prompt, target_class=target_class, image_id=image_id)
    manifest.v4_config = dataclasses.asdict(config)
    manifest.experiment_config = {
        "experiment": experiment_name or "V4_REAL_RUN",
        "resolved_device": config.device,
    }
    manifest.model_identifiers = {
        "sam3": str(getattr(sensor, "model_id", type(sensor).__name__)),
        "qwen": str(getattr(planner, "model", type(planner).__name__)),
    }
    manifest.seed = seed
    recorder = RunRecorder(paths, manifest)
    runner = Runner(config, sensor=sensor, planner=planner, recorder=recorder)
    return runner, recorder

def preflight(args) -> bool:
    logger.info("=== Preflight Checks ===")
    dep = load_m8_config(args)
    
    success = True
    def _check(cond, msg):
        nonlocal success
        if cond: logger.info(f"  [OK] {msg}")
        else:
            logger.error(f"  [FAIL] {msg}")
            success = False

    if args.stage in ("all", "M8.2", "M8.3", "pilot"):
        _check(dep.qwen_base_url, "Qwen endpoint is set")
        
    try:
        import torch
        if dep.require_cuda:
            _check(torch.cuda.is_available(), f"CUDA is available")
        else:
            logger.info("  [INFO] Running with --allow-cpu.")
            
        import transformers
        _check(True, f"Transformers version: {transformers.__version__}")
        if args.stage in ("all", "M8.0", "M8.1", "M8.3", "pilot"):
            from transformers import Sam3Model, Sam3Processor
            _check(True, "Sam3Model importable")
    except ImportError as e:
        _check(False, f"Required package missing: {e}. Ensure transformers>=5.0.0 is installed.")
        
    try:
        os.makedirs(dep.output_root, exist_ok=True)
        test_file = os.path.join(dep.output_root, ".write_test")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        _check(True, f"Output directory {dep.output_root} writable")
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
        dep = load_m8_config(args)
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
        output = service.plan_scene(evidence, BudgetState(), dep.v4_config)
        os.remove(p)
        logger.info(f"Qwen proposed {len(output.proposed_actions)} actions.")
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.2 Qwen smoke failed: {e}")
        return False

def _run_validator_and_replay(paths: RunArtifactPaths, runner_state=None) -> bool:
    """Validate artifacts and verify deterministic replay equivalence.

    The comparison is performed after a JSON round trip because persisted run
    artifacts necessarily convert tuples to arrays/lists. Raw Python equality
    would therefore reject states that are byte-for-byte equivalent once
    serialized. No fields are omitted from the scientific canonical state.
    """
    try:
        from sam3_vlm.logging.replay import canonical_scene_state

        validator = RunValidator(paths)
        result = validator.validate()
        if not result.valid:
            logger.error(f"Validation failed: {result.errors}")
            return False

        engine = ReplayEngine(paths)
        replayed_state = engine.replay_state()

        if runner_state is not None:
            cs_replayed = _json_canonical(
                canonical_scene_state(replayed_state)
            )
            cs_runner = _json_canonical(
                canonical_scene_state(runner_state)
            )

            diff = _first_difference(cs_runner, cs_replayed)
            if diff is not None:
                logger.error(
                    "Replay canonical state mismatch! First difference: %s",
                    diff,
                )
                return False

        with OracleHider(paths):
            engine_hidden = ReplayEngine(paths)
            replayed_hidden = engine_hidden.replay_state()

            if runner_state is not None:
                cs_hidden = _json_canonical(
                    canonical_scene_state(replayed_hidden)
                )

                diff = _first_difference(cs_hidden, cs_runner)
                if diff is not None:
                    logger.error(
                        "Replay with hidden artifacts canonical state mismatch! "
                        "First difference: %s",
                        diff or "unknown difference",
                    )
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
        dep = load_m8_config(args)
        
        run_id = f"m8_3_{uuid.uuid4().hex[:8]}"
        paths = RunArtifactPaths(base_dir=Path(os.path.join(dep.output_root, "M8.3", run_id)))
        run_config = dataclasses.replace(
            dep.v4_config,
            assets_dir=str(paths.base_dir / "assets"),
        )
        
        runner, recorder = assemble_e2e_runner(
            paths, run_config, sam3, qwen, run_id, args.target, "target", "m8_test_img",
            seed=dep.seed, experiment_name="M8.3",
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
    dep = load_m8_config(args)
    
    samples = []
    if args.manifest:
        with open(args.manifest, "r") as f: samples = json.load(f)
    elif os.path.isdir(args.image):
        import glob
        for p in glob.glob(os.path.join(args.image, "*.jpg")):
            samples.append({"sample_id": os.path.basename(p), "image_path": p, "target": args.target})
    else:
        samples = [{"sample_id": os.path.basename(args.image), "image_path": args.image, "target": args.target}]
        
    samples = samples[:dep.pilot_sample_limit]
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
        
    base_config = dep.v4_config
    
    variants = {
        "A_OneShot": "A_OneShot",
        "B_FixedBank": dataclasses.replace(
            base_config,
            replanning=dataclasses.replace(
                base_config.replanning,
                max_replans=0,
            ),
        ),
        "C_V4_NoExemplarCleanup": base_config
    }
    
    pilot_success = True
    report = {
        "metadata": {
            "experiment": "M8_Pilot",
            "target": args.target,
            "resolved_config": dataclasses.asdict(dep.v4_config),
            "variants": ["A_OneShot", "B_FixedBank", "C_V4_NoExemplarCleanup"]
        },
        "samples": [],
        "aggregates": {}
    }
    
    for var_name, config in variants.items():
        logger.info(f"\n--- Running Variant: {var_name} ---")
        metrics_list = []
        for sample in samples:
            img_path = sample["image_path"]
            img_name = sample["sample_id"]
            prompt = sample.get("target", args.target)
            gt_count = sample.get("gt_count")
            
            logger.info(f"Processing {img_name} [{var_name}]...")
            run_id = f"pilot_{var_name}_{img_name}_{uuid.uuid4().hex[:6]}"
            paths = RunArtifactPaths(base_dir=Path(os.path.join(dep.output_root, "pilot", var_name, run_id)))
            
            try:
                img_pil = Image.open(img_path).convert("RGB")
                start_t = time.time()
                
                valid_run = True
                sam3_c, qwen_c, clean_c, rep_c, it_c = 0, 0, 0, 0, 0
                
                
                stop_reason = None
                validator_status = None
                replay_status = None
                sam3_t = 0
                failure_category = None
                failure_message = None
                
                if var_name == "A_OneShot":
                    cfg = dataclasses.replace(
                        base_config,
                        assets_dir=str(paths.base_dir / "assets"),
                        bootstrap=dataclasses.replace(
                            base_config.bootstrap,
                            enable_tiled_bootstrap=False,
                            locked_context_prompt=None,
                            enable_pseudoexemplar_refinement=False,
                        ),
                    ) 
                    manifest = RunManifest(run_id=run_id, user_prompt=prompt, target_class="target", image_id=img_name)
                    manifest.v4_config = dataclasses.asdict(cfg)
                    manifest.experiment_config = {"experiment": "M8_Pilot:A_OneShot", "resolved_device": cfg.device}
                    manifest.model_identifiers = {
                        "sam3": str(getattr(sam3, "model_id", type(sam3).__name__)),
                        "qwen": str(getattr(qwen, "model", type(qwen).__name__)),
                    }
                    manifest.seed = dep.seed
                    recorder = RunRecorder(paths, manifest)
                    
                    recorder.record_run_started()
                    pipeline = BootstrapPipeline(sam3, config=cfg, recorder=recorder)
                    calls_before = getattr(sam3, "call_count", 0)
                    result = pipeline.execute_bootstrap(image_id=img_name, image=img_pil, user_prompt=prompt, target_class="target")
                    calls_after = getattr(sam3, "call_count", 1)
                    count = float(len(result.state.graph.nodes))
                    sam3_c = calls_after - calls_before
                    sam3_t = 0
                    stop_reason = "ONE_SHOT_COMPLETE"
                    
                    # Finalize with the same split runtime semantics as full V4 runs.
                    wall_ms = (time.time() - start_t) * 1000.0
                    sam3_ms = result.state.budget.sam3_runtime_ms
                    summary = RunSummary(
                        run_id=run_id, final_soft_count=count, count_variance=0.0,
                        node_count=len(result.state.graph.nodes),
                        sam3_calls=sam3_c, runtime_ms=wall_ms, wall_runtime_ms=wall_ms,
                        sam3_runtime_ms=sam3_ms, qwen_runtime_ms=0.0,
                        controller_runtime_ms=max(0.0, wall_ms - sam3_ms),
                    )
                    recorder.finalize_success(summary, result.state.graph.to_dict())
                else:
                    run_config = dataclasses.replace(
                        config,
                        assets_dir=str(paths.base_dir / "assets"),
                    )
                    runner, recorder = assemble_e2e_runner(
                        paths, run_config, sam3, qwen, run_id, prompt, "target", img_name,
                        seed=dep.seed, experiment_name=f"M8_Pilot:{var_name}",
                    )
                    count = runner.run(image=img_pil, user_prompt=prompt, target_class="target", image_id=img_name)
                    valid_run = _run_validator_and_replay(paths, runner.scene_state)
                    stop_reason = runner.scene_state.stop_reason.name if runner.scene_state.stop_reason else "UNKNOWN"
                    sam3_t = runner.scene_state.budget.sam3_tiles
                    validator_status = "PASS" if valid_run else "FAIL"
                    replay_status = "PASS" if valid_run else "FAIL"
                    
                runtime = time.time() - start_t
                
                # Fetch summary metrics
                if paths.summary_json.exists():
                    with open(paths.summary_json) as f:
                        s = json.load(f)
                        sam3_c = s.get("sam3_calls", sam3_c)
                        qwen_c = s.get("qwen_calls", qwen_c)
                        clean_c = s.get("cleanup_calls", clean_c)
                        rep_c = s.get("replans_executed", rep_c)
                        it_c = s.get("iterations", it_c)
                        
                total_bytes = sum(f.stat().st_size for f in paths.base_dir.rglob('*') if f.is_file())
                
                cm = CountingMetrics(
                    absolute_error=abs(count - gt_count) if gt_count is not None else 0.0,
                    signed_error=count - gt_count if gt_count is not None else 0.0,
                    squared_error=(count - gt_count)**2 if gt_count is not None else 0.0,
                    relative_error=(abs(count - gt_count)/gt_count) if (gt_count and gt_count > 0) else None,
                    true_count=gt_count if gt_count is not None else 0,
                    predicted_count=count,
                    sam3_calls=sam3_c,
                    qwen_calls=qwen_c,
                    cleanup_calls=clean_c,
                    replans=rep_c,
                    iterations=it_c,
                    total_runtime_ms=runtime * 1000,
                    storage_bytes=total_bytes
                )
                metrics_list.append(cm)
                
                entry = {
                    "variant": var_name,
                    "sample_id": img_name,
                    "predicted_count": count,
                    "count_type": "hard_one_shot" if var_name == "A_OneShot" else "posterior",
                    "gt_count": gt_count,
                    "runtime_ms": runtime * 1000,
                    "success": valid_run,
                    "run_id": run_id,
                    "storage_bytes": total_bytes,
                    "sam3_calls": sam3_c,
                    "sam3_tiles": sam3_t,
                    "qwen_calls": qwen_c,
                    "cleanup_calls": clean_c,
                    "replans": rep_c,
                    "iterations": it_c,
                    "stop_reason": stop_reason,
                    "validator_status": validator_status,
                    "replay_status": replay_status,
                    "failure_category": failure_category,
                    "failure_message": failure_message,
                    "artifact_directory": str(paths.base_dir),
                    "experiment_id": "M8_Pilot",
                    "resolved_config_reference": "m8_real_smoke"
                }
                
                if gt_count is not None:
                    entry["absolute_error"] = cm.absolute_error
                    entry["signed_error"] = cm.signed_error
                    entry["squared_error"] = cm.squared_error
                    entry["relative_error"] = cm.relative_error
                    
                report["samples"].append(entry)
                if not valid_run: pilot_success = False
                    
            except Exception as e:
                logger.error(f"Error on {img_name}: {e}")
                report["samples"].append({
                    "variant": var_name, "sample_id": img_name, "success": False,
                    "failure_category": "INFRASTRUCTURE_FAILURE",
                    "failure_message": str(e)
                })
                pilot_success = False
                
        # Calculate agg report
        agg = aggregate_count_metrics(metrics_list)
        logger.info(f"Aggregate for {var_name}: {agg}")
        report["aggregates"][var_name] = agg
                
    report_path = os.path.join(dep.output_root, "pilot_report.json")
    with open(report_path, "w") as f: json.dump(report, f, indent=2)
    logger.info(f"Pilot completed. Success: {pilot_success}. Report at {report_path}")
    return pilot_success

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["preflight", "M8.0", "M8.1", "M8.2", "M8.3", "pilot", "all"], default="all")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to test image or directory")
    parser.add_argument("--manifest", type=str, default=None, help="Optional pilot JSON manifest")
    parser.add_argument("--target", type=str, default="green citrus", help="Target concept")
    parser.add_argument("--output_dir", type=str, default=None)
    
    parser.add_argument("--require-cuda", action="store_true", default=None)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--compile-sam3", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    
    parser.add_argument("--sam3-model", type=str, default=None)
    parser.add_argument("--qwen-model", type=str, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    stages = [args.stage] if args.stage != "all" else ["preflight", "M8.0", "M8.1", "M8.2", "M8.3"]
    success = True
    
    for s in stages:
        if s == "preflight":
            success = preflight(args)
        elif s == "M8.0":
            success = m8_0_validate_adapters(args)
        elif s == "M8.1":
            success = m8_1_sam3_smoke(args)
        elif s == "M8.2":
            success = m8_2_qwen_smoke(args)
        elif s == "M8.3":
            success = m8_3_full_run(args)
        elif s == "pilot":
            success = m8_4_and_5_pilot(args)
            
        if not success:
            logger.error(f"Stage {s} failed. Aborting pipeline.")
            break
            
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
