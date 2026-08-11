"""M8 Real-Model Validation and Experimental Smoke Testing (V4)."""

import argparse
import os
import json
import time
import logging
from PIL import Image

from sam3_vlm.core.config import V4Config, BudgetConfig, StoppingConfig, ReplanningConfig
from sam3_vlm.experiments.config import ExperimentConfig
from sam3_vlm.models.sam3 import RealSAM3Sensor
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def m8_0_validate_adapters(args):
    """M8.0 Real adapter validation."""
    logger.info("=== M8.0 Real adapter validation ===")
    try:
        qwen = RealQwenPlanner()
        logger.info("RealQwenPlanner loaded successfully.")
    except Exception as e:
        logger.error(f"FAIL: Could not load RealQwenPlanner: {e}")
        return False

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3 = RealSAM3Sensor(device=device)
        logger.info(f"RealSAM3Sensor loaded successfully on {device}.")
    except Exception as e:
        logger.error(f"FAIL: Could not load RealSAM3Sensor: {e}")
        return False

    return True

def m8_1_sam3_smoke(args):
    """M8.1 Single-pass real SAM3 validation."""
    logger.info("=== M8.1 Single-pass real SAM3 validation ===")
    if not os.path.exists(args.image):
        logger.error(f"FAIL: Image not found at {args.image}")
        return False
        
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3 = RealSAM3Sensor(device=device)
        
        img_pil = Image.open(args.image).convert("RGB")
        action = SensingAction(
            action_id="smoke_01",
            semantic_key="target",
            prompt=args.target,
            family=ActionFamily.DISCOVERY,
            threshold=0.25,
            spatial_mode=SpatialMode.GLOBAL,
        )
        
        start_time = time.time()
        obs = sam3.observe(img_pil, action)
        elapsed = time.time() - start_time
        
        logger.info(f"SAM3 Global pass took {elapsed:.2f}s")
        logger.info(f"Returned {len(obs.detections)} detections.")
        
        for i, det in enumerate(obs.detections[:3]):
            mask_status = "present" if "mask" in det.raw_metadata else "missing"
            logger.info(f"  Det {i}: score={det.score:.2f}, mask={mask_status}")
            
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.1 SAM3 smoke failed: {e}")
        return False

def m8_2_qwen_smoke(args):
    """M8.2 Real Qwen planning validation."""
    logger.info("=== M8.2 Real Qwen planning validation ===")
    try:
        qwen = RealQwenPlanner()
        
        evidence = QwenEvidencePack(
            original_image_id="m8_smoke",
            user_prompt=args.target,
            target_class="target",
            contact_sheet=ContactSheet(crops=[], total_candidates=0)
        )
        
        from sam3_vlm.core.config import V4Config
        from sam3_vlm.core.types import BudgetState
        from sam3_vlm.planning.qwen_planner import QwenPlannerService
        
        service = QwenPlannerService(qwen)
        output = service.plan_scene(evidence, BudgetState(), V4Config())
        
        logger.info(f"Qwen proposed {len(output.proposed_actions)} actions.")
        for a in output.proposed_actions:
            logger.info(f"  [{a.family.name}] {a.prompt} (mode={a.suggested_spatial_mode.name}, p={a.priority})")
            
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.2 Qwen smoke failed: {e}")
        return False

def m8_3_full_run(args):
    """M8.3 One full real-image V4 run."""
    logger.info("=== M8.3 One full real-image V4 run ===")
    if not os.path.exists(args.image):
        logger.error(f"FAIL: Image not found at {args.image}")
        return False
        
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam3 = RealSAM3Sensor(device=device)
        qwen = RealQwenPlanner()
        
        config = V4Config(
            output_dir=args.output_dir,
            assets_dir=os.path.join(args.output_dir, "assets"),
        )
        
        runner = Runner(config, sensor=sam3, planner=qwen)
        
        img_pil = Image.open(args.image).convert("RGB")
        result = runner.run(
            image=img_pil,
            user_prompt=args.target,
            target_class="target",
            image_id="m8_test_img"
        )
        
        logger.info(f"Full run complete! Soft count: {result.soft_count:.2f}")
        logger.info(f"Check {args.output_dir} for artifacts.")
        
        from sam3_vlm.logging.validator import RunValidator
        validator = RunValidator(args.output_dir)
        valid, errors = validator.validate()
        if not valid:
            logger.error(f"Validation failed: {errors}")
            return False
            
        logger.info("RunValidator passed!")
        return True
    except Exception as e:
        logger.error(f"FAIL: M8.3 Full run failed: {e}")
        return False


def m8_4_and_5_pilot(args):
    """M8.4 Small multi-image pilot and M8.5 Baseline comparison."""
    logger.info("=== M8.4 & M8.5 Small Multi-Image Pilot ===")
    
    image_paths = []
    if os.path.isdir(args.image):
        import glob
        image_paths = glob.glob(os.path.join(args.image, "*.jpg"))[:5]
    else:
        image_paths = [args.image]
        
    if not image_paths:
        logger.error("No images found for pilot.")
        return False
        
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3 = RealSAM3Sensor(device=device)
        qwen = RealQwenPlanner()
    except Exception as e:
        logger.error(f"FAIL: Models missing for pilot: {e}")
        return False
        
    variants = {
        "A_OneShot": V4Config(
            output_dir=os.path.join(args.output_dir, "pilot", "A_OneShot"),
            budget=BudgetConfig(max_qwen_calls=0),
            stopping=StoppingConfig(max_iterations=0)
        ),
        "B_FixedBank": V4Config(
            output_dir=os.path.join(args.output_dir, "pilot", "B_FixedBank"),
            replanning=ReplanningConfig(max_replans=0)
        ),
        "C_FullV4": V4Config(
            output_dir=os.path.join(args.output_dir, "pilot", "C_FullV4")
        )
    }
    
    results = []
    for var_name, config in variants.items():
        logger.info(f"\n--- Running Variant: {var_name} ---")
        var_count = 0
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            logger.info(f"Processing {img_name}...")
            runner = Runner(config, sensor=sam3, planner=qwen)
            img_pil = Image.open(img_path).convert("RGB")
            
            try:
                count = runner.run(
                    image=img_pil,
                    user_prompt=args.target,
                    target_class="target",
                    image_id=img_name
                )
                var_count += count
                logger.info(f"  Result for {img_name}: {count:.2f}")
            except Exception as e:
                logger.error(f"  Error on {img_name}: {e}")
                
        results.append({"variant": var_name, "total_count": var_count})
        
    logger.info("\n=== Pilot Results ===")
    for r in results:
        logger.info(f"Variant {r['variant']}: Total Soft Count = {r['total_count']:.2f}")
        
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["M8.0", "M8.1", "M8.2", "M8.3", "pilot", "all"], default="all")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to test image or directory of images")
    parser.add_argument("--target", type=str, default="green citrus", help="Target concept")
    parser.add_argument("--output_dir", type=str, default="runs/m8_real_smoke")
    
    args = parser.parse_args()
    
    stages = [args.stage] if args.stage != "all" else ["M8.0", "M8.1", "M8.2", "M8.3", "pilot"]
    
    for s in stages:
        if s == "M8.0":
            if not m8_0_validate_adapters(args): break
        elif s == "M8.1":
            if not m8_1_sam3_smoke(args): break
        elif s == "M8.2":
            if not m8_2_qwen_smoke(args): break
        elif s == "M8.3":
            if not m8_3_full_run(args): break
        elif s == "pilot":
            if not m8_4_and_5_pilot(args): break
