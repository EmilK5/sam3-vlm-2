"""Single-image execution, lazy model reuse, and downloadable run artifacts."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from threading import Lock
import time
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageOps

from sam3_vlm.core.types import SpatialMode
from sam3_vlm.evaluation.metrics import compute_count_metrics
from sam3_vlm.experiments.final_outputs import _write_csv, render_candidate_boxes
from sam3_vlm.experiments.m8_smoke import (
    M8DeploymentConfig, _run_sam3_baseline, assemble_e2e_runner,
)
from sam3_vlm.gui.settings import Control
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.models.sam3 import RealSAM3Sensor
from sam3_vlm.scene.state import CountEstimator


class RunModels:
    """Per-run guards also cover bootstrap, which has no internal budget gate.

Limits are checked between model calls; an in-flight inference is not preempted.
Exhaustion outside the runner's normal stopping path is an explicit failed run.
The wrapped model weights/clients can be shared, but these counters cannot.
"""

    strict_model_errors = True

    def __init__(self, sensor, planner, config, clock=time.perf_counter):
        self.sensor, self.planner, self.config = sensor, planner, config
        self.clock, self.start = clock, clock()
        self.sam3_calls = self.qwen_calls = self.tiles = 0
        self.model_id = getattr(sensor, "model_id", type(sensor).__name__)
        self.model = getattr(planner, "model", type(planner).__name__)

    @property
    def last_request_text(self):
        return getattr(self.planner, "last_request_text", None)

    def _check_time(self):
        limit = self.config.budget.max_runtime_seconds
        if limit is not None and self.clock() - self.start >= limit:
            raise RuntimeError("Runtime budget reached before a model call; partial artifacts were retained.")

    def observe(self, image, action):
        self._check_time()
        tiling = action.tiling or self.config.tiling
        tiles = tiling.grid_rows * tiling.grid_cols if action.spatial_mode == SpatialMode.TILED else 0
        budget = self.config.budget
        if self.sam3_calls >= budget.max_sam3_calls:
            raise RuntimeError("SAM3 budget reached before a model call; increase it or disable bootstrap stages.")
        if budget.max_sam3_tiles is not None and self.tiles + tiles > budget.max_sam3_tiles:
            raise RuntimeError("Tile budget reached before a model call; increase it or reduce tiling.")
        self.sam3_calls += 1
        self.tiles += tiles
        return self.sensor.observe(image, action)

    def plan_scene(self, evidence, budget, config):
        self._check_time()
        if self.qwen_calls >= self.config.budget.max_qwen_calls:
            raise RuntimeError("Qwen call budget reached.")
        self.qwen_calls += 1
        return self.planner.plan_scene(evidence, budget, config)


class SingleImageService:
    """One shared model pair, a fresh Runner/state/directory for every submission."""

    def __init__(self, deployment: M8DeploymentConfig, sensor_factory=None, planner_factory=None):
        self.deployment = deployment
        self.sensor_factory = sensor_factory or self._load_sensor
        self.planner_factory = planner_factory or self._load_planner
        self.sensor = self.planner = None
        self.lock = Lock()

    def _load_sensor(self):
        import torch
        dep = self.deployment
        if dep.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Start the app in your GPU environment.")
        return RealSAM3Sensor(model_id=dep.sam3_model, device=dep.v4_config.device,
                              compile_model=dep.compile_sam3)

    def _load_planner(self):
        dep = self.deployment
        return RealQwenPlanner(base_url=dep.qwen_base_url, model=dep.qwen_model,
                               strict_model_errors=True)

    def run(self, image, prompt, config, enable_qwen=True, gt_count=None, seed=42):
        if not isinstance(image, Image.Image):
            raise ValueError("Upload an image before running.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter the target description, for example green fruit.")
        gt_count = Control("gt", "Ground-truth count", "int", nullable=True).parse(gt_count)
        seed = Control("seed", "Random seed", "int", 0, 2**32 - 1).parse(seed)
        if enable_qwen != (config.budget.max_qwen_calls > 0):
            raise ValueError("Qwen switch and effective call budget disagree.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Serial execution also protects Torch's global seed and mutable adapters.
        with self.lock:
            run_id = "gui_" + uuid4().hex
            root = Path(self.deployment.output_root).resolve() / run_id
            root.mkdir(parents=True, exist_ok=False)
            config = replace(config, output_dir=str(root), assets_dir=str(root / "assets"))
            request = {"user_prompt": prompt.strip(), "enable_qwen": enable_qwen,
                       "gt_count": gt_count, "seed": seed, "v4_config": asdict(config),
                       "models": {"sam3": self.deployment.sam3_model, "qwen": self.deployment.qwen_model}}
            (root / "settings.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            image.save(root / "input.png")
            try:
                import torch
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if self.sensor is None:
                    self.sensor = self.sensor_factory()
                if enable_qwen and self.planner is None:
                    self.planner = self.planner_factory()
                models = RunModels(self.sensor, self.planner if enable_qwen else None, config)
                paths = RunArtifactPaths(root)
                args = dict(paths=paths, config=config, sensor=models, run_id=run_id,
                            prompt=prompt.strip(), image_id=run_id, seed=seed, experiment_name="V4_GUI")
                if enable_qwen:
                    runner, _ = assemble_e2e_runner(**args, planner=models, target_class="target")
                    count = runner.run(image, prompt.strip(), target_class="target", image_id=run_id)
                    state = runner.scene_state
                    count_type = ("hard_posterior_count" if config.belief.target_count_hard_threshold is not None
                                  else "committed_soft_count" if config.belief.target_count_commit_threshold is not None
                                  else "soft_posterior_count")
                else:
                    count, state = _run_sam3_baseline(**args, image=image)
                    count_type = "hard_candidate_count"
                posterior = CountEstimator.estimate(state.graph, "target")
                row = {"run_id": run_id, "success": True, "target": prompt.strip(),
                       "predicted_count": count, "count_type": count_type, "gt_count": gt_count,
                       "candidate_count": len(state.graph.active_nodes()),
                       "raw_soft_count": posterior.raw_soft_count, "count_variance": posterior.variance,
                       "qwen_calls": state.budget.qwen_calls, "sam3_calls": state.budget.sam3_calls,
                       "sam3_tiles": state.budget.sam3_tiles, "replans": state.replans_executed,
                       "runtime_seconds": state.budget.wall_runtime_ms / 1000,
                       "stop_reason": state.stop_reason.value if state.stop_reason else "SAM3_BASELINE_COMPLETE"}
                if gt_count is not None:
                    row.update(compute_count_metrics(count, gt_count))
                row.update(render_candidate_boxes(image, state.graph, root / "bboxes.png"))
                (root / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                _write_csv(root / "result.csv", [row], list(row))
                archive = root / "run_bundle.zip"
                with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                    for path in sorted(root.rglob("*")):
                        if path.is_file() and path != archive:
                            bundle.write(path, path.relative_to(root))
                return row, request, str(root / "bboxes.png"), [str(root / "result.csv"), str(archive)]
            except Exception as exc:
                (root / "failure.json").write_text(json.dumps({"success": False, "error": str(exc)}, indent=2), encoding="utf-8")
                raise RuntimeError(f"{exc}\nRun files: {root}") from exc
