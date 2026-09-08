"""Small SAM3-only probe: prompt x threshold x exemplars, fixed bootstrap.

Association policies are evaluated offline on copies of the same detections.
This measures candidate discovery, not precision/recall or a new counting model.
"""

import argparse
import copy
import dataclasses
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode
from sam3_vlm.experiments.m8_smoke import load_m8_config, _load_pilot_samples, _get_sam3_only
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.scene.association import IoUAssociationPolicy
from sam3_vlm.scene.association_dual import IoUIoMAssociationPolicy
from sam3_vlm.scene.exemplars import select_target_pseudoexemplars
from sam3_vlm.sensing.action import SensingAction, validate_sam3_prompt_contract


def _associate_probe(state, observation, action, id_gen, config):
    results = {}
    for name, policy in (("iou_only", IoUAssociationPolicy()), ("iou_iom", IoUIoMAssociationPolicy())):
        # Association mutates graph observations, nodes and diagnostics. Each
        # policy/arm must start from the identical bootstrap graph and counters.
        graph = copy.deepcopy(state.graph)
        result = policy.associate(
            graph, copy.deepcopy(observation.detections), observation.call_id,
            action.action_id, "target", copy.deepcopy(id_gen),
            config=config.association, correlation_group="target",
        )
        results[name] = {
            "matched_detections": len(result.matched_observations),
            "new_candidates": len(result.new_nodes),
            "new_boxes": [list(n.geometry.bbox().as_tuple()) for n in result.new_nodes],
        }
    return results


class _CappedSensor:
    def __init__(self, sensor, max_calls):
        self.sensor, self.max_calls, self.calls = sensor, max_calls, 0

    def observe(self, image, action):
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Diagnostic SAM3 call cap reached ({self.max_calls})")
        self.calls += 1
        return self.sensor.observe(image, action)


def run_diagnostic(samples, sensor, config, prompts, output_dir, max_calls=100):
    if not 1 <= len(samples) <= 5:
        raise ValueError("Choose between one and five diagnostic images")
    if not 1 <= len(prompts) <= 4:
        raise ValueError("Choose between one and four diagnostic prompts")
    for prompt in prompts:
        validate_sam3_prompt_contract(prompt)
    if len({p.strip().lower() for p in prompts}) != len(prompts):
        raise ValueError("Diagnostic prompts must be distinct")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capped = _CappedSensor(sensor, max_calls)
    report = {
        "experiment": "SAM3_Discovery_Diagnostic", "config": dataclasses.asdict(config),
        "prompts": prompts, "thresholds": [0.25, 0.5], "max_sam3_calls": max_calls,
        "qwen_calls": 0, "samples": [], "complete": True,
        "notes": [
            "Each probe starts from the same bootstrap state; probes do not accumulate discoveries.",
            "Threshold and exemplar arms use separate SAM3 calls; association policies reuse identical detections.",
            "sensor_detections are after adapter postprocessing, before graph association.",
            "Extra candidates are not confirmed fruit; count-only ground truth cannot establish precision or recall.",
            "The probe uses TILED mode and the bootstrap-locked search region in every arm.",
        ],
    }
    for index, sample in enumerate(samples, 1):
        row = {"sample_id": sample["sample_id"], "target": sample["target"],
               "gt_count": sample["gt_count"], "probes": [], "success": False}
        report["samples"].append(row)
        try:
            with Image.open(sample["image_path"]) as source:
                image = source.convert("RGB")
            cfg = dataclasses.replace(config, assets_dir=str(output_dir / "assets" / str(index)))
            bootstrap = BootstrapPipeline(capped, config=cfg)
            before = capped.calls
            state = bootstrap.execute_bootstrap(sample["sample_id"], image, sample["target"]).state
            row["bootstrap_sam3_calls"] = capped.calls - before
            row["bootstrap_nodes"] = len(state.graph.active_nodes())
            row["bootstrap_target_mass"] = sum(n.class_belief.probabilities.get("target", 0) for n in state.graph.active_nodes())
            row["search_region"] = list(state.search_region.bbox().as_tuple())
            row["bootstrap_boxes"] = [list(n.geometry.bbox().as_tuple()) for n in state.graph.active_nodes()]
            seeds = select_target_pseudoexemplars(
                state.graph, max_count=config.bootstrap.pseudoexemplar_max_count,
                min_score=config.bootstrap.pseudoexemplar_min_score,
            )
            row["available_exemplars"] = len(seeds.node_ids)
            row["exemplar_boxes"] = seeds.boxes
            for prompt in prompts:
                for threshold in (0.25, 0.5):
                    for use_exemplars in (False, True):
                        probe = {"prompt": prompt, "threshold": threshold,
                                 "exemplars_requested": use_exemplars,
                                 "exemplars_used": len(seeds.node_ids) if use_exemplars else 0}
                        row["probes"].append(probe)
                        if use_exemplars and not seeds.node_ids:
                            probe["skipped"] = "No bootstrap exemplars available; this arm cannot test exemplar effects"
                            continue
                        action = SensingAction(
                            action_id=bootstrap.id_gen.next_action_id(), semantic_key="target",
                            prompt=prompt, family=ActionFamily.DISCOVERY, source=ActionSource.USER_BOOTSTRAP,
                            threshold=threshold, spatial_mode=SpatialMode.TILED, tiling=config.tiling,
                            search_region=state.search_region, semantic_prior={"target": 1.0},
                            positive_exemplar_ids=seeds.node_ids if use_exemplars else (),
                            positive_exemplar_boxes=seeds.boxes if use_exemplars else (),
                        )
                        observation = capped.observe(image, action)
                        probe.update(
                            sensor_detections=len(observation.detections),
                            runtime_ms=observation.runtime_ms,
                            detections=[{"score": d.score, "box": list(d.geometry.bbox().as_tuple())}
                                        for d in observation.detections],
                            association=_associate_probe(state, observation, action, bootstrap.id_gen, config),
                        )
            row["success"] = True
        except Exception as exc:
            row["error"] = str(exc)
            report["complete"] = False
        # Preserve numeric results if a later image or model call fails.
        report["sam3_calls"] = capped.calls
        (output_dir / "discovery_report.json").write_text(json.dumps(report, indent=2))
    with ZipFile(output_dir / "discovery_review.zip", "w", ZIP_DEFLATED) as archive:
        previews = {}
        warnings = []
        for index, sample in enumerate(samples, 1):
            try:
                with Image.open(sample["image_path"]) as source:
                    preview = source.convert("RGB")
                    preview.thumbnail((1280, 1280))
                    data = BytesIO()
                    preview.save(data, format="JPEG", quality=80)
                name = f"previews/image_{index}.jpg"
                archive.writestr(name, data.getvalue())
                previews[sample["sample_id"]] = name
            except (OSError, ValueError) as exc:
                warnings.append(f"{sample['sample_id']}: {exc}")
        report["preview_files"], report["export_warnings"] = previews, warnings
        archive.writestr("discovery_report.json", json.dumps(report, indent=2))
    (output_dir / "discovery_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One to four explicit probe phrases; not a vocabulary restriction on Qwen")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="green fruit")
    parser.add_argument("--require-cuda", action="store_true", default=None)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--sam3-model")
    parser.add_argument("--max-sam3-calls", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dep = load_m8_config(args)
    try:
        if not 1 <= len(args.sample_ids) <= 5 or len(set(args.sample_ids)) != len(args.sample_ids):
            raise ValueError("Select one to five distinct sample IDs")
        if not 1 <= len(args.prompts) <= 4:
            raise ValueError("Choose one to four prompts")
        for prompt in args.prompts:
            validate_sam3_prompt_contract(prompt)
        if len({p.strip().lower() for p in args.prompts}) != len(args.prompts):
            raise ValueError("Diagnostic prompts must be distinct")
        if args.max_sam3_calls < 1:
            raise ValueError("--max-sam3-calls must be positive")
        samples = _load_pilot_samples(args, limit=5)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"{len(samples)} images; up to {len(samples) * len(args.prompts) * 4} probes plus bootstrap; "
              f"SAM3 cap {args.max_sam3_calls}; zero Qwen calls")
        return 0
    report = run_diagnostic(samples, _get_sam3_only(args), dep.v4_config,
                            args.prompts, dep.output_root, args.max_sam3_calls)
    print(f"Review: {Path(dep.output_root) / 'discovery_review.zip'}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
