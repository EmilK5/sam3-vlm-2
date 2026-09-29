"""Run the supported single-image MVP entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters import RealSAM3, RealVLM
from .core import Config, Controller, Result


def result_dict(result: Result) -> dict:
    return {
        "counts": result.counts, "calls": result.calls, "stop_reason": result.stop_reason,
        "partial": result.partial, "errors": result.errors,
        "unexecuted_bootstrap": result.unexecuted_bootstrap,
        "unexecuted_batch": result.unexecuted_batch,
        "elapsed_seconds": result.elapsed_seconds, "model_seconds": result.model_seconds,
        "tiling": result.tiling, "roi": result.roi,
        "proposals": result.proposals, "actions": result.actions,
        "nodes": {key: {"box": node.box, "belief": node.belief,
                         "canonical_detection_id": node.canonical_detection_id,
                         "canonical_score": node.canonical_score,
                         "detection_ids": node.detection_ids, "positives": node.positives,
                         "negatives": node.negatives}
                  for key, node in result.nodes.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("target")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--masks", type=Path, help="Optional NPZ file for canonical original-image masks")
    parser.add_argument("--sam3-model", default="facebook/sam3")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    data = json.loads(args.config.read_text())
    if "bootstrap_regions" in data:
        data["bootstrap_regions"] = tuple(tuple(r) for r in data["bootstrap_regions"])
    config = Config(**data)
    image = Image.open(args.image).convert("RGB")
    sam3 = RealSAM3(model_id=args.sam3_model, device=args.device)
    vlm = RealVLM(coordinate_mode=config.vlm_coordinate_mode) if config.max_vlm_calls else None
    result = Controller(sam3, vlm, config).run(image, args.target)
    data = result_dict(result)
    data["metadata"] = {"image": str(args.image), "target": args.target,
                        "config": config.__dict__, "sam3_model": args.sam3_model,
                        "vlm_model": getattr(vlm, "model", None)}
    if args.masks:
        np.savez_compressed(args.masks, **{key: node.mask for key, node in result.nodes.items()})
        data["canonical_masks_npz"] = str(args.masks)
    output = json.dumps(data, indent=2, default=str)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
