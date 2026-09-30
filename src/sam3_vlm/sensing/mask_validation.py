"""Validate mask-only observations and log zero-area sensor proposals."""

from sam3_vlm.core.geometry import EmptyMaskError, detection_mask_geometry


def prepare_mask_observation(observation, *, mask_only):
    if not mask_only:
        return observation
    kept, skipped = [], []
    for detection in observation.detections:
        try:
            detection_mask_geometry(detection)
        except EmptyMaskError:
            skipped.append({"detection_id": detection.detection_id, "score": float(detection.score),
                            "source_tile_id": detection.source_tile_id})
        else:
            kept.append(detection)
    previous = observation.model_metadata.get("mask_validation", {})
    skipped_count = previous.get("empty_masks_skipped", 0) + len(skipped)
    observation.detections = kept
    observation.model_metadata = {**observation.model_metadata, "mask_validation": {
        "raw_detections": len(kept) + skipped_count, "accepted_detections": len(kept),
        "empty_masks_skipped": skipped_count,
        "empty_mask_examples": (previous.get("empty_mask_examples", []) + skipped)[:20],
    }}
    return observation
