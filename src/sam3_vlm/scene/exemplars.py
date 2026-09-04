"""Controller-owned pseudoexemplar selection.

Pseudoexemplars are sensor-grounded hypotheses, not labels.  They are selected
deterministically from high-confidence target-bootstrap observations and may be
used immediately as positive SAM3 box prompts.  Qwen does not choose them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sam3_vlm.scene.graph import SceneGraph


BoxTuple = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PseudoexemplarSelection:
    node_ids: Tuple[str, ...] = ()
    boxes: Tuple[BoxTuple, ...] = ()
    scores: Tuple[float, ...] = ()


def select_target_pseudoexemplars(
    graph: SceneGraph,
    *,
    max_count: int,
    min_score: float,
) -> PseudoexemplarSelection:
    """Choose the strongest active target hypotheses by observed SAM3 score.

    The bootstrap threshold is intentionally recall-oriented, so forwarding all
    candidates would turn weak false positives into visual prompts.  We instead
    recover the previous architecture's exemplar propagation from only the
    strongest seed detections.  Target posterior is used only as a tie-breaker;
    it is not treated as independent ground truth.
    """
    if max_count <= 0:
        return PseudoexemplarSelection()

    candidates = []
    for node in graph.active_nodes():
        observed_scores = [
            float(obs.score)
            for obs in node.observations
            if obs.score is not None
        ]
        if not observed_scores:
            continue
        best_score = max(observed_scores)
        if best_score < float(min_score):
            continue
        target_prob = float(node.class_belief.probabilities.get("target", 0.0))
        box = node.geometry.bbox()
        candidates.append(
            (
                -best_score,
                -target_prob,
                str(node.node_id),
                tuple(float(v) for v in box.as_tuple()),
                best_score,
            )
        )

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    chosen = candidates[: int(max_count)]
    return PseudoexemplarSelection(
        node_ids=tuple(row[2] for row in chosen),
        boxes=tuple(row[3] for row in chosen),
        scores=tuple(row[4] for row in chosen),
    )


__all__ = ["PseudoexemplarSelection", "select_target_pseudoexemplars"]
