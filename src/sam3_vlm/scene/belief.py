"""Semantic memory and Bayesian belief updating primitives (V4 Design Spec §3.5 / §11)."""

from dataclasses import dataclass, field
import math
from typing import Dict, List
from sam3_vlm.core.config import BeliefConfig
from sam3_vlm.core.types import (
    ActionFamily,
    ClassBelief,
    NodeObservationRef,
    ObservationRelation,
)
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction


@dataclass
class SemanticRecord:
    """Tracking structure for attempted semantic experiments (V4 Design Spec §3.5)."""

    semantic_key: str
    prompts: List[str] = field(default_factory=list)
    family: ActionFamily = ActionFamily.DISCOVERY
    execution_count: int = 0
    sam3_call_ids: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    new_nodes_by_execution: List[int] = field(default_factory=list)
    realized_utility_by_execution: List[float] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """Persistent history of semantic keys and queries tested."""

    records: Dict[str, SemanticRecord] = field(default_factory=dict)

    def record_execution(self, action: SensingAction, sam3_call_id: str) -> SemanticRecord:
        if action.semantic_key not in self.records:
            self.records[action.semantic_key] = SemanticRecord(
                semantic_key=action.semantic_key,
                family=action.family,
            )
        rec = self.records[action.semantic_key]
        if action.prompt not in rec.prompts:
            rec.prompts.append(action.prompt)
        rec.execution_count += 1
        rec.sam3_call_ids.append(sam3_call_id)
        return rec


class BeliefUpdater:
    """Class belief distribution updater enforcing presence/absence asymmetry (V4 Design Spec §11)."""

    @staticmethod
    def calculate_entropy(probabilities: Dict[str, float]) -> float:
        """Compute Shannon entropy H(P) = -sum p_i log2(p_i)."""
        h = 0.0
        for p in probabilities.values():
            if p > 0.0:
                h -= p * math.log2(p)
        return max(0.0, h)

    def update_node_belief(
        self,
        node: Node,
        action: SensingAction,
        obs_ref: NodeObservationRef,
        target_class: str = "target",
        confounder_class: str = "leaf",
        background_class: str = "background",
        event_id: str | None = None,
        config: BeliefConfig = BeliefConfig(),
    ) -> None:
        """Update node class belief distribution based on observation relation and action family."""
        # Initialize default probabilities if empty
        if not node.class_belief.probabilities:
            probs = {target_class: 0.5, confounder_class: 0.4, background_class: 0.1}
        else:
            probs = dict(node.class_belief.probabilities)

        score = obs_ref.score if obs_ref.score is not None else 0.5
        relation = obs_ref.relation

        # Discount repeat weight if same semantic key was already used
        same_key_count = sum(
            1 for o in node.observations if o.semantic_key == action.semantic_key
        )
        weight = (config.discount_repeat_weight ** max(0, same_key_count - 1))

        # Presence / absence asymmetric likelihood update
        if relation in (ObservationRelation.STRONG_MATCH, ObservationRelation.NEW_DETECTION):
            if action.family == ActionFamily.DISCOVERY or action.semantic_key == target_class:
                # Strong presence under target prompt -> increase target probability
                likelihoods = {target_class: 2.5 * score * weight, confounder_class: 0.8, background_class: 0.3}
            elif action.family == ActionFamily.CONFOUNDER:
                # Strong presence under confounder prompt -> increase confounder probability & reduce target probability
                likelihoods = {target_class: 0.25, confounder_class: 4.0 * score * weight, background_class: 0.5}
            else:
                likelihoods = {target_class: 1.2, confounder_class: 1.0, background_class: 0.8}
        elif relation == ObservationRelation.WEAK_MATCH:
            if action.family == ActionFamily.CONFOUNDER:
                likelihoods = {target_class: 0.8, confounder_class: 1.5 * score * weight, background_class: 0.8}
            else:
                likelihoods = {target_class: 1.3 * score * weight, confounder_class: 1.0, background_class: 0.8}
        elif relation == ObservationRelation.NOT_RETRIEVED:
            # Presence/absence asymmetry: NOT_RETRIEVED is weak evidence (imperfect SAM3 recall)
            if action.family == ActionFamily.DISCOVERY:
                likelihoods = {target_class: 0.85, confounder_class: 1.1, background_class: 1.1}
            else:
                likelihoods = {target_class: 1.0, confounder_class: 1.0, background_class: 1.0}
        else:
            likelihoods = {target_class: 1.0, confounder_class: 1.0, background_class: 1.0}

        # Apply likelihood update and normalize
        unnormalized = {}
        for cls_name, p in probs.items():
            l_val = likelihoods.get(cls_name, 1.0)
            unnormalized[cls_name] = p * l_val

        total = sum(unnormalized.values())
        if total > 0:
            updated_probs = {k: v / total for k, v in unnormalized.items()}
        else:
            updated_probs = probs

        entropy = self.calculate_entropy(updated_probs)

        node.class_belief = ClassBelief(
            probabilities=updated_probs,
            update_count=node.class_belief.update_count + 1,
            entropy=entropy,
            last_update_event_id=event_id,
        )
