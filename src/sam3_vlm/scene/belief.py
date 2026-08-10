"""Semantic memory and Bayesian belief updating primitives (V4 Design Spec §3.5 / §11 / §35.6)."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional
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
    predicted_utility_by_execution: List[float] = field(default_factory=list)

@dataclass
class SemanticMemory:
    """Persistent history of semantic keys and queries tested."""

    records: Dict[str, SemanticRecord] = field(default_factory=dict)

    def record_execution(
        self, 
        action: SensingAction, 
        sam3_call_id: str,
        new_nodes: int = 0,
        runtime_ms: float = 0.0,
        predicted_utility: float = 0.0,
    ) -> SemanticRecord:
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
        rec.total_cost += runtime_ms
        rec.new_nodes_by_execution.append(new_nodes)
        rec.predicted_utility_by_execution.append(predicted_utility)
        return rec


class BeliefUpdater:
    """Dataset-agnostic class belief distribution updater enforcing presence/absence asymmetry (V4 Design Spec §11)."""

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
        target_class: Optional[str] = None,
        confounder_class: Optional[str] = None,
        event_id: Optional[str] = None,
        config: BeliefConfig = BeliefConfig(),
    ) -> None:
        """Update node class belief distribution based on observation relation and action family."""
        target_cls = target_class or (
            action.semantic_key if action.family == ActionFamily.DISCOVERY else None
        )
        confounder_cls = confounder_class or (
            action.semantic_key if action.family == ActionFamily.CONFOUNDER else None
        )

        # Initialize probabilities if node has empty beliefs
        if not node.class_belief.probabilities:
            vocabulary = []
            if target_cls:
                vocabulary.append(target_cls)
            if confounder_cls and confounder_cls not in vocabulary:
                vocabulary.append(confounder_cls)
            if not vocabulary:
                vocabulary = [action.semantic_key]
            
            equal_p = 1.0 / len(vocabulary)
            probs = {cls_name: equal_p for cls_name in vocabulary}
        else:
            probs = dict(node.class_belief.probabilities)
            # Ensure target_cls and confounder_cls exist in distribution if provided
            if target_cls and target_cls not in probs:
                probs[target_cls] = 0.0
            if confounder_cls and confounder_cls not in probs:
                probs[confounder_cls] = 0.0

        relation = obs_ref.relation
        score = obs_ref.score if obs_ref.score is not None else 0.5

        # NOT_OBSERVABLE invariant (Spec §34.6): leave belief unchanged
        if relation == ObservationRelation.NOT_OBSERVABLE:
            return

        # Discount repeat weight if same correlation group was already executed on this node
        corr_group = action.correlation_group or action.semantic_key
        same_key_count = sum(
            1 for o in node.observations if (getattr(o, 'correlation_group', None) or o.semantic_key) == corr_group
        )
        weight = config.discount_repeat_weight ** same_key_count

        # Build likelihood multiplier per class
        likelihoods: Dict[str, float] = {cls_name: 1.0 for cls_name in probs}

        if relation in (ObservationRelation.STRONG_MATCH, ObservationRelation.NEW_DETECTION):
            if action.family == ActionFamily.DISCOVERY or (target_cls and action.semantic_key == target_cls):
                if target_cls:
                    likelihoods[target_cls] = 1.0 + (1.5 * score * weight)
            elif action.family == ActionFamily.CONFOUNDER or (confounder_cls and action.semantic_key == confounder_cls):
                if confounder_cls:
                    likelihoods[confounder_cls] = 1.0 + (3.0 * score * weight)
                if target_cls:
                    # Gentle penalty for target on confounder match
                    likelihoods[target_cls] = max(0.1, 1.0 - (0.75 * score * weight))
        elif relation == ObservationRelation.WEAK_MATCH:
            if action.family == ActionFamily.CONFOUNDER or (confounder_cls and action.semantic_key == confounder_cls):
                if confounder_cls:
                    likelihoods[confounder_cls] = 1.0 + (0.8 * score * weight)
                if target_cls:
                    likelihoods[target_cls] = max(0.1, 1.0 - (0.4 * score * weight))
            elif target_cls:
                likelihoods[target_cls] = 1.0 + (0.4 * score * weight)
        elif relation == ObservationRelation.NOT_RETRIEVED:
            # Presence/absence asymmetry: NOT_RETRIEVED is mild negative evidence
            if target_cls and (action.family == ActionFamily.DISCOVERY or action.semantic_key == target_cls):
                # We need effect(NOT_RETRIEVED) < effect(strong contradictory match)
                likelihoods[target_cls] = max(0.1, 1.0 - (0.15 * weight))

        # Apply likelihood update and normalize
        unnormalized = {cls_name: probs[cls_name] * likelihoods.get(cls_name, 1.0) for cls_name in probs}
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
