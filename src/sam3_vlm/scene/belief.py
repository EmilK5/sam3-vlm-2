"""Semantic memory and Bayesian belief updating primitives (V4 Design Spec §3.5 / §11 / §35.6)."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence
from sam3_vlm.core.config import BeliefConfig
from sam3_vlm.core.types import (
    ActionFamily,
    ClassBelief,
    NodeObservationRef,
    ObservationRelation,
)
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction


def canonical_belief_classes(num_confounders: int) -> List[str]:
    """Return the only legal posterior dimensions for a run."""
    if num_confounders < 0:
        raise ValueError("num_confounders must be non-negative")
    return ["target"] + [f"confounder{i}" for i in range(1, num_confounders + 1)]


def validate_belief_vocabulary(vocabulary: Sequence[str]) -> None:
    """Reject semantic labels/aliases from entering posterior state."""
    values = list(vocabulary)
    if not values or values[0] != "target":
        raise ValueError("Belief vocabulary must begin with canonical class 'target'.")
    expected = ["target"] + [f"confounder{i}" for i in range(1, len(values))]
    if values != expected:
        raise ValueError(
            f"Belief vocabulary is not canonical. Expected {expected}, received {values}."
        )


@dataclass
class SemanticRecord:
    """Tracking structure for attempted semantic experiments (V4 Design Spec §3.5)."""

    correlation_group: str
    semantic_keys: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    family: ActionFamily = ActionFamily.DISCOVERY
    execution_count: int = 0
    sam3_call_ids: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    new_nodes_by_execution: List[int] = field(default_factory=list)
    predicted_utility_by_execution: List[float] = field(default_factory=list)
    affected_nodes_by_execution: List[int] = field(default_factory=list)
    entropy_change_by_execution: List[float] = field(default_factory=list)
    variance_change_by_execution: List[float] = field(default_factory=list)
    realized_discrimination_proxy_by_execution: List[float] = field(default_factory=list)
    realized_utility_by_execution: List[float] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """Persistent history of semantic keys and queries tested."""

    records: Dict[str, SemanticRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import dataclasses
        return {"records": {k: dataclasses.asdict(v) for k, v in self.records.items()}}

    def record_execution(
        self,
        action: SensingAction,
        sam3_call_id: str,
        new_nodes: int = 0,
        runtime_ms: float = 0.0,
        predicted_utility: float = 0.0,
        affected_nodes: int = 0,
        entropy_change: float = 0.0,
        variance_change: float = 0.0,
        realized_discrimination_proxy: float = 0.0,
    ) -> SemanticRecord:
        group = action.correlation_group or action.semantic_key
        if group not in self.records:
            self.records[group] = SemanticRecord(correlation_group=group, family=action.family)
        rec = self.records[group]
        if action.semantic_key not in rec.semantic_keys:
            rec.semantic_keys.append(action.semantic_key)
        if action.prompt not in rec.prompts:
            rec.prompts.append(action.prompt)
        rec.execution_count += 1
        rec.sam3_call_ids.append(sam3_call_id)
        rec.total_cost += runtime_ms
        rec.new_nodes_by_execution.append(new_nodes)
        rec.predicted_utility_by_execution.append(predicted_utility)
        rec.affected_nodes_by_execution.append(affected_nodes)
        rec.entropy_change_by_execution.append(entropy_change)
        rec.variance_change_by_execution.append(variance_change)
        rec.realized_discrimination_proxy_by_execution.append(realized_discrimination_proxy)
        rec.realized_utility_by_execution.append(realized_discrimination_proxy)
        return rec


@dataclass
class ProxyEvidenceConfig:
    """Configurable coefficients for uncalibrated sensor likelihood proxy."""

    strong_target_multiplier: float = 1.5
    strong_confounder_multiplier: float = 3.0
    weak_confounder_multiplier: float = 0.8
    weak_target_multiplier: float = 0.4
    not_retrieved_penalty: float = 0.15


class ProxyEvidenceModel:
    """Explicitly uncalibrated proxy for sensor likelihoods (V4 Design Spec §11)."""

    def __init__(self, config: ProxyEvidenceConfig = ProxyEvidenceConfig()):
        self.config = config

    def compute_likelihoods(
        self,
        action: SensingAction,
        relation: ObservationRelation,
        score: float,
        weight: float,
        vocabulary: List[str],
        target_class: Optional[str] = None,
        confounder_class: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute bounded likelihood multipliers without inventing classes."""
        likelihoods = {cls: 1.0 for cls in vocabulary}
        semantic_weights = action.semantic_prior
        if semantic_weights is None:
            if action.family in (ActionFamily.DISCOVERY, ActionFamily.VERIFICATION):
                semantic_weights = (
                    {target_class: 1.0}
                    if target_class
                    else {action.semantic_key: 1.0}
                )
            else:
                # Generic M4-M7 semantics: an uncalibrated confounder/context
                # action refers to its own semantic coordinate. Canonical M8
                # actions normally provide an explicit generic-slot prior.
                semantic_weights = {action.semantic_key: 1.0}

        is_confounder = action.family == ActionFamily.CONFOUNDER
        for cls_name in vocabulary:
            sem_w = semantic_weights.get(cls_name, 0.0)
            if relation in (ObservationRelation.STRONG_MATCH, ObservationRelation.NEW_DETECTION):
                if sem_w > 0:
                    mult = (
                        self.config.strong_confounder_multiplier
                        if is_confounder
                        else self.config.strong_target_multiplier
                    )
                    likelihoods[cls_name] = 1.0 + (mult * score * weight * sem_w)
            elif relation == ObservationRelation.WEAK_MATCH:
                if sem_w > 0:
                    mult = (
                        self.config.weak_confounder_multiplier
                        if is_confounder
                        else self.config.weak_target_multiplier
                    )
                    likelihoods[cls_name] = 1.0 + (mult * score * weight * sem_w)
            elif relation == ObservationRelation.NOT_RETRIEVED:
                if sem_w > 0:
                    likelihoods[cls_name] = max(
                        0.1,
                        1.0 - (self.config.not_retrieved_penalty * weight * sem_w),
                    )
        return likelihoods


class BeliefUpdater:
    """Dataset-agnostic updater with a frozen canonical class ontology."""

    def __init__(self, proxy_config: ProxyEvidenceConfig = ProxyEvidenceConfig()):
        self.evidence_model = ProxyEvidenceModel(proxy_config)

    @staticmethod
    def calculate_entropy(probabilities: Dict[str, float]) -> float:
        h = 0.0
        for p in probabilities.values():
            if p > 0.0:
                h -= p * math.log2(p)
        return max(0.0, h)

    def _update_dynamic_node_belief(
        self,
        node: Node,
        action: SensingAction,
        obs_ref: NodeObservationRef,
        target_class: Optional[str],
        confounder_class: Optional[str],
        event_id: Optional[str],
        config: BeliefConfig,
    ) -> None:
        """Legacy/dataset-generic M4-M7 belief update.

        This path intentionally permits arbitrary finite class names. The M8
        production path opts into the frozen canonical vocabulary by passing
        class_vocabulary explicitly.
        """
        vocabulary_set = (
            set(node.class_belief.probabilities.keys())
            if node.class_belief.probabilities
            else set()
        )
        if action.semantic_prior:
            vocabulary_set.update(action.semantic_prior.keys())
        if target_class:
            vocabulary_set.add(target_class)
        if confounder_class:
            vocabulary_set.add(confounder_class)
        if not vocabulary_set:
            vocabulary_set.add(action.semantic_key)
        vocabulary = list(vocabulary_set)

        if not node.class_belief.probabilities:
            equal_p = 1.0 / len(vocabulary)
            probs = {cls_name: equal_p for cls_name in vocabulary}
        else:
            probs = dict(node.class_belief.probabilities)
            new_classes = [c for c in vocabulary if c not in probs]
            if new_classes:
                pseudocount_mass = config.prior_pseudocount / (
                    node.class_belief.update_count
                    + config.prior_pseudocount * len(vocabulary)
                )
                scale = max(0.01, 1.0 - pseudocount_mass * len(new_classes))
                for key in probs:
                    probs[key] *= scale
                for new_cls in new_classes:
                    probs[new_cls] = pseudocount_mass
                total = sum(probs.values())
                if total > 0.0:
                    probs = {k: v / total for k, v in probs.items()}

        score = obs_ref.score if obs_ref.score is not None else 0.5
        corr_group = action.correlation_group or action.semantic_key
        same_key_count = sum(
            1
            for old_obs in node.observations
            if (getattr(old_obs, "correlation_group", None) or old_obs.semantic_key)
            == corr_group
            and old_obs.observation_id != obs_ref.observation_id
            and old_obs.relation != ObservationRelation.NOT_OBSERVABLE
        )
        weight = config.discount_repeat_weight ** same_key_count
        likelihoods = self.evidence_model.compute_likelihoods(
            action=action,
            relation=obs_ref.relation,
            score=score,
            weight=weight,
            vocabulary=vocabulary,
            target_class=target_class,
            confounder_class=confounder_class,
        )
        unnormalized = {
            cls_name: probs[cls_name] * likelihoods.get(cls_name, 1.0)
            for cls_name in probs
        }
        total = sum(unnormalized.values())
        updated_probs = (
            {k: v / total for k, v in unnormalized.items()}
            if total > 0.0
            else probs
        )
        node.class_belief = ClassBelief(
            probabilities=updated_probs,
            update_count=node.class_belief.update_count + 1,
            entropy=self.calculate_entropy(updated_probs),
            last_update_event_id=event_id,
        )

    @staticmethod
    def _resolve_vocabulary(
        node: Node,
        class_vocabulary: Optional[Sequence[str]],
        config: BeliefConfig,
    ) -> List[str]:
        vocabulary = list(class_vocabulary or canonical_belief_classes(config.num_confounders))
        validate_belief_vocabulary(vocabulary)
        if node.class_belief.probabilities:
            actual = list(node.class_belief.probabilities.keys())
            if set(actual) != set(vocabulary):
                raise ValueError(
                    "Belief ontology invariant violated: node posterior keys changed. "
                    f"Expected {vocabulary}, found {actual}."
                )
        return vocabulary

    def update_node_belief(
        self,
        node: Node,
        action: SensingAction,
        obs_ref: NodeObservationRef,
        target_class: Optional[str] = None,
        confounder_class: Optional[str] = None,
        event_id: Optional[str] = None,
        config: BeliefConfig = BeliefConfig(),
        class_vocabulary: Optional[Sequence[str]] = None,
    ) -> None:
        """Update a node; explicit class_vocabulary opts into frozen M8 ontology."""
        # NOT_OBSERVABLE is a hard no-evidence invariant in every mode and
        # must return before any vocabulary validation/mutation.
        if obs_ref.relation == ObservationRelation.NOT_OBSERVABLE:
            return

        if class_vocabulary is None:
            return self._update_dynamic_node_belief(
                node=node,
                action=action,
                obs_ref=obs_ref,
                target_class=target_class,
                confounder_class=confounder_class,
                event_id=event_id,
                config=config,
            )

        vocabulary = self._resolve_vocabulary(node, class_vocabulary, config)
        canonical_target = "target"
        if target_class not in (None, "target"):
            raise ValueError(
                f"Posterior target class is canonical 'target'; received alias {target_class!r}."
            )

        if action.semantic_prior:
            unknown = set(action.semantic_prior) - set(vocabulary)
            if unknown:
                raise ValueError(
                    "Qwen semantic_prior attempted to expand belief ontology with: "
                    + ", ".join(sorted(unknown))
                )

        if obs_ref.relation == ObservationRelation.NOT_OBSERVABLE:
            return

        if not node.class_belief.probabilities:
            equal_p = 1.0 / len(vocabulary)
            probs = {cls_name: equal_p for cls_name in vocabulary}
        else:
            probs = {cls_name: node.class_belief.probabilities[cls_name] for cls_name in vocabulary}

        score = obs_ref.score if obs_ref.score is not None else 0.5
        corr_group = action.correlation_group or action.semantic_key
        same_key_count = sum(
            1
            for o in node.observations
            if (getattr(o, "correlation_group", None) or o.semantic_key) == corr_group
            and o.observation_id != obs_ref.observation_id
            and o.relation != ObservationRelation.NOT_OBSERVABLE
        )
        weight = config.discount_repeat_weight ** same_key_count

        likelihoods = self.evidence_model.compute_likelihoods(
            action=action,
            relation=obs_ref.relation,
            score=score,
            weight=weight,
            vocabulary=vocabulary,
            target_class=canonical_target,
            confounder_class=confounder_class,
        )
        unnormalized = {
            cls_name: probs[cls_name] * likelihoods.get(cls_name, 1.0)
            for cls_name in vocabulary
        }
        total = sum(unnormalized.values())
        updated_probs = (
            {k: v / total for k, v in unnormalized.items()} if total > 0 else probs
        )
        if list(updated_probs.keys()) != vocabulary:
            raise AssertionError("Belief updater changed canonical vocabulary ordering.")

        node.class_belief = ClassBelief(
            probabilities=updated_probs,
            update_count=node.class_belief.update_count + 1,
            entropy=self.calculate_entropy(updated_probs),
            last_update_event_id=event_id,
        )
