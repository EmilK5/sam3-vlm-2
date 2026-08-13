"""Residual targeted cleanup phase (V4 Design Spec §13)."""

from typing import List, Optional
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box
from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.id_generator import IDGenerator


class CleanupController:
    """Manages residual cleanup selection and batching."""

    def __init__(self, id_gen: IDGenerator):
        self.id_gen = id_gen
        self.batch_attempts = {}
        self.last_batch_utility = {}

    def select_residual_nodes(self, graph: SceneGraph, config: V4Config, target_class: str) -> List[Node]:
        """Select a small subset of highly ambiguous nodes for cleanup."""
        ambiguous = []
        for node in graph.active_nodes():
            p = node.class_belief.probabilities.get(target_class, 0.0)
            variance = p * (1.0 - p)
            entropy = node.class_belief.entropy

            # Consider ambiguous if entropy is high or it contributes significantly to count variance
            if (entropy > config.cleanup.cleanup_ambiguity_threshold or 
                variance > (config.cleanup.cleanup_ambiguity_threshold / 4.0)):
                ambiguous.append((node, variance, entropy))

        # Sort by variance descending, then entropy descending
        ambiguous.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Take the top N
        return [x[0] for x in ambiguous[:config.cleanup.cleanup_residual_max_nodes]]

    def _select_trusted_exemplars(self, graph: SceneGraph, target_class: str, residual_nodes: List[Node]) -> tuple[str, ...]:
        """Select trusted positive exemplars from already-resolved nodes outside the residual set."""
        trusted = []
        residual_ids = {n.node_id for n in residual_nodes}
        for node in graph.active_nodes():
            if node.node_id in residual_ids:
                continue
            
            p = node.class_belief.probabilities.get(target_class, 0.0)
            entropy = node.class_belief.entropy
            
            # Trusted if high probability and low entropy
            if p > 0.8 and entropy < 0.2:
                trusted.append(node.node_id)
                
        return tuple(trusted)

    def generate_cleanup_action(
        self, residual_nodes: List[Node], graph: SceneGraph, target_class: str,
        config: V4Config, user_prompt: Optional[str] = None
    ) -> "CleanupDecision":
        """Generate a batched or local cleanup action for the residual nodes."""
        from sam3_vlm.core.types import CleanupDecision, StopReason
        
        if not residual_nodes:
            return CleanupDecision(action=None, reason=StopReason.CLEANUP_COMPLETE)

        # Partition residual nodes spatially into batches of max roi_batch_size
        residual_nodes_sorted = sorted(residual_nodes, key=lambda n: n.geometry.bbox().x1)
        batches = [residual_nodes_sorted[i:i + config.cleanup.roi_batch_size] for i in range(0, len(residual_nodes_sorted), config.cleanup.roi_batch_size)]

        best_batch = None
        best_utility = -1.0
        best_raw_utility = -1.0
        low_utility_count = 0

        for batch in batches:
            batch_ids = frozenset(n.node_id for n in batch)
            attempts = self.batch_attempts.get(batch_ids, 0)

            total_variance = sum(
                n.class_belief.probabilities.get(target_class, 0.0) * (1.0 - n.class_belief.probabilities.get(target_class, 0.0))
                for n in batch
            )
            total_entropy = sum(n.class_belief.entropy for n in batch)
            raw_utility = (total_variance + total_entropy) / (len(batch) + 1)

            if raw_utility < config.cleanup.cleanup_min_utility:
                low_utility_count += 1
                continue

            utility = raw_utility
            if attempts > 0:
                last_util = self.last_batch_utility.get(batch_ids, 0.0)
                gain = last_util - raw_utility
                if gain < 0.05:
                    low_utility_count += 1
                    continue
                utility /= (attempts + 1)

            if utility > best_utility:
                best_utility = utility
                best_raw_utility = raw_utility
                best_batch = batch

        if not best_batch:
            if low_utility_count == len(batches):
                return CleanupDecision(action=None, reason=StopReason.LOW_MARGINAL_UTILITY)
            else:
                return CleanupDecision(action=None, reason=StopReason.NO_VALID_ACTIONS)
            
        batch = best_batch
        batch_ids = frozenset(n.node_id for n in batch)
        self.batch_attempts[batch_ids] = self.batch_attempts.get(batch_ids, 0) + 1
        self.last_batch_utility[batch_ids] = best_raw_utility

        # Determine spatial mode based on batch size
        if len(batch) > 1:
            mode = SpatialMode.ROI_BATCH
            min_x = min(n.geometry.bbox().x1 for n in batch)
            min_y = min(n.geometry.bbox().y1 for n in batch)
            max_x = max(n.geometry.bbox().x2 for n in batch)
            max_y = max(n.geometry.bbox().y2 for n in batch)
            roi = Box(x1=min_x, y1=min_y, x2=max_x, y2=max_y)
        else:
            mode = SpatialMode.LOCAL
            roi = batch[0].geometry.bbox()

        trusted_exemplars = self._select_trusted_exemplars(graph, target_class, batch)

        # Dataset-independent cleanup prior
        semantic_prior = {target_class: 0.9}

        action = SensingAction(
            action_id=self.id_gen.next_action_id(),
            semantic_key=f"cleanup_{target_class}",
            prompt=user_prompt or "ambiguous object",
            family=ActionFamily.VERIFICATION,
            spatial_mode=mode,
            source=ActionSource.CLEANUP,
            roi=roi,
            positive_exemplar_ids=trusted_exemplars,
            qwen_priority=1.0,
            semantic_prior=semantic_prior
        )
        return CleanupDecision(action=action)
