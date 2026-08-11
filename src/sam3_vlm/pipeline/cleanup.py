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

    def generate_cleanup_action(self, residual_nodes: List[Node], config: V4Config) -> Optional[SensingAction]:
        """Generate a batched or local cleanup action for the residual nodes."""
        if not residual_nodes:
            return None

        # Prioritize batched if more than 1 node, otherwise LOCAL
        if len(residual_nodes) > 1:
            mode = SpatialMode.ROI_BATCH
            # Create a bounding box covering all residual nodes
            min_x = min(n.geometry.bbox().x1 for n in residual_nodes)
            min_y = min(n.geometry.bbox().y1 for n in residual_nodes)
            max_x = max(n.geometry.bbox().x2 for n in residual_nodes)
            max_y = max(n.geometry.bbox().y2 for n in residual_nodes)
            roi = Box(x1=min_x, y1=min_y, x2=max_x, y2=max_y)
            # Use positive_exemplar_ids to pass the individual nodes being checked for batched
            exemplars = tuple(n.node_id for n in residual_nodes)
        else:
            mode = SpatialMode.LOCAL
            roi = residual_nodes[0].geometry.bbox()
            exemplars = (residual_nodes[0].node_id,)

        # In a real implementation, Qwen might propose the semantic key for the residual.
        # For M6 infrastructure, we generate a generic verification action.
        return SensingAction(
            action_id=self.id_gen.next_action_id(),
            semantic_key="cleanup_verification",
            prompt="verify target objects",
            family=ActionFamily.VERIFICATION,
            spatial_mode=mode,
            source=ActionSource.CLEANUP,
            roi=roi,
            positive_exemplar_ids=exemplars,
            qwen_priority=1.0,
            semantic_prior={"target": 0.8, "leaf": 0.2} # Generic prior
        )
