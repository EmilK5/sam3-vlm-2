"""Bootstrap pipeline implementation for global & conditional tiled sensing (V4 Design Spec §5)."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.scene.association import AssociationPolicy, IoUAssociationPolicy
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import DiscoveryState, SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import (
    ContactSheetBuilder,
    QwenEvidencePack,
)
from sam3_vlm.sensing.tiling import DefaultTilingPolicy, TilingPolicy


@dataclass
class BootstrapResult:
    """Output bundle from bootstrap pipeline execution (V4 Design Spec §5.4)."""

    state: SceneState
    qwen_evidence_pack: QwenEvidencePack


class BootstrapStage(Protocol):
    """Protocol for initial global/tiled bootstrap pass."""

    def execute_bootstrap(
        self,
        image_id: str,
        image: Any,
        user_prompt: str,
        target_class: str = "target",
        confounder_class: str = "confounder",
    ) -> BootstrapResult:
        ...


class BootstrapPipeline:
    """Orchestrates initial global and conditional tiled user-prompt sensing (V4 Design Spec §5).

    Invariant §5.4: Bootstrap MUST NOT execute Qwen planning internally.
    """

    def __init__(
        self,
        sensor: SAM3Sensor,
        association_policy: Optional[AssociationPolicy] = None,
        belief_updater: Optional[BeliefUpdater] = None,
        tiling_policy: Optional[TilingPolicy] = None,
        id_gen: Optional[IDGenerator] = None,
        config: V4Config = V4Config(),
    ) -> None:
        self.sensor = sensor
        self.association_policy = association_policy or IoUAssociationPolicy()
        self.belief_updater = belief_updater or BeliefUpdater()
        self.tiling_policy = tiling_policy or DefaultTilingPolicy()
        self.id_gen = id_gen or IDGenerator()
        self.config = config

    def execute_bootstrap(
        self,
        image_id: str,
        image: Any,
        user_prompt: str,
        target_class: str = "target",
        confounder_class: str = "confounder",
    ) -> BootstrapResult:
        """Run global user-prompt pass, optional tiled pass, candidate registration, and evidence assembly."""
        graph = SceneGraph()
        semantic_memory = SemanticMemory()
        discovery_state = DiscoveryState()

        state = SceneState(
            image_id=image_id,
            user_prompt=user_prompt,
            target_class=target_class,
            graph=graph,
            semantic_memory=semantic_memory,
            discovery_state=discovery_state,
            iteration=0,
            qwen_round=0,
        )

        # Stage 1: Global User-Prompt Sensing Pass (Spec §5.1)
        global_action = SensingAction(
            action_id=self.id_gen.next_action_id(),
            semantic_key=target_class,
            prompt=user_prompt,
            family=ActionFamily.DISCOVERY,
            spatial_mode=SpatialMode.GLOBAL,
            source=ActionSource.USER_BOOTSTRAP,
            threshold=self.config.sam3.default_threshold,
        )

        semantic_memory.record_execution(global_action, "global_bootstrap")

        obs_global = self.sensor.observe(image, global_action)
        state.budget.sam3_calls += 1
        state.budget.total_runtime_ms += obs_global.runtime_ms

        # Associate global observations into graph
        assoc_global = self.association_policy.associate(
            graph=state.graph,
            detections=obs_global.detections,
            sam3_call_id=obs_global.call_id,
            action_id=global_action.action_id,
            semantic_key=global_action.semantic_key,
            id_gen=self.id_gen,
            config=self.config.association,
        )

        # Update beliefs for matched and new nodes
        for node_id, obs_ref in assoc_global.matched_observations:
            node = state.graph.get_node(node_id)
            if node:
                self.belief_updater.update_node_belief(
                    node, global_action, obs_ref, target_class=target_class, confounder_class=confounder_class
                )

        for new_node in assoc_global.new_nodes:
            self.belief_updater.update_node_belief(
                new_node, global_action, new_node.observations[0], target_class=target_class, confounder_class=confounder_class
            )

        # Stage 2: Conditional Tiled Bootstrap Pass (Spec §5.2)
        img_w, img_h = 1000, 1000
        if isinstance(image, (tuple, list)) and len(image) == 2:
            img_w, img_h = int(image[0]), int(image[1])
        elif hasattr(image, "size"):
            img_w, img_h = image.size[0], image.size[1]
        elif hasattr(image, "shape"):
            img_h, img_w = int(image.shape[0]), int(image.shape[1])

        tiling_decision = self.tiling_policy.evaluate_tiling(
            image_width=img_w,
            image_height=img_h,
            config=self.config.tiling,
            graph=state.graph,
        )

        if tiling_decision.should_tile and self.config.bootstrap.enable_tiled_bootstrap:
            tiled_action = SensingAction(
                action_id=self.id_gen.next_action_id(),
                semantic_key=target_class,
                prompt=user_prompt,
                family=ActionFamily.DISCOVERY,
                spatial_mode=SpatialMode.TILED,
                source=ActionSource.USER_BOOTSTRAP,
                tiling=self.config.tiling,
                threshold=self.config.sam3.default_threshold,
            )

            semantic_memory.record_execution(tiled_action, "tiled_bootstrap")

            obs_tiled = self.sensor.observe(image, tiled_action)
            state.budget.sam3_calls += 1
            state.budget.sam3_tiles += len(tiling_decision.tiles)
            state.budget.total_runtime_ms += obs_tiled.runtime_ms

            assoc_tiled = self.association_policy.associate(
                graph=state.graph,
                detections=obs_tiled.detections,
                sam3_call_id=obs_tiled.call_id,
                action_id=tiled_action.action_id,
                semantic_key=tiled_action.semantic_key,
                id_gen=self.id_gen,
                config=self.config.association,
            )

            for node_id, obs_ref in assoc_tiled.matched_observations:
                node = state.graph.get_node(node_id)
                if node:
                    self.belief_updater.update_node_belief(
                        node, tiled_action, obs_ref, target_class=target_class, confounder_class=confounder_class
                    )

            for new_node in assoc_tiled.new_nodes:
                self.belief_updater.update_node_belief(
                    new_node, tiled_action, new_node.observations[0], target_class=target_class, confounder_class=confounder_class
                )

            discovery_state.tiled_bootstrap_gain = float(len(assoc_tiled.new_nodes))

        # Stage 3: Contact-Sheet & Qwen Evidence Pack Assembly (Spec §5.3 / §6.1)
        contact_sheet = ContactSheetBuilder().build_contact_sheet(state.graph, max_crops=24)

        qwen_evidence_pack = QwenEvidencePack(
            original_image_id=image_id,
            user_prompt=user_prompt,
            target_class=target_class,
            contact_sheet=contact_sheet,
            scene_summary=f"Bootstrap complete. Total candidates: {contact_sheet.total_candidates}.",
            discovery_diagnostics={
                "sam3_calls": state.budget.sam3_calls,
                "active_nodes": len(state.graph.active_nodes()),
                "tiled_bootstrap_executed": tiling_decision.should_tile and self.config.bootstrap.enable_tiled_bootstrap,
            },
        )

        return BootstrapResult(
            state=state,
            qwen_evidence_pack=qwen_evidence_pack,
        )
