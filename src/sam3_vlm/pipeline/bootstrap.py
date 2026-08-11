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
        recorder: Optional[Any] = None
    ) -> None:
        self.sensor = sensor
        self.association_policy = association_policy or IoUAssociationPolicy()
        self.belief_updater = belief_updater or BeliefUpdater()
        self.tiling_policy = tiling_policy or DefaultTilingPolicy()
        self.id_gen = id_gen or IDGenerator()
        self.config = config
        self.recorder = recorder

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

        if self.recorder:
            self.recorder.record_sam3_action_selected(global_action.action_id, global_action.semantic_key)
            self.recorder.record_sam3_action_started(global_action.action_id)

        obs_global = self.sensor.observe(image, global_action)
        
        if self.recorder:
            mask_artifacts = []
            compact_detections = []
            for det in obs_global.detections:
                if "mask" in det.raw_metadata:
                    art_ref = self.recorder.save_mask_artifact(det.detection_id, det.raw_metadata["mask"])
                    det.mask_artifact = art_ref["relative_path"]
                    mask_artifacts.append(art_ref)
                    
                box = det.geometry.bbox()
                compact_detections.append({
                    "detection_id": det.detection_id,
                    "geometry": {"box": box.as_tuple(), "coordinate_space": box.coordinate_space},
                    "score": det.score,
                    "source_tile_id": getattr(det, 'source_tile_id', None),
                    "mask_artifact": getattr(det, 'mask_artifact', None)
                })

            self.recorder.record_sam3_action_completed(
                global_action.action_id,
                obs_global.call_id,
                {
                    "num_detections": len(obs_global.detections), 
                    "runtime_ms": obs_global.runtime_ms, 
                    "mask_artifacts": mask_artifacts,
                    "searched_regions": [{"box": r.bbox().as_tuple(), "coordinate_space": r.bbox().coordinate_space} for r in obs_global.searched_regions] if hasattr(obs_global, 'searched_regions') else [],
                    "detections": compact_detections
                }
            )
        semantic_memory.record_execution(global_action, obs_global.call_id)
        if self.recorder:
            self.recorder.record_semantic_memory_updated(semantic_memory.to_dict())
            
        state.budget.sam3_calls += 1
        state.budget.total_runtime_ms += obs_global.runtime_ms
        if self.recorder:
            self.recorder.record_budget_updated(state.budget.__dict__)

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
        if self.recorder:
            self.recorder.record_association_completed(global_action.action_id, len(assoc_global.matched_observations), len(assoc_global.new_nodes))
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
            if self.recorder:
                prov = {
                    "action_id": global_action.action_id,
                    "sam3_call_id": obs_global.call_id,
                    "detection_id": new_node.observations[0].detection_id,
                    "observation_id": new_node.observations[0].observation_id,
                    "semantic_key": global_action.semantic_key
                }
                self.recorder.record_node_created(new_node.node_id, new_node.to_dict(), prov)

        # Stage 2: Conditional Tiled Bootstrap Pass (Spec §5.2)
        img_w, img_h = 1000, 1000
        if isinstance(image, (tuple, list)) and len(image) == 2:
            img_w, img_h = int(image[0]), int(image[1])
        elif hasattr(image, "shape") and len(image.shape) >= 2:
            img_h, img_w = int(image.shape[0]), int(image.shape[1])
        elif hasattr(image, "size") and isinstance(image.size, tuple):
            img_w, img_h = image.size[0], image.size[1]

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

            if self.recorder:
                self.recorder.record_sam3_action_selected(tiled_action.action_id, tiled_action.semantic_key)
                self.recorder.record_sam3_action_started(tiled_action.action_id)

            obs_tiled = self.sensor.observe(image, tiled_action)
            
            if self.recorder:
                mask_artifacts = []
                compact_detections = []
                for det in obs_tiled.detections:
                    if "mask" in det.raw_metadata:
                        art_ref = self.recorder.save_mask_artifact(det.detection_id, det.raw_metadata["mask"])
                        det.mask_artifact = art_ref["relative_path"]
                        mask_artifacts.append(art_ref)
                    
                    box = det.geometry.bbox()
                    compact_detections.append({
                        "detection_id": det.detection_id,
                        "geometry": {"box": box.as_tuple(), "coordinate_space": box.coordinate_space},
                        "score": det.score,
                        "source_tile_id": getattr(det, 'source_tile_id', None),
                        "mask_artifact": getattr(det, 'mask_artifact', None)
                    })

                self.recorder.record_sam3_action_completed(
                    tiled_action.action_id,
                    obs_tiled.call_id,
                    {
                        "num_detections": len(obs_tiled.detections), 
                        "runtime_ms": obs_tiled.runtime_ms, 
                        "mask_artifacts": mask_artifacts,
                        "searched_regions": [{"box": r.bbox().as_tuple(), "coordinate_space": r.bbox().coordinate_space} for r in obs_tiled.searched_regions] if hasattr(obs_tiled, 'searched_regions') else [],
                        "detections": compact_detections
                    }
                )
            
            semantic_memory.record_execution(tiled_action, obs_tiled.call_id)
            if self.recorder:
                self.recorder.record_semantic_memory_updated(semantic_memory.to_dict())
                
            state.budget.sam3_calls += 1
            state.budget.sam3_tiles += len(tiling_decision.tiles)
            state.budget.total_runtime_ms += obs_tiled.runtime_ms
            if self.recorder:
                self.recorder.record_budget_updated(state.budget.__dict__)

            assoc_tiled = self.association_policy.associate(
                graph=state.graph,
                detections=obs_tiled.detections,
                sam3_call_id=obs_tiled.call_id,
                action_id=tiled_action.action_id,
                semantic_key=tiled_action.semantic_key,
                id_gen=self.id_gen,
                config=self.config.association,
            )
            if self.recorder:
                self.recorder.record_association_completed(tiled_action.action_id, len(assoc_tiled.matched_observations), len(assoc_tiled.new_nodes))

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
                if self.recorder:
                    prov = {
                        "action_id": tiled_action.action_id,
                        "sam3_call_id": obs_tiled.call_id,
                        "detection_id": new_node.observations[0].detection_id,
                        "observation_id": new_node.observations[0].observation_id,
                        "semantic_key": tiled_action.semantic_key
                    }
                    self.recorder.record_node_created(new_node.node_id, new_node.to_dict(), prov)

            discovery_state.tiled_bootstrap_gain = float(len(assoc_tiled.new_nodes))
        
        if self.recorder:
            self.recorder.record_discovery_state_updated(discovery_state.to_dict())
            self.recorder.record_budget_updated(state.budget.__dict__)

        # Stage 3: Contact-Sheet & Qwen Evidence Pack Assembly (Spec §5.3 / §6.1)
        contact_sheet = ContactSheetBuilder().build_contact_sheet(
            graph=state.graph,
            max_crops=24,
            image=image,
            assets_dir=self.config.assets_dir,
            image_id=image_id,
        )

        image_path_str = None
        if image is not None:
            # Try to save the full image to assets as well, or just keep original path if it's a string
            from pathlib import Path
            if isinstance(image, (str, Path)):
                image_path_str = str(image)
            else:
                from sam3_vlm.sensing.visuals import save_image, to_numpy_image
                img_arr = to_numpy_image(image)
                if img_arr is not None:
                    full_img_path = Path(self.config.assets_dir) / f"{image_id}.jpg"
                    if save_image(img_arr, str(full_img_path)):
                        image_path_str = str(full_img_path)
                        
        state.image_path = image_path_str

        qwen_evidence_pack = QwenEvidencePack(
            original_image_id=image_id,
            user_prompt=user_prompt,
            target_class=target_class,
            contact_sheet=contact_sheet,
            image_path=image_path_str,
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

