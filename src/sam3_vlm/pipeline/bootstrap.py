"""Sensor-first bootstrap with optional locked context search domain."""

from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.scene.association import AssociationPolicy, IoUAssociationPolicy
from sam3_vlm.scene.association_dual import IoUIoMAssociationPolicy
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory, canonical_belief_classes
from sam3_vlm.scene.exemplars import select_target_pseudoexemplars
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import DiscoveryState, SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheetBuilder, QwenEvidencePack
from sam3_vlm.sensing.tiling import DefaultTilingPolicy, TilingPolicy


@dataclass
class BootstrapResult:
    state: SceneState
    qwen_evidence_pack: QwenEvidencePack


class BootstrapStage(Protocol):
    def execute_bootstrap(
        self,
        image_id: str,
        image: Any,
        user_prompt: str,
        target_class: str = "target",
        confounder_class: Optional[str] = None,
    ) -> BootstrapResult:
        ...


class BootstrapPipeline:
    """Bootstrap without Qwen: optional context lock, global target, optional tiled target."""

    def __init__(
        self,
        sensor: SAM3Sensor,
        association_policy: Optional[AssociationPolicy] = None,
        belief_updater: Optional[BeliefUpdater] = None,
        tiling_policy: Optional[TilingPolicy] = None,
        id_gen: Optional[IDGenerator] = None,
        config: V4Config = V4Config(),
        recorder: Optional[Any] = None,
    ) -> None:
        self.sensor = sensor
        self.association_policy = association_policy or (
            IoUIoMAssociationPolicy()
            if config.association.enable_iom_dedup
            else IoUAssociationPolicy()
        )
        self.belief_updater = belief_updater or BeliefUpdater()
        self.tiling_policy = tiling_policy or DefaultTilingPolicy()
        self.id_gen = id_gen or IDGenerator()
        self.config = config
        self.recorder = recorder

    @staticmethod
    def _image_size(image: Any) -> tuple[int, int]:
        if isinstance(image, (tuple, list)) and len(image) == 2:
            return int(image[0]), int(image[1])
        if hasattr(image, "shape") and len(image.shape) >= 2:
            return int(image.shape[1]), int(image.shape[0])
        if hasattr(image, "size") and isinstance(image.size, tuple):
            return int(image.size[0]), int(image.size[1])
        return 1000, 1000

    def _record_observation(self, action: SensingAction, observation) -> None:
        if not self.recorder:
            return
        mask_artifacts = []
        compact_detections = []
        for det in observation.detections:
            if "mask" in det.raw_metadata:
                art_ref = self.recorder.save_mask_artifact(det.detection_id, det.raw_metadata["mask"])
                det.mask_artifact = art_ref["relative_path"]
                mask_artifacts.append(art_ref)
            box = det.geometry.bbox()
            compact_detections.append({
                "detection_id": det.detection_id,
                "geometry": {"box": box.as_tuple(), "coordinate_space": box.coordinate_space},
                "score": det.score,
                "source_tile_id": getattr(det, "source_tile_id", None),
                "mask_artifact": getattr(det, "mask_artifact", None),
            })
        self.recorder.record_sam3_action_completed(
            action.action_id,
            observation.call_id,
            {
                "num_detections": len(observation.detections),
                "runtime_ms": observation.runtime_ms,
                "model_metadata": dict(observation.model_metadata),
                "mask_artifacts": mask_artifacts,
                "searched_regions": [
                    {"box": r.bbox().as_tuple(), "coordinate_space": r.bbox().coordinate_space}
                    for r in observation.searched_regions
                ],
                "detections": compact_detections,
            },
        )

    def _execute_sensor_action(self, state: SceneState, image: Any, action: SensingAction):
        if self.recorder:
            self.recorder.record_sam3_action_selected(action.action_id, action.semantic_key)
            self.recorder.record_sam3_action_started(action.action_id)
        observation = self.sensor.observe(image, action)
        predicted_tiles = (
            action.tiling.grid_rows * action.tiling.grid_cols
            if action.spatial_mode == SpatialMode.TILED and action.tiling else 0
        )
        state.budget.sam3_calls += 1
        state.budget.sam3_tiles += predicted_tiles
        state.budget.sam3_runtime_ms += observation.runtime_ms
        state.budget.model_runtime_ms += observation.runtime_ms
        state.budget.total_runtime_ms += observation.runtime_ms
        self._record_observation(action, observation)
        if self.recorder:
            self.recorder.record_budget_updated(state.budget.__dict__)
        return observation

    @staticmethod
    def _enclosing_detection_region(detections, img_w: int, img_h: int) -> Optional[BoxGeometry]:
        if not detections:
            return None
        boxes = [det.geometry.bbox() for det in detections]
        box = Box(
            x1=max(0.0, min(b.x1 for b in boxes)),
            y1=max(0.0, min(b.y1 for b in boxes)),
            x2=min(float(img_w), max(b.x2 for b in boxes)),
            y2=min(float(img_h), max(b.y2 for b in boxes)),
        )
        return BoxGeometry(box) if box.area > 0.0 else None

    def _update_beliefs(self, state: SceneState, action: SensingAction, observation, assoc_result) -> None:
        for node_id, obs_ref in assoc_result.matched_observations:
            node = state.graph.get_node(node_id)
            if node:
                self.belief_updater.update_node_belief(
                    node,
                    action,
                    obs_ref,
                    target_class=state.target_class,
                    config=self.config.belief,
                    class_vocabulary=state.belief_classes or None,
                )
                if self.recorder:
                    prov = {
                        "action_id": action.action_id,
                        "sam3_call_id": observation.call_id,
                        "detection_id": obs_ref.detection_id,
                        "observation_id": obs_ref.observation_id,
                        "semantic_key": action.semantic_key,
                    }
                    self.recorder.record_node_updated(node.node_id, node.to_dict(), prov)
        for new_node in assoc_result.new_nodes:
            self.belief_updater.update_node_belief(
                new_node,
                action,
                new_node.observations[0],
                target_class=state.target_class,
                config=self.config.belief,
                class_vocabulary=state.belief_classes or None,
            )
            if self.recorder:
                prov = {
                    "action_id": action.action_id,
                    "sam3_call_id": observation.call_id,
                    "detection_id": new_node.observations[0].detection_id,
                    "observation_id": new_node.observations[0].observation_id,
                    "semantic_key": action.semantic_key,
                }
                self.recorder.record_node_created(new_node.node_id, new_node.to_dict(), prov)

    def _associate_discovery(self, state: SceneState, action: SensingAction, observation):
        result = self.association_policy.associate(
            graph=state.graph,
            detections=observation.detections,
            sam3_call_id=observation.call_id,
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            id_gen=self.id_gen,
            correlation_group=action.correlation_group,
            config=self.config.association,
        )
        if self.recorder:
            self.recorder.record_association_completed(
                action.action_id, len(result.matched_observations), len(result.new_nodes)
            )
        self._update_beliefs(state, action, observation, result)
        state.discovery_state.record_search(observation.searched_regions, state.search_region)
        state.discovery_state.record_discovery_gain(
            len(result.new_nodes),
            [node.node_id for node in result.new_nodes],
            plateau_window=self.config.replanning.discovery_plateau_steps + 1,
        )
        state.semantic_memory.record_execution(
            action,
            observation.call_id,
            new_nodes=len(result.new_nodes),
            runtime_ms=observation.runtime_ms,
        )
        if self.recorder:
            self.recorder.record_semantic_memory_updated(state.semantic_memory.to_dict())
        return result

    def execute_bootstrap(
        self,
        image_id: str,
        image: Any,
        user_prompt: str,
        target_class: str = "target",
        confounder_class: Optional[str] = None,
    ) -> BootstrapResult:
        canonical_mode = target_class == "target" and confounder_class is None
        effective_target = "target" if canonical_mode else target_class

        img_w, img_h = self._image_size(image)
        full_image = BoxGeometry(Box(0.0, 0.0, float(img_w), float(img_h)))
        belief_classes = (
            canonical_belief_classes(self.config.belief.num_confounders)
            if canonical_mode
            else []
        )
        state = SceneState(
            image_id=image_id,
            user_prompt=user_prompt,
            target_class=effective_target,
            graph=SceneGraph(),
            semantic_memory=SemanticMemory(),
            discovery_state=DiscoveryState(),
            belief_classes=belief_classes,
            search_region=full_image,
            search_region_locked=False,
            search_region_source="FULL_IMAGE",
            iteration=0,
            qwen_round=0,
        )

        # Pass 0: deployment-configured context localization.  It never enters
        # the counted graph.  Green citrus config uses "tree canopy".
        context_prompt = self.config.bootstrap.locked_context_prompt
        if context_prompt:
            context_action = SensingAction(
                action_id=self.id_gen.next_action_id(),
                semantic_key="locked_context_region",
                prompt=context_prompt,
                family=ActionFamily.CONTEXT,
                spatial_mode=SpatialMode.GLOBAL,
                source=ActionSource.CONTROLLER,
                threshold=self.config.bootstrap.locked_context_threshold,
            )
            context_obs = self._execute_sensor_action(state, image, context_action)
            state.semantic_memory.record_execution(
                context_action,
                context_obs.call_id,
                new_nodes=0,
                runtime_ms=context_obs.runtime_ms,
            )
            state.search_region_call_id = context_obs.call_id
            context_region = self._enclosing_detection_region(context_obs.detections, img_w, img_h)
            if context_region is not None:
                state.search_region = context_region
                state.search_region_locked = True
                state.search_region_source = f"SAM3_CONTEXT:{context_prompt}"
            elif self.config.bootstrap.locked_context_fallback_full_image:
                state.search_region = full_image
                state.search_region_locked = True
                state.search_region_source = f"SAM3_CONTEXT_FALLBACK:{context_prompt}"
                state.search_region_fallback_used = True
            else:
                raise RuntimeError(f"Locked context prompt {context_prompt!r} returned no usable region.")
            if self.recorder:
                self.recorder.record_semantic_memory_updated(state.semantic_memory.to_dict())
                self.recorder.record_controller_state_updated({
                    "search_region": state.search_region.bbox().as_tuple(),
                    "search_region_locked": state.search_region_locked,
                    "search_region_source": state.search_region_source,
                    "search_region_fallback_used": state.search_region_fallback_used,
                    "search_region_call_id": state.search_region_call_id,
                })

        # Pass 1: target bootstrap across the active search domain.
        global_action = SensingAction(
            action_id=self.id_gen.next_action_id(),
            semantic_key=state.target_class,
            prompt=user_prompt,
            family=ActionFamily.DISCOVERY,
            spatial_mode=SpatialMode.GLOBAL,
            source=ActionSource.USER_BOOTSTRAP,
            search_region=state.search_region,
            threshold=self.config.sam3.default_threshold,
            semantic_prior={state.target_class: 1.0},
        )
        obs_global = self._execute_sensor_action(state, image, global_action)
        self._associate_discovery(state, global_action, obs_global)

        pseudo = select_target_pseudoexemplars(
            state.graph,
            max_count=self.config.bootstrap.pseudoexemplar_max_count,
            min_score=self.config.bootstrap.pseudoexemplar_min_score,
        )

        # Pass 2: same target text + strong seed boxes. This happens before
        # Qwen so semantic planning sees the visually refined bootstrap.
        if self.config.bootstrap.enable_pseudoexemplar_refinement and pseudo.node_ids:
            refine_action = SensingAction(
                action_id=self.id_gen.next_action_id(),
                semantic_key="target",
                prompt=user_prompt,
                family=ActionFamily.DISCOVERY,
                spatial_mode=SpatialMode.GLOBAL,
                source=ActionSource.USER_BOOTSTRAP,
                search_region=state.search_region,
                threshold=self.config.sam3.default_threshold,
                semantic_prior={"target": 1.0},
                correlation_group="target",
                positive_exemplar_ids=pseudo.node_ids,
                positive_exemplar_boxes=pseudo.boxes,
            )
            obs_refined = self._execute_sensor_action(state, image, refine_action)
            self._associate_discovery(state, refine_action, obs_refined)
            pseudo = select_target_pseudoexemplars(
                state.graph,
                max_count=self.config.bootstrap.pseudoexemplar_max_count,
                min_score=self.config.bootstrap.pseudoexemplar_min_score,
            )

        # Pass 3: optional same-prompt tiling, strictly inside the same domain.
        domain_box = state.search_region.bbox()
        tiling_decision = self.tiling_policy.evaluate_tiling(
            image_width=max(1, int(round(domain_box.width))),
            image_height=max(1, int(round(domain_box.height))),
            config=self.config.tiling,
            graph=state.graph,
        )
        tiled_executed = False
        if tiling_decision.should_tile and self.config.bootstrap.enable_tiled_bootstrap:
            tiled_action = SensingAction(
                action_id=self.id_gen.next_action_id(),
                semantic_key=state.target_class,
                prompt=user_prompt,
                family=ActionFamily.DISCOVERY,
                spatial_mode=SpatialMode.TILED,
                source=ActionSource.USER_BOOTSTRAP,
                search_region=state.search_region,
                tiling=self.config.tiling,
                threshold=self.config.sam3.default_threshold,
                semantic_prior={state.target_class: 1.0},
                correlation_group=state.target_class,
            )
            if self.config.bootstrap.enable_pseudoexemplar_refinement and pseudo.node_ids:
                tiled_action = replace(
                    tiled_action,
                    positive_exemplar_ids=pseudo.node_ids,
                    positive_exemplar_boxes=pseudo.boxes,
                )
            obs_tiled = self._execute_sensor_action(state, image, tiled_action)
            assoc_tiled = self._associate_discovery(state, tiled_action, obs_tiled)
            state.discovery_state.tiled_bootstrap_gain = float(len(assoc_tiled.new_nodes))
            tiled_executed = True

        if self.recorder:
            self.recorder.record_discovery_state_updated(state.discovery_state.to_dict())
            self.recorder.record_budget_updated(state.budget.__dict__)

        contact_sheet = ContactSheetBuilder().build_contact_sheet(
            graph=state.graph,
            max_crops=24,
            image=image,
            assets_dir=self.config.assets_dir,
            image_id=image_id,
            semantic_memory=state.semantic_memory,
            target_class=state.target_class,
        )

        image_path_str = None
        if image is not None:
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
            target_class=state.target_class,
            contact_sheet=contact_sheet,
            image_path=image_path_str,
            scene_summary=f"Bootstrap complete. Total candidates: {contact_sheet.total_candidates}.",
            discovery_diagnostics={
                "sam3_calls": state.budget.sam3_calls,
                "active_nodes": len(state.graph.active_nodes()),
                "tiled_bootstrap_executed": tiled_executed,
                "coverage_ratio": state.discovery_state.spatial_coverage.coverage_ratio,
                "search_region": state.search_region.bbox().as_tuple(),
                "search_region_locked": state.search_region_locked,
                "search_region_source": state.search_region_source,
                "search_region_fallback_used": state.search_region_fallback_used,
            },
            belief_classes=list(state.belief_classes),
            confounder_labels=dict(state.confounder_labels),
        )
        return BootstrapResult(state=state, qwen_evidence_pack=qwen_evidence_pack)
