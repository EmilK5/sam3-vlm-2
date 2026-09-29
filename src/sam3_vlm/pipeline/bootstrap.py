"""Full-image target bootstrap, refinement and adaptive spatial sensing."""

from dataclasses import asdict, dataclass, replace
import time
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
from sam3_vlm.sensing.adaptive_tiling import plan_adaptive_tiles


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
    """Bootstrap without Qwen or ROI selection; every search uses the input target."""

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
            if config.association.enable_iom_dedup or config.association.mask_only
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

    def _execute_sensor_action(self, state: SceneState, image: Any, action: SensingAction):
        if self.recorder:
            self.recorder.record_sam3_action_selected(action.action_id, action.semantic_key)
            self.recorder.record_sam3_action_started(action.action_id)
        budget = self.config.budget
        if state.budget.sam3_calls >= budget.max_sam3_calls:
            raise RuntimeError("SAM3 budget exhausted during bootstrap; partial artifacts retained")
        if action.tile_id and budget.max_sam3_tiles is not None and state.budget.sam3_tiles >= budget.max_sam3_tiles:
            raise RuntimeError("Tile budget exhausted during bootstrap; partial artifacts retained")
        if budget.max_runtime_seconds is not None and max(
            time.perf_counter() - self._bootstrap_start, state.budget.total_runtime_ms / 1000
        ) >= budget.max_runtime_seconds:
            raise RuntimeError("Runtime budget exhausted during bootstrap; partial artifacts retained")
        observation = self.sensor.observe(image, action)
        predicted_tiles = (
            action.tiling.grid_rows * action.tiling.grid_cols
            if action.spatial_mode == SpatialMode.TILED and action.tiling else int(action.tile_id is not None)
        )
        state.budget.sam3_calls += 1
        state.budget.sam3_tiles += predicted_tiles
        state.budget.sam3_runtime_ms += observation.runtime_ms
        state.budget.model_runtime_ms += observation.runtime_ms
        state.budget.total_runtime_ms += observation.runtime_ms
        if self.recorder:
            self.recorder.record_sam3_observation(action, observation)
            self.recorder.record_budget_updated(state.budget.__dict__)
        return observation

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
        # Association can change overlap diagnostics on existing nodes that
        # receive no detection. They still need an event for exact replay.
        previous_diagnostics = (
            {node.node_id: node.to_dict()["diagnostics"] for node in state.graph.active_nodes()}
            if self.recorder else {}
        )
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
        if self.recorder:
            matched_ids = {node_id for node_id, _ in result.matched_observations}
            for node_id, diagnostics in previous_diagnostics.items():
                node = state.graph.get_node(node_id)
                if (
                    node is not None
                    and node_id not in matched_ids
                    and node.to_dict()["diagnostics"] != diagnostics
                ):
                    self.recorder.record_node_updated(
                        node_id,
                        node.to_dict(),
                        {
                            "action_id": action.action_id,
                            "sam3_call_id": observation.call_id,
                            "semantic_key": action.semantic_key,
                            "reason": "ASSOCIATION_DIAGNOSTICS_CHANGED",
                        },
                    )
        state.discovery_state.record_search(observation.searched_regions, state.search_region)
        state.discovery_state.record_discovery_gain(
            len(result.new_nodes),
            [node.node_id for node in result.new_nodes],
            plateau_window=max(
                1, self.config.replanning.discovery_plateau_steps
            ),
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

        self._bootstrap_start = time.perf_counter()

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

        adaptive_plan = None
        if self.config.tiling.enable_adaptive:
            seeds = [node for node in state.graph.active_nodes()
                     if max((obs.score or 0 for obs in node.observations), default=0)
                     >= self.config.tiling.adaptive_seed_min_score]
            boxes = [tuple(int(v) for v in node.geometry.bbox().as_tuple()) for node in seeds]
            adaptive_plan = plan_adaptive_tiles(
                boxes, img_w, img_h,
                density_threshold=self.config.tiling.adaptive_density_threshold,
                min_tile_size=self.config.tiling.adaptive_min_tile_size,
                max_tile_size=self.config.tiling.adaptive_max_tile_size,
            )
            state.discovery_state.adaptive_tiling = asdict(adaptive_plan)
            if self.recorder:
                self.recorder.record_discovery_state_updated(state.discovery_state.to_dict())

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
        if adaptive_plan is not None and self.config.bootstrap.enable_tiled_bootstrap:
            gain = 0
            for index, coords in enumerate(adaptive_plan.tiles):
                if coords == (0, 0, img_w, img_h):
                    continue  # The identical text-only full-image search already ran.
                tile_action = SensingAction(
                    action_id=self.id_gen.next_action_id(),
                    semantic_key=state.target_class, prompt=user_prompt,
                    family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.LOCAL,
                    source=ActionSource.USER_BOOTSTRAP,
                    roi=BoxGeometry(Box(*coords)), tile_id=f"adaptive_{index:03d}",
                    threshold=self.config.sam3.default_threshold,
                    semantic_prior={state.target_class: 1.0},
                    correlation_group=state.target_class,
                    positive_exemplar_ids=pseudo.node_ids if self.config.bootstrap.enable_pseudoexemplar_refinement else (),
                    positive_exemplar_boxes=pseudo.boxes if self.config.bootstrap.enable_pseudoexemplar_refinement else (),
                )
                tile_obs = self._execute_sensor_action(state, image, tile_action)
                gain += len(self._associate_discovery(state, tile_action, tile_obs).new_nodes)
                tiled_executed = True
            state.discovery_state.tiled_bootstrap_gain = float(gain)
        elif tiling_decision.should_tile and self.config.bootstrap.enable_tiled_bootstrap:
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
                "adaptive_tiling": asdict(adaptive_plan) if adaptive_plan is not None else None,
                "coverage_ratio": state.discovery_state.spatial_coverage.coverage_ratio,
                "discovery_saturated": state.discovery_state.saturated,
                "plateau_score": state.discovery_state.plateau_score,
                "tried_sam3_prompts": [
                    prompt
                    for record in state.semantic_memory.records.values()
                    for prompt in record.prompts
                ],
                "search_region": state.search_region.bbox().as_tuple(),
                "search_region_locked": state.search_region_locked,
                "search_region_source": state.search_region_source,
                "search_region_fallback_used": state.search_region_fallback_used,
            },
            belief_classes=list(state.belief_classes),
            confounder_labels=dict(state.confounder_labels),
        )
        return BootstrapResult(state=state, qwen_evidence_pack=qwen_evidence_pack)
