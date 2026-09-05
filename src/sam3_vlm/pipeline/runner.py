"""Runner state machine definitions and implementation (V4 Design Spec §24)."""

import dataclasses
import time
from enum import Enum
from typing import Any, List, Optional
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import (
    ActionFamily,
    ActionSource,
    NodeObservationRef,
    ObservationRelation,
    SpatialMode,
    StopReason,
)
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.models.qwen import QwenPlanner
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import QwenPlannerService
from sam3_vlm.planning.stopping import (
    CompositeStoppingCondition,
    BudgetStoppingCondition,
    DiscoveryAndUncertaintySaturatedStoppingCondition,
    IterationStoppingCondition,
)
from sam3_vlm.planning.replanning import (
    ReplanningPolicy,
    ReplanEvidenceBuilder,
    discovery_is_plateaued,
)
from sam3_vlm.planning.utility import DefaultUtilityEvaluator
from sam3_vlm.pipeline.cleanup import CleanupController
from sam3_vlm.scene.association import AssociationPolicy, IoUAssociationPolicy
from sam3_vlm.scene.association_dual import IoUIoMAssociationPolicy
from sam3_vlm.scene.belief import SemanticMemory, BeliefUpdater
from sam3_vlm.scene.exemplars import select_target_pseudoexemplars
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import SceneState, CountEstimator
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.sensing.evidence import ContactSheetBuilder
from sam3_vlm.sensing.tiling import TilingPolicy
from sam3_vlm.core.geometry import Box


class RunnerState(str, Enum):
    """Explicit pipeline runner stage enum."""

    INITIALIZE = "INITIALIZE"
    BOOTSTRAP_GLOBAL = "BOOTSTRAP_GLOBAL"
    PLAN = "PLAN"
    GLOBAL_SENSING = "GLOBAL_SENSING"
    ASSESS = "ASSESS"
    REPLAN = "REPLAN"
    CLEANUP = "CLEANUP"
    ASSESS_CLEANUP = "ASSESS_CLEANUP"
    FINALIZE = "FINALIZE"
    DONE = "DONE"


class Runner:
    """Central state machine orchestrating SAM3 and Qwen across the active sensing loop."""

    def __init__(self, config: V4Config, sensor: SAM3Sensor, planner: QwenPlanner, recorder=None):
        self.config = config
        self.sensor = sensor
        self.planner = planner
        self.recorder = recorder
        self.state = RunnerState.INITIALIZE
        self.scene_state: Optional[SceneState] = None
        self.id_gen = IDGenerator()
        self.target_class = "target"
        self._pending_replan_reason: Optional[str] = None
        
        # Sub-components
        self.bootstrap = BootstrapPipeline(sensor, config=config, id_gen=self.id_gen, recorder=self.recorder)
        self.planner_service = QwenPlannerService(planner)
        self.bank_generator = ActionBankGenerator()
        self.association_policy = (
            IoUIoMAssociationPolicy()
            if self.config.association.enable_iom_dedup
            else IoUAssociationPolicy()
        )
        self.belief_updater = BeliefUpdater()
        self.utility_evaluator = DefaultUtilityEvaluator()
        
        self.stopping_condition = CompositeStoppingCondition([
            BudgetStoppingCondition(),
            DiscoveryAndUncertaintySaturatedStoppingCondition(),
            IterationStoppingCondition(),
        ])
        
        self.replanning_policy = ReplanningPolicy()
        self.replan_evidence_builder = ReplanEvidenceBuilder(ContactSheetBuilder())
        self.cleanup_controller = CleanupController(self.id_gen)

    def run(self, image: Any, user_prompt: str, target_class: str = "target", image_id: str = "img1") -> float:
        """Execute the active perception state machine until DONE."""
        self._run_start_perf = time.perf_counter()
        self.image = image
        self.user_prompt = user_prompt
        self.target_class = target_class
        self.image_id = image_id
        
        if self.recorder:
            self.recorder.update_manifest({
                "image_id": image_id,
                "user_prompt": user_prompt,
                "target_class": target_class,
                "v4_config": dataclasses.asdict(self.config)
            })
            self.recorder.record_run_started()
            self.recorder.record_bootstrap_started()
            
        try:
            while self.state != RunnerState.DONE:
                self._step()
                
            final_count = self._compute_final_count()
            self._finalize_runtime_accounting()

            # _finalize_runtime_accounting() mutates persistent BudgetState after
            # the final model-action budget event. Persist the finalized budget so
            # deterministic replay sees exactly the same terminal state.
            if self.recorder:
                self.recorder.record_budget_updated(
                    self.scene_state.budget.__dict__
                )

                # Also persist the terminal controller state after all final mutations.
                self._record_controller_state()
                
            if self.recorder:
                from sam3_vlm.logging.schema import RunSummary
                summary = RunSummary(
                    run_id=self.recorder.manifest.run_id,
                    final_soft_count=final_count,
                    count_variance=self.scene_state.count_estimate.variance,
                    final_stop_reason=self.scene_state.stop_reason.value if self.scene_state.stop_reason else None,
                    node_count=len(self.scene_state.graph.nodes),
                    qwen_calls=self.scene_state.budget.qwen_calls,
                    sam3_calls=self.scene_state.budget.sam3_calls,
                    sam3_tiles=self.scene_state.budget.sam3_tiles,
                    cleanup_calls=self.scene_state.budget.cleanup_calls,
                    runtime_ms=self.scene_state.budget.wall_runtime_ms,
                    wall_runtime_ms=self.scene_state.budget.wall_runtime_ms,
                    sam3_runtime_ms=self.scene_state.budget.sam3_runtime_ms,
                    qwen_runtime_ms=self.scene_state.budget.qwen_runtime_ms,
                    controller_runtime_ms=self.scene_state.budget.controller_runtime_ms,
                    number_of_replans=self.scene_state.replans_executed,
                    discovery_statistics={
                        "coverage_ratio": self.scene_state.discovery_state.spatial_coverage.coverage_ratio,
                        "saturated": discovery_is_plateaued(
                            self.scene_state, self.config
                        ),
                        "raw_soft_count": self.scene_state.count_estimate.raw_soft_count,
                        "target_count_commit_threshold": self.config.belief.target_count_commit_threshold,
                        "committed_target_nodes": self.scene_state.count_estimate.committed_node_count,
                    }
                )
                # Serialize graph properly
                final_graph_dict = self.scene_state.graph.to_dict()
                self.recorder.finalize_success(summary, final_graph_dict)
                
            return final_count
            
        except Exception as e:
            if self.recorder:
                self.recorder.record_run_failed(str(e))
            raise e

    def _record_controller_state(self):
        if self.recorder and self.scene_state:
           self.recorder.record_controller_state_updated({
                "iteration": self.scene_state.iteration,
                "qwen_round": self.scene_state.qwen_round,
                "replans_executed": self.scene_state.replans_executed,
                "actions_since_replan": self.scene_state.actions_since_replan,

                "last_plan_accepted_actions": (
                    self.scene_state.last_plan_accepted_actions
                ),

                "last_plan_action_ids": list(
                    self.scene_state.last_plan_action_ids
                ),

                "belief_classes": list(
                    self.scene_state.belief_classes or []
                ),

                "confounder_labels": dict(
                    self.scene_state.confounder_labels or {}
                ),

                "search_region": (
                    self.scene_state.search_region.bbox().as_tuple()
                    if self.scene_state.search_region is not None
                    else None
                ),

                "search_region_locked": (
                    self.scene_state.search_region_locked
                ),

                "search_region_source": (
                    self.scene_state.search_region_source
                ),

                "search_region_fallback_used": (
                    self.scene_state.search_region_fallback_used
                ),

                "search_region_call_id": (
                    self.scene_state.search_region_call_id
                ),
            }) 

    def _uses_canonical_m8_policy(self) -> bool:
        return bool(
            self.scene_state and self.scene_state.uses_canonical_m8_policy
        )

    def _belief_class_vocabulary(self):
        return (
            self.scene_state.belief_classes
            if self._uses_canonical_m8_policy()
            else None
        )

    def _estimate_count(self):
        return CountEstimator.estimate(
            self.scene_state.graph,
            self.target_class,
            target_commit_threshold=(
                self.config.belief.target_count_commit_threshold
            ),
        )

    def _estimated_tile_count(self, action: 'SensingAction') -> int:
        if action.spatial_mode == SpatialMode.TILED:
            if action.tiling:
                return action.tiling.grid_rows * action.tiling.grid_cols
            return self.config.tiling.grid_rows * self.config.tiling.grid_cols
        return 0

    def _check_hard_budgets(self, predicted_tiles: int = 0) -> Optional[StopReason]:
        """Check all global hard budgets and return a StopReason if any are exhausted."""
        budget = self.scene_state.budget
        cfg_budget = self.config.budget
        
        if budget.sam3_calls >= cfg_budget.max_sam3_calls:
            return StopReason.SAM3_BUDGET
            
        if budget.sam3_tiles + predicted_tiles > cfg_budget.max_sam3_tiles:
            return StopReason.TILE_BUDGET
            
        elapsed_wall_ms = self._elapsed_wall_ms()
        if cfg_budget.max_runtime_seconds and (elapsed_wall_ms / 1000.0) >= cfg_budget.max_runtime_seconds:
            return StopReason.RUNTIME_BUDGET
            
        if self.scene_state.iteration >= self.config.stopping.max_iterations:
            return StopReason.MAX_ITERATIONS
            
        return None

    def _step(self):
        """Execute one transition of the state machine."""
        if self.state == RunnerState.INITIALIZE:
            self.state = RunnerState.BOOTSTRAP_GLOBAL
            
        elif self.state == RunnerState.BOOTSTRAP_GLOBAL:
            # We combine Bootstrap stages via BootstrapPipeline to simplify
            self.bootstrap_result = self.bootstrap.execute_bootstrap(
                image_id=self.image_id,
                image=self.image,
                user_prompt=self.user_prompt,
                target_class=self.target_class
            )
            # Use the state created by BootstrapPipeline
            self.scene_state = self.bootstrap_result.state
            if self.scene_state.action_bank is None:
                self.scene_state.action_bank = ActionBank()
                
            self.evidence_pack = self.bootstrap_result.qwen_evidence_pack
            
            if self.recorder:
                self.recorder.record_bootstrap_completed(len(self.scene_state.graph.nodes))
                self.recorder.record_budget_updated(self.scene_state.budget.__dict__)
                
            self.state = RunnerState.PLAN
            
        elif self.state == RunnerState.PLAN:
            if self.recorder:
                self.recorder.record_qwen_plan_started(self.scene_state.qwen_round)
                
            self._execute_plan()
            
            if self.recorder:
                self.recorder.record_budget_updated(self.scene_state.budget.__dict__)
            if (
                self._uses_canonical_m8_policy()
                and self.scene_state.last_plan_accepted_actions == 0
            ):
                self.scene_state.set_stop_reason(StopReason.NO_VALID_ACTIONS)
                self.state = RunnerState.CLEANUP
            else:
                self.state = RunnerState.GLOBAL_SENSING
            
        elif self.state == RunnerState.GLOBAL_SENSING:
            # 1. Recompute utility & 2. Choose one action
            best_entry = self._choose_best_action()
            
            if not best_entry:
                # Strict M8: if Qwen just produced accepted actions but none
                # clears the numerical utility threshold, there is no new sensor
                # evidence to justify another Qwen call. Stop instead of
                # replanning on an unchanged evidence state.
                if (
                    self._uses_canonical_m8_policy()
                    and self.scene_state.last_plan_accepted_actions > 0
                    and self.scene_state.actions_since_replan == 0
                ):
                    self.scene_state.set_stop_reason(
                        StopReason.ACTION_BANK_EXHAUSTED
                    )
                    if self.recorder:
                        self.recorder.record_stop_decided(
                            self.scene_state.stop_reason.value
                        )
                    self.state = RunnerState.CLEANUP
                else:
                    self._pending_replan_reason = "ACTION_BANK_EXHAUSTED"
                    self.state = RunnerState.REPLAN
                return
                
            action = self._constrain_action_to_search_region(best_entry.action)
            action = self._attach_target_pseudoexemplars(action)
            
            if self.recorder:
                self.recorder.record_sam3_action_selected(action.action_id, action.semantic_key)
                
            # 3. Validate budget BEFORE execution
            predicted_tiles = self._estimated_tile_count(action)
                
            stop_reason = self._check_hard_budgets(predicted_tiles=predicted_tiles)
            if stop_reason:
                if self.recorder:
                    self.recorder.record_stop_decided(stop_reason.value)
                self.scene_state.set_stop_reason(stop_reason)
                self.state = RunnerState.CLEANUP
                return
                
            if self.recorder:
                self.recorder.record_sam3_action_started(action.action_id)
                
            # 4. Execute SAM3 once
            observation = self.sensor.observe(self.image, action)
            best_entry.executed = True  # Now it's executed
            self.scene_state.actions_since_replan += 1
            self._record_controller_state()
            
            budget = self.scene_state.budget
            # Update budgets
            budget.sam3_calls += 1
            budget.sam3_tiles += predicted_tiles
            budget.sam3_runtime_ms += observation.runtime_ms
            budget.model_runtime_ms += observation.runtime_ms
            budget.total_runtime_ms += observation.runtime_ms
            
            if self.recorder:
                self.recorder.record_sam3_observation(action, observation)
                self.recorder.record_budget_updated(budget.__dict__)
            
            new_nodes_count = 0
            
            # Compute pre-update stats for logging
            pre_count = self._estimate_count()
            pre_entropy = sum(n.class_belief.entropy for n in self.scene_state.graph.active_nodes())
            
            # 5-8. Associate, Create Nodes, Update Beliefs
            if action.family == ActionFamily.CONTEXT:
                # Intercept CONTEXT actions to avoid creating countable target nodes
                # Defer logic to future M5 context handling.
                matched_node_ids = set()
                new_node_ids = set()
            else:
                assoc_result = self.association_policy.associate(
                    self.scene_state.graph,
                    observation.detections,
                    observation.call_id,
                    action.action_id,
                    action.semantic_key,
                    self.id_gen,
                    correlation_group=action.correlation_group,
                    config=self.config.association
                )
                
                new_nodes_count, not_retrieved_nodes_count = self._project_observations(
                    action, observation, assoc_result
                )

                if self.recorder:
                    # Association may provisionally construct unmatched nodes.
                    # Report only nodes actually admitted to the counting graph
                    # after the Runner's family-level graph-expansion policy.
                    self.recorder.record_association_completed(
                        action.action_id,
                        len(assoc_result.matched_observations),
                        new_nodes_count,
                    )

            # Recompute soft count and variance
            self.scene_state.count_estimate = self._estimate_count()
            
            post_entropy = sum(n.class_belief.entropy for n in self.scene_state.graph.active_nodes())
            
            if self.recorder:
                self.recorder.record_belief_update_completed(len(self.scene_state.graph.nodes), post_entropy)
            
            # Simple discrimination proxy for logging
            discrimination_proxy = max(0.0, pre_entropy - post_entropy)
                    
            # 10. Update semantic memory
            affected = 0
            if action.family == ActionFamily.CONTEXT:
                affected = 0
            else:
                affected = len(assoc_result.matched_observations) + new_nodes_count + not_retrieved_nodes_count

            self.scene_state.semantic_memory.record_execution(
                action=action, 
                sam3_call_id=observation.call_id,
                new_nodes=new_nodes_count,
                runtime_ms=observation.runtime_ms,
                predicted_utility=best_entry.total_utility if best_entry.total_utility else 0.0,
                affected_nodes=affected,
                entropy_change=post_entropy - pre_entropy,
                variance_change=self.scene_state.count_estimate.variance - pre_count.variance,
                realized_discrimination_proxy=discrimination_proxy,
            )
            if self.recorder:
                records_dict = {k: dataclasses.asdict(v) for k, v in self.scene_state.semantic_memory.records.items()}
                self.recorder.record_semantic_memory_updated({"records": records_dict})
            
            # 11. Update spatial coverage for every non-context sensor search.
            if action.family != ActionFamily.CONTEXT:
                self.scene_state.discovery_state.record_search(
                    observation.searched_regions, self.scene_state.search_region
                )
            # Discovery plateau history MUST include zero-gain discovery executions.
            if action.family == ActionFamily.DISCOVERY:
                self.scene_state.discovery_state.record_discovery_gain(
                    new_nodes_count,
                    [n.node_id for n in assoc_result.new_nodes],
                    plateau_window=max(
                        1, self.config.replanning.discovery_plateau_steps
                    ),
                )
            if self.recorder:
                self.recorder.record_discovery_state_updated(self.scene_state.discovery_state.to_dict())
            
            # 13. Evaluate replanning triggers & 14. Global stopping
            self.scene_state.iteration += 1
            self._record_controller_state()
            
            self.state = RunnerState.ASSESS
                
        elif self.state == RunnerState.ASSESS:
            # Check stopping
            stop_reason = self.stopping_condition.should_stop(self.scene_state, self.config)
            if stop_reason:
                self.scene_state.set_stop_reason(stop_reason)
                self.state = RunnerState.CLEANUP
                return

            # Check replanning
            should_replan, replan_reason = self.replanning_policy.should_replan(self.scene_state, self.config)
            if should_replan:
                self._pending_replan_reason = replan_reason
                self.state = RunnerState.REPLAN
            else:
                self.state = RunnerState.GLOBAL_SENSING
                
        elif self.state == RunnerState.REPLAN:
            if self.recorder:
                self.recorder.record_replan_triggered(
                    self._pending_replan_reason or "ACTION_BANK_EXHAUSTED"
                )
            self._pending_replan_reason = None
            self._request_replan()
                
        elif self.state == RunnerState.CLEANUP:
            if self.recorder:
                self.recorder.record_cleanup_started()
                
            # First evaluate global hard budgets!
            hard_reason = self._check_hard_budgets(predicted_tiles=0)
            if hard_reason:
                if self.recorder:
                    self.recorder.record_stop_decided(hard_reason.value)
                self.scene_state.set_stop_reason(hard_reason)
                self.state = RunnerState.FINALIZE
                return
                
            max_cleanup_calls = self.config.budget.max_cleanup_calls
            if max_cleanup_calls <= 0 and self._uses_canonical_m8_policy():
                # In real M8 cleanup can be deliberately disabled; that is not
                # cleanup budget exhaustion. Frozen generic M6 semantics below
                # remain unchanged.
                if self.scene_state.stop_reason is None:
                    self.scene_state.set_stop_reason(StopReason.NO_VALID_ACTIONS)
                if self.recorder:
                    self.recorder.record_stop_decided(self.scene_state.stop_reason.value)
                self.state = RunnerState.FINALIZE
                return
            if self.scene_state.budget.cleanup_calls >= max_cleanup_calls:
                if self.recorder:
                    self.recorder.record_stop_decided(StopReason.CLEANUP_BUDGET.value)
                self.scene_state.set_stop_reason(StopReason.CLEANUP_BUDGET)
                self.state = RunnerState.FINALIZE
                return

            residual_nodes = self.cleanup_controller.select_residual_nodes(
                self.scene_state.graph, self.config, self.target_class
            )
            
            decision = self.cleanup_controller.generate_cleanup_action(
                residual_nodes, self.scene_state.graph, self.target_class, self.config,
                user_prompt=self.scene_state.user_prompt,
            )
            
            if not decision.action:
                self.scene_state.set_stop_reason(decision.reason)
                if self.recorder:
                    resolved_reason = self.scene_state.stop_reason or decision.reason
                    self.recorder.record_stop_decided(
                        resolved_reason.value if resolved_reason else "CLEANUP_COMPLETE"
                    )
                self.state = RunnerState.FINALIZE
                return
            
            cleanup_action = self._constrain_action_to_search_region(decision.action)
            if self.recorder:
                self.recorder.record_sam3_action_selected(cleanup_action.action_id, cleanup_action.semantic_key)
            
            predicted_tiles = self._estimated_tile_count(cleanup_action)
            stop_reason = self._check_hard_budgets(predicted_tiles=predicted_tiles)
            if stop_reason:
                if self.recorder:
                    self.recorder.record_stop_decided(stop_reason.value)
                self.scene_state.set_stop_reason(stop_reason)
                self.state = RunnerState.FINALIZE
                return

            if self.recorder:
                self.recorder.record_sam3_action_started(cleanup_action.action_id)
                
            # Execute cleanup
            observation = self.sensor.observe(self.image, cleanup_action)
            self.scene_state.actions_since_replan += 1
            self.scene_state.budget.sam3_calls += 1
            self.scene_state.budget.cleanup_calls += 1
            self.scene_state.budget.sam3_tiles += predicted_tiles
            self.scene_state.budget.sam3_runtime_ms += observation.runtime_ms
            self.scene_state.budget.model_runtime_ms += observation.runtime_ms
            self.scene_state.budget.total_runtime_ms += observation.runtime_ms
            self._record_controller_state()
            
            if self.recorder:
                self.recorder.record_sam3_observation(
                    cleanup_action,
                    observation,
                )
                self.recorder.record_cleanup_action_completed(cleanup_action.action_id)
                self.recorder.record_budget_updated(self.scene_state.budget.__dict__)

            assoc_result = self.association_policy.associate(
                self.scene_state.graph,
                observation.detections,
                observation.call_id,
                cleanup_action.action_id,
                cleanup_action.semantic_key,
                self.id_gen,
                correlation_group=cleanup_action.correlation_group,
                config=self.config.association
            )
            
            new_nodes_count, not_retrieved_nodes_count = self._project_observations(
                cleanup_action, observation, assoc_result
            )

            if self.recorder:
                self.recorder.record_association_completed(
                    cleanup_action.action_id,
                    len(assoc_result.matched_observations),
                    new_nodes_count,
                )

            # Compute stats
            pre_count = self.scene_state.count_estimate
            pre_entropy = sum(n.class_belief.entropy for n in self.scene_state.graph.active_nodes())
            
            self.scene_state.count_estimate = self._estimate_count()
            post_entropy = sum(n.class_belief.entropy for n in self.scene_state.graph.active_nodes())
            
            if self.recorder:
                self.recorder.record_belief_update_completed(len(self.scene_state.graph.nodes), post_entropy)
            
            self.scene_state.semantic_memory.record_execution(
                action=cleanup_action,
                sam3_call_id=observation.call_id,
                new_nodes=new_nodes_count,
                runtime_ms=observation.runtime_ms,
                predicted_utility=0.0,
                affected_nodes=len(assoc_result.matched_observations) + new_nodes_count + not_retrieved_nodes_count,
                entropy_change=post_entropy - pre_entropy,
                variance_change=self.scene_state.count_estimate.variance - pre_count.variance,
                realized_discrimination_proxy=max(0.0, pre_entropy - post_entropy)
            )
            if self.recorder:
                records_dict = {k: dataclasses.asdict(v) for k, v in self.scene_state.semantic_memory.records.items()}
                self.recorder.record_semantic_memory_updated({"records": records_dict})
                self.recorder.record_discovery_state_updated(self.scene_state.discovery_state.to_dict())

            self.scene_state.iteration += 1
            self._record_controller_state()
            self.state = RunnerState.ASSESS_CLEANUP

        elif self.state == RunnerState.ASSESS_CLEANUP:
            # Simple loop back to cleanup to see if there are more residuals
            self.state = RunnerState.CLEANUP
            
        elif self.state == RunnerState.FINALIZE:
            self.state = RunnerState.DONE

    def _admitted_new_nodes_for_action(self, action, assoc_result):
        """Return provisional association nodes that may enter the count graph.

        ``IoUAssociationPolicy.associate`` registers unmatched SAM3 detections
        provisionally in the graph.  Graph dimensionality, however, is a
        controller-level scientific decision: discovery experiments may enlarge
        the represented object set, while confounder/verification/cleanup
        experiments are discriminative measurements over the represented set.

        For non-discovery actions we therefore remove provisional unmatched
        nodes immediately.  Their raw SAM3 detections remain fully preserved in
        the SAM3 action-completed artifact, so this does not discard sensor
        provenance; it only prevents confounder/background detections from
        becoming count-bearing object hypotheses.
        """
        provisional_new_nodes = list(assoc_result.new_nodes)
        if action.family == ActionFamily.DISCOVERY:
            return provisional_new_nodes

        for node in provisional_new_nodes:
            self.scene_state.graph.nodes.pop(node.node_id, None)
        return []

    def _project_observations(self, action, observation, assoc_result) -> tuple[int, int]:
        """Project one SAM3 observation onto the represented scene graph.

        Only DISCOVERY actions may admit unmatched detections as new graph
        nodes.  All other action families can still strongly update matched
        nodes and contribute NOT_RETRIEVED/NOT_OBSERVABLE evidence.
        """
        admitted_new_nodes = self._admitted_new_nodes_for_action(action, assoc_result)
        new_nodes_count = len(admitted_new_nodes)
        matched_node_ids = {nid for nid, _ in assoc_result.matched_observations}
        new_node_ids = {n.node_id for n in admitted_new_nodes}

        for node_id, obs_ref in assoc_result.matched_observations:
            node = self.scene_state.graph.get_node(node_id)
            if node:
                self.belief_updater.update_node_belief(
                    node, action, obs_ref, target_class=self.target_class,
                    config=self.config.belief, class_vocabulary=self._belief_class_vocabulary(),
                )
                if self.recorder:
                    prov = {
                        "action_id": action.action_id,
                        "sam3_call_id": observation.call_id,
                        "detection_id": obs_ref.detection_id,
                        "observation_id": obs_ref.observation_id,
                        "semantic_key": action.semantic_key
                    }
                    self.recorder.record_node_updated(node_id, node.to_dict(), prov)

        for new_node in admitted_new_nodes:
            self.belief_updater.update_node_belief(
                new_node, action, new_node.observations[-1], target_class=self.target_class,
                config=self.config.belief, class_vocabulary=self._belief_class_vocabulary(),
            )
            if self.recorder:
                prov = {
                    "action_id": action.action_id,
                    "sam3_call_id": observation.call_id,
                    "detection_id": new_node.observations[-1].detection_id,
                    "observation_id": new_node.observations[-1].observation_id,
                    "semantic_key": action.semantic_key
                }
                self.recorder.record_node_created(new_node.node_id, new_node.to_dict(), prov)

        not_retrieved_nodes_count = 0
        for node in self.scene_state.graph.active_nodes():
            if node.node_id not in matched_node_ids and node.node_id not in new_node_ids:
                relation = ObservationRelation.NOT_OBSERVABLE

                if observation.searched_regions:
                    node_box = node.geometry.bbox()
                    for region in observation.searched_regions:
                        if region.bbox().iou(node_box) > 0.0 or region.bbox().intersection(node_box) > 0.0:
                            relation = ObservationRelation.NOT_RETRIEVED
                            not_retrieved_nodes_count += 1
                            break
                else:
                    if action.spatial_mode in (SpatialMode.GLOBAL, SpatialMode.TILED):
                        relation = ObservationRelation.NOT_RETRIEVED
                        not_retrieved_nodes_count += 1

                obs_ref = NodeObservationRef(
                    observation_id=self.id_gen.next_observation_id(),
                    sam3_call_id=observation.call_id,
                    action_id=action.action_id,
                    semantic_key=action.semantic_key,
                    correlation_group=action.correlation_group,
                    relation=relation,
                    score=0.0
                )
                node.observations.append(obs_ref)
                self.belief_updater.update_node_belief(
                    node, action, obs_ref, target_class=self.target_class,
                    config=self.config.belief, class_vocabulary=self._belief_class_vocabulary(),
                )
                if self.recorder:
                    prov = {
                        "action_id": action.action_id,
                        "sam3_call_id": observation.call_id,
                        "observation_id": obs_ref.observation_id,
                        "semantic_key": action.semantic_key,
                        "relation": obs_ref.relation.value
                    }
                    self.recorder.record_node_updated(node.node_id, node.to_dict(), prov)

        return new_nodes_count, not_retrieved_nodes_count

    def _execute_plan(self, *, is_replan: bool = False) -> None:
        """Run one Qwen round and add its valid actions to the bank."""
        call_id = self.id_gen.next_qwen_call_id()
        planner_output = self.planner_service.plan_scene(self.evidence_pack, self.scene_state.budget, self.config)

        strict_m8 = self._uses_canonical_m8_policy()
        valid_node_ids = {n.node_id for n in self.scene_state.graph.active_nodes()}
        new_entries = self.bank_generator.generate_entries(
            planner_output,
            self.scene_state.semantic_memory,
            self.scene_state.action_bank,
            self.id_gen,
            valid_node_ids=valid_node_ids,
            config=self.config,
            search_region=self.scene_state.search_region,
            enforce_qwen_contract=strict_m8,
            allowed_belief_classes=(
                self.scene_state.belief_classes
                if strict_m8
                else None
            ),
        )
        
        if self.recorder:
            cs_ref = None
            if self.evidence_pack.contact_sheet.contact_sheet_image_path:
                try:
                    with open(self.evidence_pack.contact_sheet.contact_sheet_image_path, "rb") as f:
                        cs_ref = self.recorder.save_contact_sheet_artifact(call_id, f.read())
                except FileNotFoundError:
                    pass
            payload = {
                "qwen_call_id": call_id,
                "qwen_round": self.scene_state.qwen_round,
                "input": {
                    "evidence_pack": self.evidence_pack.to_dict(),
                    "contact_sheet_ref": cs_ref
                },
                "output": planner_output.to_dict(),
                "metadata": {
                    "repair_attempted": self.planner_service.last_repair_attempted,
                    "fallback_used": self.planner_service.last_fallback_used,
                    "qwen_runtime_ms": self.planner_service.last_call_runtime_ms,
                    "contract_diagnostic": self.planner_service.last_contract_diagnostic,
                    "rejections": [r.to_dict() for r in self.bank_generator.last_rejections],
                }
            }
            path = self.recorder.save_qwen_artifact(call_id, payload)
            action_ids = [e.action.action_id for e in new_entries]
            self.recorder.record_qwen_plan_completed(path, action_ids)
            
        self._freeze_confounder_labels(planner_output)
        self.scene_state.last_plan_accepted_actions = len(new_entries)
        self.scene_state.last_plan_action_ids = [
            entry.action.action_id for entry in new_entries
        ]
        if self.recorder:
            invalid_total = sum(1 for e in self.scene_state.action_bank.entries if e.invalid_reason is not None)
            invalid_total += len(self.bank_generator.last_rejections)
            self.recorder.record_action_bank_refreshed(len(self.scene_state.action_bank.entries), invalid_total)
        self.scene_state.action_bank.purge_stale_actions(self.config.stopping.utility_min_threshold)
        self.scene_state.qwen_round += 1
        self.scene_state.actions_since_replan = 0
        if is_replan:
            self.scene_state.replans_executed += 1
        self._record_controller_state()
        
        if self.recorder:
            self.recorder.record_budget_updated(self.scene_state.budget.__dict__)

    def _request_replan(self):
        """Centralized handler preserving frozen M6 behavior outside strict M8."""
        qwen_exhausted = self.scene_state.budget.qwen_calls >= self.config.budget.max_qwen_calls
        replans_exhausted = self.scene_state.replans_executed >= self.config.replanning.max_replans
        strict_m8 = self._uses_canonical_m8_policy()

        if (
            strict_m8
            and self.scene_state.actions_since_replan > 0
            and not self._last_plan_had_marginal_value()
        ):
            # The preceding replan was measured and produced no new discovery
            # or useful discrimination.  Another expensive Qwen call would be
            # based on no materially improved scene state.
            self.scene_state.set_stop_reason(StopReason.LOW_MARGINAL_UTILITY)
            self.state = RunnerState.CLEANUP
            return

        if qwen_exhausted or replans_exhausted:
            best_entry = self._choose_best_action()
            if strict_m8:
                if best_entry is not None:
                    self.state = RunnerState.GLOBAL_SENSING
                else:
                    self.scene_state.set_stop_reason(
                        StopReason.QWEN_BUDGET
                        if qwen_exhausted
                        else StopReason.LOW_MARGINAL_UTILITY
                    )
                    self.state = RunnerState.CLEANUP
            else:
                # Frozen M6 contract: only continue globally when a remaining
                # action has sufficient measured utility; otherwise enter cleanup.
                if (
                    best_entry
                    and best_entry.total_utility
                    and best_entry.total_utility >= self.config.stopping.utility_min_threshold
                ):
                    self.state = RunnerState.GLOBAL_SENSING
                else:
                    self.state = RunnerState.CLEANUP
            return

        if strict_m8 and (
            self.scene_state.replans_executed > 0
            and self.scene_state.last_plan_accepted_actions == 0
            and self.scene_state.actions_since_replan == 0
        ):
            self.scene_state.set_stop_reason(StopReason.NO_VALID_ACTIONS)
            self.state = RunnerState.CLEANUP
            return

        self.evidence_pack = self.replan_evidence_builder.build(
            self.scene_state,
            self.image,
            assets_dir=self.config.assets_dir,
            config=self.config,
        )
        self._execute_plan(is_replan=True)
        if strict_m8 and self.scene_state.last_plan_accepted_actions == 0:
            self.scene_state.set_stop_reason(StopReason.NO_VALID_ACTIONS)
            self.state = RunnerState.CLEANUP
        else:
            self.state = RunnerState.GLOBAL_SENSING

    def _last_plan_had_marginal_value(self) -> bool:
        """Check whether the last target plan materially improved count state."""
        action_ids = set(self.scene_state.last_plan_action_ids or [])
        if not action_ids:
            return False

        for record in self.scene_state.semantic_memory.records.values():
            recorded_action_ids = list(
                getattr(record, "action_ids_by_execution", []) or []
            )
            for index, action_id in enumerate(recorded_action_ids):
                if action_id not in action_ids:
                    continue
                gains = record.new_nodes_by_execution
                if index < len(gains) and gains[index] > (
                    self.config.stopping.discovery_saturation_threshold
                ):
                    return True

                variance_changes = record.variance_change_by_execution
                if index >= len(variance_changes):
                    continue
                change = float(variance_changes[index])
                variance_after = self.scene_state.count_estimate.variance
                variance_before = max(0.0, variance_after - change)
                relative_reduction = max(0.0, -change) / max(
                    variance_before, 1e-12
                )
                if relative_reduction >= (
                    self.config.replanning.min_relative_count_variance_reduction
                ):
                    return True

        return False

    def _choose_best_action(self):
        """Recompute utility for all unexecuted actions and return the best."""
        from sam3_vlm.core.types import SpatialMode
        
        best_entry = None
        best_score = -9999.0
        
        for entry in self.scene_state.action_bank.unexecuted_entries():
            # Defer LOCAL actions for global sensing
            if entry.action.spatial_mode == SpatialMode.LOCAL:
                continue
                
            utility = self.utility_evaluator.evaluate_utility(
                entry, state=self.scene_state, config=self.config
            )
            entry.total_utility = utility.total_utility  # cache for SemanticRecord
            if utility.total_utility > best_score:
                best_score = utility.total_utility
                best_entry = entry
                
        if best_score < self.config.stopping.utility_min_threshold:
            return None
            
        return best_entry

    def _elapsed_wall_ms(self) -> float:
        start = getattr(self, "_run_start_perf", None)
        if start is None:
            return 0.0
        return max(0.0, (time.perf_counter() - start) * 1000.0)

    def _finalize_runtime_accounting(self) -> None:
        if not self.scene_state:
            return
        budget = self.scene_state.budget
        budget.wall_runtime_ms = self._elapsed_wall_ms()
        budget.model_runtime_ms = budget.sam3_runtime_ms + budget.qwen_runtime_ms
        budget.controller_runtime_ms = max(0.0, budget.wall_runtime_ms - budget.model_runtime_ms)
        # Adapter-reported model latency can exceed locally observed wall time
        # (for example in mocks, remote services, or rounded telemetry).  Never
        # let finalization move the aggregate runtime backwards or below either
        # of its authoritative measurements.
        budget.total_runtime_ms = max(
            budget.total_runtime_ms,
            budget.wall_runtime_ms,
            budget.model_runtime_ms,
        )

    def _freeze_confounder_labels(self, planner_output) -> None:
        """Bind generic confounder slots once; replans may not rename them."""
        if not self._uses_canonical_m8_policy():
            return
        slots = [c for c in self.scene_state.belief_classes if c != "target"]
        for slot, label in zip(slots, planner_output.likely_confounders):
            if slot not in self.scene_state.confounder_labels and label:
                self.scene_state.confounder_labels[slot] = str(label)
        if hasattr(self, "evidence_pack"):
            self.evidence_pack.belief_classes = list(self.scene_state.belief_classes)
            self.evidence_pack.confounder_labels = dict(self.scene_state.confounder_labels)

    def _constrain_action_to_search_region(self, action):
        """Enforce the immutable run-level spatial search domain."""
        if not self.scene_state or not self.scene_state.search_region:
            return action
        domain = self.scene_state.search_region.bbox()
        if action.spatial_mode in (SpatialMode.GLOBAL, SpatialMode.TILED):
            return dataclasses.replace(action, search_region=self.scene_state.search_region)
        if action.roi is None:
            return action
        roi = action.roi.bbox() if hasattr(action.roi, "bbox") else action.roi
        x1, y1 = max(domain.x1, roi.x1), max(domain.y1, roi.y1)
        x2, y2 = min(domain.x2, roi.x2), min(domain.y2, roi.y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Controller cleanup ROI lies outside the locked search region.")
        return dataclasses.replace(action, roi=Box(x1=x1, y1=y1, x2=x2, y2=y2))

    def _attach_target_pseudoexemplars(self, action):
        """Ground target-oriented M8 actions in controller-selected visual seeds."""
        if (
            not self.scene_state
            or not self.config.bootstrap.enable_pseudoexemplar_refinement
            or not self._uses_canonical_m8_policy()
            or action.family not in (ActionFamily.DISCOVERY, ActionFamily.VERIFICATION)
        ):
            return action

        # A non-target semantic prior marks a genuinely different concept query;
        # do not silently turn confounder sensing into target sensing.
        prior = action.semantic_prior or {}
        if prior:
            target_weight = float(prior.get("target", 0.0))
            other_weight = max(
                (float(v) for key, v in prior.items() if key != "target"),
                default=0.0,
            )
            if target_weight <= other_weight:
                return action

        pseudo = select_target_pseudoexemplars(
            self.scene_state.graph,
            max_count=self.config.bootstrap.pseudoexemplar_max_count,
            min_score=self.config.bootstrap.pseudoexemplar_min_score,
        )
        if not pseudo.node_ids:
            return action
        return dataclasses.replace(
            action,
            positive_exemplar_ids=pseudo.node_ids,
            positive_exemplar_boxes=pseudo.boxes,
        )

    def _compute_final_count(self) -> float:
        """Compute the final soft count from graph beliefs."""
        if not self.scene_state:
            return 0.0
            
        self.scene_state.count_estimate = self._estimate_count()
        return self.scene_state.count_estimate.mean_count
