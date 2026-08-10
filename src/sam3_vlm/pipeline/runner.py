"""Runner state machine definitions and implementation (V4 Design Spec §24)."""

import time
from enum import Enum
from typing import Any, List, Optional
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import ActionSource
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.models.qwen import QwenPlanner
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import QwenPlannerService
from sam3_vlm.planning.stopping import (
    CompositeStoppingCondition,
    BudgetStoppingCondition,
    DiscoveryPlateauStoppingCondition,
    IterationStoppingCondition,
)
from sam3_vlm.planning.utility import DefaultUtilityEvaluator
from sam3_vlm.scene.association import AssociationPolicy, IoUAssociationPolicy
from sam3_vlm.scene.belief import SemanticMemory, BeliefUpdater
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import SceneState
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.sensing.tiling import TilingPolicy


class RunnerState(str, Enum):
    """Explicit pipeline runner stage enum."""

    INITIALIZE = "INITIALIZE"
    BOOTSTRAP_GLOBAL = "BOOTSTRAP_GLOBAL"
    BOOTSTRAP_TILE_DECISION = "BOOTSTRAP_TILE_DECISION"
    BOOTSTRAP_TILED = "BOOTSTRAP_TILED"
    BUILD_QWEN_EVIDENCE = "BUILD_QWEN_EVIDENCE"
    PLAN = "PLAN"
    GLOBAL_SENSING = "GLOBAL_SENSING"
    REPLAN = "REPLAN"
    CLEANUP_DECISION = "CLEANUP_DECISION"
    CLEANUP = "CLEANUP"
    FINALIZE = "FINALIZE"
    DONE = "DONE"


class Runner:
    """Central state machine orchestrating SAM3 and Qwen across the active sensing loop."""

    def __init__(self, config: V4Config, sensor: SAM3Sensor, planner: QwenPlanner):
        self.config = config
        self.sensor = sensor
        self.planner = planner
        self.state = RunnerState.INITIALIZE
        self.scene_state: Optional[SceneState] = None
        self.id_gen = IDGenerator()
        
        # Sub-components
        self.bootstrap = BootstrapPipeline(sensor, config=config, id_gen=self.id_gen)
        self.planner_service = QwenPlannerService(planner)
        self.bank_generator = ActionBankGenerator()
        self.association_policy = IoUAssociationPolicy()
        self.belief_updater = BeliefUpdater()
        self.utility_evaluator = DefaultUtilityEvaluator()
        
        self.stopping_condition = CompositeStoppingCondition([
            BudgetStoppingCondition(),
            DiscoveryPlateauStoppingCondition(),
            IterationStoppingCondition(),
        ])

    def run(self, image: Any, user_prompt: str, target_class: str = "target", image_id: str = "img1") -> float:
        """Execute the active perception state machine until DONE."""
        self.image = image
        self.user_prompt = user_prompt
        self.target_class = target_class
        self.image_id = image_id
        
        while self.state != RunnerState.DONE:
            self._step()
            
        return self._compute_final_count()

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
            self.state = RunnerState.PLAN
            
        elif self.state == RunnerState.PLAN:
            self._execute_replan()
            self.state = RunnerState.GLOBAL_SENSING
            
        elif self.state == RunnerState.GLOBAL_SENSING:
            # 1. Recompute utility & 2. Choose one action
            best_entry = self._choose_best_action()
            
            if not best_entry:
                self.state = RunnerState.REPLAN
                return
                
            # 3. Validate budget BEFORE execution
            budget = self.scene_state.budget
            cfg_budget = self.config.budget
            if budget.sam3_calls >= cfg_budget.max_sam3_calls:
                self.state = RunnerState.CLEANUP_DECISION
                return
            if cfg_budget.max_runtime_seconds and budget.total_runtime_ms / 1000.0 >= cfg_budget.max_runtime_seconds:
                self.state = RunnerState.CLEANUP_DECISION
                return
                
            action = best_entry.action
            from sam3_vlm.core.types import SpatialMode, ActionFamily
            
            # Predict tiles if tiled (approx for budget check)
            predicted_tiles = 1
            if action.spatial_mode == SpatialMode.TILED and action.tiling:
                predicted_tiles = action.tiling.grid_rows * action.tiling.grid_cols
                if budget.sam3_tiles + predicted_tiles > cfg_budget.max_sam3_tiles:
                    best_entry.invalid_reason = "Budget Exceeded"
                    return # skip this action because it exceeds tile budget
                    
            # 4. Execute SAM3 once
            observation = self.sensor.observe(self.image, action)
            best_entry.executed = True  # Now it's executed
            
            # Update budgets
            budget.sam3_calls += 1
            if action.spatial_mode == SpatialMode.TILED:
                budget.sam3_tiles += predicted_tiles
            budget.model_runtime_ms += observation.runtime_ms
            budget.total_runtime_ms += observation.runtime_ms
            
            new_nodes_count = 0
            
            # 5-8. Associate, Create Nodes, Update Beliefs
            if action.family == ActionFamily.CONTEXT:
                # Intercept CONTEXT actions to avoid creating countable target nodes
                # Defer logic to future M5 context handling.
                pass
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
                
                new_nodes_count = len(assoc_result.new_nodes)
                
                # Scene-wide projection (M4/M5 interface)
                # First, collect matched node IDs and new node IDs
                matched_node_ids = {nid for nid, _ in assoc_result.matched_observations}
                new_node_ids = {n.node_id for n in assoc_result.new_nodes}
                
                for node_id, obs_ref in assoc_result.matched_observations:
                    node = self.scene_state.graph.get_node(node_id)
                    if node:
                        self.belief_updater.update_node_belief(
                            node, action, obs_ref, target_class=self.target_class
                        )
                for new_node in assoc_result.new_nodes:
                    self.belief_updater.update_node_belief(
                        new_node, action, new_node.observations[-1], target_class=self.target_class
                    )
                    
                # Project absences for unmatched active nodes
                from sam3_vlm.core.types import ObservationRelation, NodeObservationRef
                for node in self.scene_state.graph.active_nodes():
                    if node.node_id not in matched_node_ids and node.node_id not in new_node_ids:
                        # Determine if node was within search ROI. 
                        # GLOBAL and TILED (without explicit sub-ROI) cover the entire scene.
                        # For M4, if action is GLOBAL/TILED, we assign NOT_RETRIEVED. Otherwise NOT_OBSERVABLE.
                        if action.spatial_mode in (SpatialMode.GLOBAL, SpatialMode.TILED):
                            relation = ObservationRelation.NOT_RETRIEVED
                        else:
                            relation = ObservationRelation.NOT_OBSERVABLE
                            
                        # Append observation
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
                            node, action, obs_ref, target_class=self.target_class
                        )

            # Recompute soft count and variance
            mean_count = 0.0
            variance = 0.0
            for node in self.scene_state.graph.active_nodes():
                p = node.class_belief.probabilities.get(self.target_class, 0.0)
                mean_count += p
                variance += p * (1.0 - p)
            self.scene_state.count_estimate.mean_count = mean_count
            self.scene_state.count_estimate.variance = variance
            self.scene_state.count_estimate.std_dev = variance ** 0.5
                    
            # 10. Update semantic memory
            self.scene_state.semantic_memory.record_execution(
                action=action, 
                sam3_call_id=observation.call_id,
                new_nodes=new_nodes_count,
                runtime_ms=observation.runtime_ms,
                predicted_utility=best_entry.total_utility if best_entry.total_utility else 0.0,
            )
            
            # 11. Update discovery state
            self.scene_state.discovery_state.recent_new_node_counts.append(float(new_nodes_count))
            if new_nodes_count > 0 and action.family != ActionFamily.CONTEXT: # skip context for recent_new_nodes
                self.scene_state.discovery_state.recent_new_nodes.extend([n.node_id for n in assoc_result.new_nodes])
            
            # 13. Evaluate replanning triggers & 14. Global stopping
            self.scene_state.iteration += 1
            
            if self.stopping_condition.should_stop(self.scene_state, self.config):
                self.state = RunnerState.CLEANUP_DECISION
                
        elif self.state == RunnerState.REPLAN:
            if self.scene_state.budget.qwen_calls >= self.config.budget.max_qwen_calls:
                self.state = RunnerState.CLEANUP_DECISION
            else:
                self._execute_replan()
                self.state = RunnerState.GLOBAL_SENSING
                
        elif self.state == RunnerState.CLEANUP_DECISION:
            # Skip cleanup for M4, proceed to FINALIZE
            self.state = RunnerState.FINALIZE
            
        elif self.state == RunnerState.FINALIZE:
            self.state = RunnerState.DONE

    def _execute_replan(self):
        """Execute Qwen planning and update action bank."""
        planner_output = self.planner_service.plan_scene(self.evidence_pack, self.scene_state.budget, self.config)
        valid_node_ids = {n.node_id for n in self.scene_state.graph.active_nodes()}
        new_entries = self.bank_generator.generate_entries(
            planner_output,
            self.scene_state.semantic_memory,
            self.scene_state.action_bank,
            self.id_gen,
            valid_node_ids=valid_node_ids
        )
        self.scene_state.qwen_round += 1

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

    def _compute_final_count(self) -> float:
        """Compute the final soft count from graph beliefs."""
        if not self.scene_state:
            return 0.0
            
        total = 0.0
        for node in self.scene_state.graph.active_nodes():
            # For M4, use simple sum of target class probabilities if it exists, else 0
            if self.target_class in node.class_belief.probabilities:
                total += node.class_belief.probabilities[self.target_class]
            else:
                # Fallback naive assumption if belief hasn't been updated properly
                total += 1.0 if node.diagnostics.support_count > 0 else 0.0
        return total
