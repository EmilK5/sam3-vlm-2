# V4 Design Specification

*Implementation source of truth for the V4 rewrite. This document is intended for a coding/refactor LLM or engineer. Scientific motivation and the compact mathematical narrative live in `V4_PAPER_TLDR.md`.*

## 0. Status and scope

V4 is a clean architectural rewrite of the current V3 research code. V3 remains frozen as a reference implementation and source of tested low-level components. V4 must not preserve legacy abstractions merely for compatibility.

### Current M8 execution policy

The current M8 experiment deliberately uses a narrower policy than the generic
architecture described in the remainder of this document:

- Qwen may identify confounders in its scene analysis, but every executable
  Qwen action must be a novel `target` / `DISCOVERY` prompt.
- Confounders are context for phrasing better target prompts; they are not sent
  to SAM3 as separate experiments.
- The information-value proxy is evaluated only for target actions.
- Each planning round admits at most one action. The production configuration
  permits one replan, for at most two Qwen calls after bootstrap.
- The production planner is the local Ollama alias `qwen3.5-9b-sam3`, built
  from `qwen3.5:9b-q4_K_M` with an 8192-token context and a 512-token output
  limit. Requests use non-thinking JSON mode, a 45-second transport timeout,
  and no hidden client retries.
- A target posterior at or above `0.8` contributes `1.0` to the reported count.
  The posterior itself remains unchanged and the raw soft count is retained.

These rules override older M8 examples below that execute confounder actions.
The generic schemas and belief model retain confounder families so historical
artifacts and non-M8 experiments remain replayable.

The primary change is the unit of control:

- **V3 dominant pattern:** select one unresolved node, generate/choose one verification action, run SAM3 locally, update that node, repeat.
- **V4 pattern:** bootstrap the scene once, let Qwen propose a scene-level semantic action bank, execute selected prompts globally or in batched regions, update the entire graph, replan only when the scene has materially changed, and use local verification only for the residual set.

V4 must support different counting tasks without hard-coding CARPK or citrus logic. Green-citrus counting remains the main research task and should be the most demanding confounder case.

---

# 1. System goals and non-goals

## 1.1 Goals

Given `(image, user_prompt)`, V4 must:

1. Discover as many relevant physical instances as possible without task-specific training labels.
2. Maintain one evolving graph of sensor-grounded object hypotheses.
3. Use Qwen to propose semantic sensing actions after observing actual SAM3 behavior.
4. Execute semantic prompts globally whenever possible so one SAM3 pass can affect many nodes.
5. Treat target-oriented and confounder-oriented prompts as first-class sensing actions.
6. Update semantic beliefs probabilistically or with explicitly documented approximations.
7. Re-query Qwen only when new observations materially change the planning problem.
8. Reserve per-object verification for the small residual subset.
9. Stop based on marginal value, discovery saturation, uncertainty, and hard compute budgets.
10. Produce a soft count, a hard count if requested, uncertainty/diagnostics, complete provenance, and replayable run artifacts.

## 1.2 Non-goals

V4 must not:

- use Qwen-generated boxes as detections;
- call Qwen once per object by default;
- equate SAM3 confidence with target-class probability;
- treat global and tiled executions of the same semantic prompt as independent semantic evidence;
- assume that a confident current graph implies discovery is complete;
- require dataset-specific prompt logic inside the core controller;
- maintain multiple historical runners/policies in the main V4 package;
- solve the full POMDP or exact Bayesian data association in V4.0.

---

# 2. Core architecture

The package should be small and layered:

```text
sam3_vlm/
  models/
    sam3_sensor.py
    qwen_planner.py
  scene/
    state.py
    graph.py
    node.py
    association.py
    belief.py
  sensing/
    action.py
    observation.py
    semantic_key.py
    tiling.py
    evidence.py
  planning/
    action_bank.py
    utility.py
    replanning.py
    stopping.py
  pipeline/
    bootstrap.py
    global_loop.py
    cleanup.py
    runner.py
  datasets/
    base.py
    carpk.py
    citrus.py
    pixmo_count.py
    blood_cells.py
  eval/
    matching.py
    metrics.py
    reporting.py
  logging/
    schema.py
    writer.py
    replay.py
    validator.py
```

Hard layering rules:

- `models/` knows model APIs, not the active-perception algorithm.
- `scene/` knows graph/state structures, not Ollama/vLLM/Gradio.
- `sensing/` defines model-independent actions and observations.
- `planning/` operates on state + candidate actions, not raw model tensors.
- `pipeline/` orchestrates the stages.
- UI/CLI code calls one runner and contains no algorithmic logic.

The only top-level orchestration API should be conceptually:

```python
result = Runner(config).run(image=image, prompt=user_prompt)
```

---

# 3. Scene state schema

## 3.1 Mathematical state

The operational state is

\[
B_t=(G_t,\rho_t,U_t,\mathcal S_t,\mathcal A_t,C_t).
\]

Implementation should map these terms directly to fields rather than hiding state in runner-local variables.

## 3.2 SceneState

Suggested schema:

```python
@dataclass
class SceneState:
    image_id: str
    user_prompt: str
    target_class: str
    graph: SceneGraph
    semantic_memory: SemanticMemory
    discovery_state: DiscoveryState
    action_bank: ActionBank | None
    budget: BudgetState
    iteration: int
    qwen_round: int
```

`SceneState` must be serializable without model objects or GPU tensors.

## 3.3 Node

Each node represents a sensor-grounded object hypothesis, not guaranteed ground truth.

```python
@dataclass
class Node:
    node_id: str
    geometry: Geometry
    class_belief: dict[str, float]
    existence_score: float
    duplicate_risk: float
    merge_risk: float
    observations: list[NodeObservationRef]
    created_by_call_id: str
    status: NodeStatus
```

Required `NodeStatus` values should be minimal, e.g. `ACTIVE`, `RESOLVED`, `REJECTED`, `AMBIGUOUS`.

Do not create citrus-specific fields such as `fruit_score` in the core node schema.

## 3.4 DiscoveryState

V4.0 does not claim a calibrated posterior over missed objects. Maintain explicit diagnostics instead:

```python
@dataclass
class DiscoveryState:
    recent_new_nodes: list[int]
    recent_new_target_mass: list[float]
    spatial_coverage: CoverageSummary
    tiled_bootstrap_gain: float | None
    plateau_score: float
    unresolved_regions: list[Geometry]
    qwen_missing_modes: list[str]
```

The scientific latent quantity is the missed target count `M_t*`; this structure is an approximation used by the controller.

## 3.5 SemanticMemory

Tracks what semantic experiments have already been attempted:

```python
@dataclass
class SemanticRecord:
    semantic_key: str
    prompts: list[str]
    family: ActionFamily
    execution_count: int
    sam3_call_ids: list[str]
    total_cost: float
    new_nodes_by_execution: list[int]
    realized_utility_by_execution: list[float]
```

Near-paraphrases must map to the same or related semantic key when possible.

---

# 4. SAM3 sensor interface

SAM3 must implement a single clean sensor contract:

```python
observation = sam3.observe(image, action)
```

## 4.1 SensingAction

```python
@dataclass(frozen=True)
class SensingAction:
    action_id: str
    semantic_key: str
    prompt: str
    family: ActionFamily
    roi: Geometry | None
    positive_exemplar_ids: tuple[str, ...]
    negative_exemplar_ids: tuple[str, ...]
    threshold: float
    spatial_mode: SpatialMode
    tiling: TilingConfig | None
    source: ActionSource
    qwen_priority: float | None
    semantic_prior: dict[str, float] | None
```

Families:

- `DISCOVERY`
- `CONFOUNDER`
- `CONTEXT`
- `VERIFICATION`

Spatial modes:

- `GLOBAL`
- `TILED`
- `ROI_BATCH`
- `LOCAL`

## 4.2 SAM3Observation

```python
@dataclass
class SAM3Observation:
    call_id: str
    action_id: str
    semantic_key: str
    detections: list[Detection]
    searched_regions: list[Geometry]
    runtime_ms: float
    model_metadata: dict
```

Each `Detection` contains geometry, score, mask reference, tile/source metadata, and no semantic class label invented by the wrapper.

## 4.3 Critical rule

The SAM3 wrapper must not:

- modify graph beliefs;
- create final graph nodes directly;
- call Qwen;
- select the next action;
- decide the count.

It returns sensor observations only.

---

# 5. Bootstrap pipeline

Bootstrap always precedes Qwen.

## 5.1 Global user-prompt pass

Construct a `DISCOVERY` action directly from the user prompt and execute it globally.

Bootstrap threshold is recall-oriented. The exact default should be configurable and dataset-independent.

Register returned detections into an initial graph. They are **candidates**, not positives.

## 5.2 Conditional tiled bootstrap

After the global pass, estimate whether a same-prompt tiled pass is worthwhile using:

- candidate size relative to image size;
- image resolution;
- density/clustering;
- known SAM3 resize behavior;
- confidence distribution;
- uncovered spatial regions;
- optional dataset adapter hints.

If triggered, execute the same semantic key tiled. Tiled/global observations improve retrieval support and may discover new nodes, but must remain semantically correlated.

## 5.3 Contact-sheet construction

Create a bounded Qwen evidence pack with the original image plus representative crops. Suggested default: 16--24 crops drawn from strata:

- high support/confidence;
- medium;
- low/suspicious;
- spatial/appearance outliers.

Each crop annotation includes:

- node ID;
- SAM3 score;
- retrieval support level;
- global/tiled provenance;
- current belief only if clearly labeled as system belief, not truth.

Qwen instructions must explicitly state that the crops are unverified sensor candidates.

## 5.4 Bootstrap output

```python
BootstrapResult(
    state=scene_state,
    qwen_evidence_pack=evidence_pack,
)
```

`bootstrap.py` must not call Qwen internally.

---

# 6. Qwen evidence pack and planner interface

Qwen operates at scene level.

## 6.1 Input

The planner receives:

- original image;
- user counting concept;
- compact contact sheet;
- crop annotations and SAM3 scores;
- current class vocabulary/confounder vocabulary if known;
- recent action-performance summary for replanning calls;
- discovery saturation summary;
- unresolved appearance clusters if available.

Do not send every graph node as verbose JSON.

## 6.2 Output schema

Qwen returns a constrained structured object:

```python
@dataclass
class PlannerOutput:
    scene_summary: str
    proposed_actions: list[ProposedAction]
    missing_appearance_modes: list[str]
    likely_confounders: list[str]
```

Each action:

```python
@dataclass
class ProposedAction:
    semantic_key: str
    prompt: str
    family: ActionFamily
    priority: float
    semantic_prior: dict[str, float]
    suggested_threshold: float | None
    suggested_spatial_mode: SpatialMode
    exemplar_policy: str | None
    rationale: str
```

The structured output should be schema-constrained to eliminate retry-heavy malformed JSON.

## 6.3 Qwen semantic prior

`semantic_prior` approximates quantities such as class-conditional property prevalence. It is not a calibrated SAM3 likelihood and must never be written into the graph as if observed evidence.

Store Qwen's proposed values separately from empirical sensor statistics.

## 6.4 Qwen call budget

Maintain a hard `max_qwen_calls` independent of SAM3 budget. Expected normal behavior:

- easy image: 1 planning call;
- difficult image: 2--3;
- hard maximum: 4 unless explicitly overridden.

No hidden retries beyond the configured endpoint retry count.

---

# 7. Action bank

`ActionBank` is the finite candidate set proposed by Qwen plus optional controller-generated actions.

```python
@dataclass
class ActionBankEntry:
    action: SensingAction
    qwen_priority: float
    predicted_discovery_value: float | None
    predicted_discrimination_value: float | None
    redundancy: float
    estimated_cost: float
    executed: bool
    invalid_reason: str | None
```

## 7.1 Deduplication

Before execution:

- canonicalize semantic keys;
- reject exact prompt duplicates;
- mark near-paraphrases as correlated;
- avoid running multiple trivially equivalent prompts simply because Qwen listed them separately.

## 7.2 Action-source distinction

`ActionSource` should identify:

- `USER_BOOTSTRAP`
- `QWEN`
- `CONTROLLER`
- `CLEANUP`

This is needed for ablations and provenance.

---

# 8. Scene-level utility and action selection

The scientific working objective is

\[
U_t(x)=\alpha_tD_t(x)+\beta_tI_t(x)-\gamma R_t(x)-\lambda C(x)+\eta_tQ_t(x).
\]

V4.0 can implement approximations to each component.

## 8.1 Discovery value `D_t(x)`

Use a predicted or empirical measure of expected new useful hypotheses. Inputs can include:

- Qwen family/priority;
- prior performance of related semantic keys;
- recent discovery gains;
- uncovered spatial regions;
- predicted target affinity;
- whether this appearance mode has been searched.

Do not pretend this is a calibrated expected count unless it is actually modeled as one.

## 8.2 Discrimination value `I_t(x)`

Preferred conceptual quantity:

\[
I(Z_{V_t};\mathbf O_x\mid B_t).
\]

Initial approximation:

\[
I_t(x)\approx\sum_iw_{i,t}I(Z_i;O_{i,x}\mid B_t).
\]

If no calibrated observation channel exists yet, V4.0 may substitute a proxy based on:

- current node entropy;
- semantic prior separation across classes;
- empirical prompt reliability;
- number of affected uncertain nodes.

Name such a quantity `discrimination_proxy`, not `information_gain`, unless it is actually computed probabilistically.

## 8.3 Redundancy `R_t(x)`

Penalize:

- same semantic key already exhausted;
- close paraphrase of recent prompt;
- repeated action with low marginal gain;
- same semantic coordinate with different tiling when spatial value is already saturated.

## 8.4 Cost `C(x)`

Charge actual model execution cost, not number of nodes updated. Track at least:

- SAM3 forward executions;
- tile count / processed pixels;
- runtime;
- optional normalized GPU cost.

Qwen planning cost is accounted separately.

## 8.5 Qwen priority `Q_t(x)`

Use as a semantic prior. It must be overridable by empirical evidence.

Initial implementation can use weighted scoring. Later work may learn or calibrate the weights.

## 8.6 Phase-dependent weighting

Early:

- high discovery weight;
- moderate Qwen prior;
- lower discrimination weight.

Late:

- lower discovery weight after plateau;
- higher count/discrimination weight;
- higher redundancy penalty;
- lower Qwen weight as empirical action history accumulates.

---

# 9. Global multipass controller

The central loop is action-first, not node-first.

Pseudo-logic:

```text
bootstrap
plan with Qwen
while not stopped:
    refresh utility for unexecuted actions
    choose best scene-level action
    execute one SAM3 sensing action
    associate detections to graph
    create genuinely new nodes
    update all affected node beliefs
    update semantic-memory statistics
    update discovery state and budgets
    if replanning trigger fires:
        call Qwen and replace/extend action bank
    if global phase should end:
        break
run residual cleanup if needed
finalize count
```

## 9.1 Global first

`DISCOVERY` and `CONFOUNDER` actions should default to whole-scene execution unless the action is explicitly context-localized or the image is too large.

## 9.2 Tiling

Tiling is a spatial execution decision, not a new semantic action. The controller may choose tiled execution when expected spatial recall gain justifies cost.

Do not independently multiply semantic evidence from global and tiled versions of the same semantic key.

## 9.3 One action, many updates

Every global action must be projected onto all relevant existing nodes. The graph update must not loop by re-running SAM3 per node.

---

# 10. Graph association and registration

Association maps a detection set to existing graph hypotheses and determines which detections can create new nodes.

## 10.1 Requirements

Association must use geometry and, where helpful, masks/appearance embeddings. It must distinguish:

- matched observation of existing node;
- new object hypothesis;
- possible duplicate;
- possible split/merge;
- ambiguous association.

## 10.2 Registration invariant

The counted target class should approach one graph entity per physical object. Confounder/background registration can be looser if they are not directly counted.

## 10.3 V4.0 approximation

Exact probabilistic data association is out of scope. Use deterministic matching plus explicit diagnostics:

- match score;
- duplicate risk;
- merge risk;
- ambiguity flag.

Never silently discard ambiguous cases from provenance.

## 10.4 New-node creation

Only unmatched SAM3 detections can create nodes. Qwen boxes, ROIs, or textual suggestions cannot.

---

# 11. Belief and evidence updates

## 11.1 Class vocabulary

The core engine supports an arbitrary finite class set. For citrus, expected competitors include target citrus, leaf, and background/null. For other tasks, dataset-specific confounder classes can be generated dynamically by Qwen.

## 11.2 Observation categories

Projection of one global SAM3 result onto a node should preserve at least:

- `STRONG_MATCH`
- `WEAK_MATCH`
- `NOT_RETRIEVED`
- `NOT_OBSERVABLE`

Continuous scores should be retained even if the belief layer uses discretized bins.

## 11.3 Presence/absence asymmetry

Do not use symmetric positive/negative votes. SAM3 has imperfect recall. Strong presence under a confounder prompt can be much stronger evidence than absence under a target prompt.

## 11.4 Positive and confounder prompts

Both update beliefs using their class-conditional semantic meaning. A leaf prompt is not a special veto; it is evidence favoring `leaf` relative to `target`.

## 11.5 Semantic-key correlation

Belief fusion must know when multiple observations use the same semantic key or near-equivalent semantic coordinate. V4.0 options, in increasing rigor:

1. hard deduplication: only first/new-best observation contributes semantic evidence;
2. discounted repeat weight;
3. latent-property joint model using the subset-moment formulation.

The implementation must isolate this policy so it can be upgraded without changing the graph API.

## 11.6 Qwen priors versus empirical channel

Store separately:

- Qwen semantic prior `beta_hat_qwen`;
- empirical prompt response statistics;
- actual node observations.

Do not mutate Qwen priors to look like measured sensor probabilities without explicit calibration logic.

---

# 12. Qwen replanning

Replanning is event-driven, not per-node and not blindly every N passes.

Trigger candidates:

- no high-utility unexecuted actions remain;
- best utility falls below threshold while uncertainty remains;
- two or more recent discovery actions yield little new coverage;
- a new recurring confounder pattern appears;
- large unresolved target mass remains;
- remaining ambiguous nodes form a new visual cluster;
- current prompt bank is exhausted.

## 12.1 Replanning evidence pack

Include:

- original image;
- updated representative contact sheet;
- newly discovered candidates;
- likely confounders;
- ambiguous residuals;
- concise table of executed prompts and realized effects;
- discovery plateau metrics.

Do not resend the entire raw event log.

## 12.2 Bank replacement policy

Qwen may extend or replace the bank. Preserve historical actions in semantic memory. New proposals are deduplicated against all past semantic keys.

---

# 13. Residual cleanup

Enter cleanup only after global sensing has low marginal value and unresolved uncertainty is concentrated in a small subset.

## 13.1 Cleanup levels

Use this order:

1. **Shared residual prompt:** Qwen sees an ambiguous-crop contact sheet and proposes a common distinction.
2. **ROI batch:** execute one prompt across all ambiguous ROIs in a batch/tiled composite.
3. **Cluster-specific batch:** group ambiguous nodes by appearance and query each cluster.
4. **Individual local verification:** final fallback only.

## 13.2 Hard guardrail

Per-node Qwen generation must not reappear in cleanup unless explicitly enabled for a diagnostic ablation. Qwen should still reason over the residual set collectively.

---

# 14. Stopping and count output

## 14.1 Stopping dimensions

The controller stops when all of the following are sufficiently satisfied or hard budgets are exhausted:

- residual discovery potential is low;
- count uncertainty is low enough;
- best remaining action has insufficient utility relative to cost;
- cleanup residual is empty or below tolerance.

Hard limits always include separate:

- `max_qwen_calls`;
- `max_sam3_calls`;
- `max_tiles` or equivalent spatial cost;
- `max_runtime_seconds` if enabled.

## 14.2 Soft count

Preferred count:

\[
\hat N=\sum_i e_i\rho_i(k^\star),
\]

with duplicate/merge corrections where implemented.

V4.0 should report:

- soft count;
- optional hard-threshold count;
- number of graph nodes;
- rejected/ambiguous nodes;
- registration diagnostics;
- discovery-saturation diagnostic;
- posterior/count uncertainty if justified by the belief model.

Do not report a mathematically unjustified confidence interval merely because the UI expects one.

## 14.3 Missed-object correction

The scientific formulation includes residual missed count `M_t*`. V4.0 may not have a calibrated estimator. Until it does, expose discovery saturation separately rather than adding an arbitrary correction to the count.

---

# 15. Cost and budget accounting

The previous code conflated pass, Qwen, and SAM3 budgets. V4 must make them explicit.

```python
@dataclass
class BudgetState:
    qwen_calls: int
    sam3_calls: int
    sam3_tiles: int
    model_runtime_ms: float
    total_runtime_ms: float
```

Suggested default research-scale policy:

- Qwen scene planning: max 2--4 calls/image;
- bootstrap SAM3: 1 global + optional 1 tiled;
- global semantic SAM3: ~4--10 useful actions;
- cleanup: a small bounded number of batched/local actions.

Defaults are not claims and should be configurable.

For research plots, always report performance versus actual compute: SAM3 executions, Qwen calls, runtime, and possibly processed pixels/tiles.

---

# 16. Logging, provenance, replay, validation

Keep the compact-logging lessons from V3.

## 16.1 Run artifacts

Each run should contain:

```text
run.json
summary.json
events.jsonl
artifacts/
  masks/*.npz
  contact_sheets/*.png
  qwen/*.json
  graph/final_graph.json
```

Do not embed dense masks or repeated graph snapshots in JSON.

## 16.2 Event types

Minimal events:

- bootstrap started/completed;
- SAM3 action proposed/executed/completed;
- detections associated;
- node created/updated/merged/rejected;
- belief update;
- Qwen planning call/request/response;
- action-bank refresh;
- replanning trigger;
- cleanup transition;
- stop decision;
- final count.

Every graph/belief change must trace back to explicit sensor observations.

## 16.3 Replay

Replay should reconstruct the scene state from logged actions and observations without rerunning models. Model nondeterminism is therefore separated from controller correctness.

## 16.4 Validation

Validator checks:

- schema versions;
- event ordering;
- referenced call/action/node IDs;
- artifact hashes;
- no Qwen-originated graph node;
- budgets respected;
- semantic-key provenance retained;
- final graph reproducible from events.

---

# 17. Dataset interface and evaluation

Core algorithm receives only an image and user concept. Dataset adapters provide evaluation metadata, not control logic.

```python
class CountingDataset(Protocol):
    def images(self) -> Iterable[Sample]: ...
    def user_prompt(self, sample) -> str: ...
    def ground_truth(self, sample) -> GroundTruth | None: ...
```

Optional adapter hints may describe expected image scale or evaluation format, but the semantic prompt bank must not be manually hard-coded per test image.

Target datasets include:

- green citrus (primary);
- CARPK;
- PiXMo-Count;
- blood cells;
- fruit-counting datasets.

Metrics should separate:

- count error;
- target detection precision/recall/F1;
- discovery recall where ground truth permits;
- duplicate/merge errors;
- classification/confounder errors;
- Qwen calls;
- SAM3 calls;
- runtime;
- storage.

---

# 18. Experiments and acceptance tests

## 18.1 Mandatory ablations

Implement configurations, not separate runners, for:

1. one-shot user-prompt SAM3;
2. + tiled bootstrap;
3. fixed global prompt bank;
4. one-shot Qwen bank with no replanning;
5. full Qwen replanning;
6. target prompts only;
7. target + confounder prompts;
8. global vectorized sensing;
9. per-node verification baseline;
10. with/without semantic-key deduplication;
11. with/without residual cleanup;
12. hard count vs soft count.

## 18.2 Unit tests

Required families:

- semantic-key deduplication;
- global/tiled same-key correlation handling;
- action-bank validation;
- budget counters independent from one another;
- Qwen proposals cannot create nodes;
- graph association edge cases;
- presence vs absence evidence;
- `NOT_OBSERVABLE` never used as negative evidence;
- global action updates multiple nodes without multiple SAM3 calls;
- replay reproduces state;
- stop rules respect both discovery and uncertainty.

## 18.3 Integration tests

Use synthetic/mock sensors first:

- known scene with fixed objects;
- configurable recall/false positives;
- controllable prompt informativeness;
- deterministic Qwen bank.

Then real-model tests:

1. one global bootstrap;
2. optional tiled bootstrap;
3. Qwen action bank generation;
4. one global discovery prompt;
5. one confounder prompt;
6. full short global loop;
7. replanning event;
8. cleanup on 2--5 residual nodes;
9. run validation/replay;
10. small multi-dataset sweep.

## 18.4 Acceptance criterion for V4.0

V4.0 is ready for research experiments when:

- all CPU/unit tests pass in a fresh environment;
- one image can run end-to-end with no hidden legacy path;
- Qwen calls are bounded scene-level calls;
- one global SAM3 pass demonstrably updates many nodes;
- positive and confounder actions both affect beliefs;
- run replay/validation succeeds;
- runtime and call counts are visible and plausible;
- no dataset-specific special case is required in the core runner.

---

# 19. Migration from V3

V3 is a reference, not a dependency target.

## 19.1 Components worth extracting after review

Likely reusable concepts/code:

- low-level SAM3 model loading/inference;
- coordinate conversion and tiling;
- mask/box utilities;
- graph geometry matching where clean;
- dataset loaders;
- evaluation matching/metrics;
- compact logging and provenance ideas;
- run validator/replay utilities.

Every extracted component must receive a V4 interface and V4 tests. Do not import large V3 modules simply to reuse one helper.

## 19.2 Components not to carry forward as architecture

Do not preserve as core orchestration:

- `policy_vlm.py` / `policy_vlm_v3.py` abstractions;
- current node-first `QwenAshtRunner`;
- per-node Qwen candidate generation;
- per-node query budgets as the primary global budget;
- legacy VIP/oracle routing as the main inference path;
- many historical experiment modes/runners;
- duplicated dashboard orchestration;
- giant root-level inference modules.

They can remain in V3 for baselines.

## 19.3 Recommended rewrite process

1. Freeze V3 and record the commit hash.
2. Start V4 with the package skeleton and schemas only.
3. Implement deterministic mock SAM3/Qwen adapters.
4. Implement bootstrap and graph registration against mocks.
5. Implement action bank, semantic memory, utility and global loop.
6. Implement belief evidence semantics.
7. Add replanning and cleanup.
8. Add logging/replay.
9. Only then connect real SAM3.
10. Connect Qwen last, with constrained output and hard call limits.
11. Port dataset adapters/evaluation.
12. Run ablations against V3 baselines.

This order ensures the controller can be tested without expensive foundation models.

---

# 20. Mathematical implementation notes

These are the scientific concepts the code must leave room to refine.

## 20.1 Semantic moments

For semantic descriptor set \(S\),

\[
\beta_{k,S}=P(\wedge_{w\in S}a_w=1\mid Z=k).
\]

Qwen can propose initial values/relative rankings. The system must not hard-wire the assumption that they are calibrated.

## 20.2 Separation order

\[
r(k,l)=\min\{|S|:\beta_{k,S}\neq\beta_{l,S}\}.
\]

The architecture must support conjunction-style prompts and confounder prompts because low-order semantic coordinates can be non-discriminative.

## 20.3 Sensor channel

The actual prompt-conditioned SAM3 response law is action-specific. Maintain a place to estimate empirical prompt reliability/channel parameters from pseudoexemplars or benchmark calibration.

## 20.4 Active-testing connection

Per represented node, the posterior is the ASHT-like information state. Scene-level action choice generalizes this by selecting one experiment that produces a vector of node observations and possible new detections.

## 20.5 Discovery extension

The graph is not complete. Therefore pure entropy reduction over current nodes cannot be the sole utility. The implementation must retain a separate discovery state and discovery action family.

## 20.6 Terminal objective

The task is counting. Future versions should support count-risk utility, e.g. expected reduction in posterior count variance per unit cost, rather than requiring every node to be fully classified.

---

# 21. Open issues intentionally not fixed in V4.0

The refactor must not fabricate certainty on these points:

1. Exact calibration of Qwen semantic priors.
2. Exact estimation of arbitrary SAM3 prompt channels.
3. A calibrated posterior over still-undiscovered object count.
4. Exact Bayesian registration/partition inference.
5. Exact scene-level mutual information with correlated nodes.
6. Optimal utility weights \(\alpha,\beta,\gamma,\lambda,\eta\).
7. Optimal trigger for tiled sensing.
8. Optimal number of Qwen replans.

Each should be implemented behind an interface or configurable policy so later research can change it without restructuring the pipeline.

---

# 22. Final architecture summary

The authoritative V4 flow is:

```text
(image, user prompt)
        |
        v
GLOBAL SAM3 BOOTSTRAP
        |
        +--> conditional same-prompt tiling
        |
        v
initial sensor-grounded scene graph
        |
        v
original image + representative candidate contact sheet
        |
        v
QWEN SCENE PLAN
  discovery / confounder / context / verification bank
        |
        v
SCENE-LEVEL CONTROLLER
  rank semantic actions by discovery + discrimination
  - redundancy - cost + Qwen prior
        |
        v
GLOBAL/TILED SAM3 ACTION
        |
        v
associate detections --> create new nodes --> update many beliefs
        |
        v
update discovery state + empirical prompt value
        |
        +--> if state changed enough: QWEN REPLAN
        |
        v
marginal global value low
        |
        v
BATCHED RESIDUAL CLEANUP
        |
        v
rare individual verification
        |
        v
soft count + diagnostics + replayable provenance
```

Any future implementation choice that reintroduces Qwen-per-node or SAM3-per-node as the default loop contradicts this specification and should be treated as an explicit baseline/ablation rather than V4 proper.

---

# 23. Detailed implementation contracts

This section turns the architectural objects above into concrete implementation contracts. The dataclasses shown here are normative at the level of field meaning, ownership, and serialization behavior, but exact Python syntax may change during implementation.

## 21.1 Identifier policy

Every persistent entity must have a stable string identifier. IDs are created once and never reused within a run.

Required ID domains:

- `run_id`
- `image_id`
- `node_id`
- `action_id`
- `semantic_key`
- `sam3_call_id`
- `qwen_call_id`
- `observation_id`
- `event_id`

Recommended format:

```text
run_000001
action_000007
sam3_000009
qwen_000002
node_000143
```

Do not derive persistent IDs from array positions because graph ordering may change after merges, rejection, replay, or serialization.

## 21.2 Geometry contract

All persistent boxes and masks must refer to original-image coordinates unless a field is explicitly marked local/tile coordinates.

```python
@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    coordinate_space: Literal["image"] = "image"
```

A `Geometry` object must expose at least:

```python
class Geometry(Protocol):
    def bbox(self) -> Box: ...
    def area(self) -> float: ...
    def iou(self, other: "Geometry") -> float: ...
```

Dense masks should be stored as external artifacts and referenced from state/log files. Do not serialize full masks inline in graph JSON.

## 21.3 Detection schema

```python
@dataclass
class Detection:
    detection_id: str
    geometry: GeometryRef
    score: float
    source_tile_id: str | None
    local_geometry: GeometryRef | None
    mask_artifact: str | None
    raw_metadata: dict[str, Any]
```

`Detection` is a sensor object. It does not contain:

- `target=True`;
- a final semantic class;
- a graph node ID before association;
- Qwen-proposed labels masquerading as observations.

## 21.4 Node observation reference

Every node update must point to the observation that caused it.

```python
@dataclass
class NodeObservationRef:
    observation_id: str
    sam3_call_id: str
    action_id: str
    semantic_key: str
    detection_id: str | None
    relation: ObservationRelation
    score: float | None
    association_score: float | None
```

`ObservationRelation` should contain at least:

- `STRONG_MATCH`
- `WEAK_MATCH`
- `NOT_RETRIEVED`
- `NOT_OBSERVABLE`
- `NEW_DETECTION`
- `AMBIGUOUS_ASSOCIATION`

The same SAM3 execution can create observation references for many graph nodes.

## 21.5 Belief state

The core semantic belief object must be class-generic:

```python
@dataclass
class ClassBelief:
    probabilities: dict[str, float]
    update_count: int
    entropy: float
    last_update_event_id: str | None
```

Invariants:

- probabilities are finite;
- all probabilities are nonnegative;
- they sum to one within numerical tolerance;
- no class name is hard-coded in generic math utilities;
- updates preserve the original evidence trail.

## 21.6 Registration state

```python
@dataclass
class RegistrationDiagnostics:
    existence_score: float
    duplicate_risk: float
    merge_risk: float
    ambiguous_with: list[str]
    support_count: int
    independent_semantic_support_count: int
```

`support_count` means the number of sensor observations geometrically supporting the node. `independent_semantic_support_count` means the number of distinct semantic keys providing useful semantic evidence. They must not be conflated.

## 21.7 Action schema validation

Every `SensingAction` must pass validation before reaching SAM3.

Validation checks:

- non-empty prompt;
- known action family;
- threshold in allowed range;
- ROI inside image bounds;
- referenced exemplar nodes exist;
- positive and negative exemplar IDs are disjoint;
- semantic key is non-empty and canonicalized;
- spatial mode and tiling config are compatible;
- action does not exceed hard budget before execution.

Invalid Qwen actions are logged and discarded; they do not terminate the run unless no valid alternatives remain.

---

# 24. Runner state machine

The V4 runner should be implemented as an explicit state machine rather than a collection of loosely coupled while-loops.

Recommended stages:

```text
INITIALIZE
  -> BOOTSTRAP_GLOBAL
  -> BOOTSTRAP_TILE_DECISION
  -> BOOTSTRAP_TILED (optional)
  -> BUILD_QWEN_EVIDENCE
  -> PLAN
  -> GLOBAL_SENSING
  -> REPLAN (zero or more times)
  -> CLEANUP_DECISION
  -> CLEANUP (optional)
  -> FINALIZE
  -> DONE
```

Every transition must be represented by a logged event.

## 22.1 Initialization

Inputs:

```python
RunInput(
    image,
    user_prompt,
    optional_class_vocabulary,
    optional_dataset_context,
)
```

Initialization creates:

- immutable `RunConfig`;
- empty `SceneGraph`;
- empty `SemanticMemory`;
- zeroed `BudgetState`;
- deterministic ID generators;
- run artifact directories.

No model may run during object construction.

## 22.2 Bootstrap global

The runner constructs exactly one user-derived global discovery action and passes it to `SAM3Sensor.observe`.

The resulting detection set is handed to association/graph initialization. The runner does not independently inspect masks to make semantic decisions.

Outputs:

- graph candidates;
- bootstrap sensor summary;
- candidate size/density statistics;
- updated discovery state;
- budget increment.

## 22.3 Tile decision

`TilingPolicy.should_tile_bootstrap(state, image_info)` returns a decision object, not merely a boolean:

```python
@dataclass
class TilingDecision:
    should_tile: bool
    reasons: list[str]
    estimated_extra_cost: float
    score: float
```

Reasons are logged for later analysis. This allows the research code to test whether tiling decisions were useful.

## 22.4 First Qwen plan

The planner is called only after bootstrap is complete. It receives a compact evidence pack and returns a structured plan. The runner validates and canonicalizes proposed actions before they enter the bank.

## 22.5 Global sensing cycle

Each global sensing iteration performs exactly this order:

1. recompute utility for currently available actions;
2. choose one action;
3. validate budget;
4. execute SAM3 once;
5. associate all returned detections;
6. create new nodes from genuinely unmatched detections;
7. project the global observation onto relevant existing nodes;
8. update beliefs;
9. update registration diagnostics;
10. update semantic memory with realized effects;
11. update discovery state;
12. update budgets;
13. evaluate replanning triggers;
14. evaluate global stopping conditions.

This order is normative. In particular, belief updating happens **after association**, because the system must know which physical hypothesis each detection supports.

## 22.6 Replanning cycle

When a replan trigger fires:

1. finish the current SAM3 state transition completely;
2. construct the new Qwen evidence pack from the updated scene;
3. call Qwen once;
4. validate/canonicalize the returned bank;
5. deduplicate against all previously executed and proposed semantic keys;
6. either extend or replace the active bank according to configuration;
7. resume global sensing.

No Qwen call should happen halfway through an incomplete SAM3 graph update.

## 22.7 Cleanup transition

Cleanup begins only if:

- global-stage stopping has fired;
- count/discovery uncertainty remains above cleanup tolerance;
- unresolved node count is below a configurable maximum suitable for batching;
- cleanup budget remains.

If hundreds of nodes remain unresolved, the system should replan or stop with a diagnostic rather than silently falling back to hundreds of local queries.

---

# 25. Qwen planner prompt and structured-output contract

The Qwen interface is strategically important and must be tightly specified to avoid recreating the V3 behavior where the model effectively became an unconstrained per-node controller.

## 23.1 Planner role statement

Every planner request should communicate, in substance:

> You are proposing semantic sensing experiments for a fixed image. You are not labeling the provided crops, not generating detections, and not producing a final count. The crops are unverified outputs of SAM3. Propose a small diverse bank of prompts that can increase target discovery, reveal important confounders, or resolve remaining ambiguities when executed by SAM3 over the scene.

The exact wording may evolve, but this role separation is an invariant.

## 23.2 Evidence-pack payload

A Qwen request contains four conceptual blocks.

### Task block

- user prompt;
- target concept;
- optional high-level class vocabulary;
- request type: initial planning or replanning.

### Scene block

- original image or controlled-resolution version;
- image dimensions;
- current candidate count;
- discovery/uncertainty summary.

### Representative visual block

A contact sheet of candidates. Each panel should display or encode:

- node ID;
- crop;
- SAM3 confidence;
- support count;
- semantic keys that retrieved it;
- current belief only for replanning, clearly labeled as model belief;
- whether the node is new since last planning call.

### Search-history block

For replanning only:

```text
semantic key | family | executions | new nodes | target-mass gain | confounder overlaps | cost
```

This is intentionally concise. Do not include raw chain-of-thought or full event history.

## 23.3 Planner output limits

Recommended defaults:

- 6--12 proposed actions;
- at least 2 discovery actions unless discovery is clearly saturated;
- at least 1 confounder action when confounders are plausible;
- at most 2 pure verification actions in the initial plan;
- rationale length capped;
- no free-form boxes unless context-localized search is explicitly requested;
- no per-node action list.

## 23.4 Semantic priors

If Qwen returns `semantic_prior`, values must be interpreted as relative semantic expectations rather than calibrated probabilities. The planner may instead return ordinal labels such as `high/medium/low` if numeric beta estimates prove unstable.

The implementation should support both without changing downstream graph structures.

## 23.5 Planner failure handling

Failures include:

- transport timeout;
- malformed structured output;
- zero valid actions;
- all actions semantic duplicates;
- references to nonexistent exemplars;
- unsafe/invalid ROI;
- nonsensical thresholds.

Policy:

1. one schema-repair retry at most;
2. if still invalid, use a deterministic fallback bank;
3. record planner failure but continue if possible;
4. never loop indefinitely retrying Qwen.

A fallback bank may contain the user prompt plus generic semantic variations generated deterministically from already known semantic keys, but it must not introduce dataset-specific hidden prompt templates into the generic controller.

---

# 26. Action-bank lifecycle and scoring details

## 24.1 Bank states

Each entry moves through:

```text
PROPOSED -> VALID -> SELECTED -> EXECUTED
                  \-> REJECTED
                  \-> EXPIRED
```

Reasons for rejection/expiration include:

- duplicate semantic key;
- budget violation;
- action rendered irrelevant by scene change;
- low utility after empirical updates;
- invalid exemplars/ROI;
- semantic coordinate already saturated.

## 24.2 Utility decomposition in V4.0

Because fully calibrated Bayesian quantities are unlikely at first, implement utility components separately and expose them in logs:

```python
@dataclass
class UtilityBreakdown:
    discovery_score: float
    discrimination_score: float
    redundancy_penalty: float
    cost_penalty: float
    qwen_prior_score: float
    total: float
```

Never log only the final scalar. Research analysis needs to know *why* an action was selected.

## 24.3 Recommended first-pass proxy formulas

These are implementation starting points, not research claims.

Discovery proxy:

```text
discovery_score =
    qwen_discovery_prior
    * unsaturated_semantic_mode_weight
    * uncovered_region_weight
    * empirical_related_prompt_gain
```

Discrimination proxy:

```text
discrimination_score =
    sum over active nodes(
        node_uncertainty
        * node_existence_score
        * semantic_class_separation
        * expected_observability
    )
```

Redundancy:

```text
redundancy = max similarity to previously executed semantic keys
           + repeat_count_penalty
           + low_recent_marginal_gain_penalty
```

Cost:

```text
cost = normalized_expected_runtime
     + tile_multiplier
     + processed_pixel_multiplier
```

Qwen prior:

```text
qwen_prior = normalized planner priority
```

All coefficients must live in configuration and be logged.

## 24.4 Empirical override

After an action is executed, record realized outcomes:

- raw new detections;
- new registered nodes;
- new target posterior mass;
- decrease in total node entropy/proxy uncertainty;
- confounder posterior mass gained;
- duplicate detections rejected;
- runtime/tiles.

Related future actions should have their utility adjusted using these empirical outcomes. This is how the numerical controller begins to override Qwen's initial ranking.

---

# 27. Association and graph-update algorithm

Registration is one of the most dangerous sources of silent counting error. It requires an explicit algorithm and tests.

## 25.1 Matching stages

For each new SAM3 detection set:

### Stage A: geometric candidate generation

Create candidate node matches using broad geometry criteria:

- bounding-box overlap;
- mask overlap where available;
- center distance normalized by object scale;
- containment;
- optional appearance similarity.

This stage should favor recall and can return multiple possible graph nodes.

### Stage B: match scoring

Compute a deterministic association score from configurable features. Do not use semantic class posterior as the dominant matching feature because the same object may be detected under a confounder prompt.

### Stage C: assignment

Use a one-to-one assignment strategy when the sensor output represents instance detections. Preserve ambiguous many-to-one or one-to-many cases as diagnostics rather than forcing every detection into an arbitrary match.

### Stage D: node creation

Only detections unmatched above a conservative threshold can create new nodes.

## 25.2 Geometry update

When a detection matches an existing node, node geometry should be updated through an explicit policy:

- keep highest-quality mask;
- weighted box average;
- canonical geometry chosen by strongest sensor score;
- or task-dependent strategy exposed via configuration.

Do not let later low-confidence confounder detections overwrite good target geometry without a policy decision.

## 25.3 Duplicate handling

Potential duplicates can arise from:

- tiled overlap;
- synonymous prompts;
- fragmented masks;
- repeated global passes.

Duplicate resolution should rely primarily on stable geometry across observations. Semantic agreement is supporting evidence, not proof.

If two nodes are merged, preserve both historical IDs in lineage metadata and emit an explicit merge event.

## 25.4 Split/merge risk

If one detection overlaps two established nodes substantially, mark merge risk rather than immediately fusing those nodes.

If two detections repeatedly partition one existing node, mark split risk and allow later graph surgery if configured.

The first V4 implementation may leave risky cases unresolved; silent structural errors are worse than explicit ambiguity.

---

# 28. Evidence update policy for V4.0

The long-term goal is a calibrated Bayesian update. V4.0 needs a robust approximate implementation that does not overclaim calibration.

## 26.1 Evidence object

After projection, construct:

```python
@dataclass
class SemanticEvidence:
    node_id: str
    semantic_key: str
    family: ActionFamily
    relation: ObservationRelation
    sensor_score: float | None
    semantic_prior: dict[str, float] | None
    empirical_channel: PromptChannelStats | None
    repeat_group_id: str
    weight: float
```

This object is the only input to the belief updater.

## 26.2 Initial update modes

Support three switchable modes.

### Mode A: conservative log-score update

Convert semantic prior separation and observation relation into bounded additive log-odds evidence. Cap per-action contribution. This is easiest to debug.

### Mode B: discretized likelihood update

Estimate a small observation likelihood table for `STRONG_MATCH`, `WEAK_MATCH`, and `NOT_RETRIEVED`; update with Bayes' rule.

### Mode C: joint semantic-moment model

Use latent semantic coordinates and correlated observation handling from the research formulation.

V4.0 should probably begin with A or B but keep interfaces compatible with C.

## 26.3 Absence evidence

`NOT_RETRIEVED` receives reduced weight unless empirical recall for that action is high. `NOT_OBSERVABLE` receives exactly zero semantic evidence.

## 26.4 Repeat handling

For evidence sharing the same semantic key:

- first execution receives full semantic weight;
- subsequent same-key evidence is discounted or used primarily for retrieval/geometry support;
- tiled/global same-key repeats belong to the same repeat group;
- paraphrases with high semantic similarity may share a repeat group.

The discount policy must be logged.

## 26.5 Belief resolution

Node resolution should require both:

- posterior/class threshold;
- sufficient registration/existence confidence.

Example configurable rule:

```text
RESOLVED if max class probability >= 0.95
        and existence_score >= 0.8
        and duplicate_risk <= 0.2
```

These numbers are defaults only, not theoretical constants.

---

# 29. Discovery-state and plateau implementation

The latent missed count cannot initially be estimated exactly. V4 therefore needs explicit, auditable discovery proxies.

## 27.1 Per-action discovery record

For each discovery-oriented execution, log:

```python
@dataclass
class DiscoveryOutcome:
    action_id: str
    new_raw_detections: int
    new_registered_nodes: int
    new_active_nodes: int
    new_target_mass: float
    duplicate_detections: int
    searched_fraction: float
    runtime_ms: float
```

## 27.2 Plateau score

A first implementation may define plateau from a window of recent discovery outcomes. Example inputs:

- median new registered nodes per cost;
- median new target mass per cost;
- semantic diversity already exhausted;
- proportion of image adequately searched;
- remaining Qwen missing modes.

The exact scalar may evolve, but all component diagnostics should remain exposed.

## 27.3 Spatial coverage

Maintain a coarse image grid or region map indicating how much sensing effort has covered each location. Coverage is not the same as target probability.

Useful fields per spatial cell:

- number of global actions covering it;
- number of tiled actions covering it;
- semantic families applied;
- local candidate density;
- unresolved candidate density.

This supports future active tiling without contaminating class beliefs.

---

# 30. Replanning trigger specification

Each trigger should be individually testable and logged with its reason.

Recommended trigger object:

```python
@dataclass
class ReplanDecision:
    should_replan: bool
    reasons: list[ReplanReason]
    metrics: dict[str, float]
```

Initial `ReplanReason` enum:

- `BANK_EXHAUSTED`
- `LOW_BEST_UTILITY`
- `DISCOVERY_PLATEAU_WITH_UNCERTAINTY`
- `NEW_CONFOUNDER_PATTERN`
- `UNRESOLVED_CLUSTER`
- `LARGE_COUNT_UNCERTAINTY`
- `MANUAL_DEBUG_TRIGGER`

Replanning must not occur if `max_qwen_calls` has been reached. In that case the controller either continues with remaining deterministic actions, enters cleanup, or stops with a diagnostic.

---

# 31. Configuration hierarchy

Configuration should be explicit and immutable for each run.

Recommended groups:

```python
@dataclass(frozen=True)
class V4Config:
    bootstrap: BootstrapConfig
    planner: PlannerConfig
    sam3: SAM3Config
    action_selection: ActionSelectionConfig
    association: AssociationConfig
    belief: BeliefConfig
    replanning: ReplanningConfig
    cleanup: CleanupConfig
    stopping: StoppingConfig
    budgets: BudgetConfig
    logging: LoggingConfig
```

No algorithmic magic numbers should be hidden inside runners.

Every run must save the complete resolved configuration.

Dataset adapters may provide *hints* through a separate `DatasetHints` object, but they may not silently mutate core defaults. The log must show when a dataset hint affected behavior.

---

# 32. Failure handling and degraded operation

V4 should finish with an interpretable degraded result whenever possible rather than crashing or looping indefinitely.

## 30.1 SAM3 failure

If a SAM3 call throws or returns invalid output:

1. log failure with action/call ID;
2. do not mutate graph state;
3. optionally retry once if configured;
4. mark action failed;
5. select another action if budget remains;
6. stop with diagnostic if sensor is unusable.

## 30.2 Qwen failure

Use the planner fallback described above. Qwen failure must not invalidate already sensor-grounded graph state.

## 30.3 Association failure

If matching produces ambiguous or impossible assignment:

- preserve raw detections;
- mark affected nodes/detections ambiguous;
- do not silently count ambiguous newly created duplicates as independent objects;
- allow later observations to resolve the structure.

## 30.4 Budget exhaustion

Budget exhaustion is a normal stopping condition, not an exception.

Final output must state which budget ended the run.

## 30.5 Empty graph

If no candidate is ever found:

- final soft count is zero under represented-object estimate;
- discovery diagnostic must state that completeness is uncertain unless the sensing process strongly supports saturation;
- do not fabricate confidence that the true count is zero.

---

# 33. Reference run: green citrus

This reference run is normative for control flow, not for exact counts or prompts.

Assume one orchard image and user prompt:

```text
green citrus fruit
```

## 31.1 Bootstrap

### Step 1: global user-prompt action

SAM3 returns 61 detections.

Association creates 58 nodes because three detections are near-duplicates.

State summary:

```text
nodes: 58
semantic keys used: [green_citrus]
qwen calls: 0
sam3 calls: 1
```

The graph does **not** label all 58 nodes fruit.

### Step 2: tile decision

Median object size is small relative to image dimensions and detections are spatially dense, so tiled bootstrap is approved.

Tiled SAM3 returns 84 raw detections. After cross-pass association:

- 55 support existing nodes;
- 16 are duplicates from tile overlap;
- 13 create new candidate nodes.

State:

```text
nodes: 71
qwen calls: 0
sam3 calls: 2
distinct semantic keys: 1
```

Even though 55 nodes were seen in both global and tiled passes, the system still has only one semantic coordinate of evidence: `green_citrus`.

## 31.2 First Qwen plan

Build a 20-crop sheet containing high-, medium-, low-support, and outlier candidates. Qwen sees the original image and learns that foliage is the dominant confounder.

Qwen proposes, for example:

```text
DISCOVERY: round green fruit
DISCOVERY: small partially occluded citrus
DISCOVERY: clustered green fruit
CONFOUNDER: flat green leaf
CONFOUNDER: veined leaf cluster
CONTEXT: dense tree canopy
VERIFICATION: smooth spherical green object
```

After validation/deduplication, six actions enter the bank.

State:

```text
qwen calls: 1
sam3 calls: 2
bank size: 6
```

## 31.3 Global action 1: round green fruit

The utility controller selects this action because it has high discovery score and low redundancy.

One global SAM3 call returns detections that:

- support 37 existing nodes;
- discover 8 new candidate nodes;
- fail to retrieve several current candidates;
- create 4 obvious duplicates rejected during association.

One SAM3 execution therefore changes evidence for dozens of nodes and expands the graph from 71 to 79 nodes.

## 31.4 Global action 2: flat green leaf

The controller selects a confounder prompt because many active nodes remain ambiguous and the query has high class-separation value.

SAM3 retrieves regions overlapping 24 graph nodes strongly. Beliefs on those nodes shift toward the leaf hypothesis. Several initial `green_citrus` candidates now become likely confounders.

No special negative-vote logic occurs; the leaf query is simply an observation whose likelihood favors `leaf`.

## 31.5 Global action 3: partially occluded citrus

This query adds three new nodes and supports several low-confidence candidates in shadowed foliage.

Recent discovery gains are now `8, 0, 3` new nodes across three actions. Discovery has not fully plateaued.

## 31.6 Global action 4: veined leaf cluster

This confounder query explains additional suspicious regions but adds no new target-like candidates.

The remaining bank has low predicted discovery utility and the graph contains a cluster of uncertain small dark-green objects.

## 31.7 Qwen replan

Replan trigger:

```text
LOW_BEST_UTILITY
UNRESOLVED_CLUSTER
```

Qwen receives:

- original image;
- new contact sheet emphasizing recently discovered and ambiguous nodes;
- concise history of four global semantic actions;
- realized discovery/confounder effects.

It proposes a smaller second bank oriented to the unresolved appearance mode.

This is Qwen call 2, not call 80.

## 31.8 Second global round

Two additional global/region-level actions are executed. Most remaining uncertainty collapses. The graph now contains:

```text
84 total nodes
51 high target probability
27 high leaf/background probability
6 ambiguous
```

Discovery gain has been near zero for several distinct semantic actions.

## 31.9 Cleanup

Because only six nodes remain ambiguous, the runner enters cleanup.

A six-crop contact sheet is used to choose one shared residual prompt. SAM3 evaluates it over the six ROIs as one batched operation. Four nodes resolve; two remain ambiguous.

If local cleanup budget allows, one final targeted batch/action may be used. No per-node Qwen generation occurs.

## 31.10 Finalization

The runner reports:

- soft target count;
- hard threshold count;
- graph size;
- unresolved nodes;
- discovery plateau diagnostics;
- duplicate/merge diagnostics;
- Qwen calls: 2;
- SAM3 calls: bootstrap 2 + global 6 + cleanup 1--2;
- runtime and storage;
- complete event/provenance log.

This reference execution captures the intended computational scaling: the number of semantic model calls depends mainly on the number of *useful scene-level experiments*, not on the number of objects.

---

# 34. Reference run: CARPK contrast case

CARPK is useful because the confounder problem is easier than green citrus but object count can be high.

Input:

```text
car
```

Bootstrap may discover dozens of cars. Qwen might propose variants such as parked vehicle, vehicle roof, white car, dark car, parking marking, or road surface.

The important invariant is unchanged:

```text
54 candidates does not imply 54 Qwen calls.
```

A plausible run uses:

- 1 global bootstrap SAM3;
- optional 1 tiled bootstrap;
- 1 initial Qwen plan;
- 3--5 global semantic SAM3 passes;
- optional 1 Qwen replan;
- 0--2 cleanup batches.

The pipeline must make no CARPK-specific assumption that every candidate is a rigid rectangle or that a parking lot ROI exists. Those facts may emerge through Qwen/context sensing, not core code.

---

# 35. Migration matrix from V3

Before copying code, classify each V3 component into one of four categories.

## 33.1 Reuse with minimal adaptation

Candidates:

- pure geometry utilities;
- mask serialization helpers;
- dataset parsers;
- evaluation metric functions;
- low-level stable model-loading helpers.

Criteria:

- no legacy runner imports;
- deterministic behavior;
- narrow responsibility;
- existing tests portable.

## 33.2 Extract and refactor

Likely examples:

- tiling logic;
- SAM3 inference wrappers;
- graph matching;
- compact logging;
- replay/validator machinery.

These may contain valuable logic but require new V4 interfaces and removal of hidden policy assumptions.

## 33.3 Reimplement from behavior tests

Likely examples:

- graph state object;
- belief updater;
- experiment runner;
- Qwen action generator;
- budget/stopping controller.

For these, V3 tests/invariants are more valuable than the implementation itself.

## 33.4 Keep only as baseline

- node-first Qwen ASHT runner;
- legacy VLM policies;
- VIP/oracle controller paths;
- old dashboard-specific orchestration;
- historical experiment modes.

They should remain executable in V3 for comparison, not imported into V4.

---

# 36. Detailed test matrix

The test plan should mirror the module architecture so every scientific invariant has a software regression test.

## 34.1 Schema tests

- serialization round-trip for every persistent dataclass;
- schema version present;
- unknown future fields tolerated where appropriate;
- invalid probability vectors rejected;
- image-coordinate geometry preserved.

## 34.2 Sensor-interface tests

With mock SAM3:

- one action -> one sensor call counter increment;
- global action can return many detections;
- sensor wrapper never mutates graph;
- tile-local detections convert correctly to image coordinates;
- dense masks stored externally.

## 34.3 Bootstrap tests

- user prompt executes before Qwen;
- zero-detection bootstrap still builds evidence pack;
- optional tiling uses same semantic key;
- global+tiled support does not count as two semantic coordinates;
- contact sheet is bounded;
- confidence strata represented;
- crops labeled as candidates, not positives.

## 34.4 Qwen planner tests

Using deterministic mock responses:

- valid plan accepted;
- malformed plan repaired at most once;
- duplicate prompts removed;
- duplicate semantic keys correlated;
- invalid exemplar references rejected;
- more than max actions truncated deterministically;
- Qwen call budget enforced independently.

## 34.5 Utility tests

- discovery-heavy early weighting favors novel global query;
- redundant prompt receives penalty;
- cheaper global query can beat expensive local query with equal expected value;
- Qwen priority can be overridden by poor empirical performance;
- utility breakdown sums to total exactly.

## 34.6 Association tests

Synthetic geometry cases:

- exact re-detection -> existing node;
- tile duplicate -> existing node;
- nearby distinct objects remain distinct;
- one large mask overlapping two objects marks merge risk;
- fragmented observations mark split risk;
- unmatched detection creates node;
- Qwen ROI cannot create node.

## 34.7 Evidence tests

- strong target match increases target belief;
- strong confounder match increases confounder belief;
- target absence has weaker effect when recall is low;
- `NOT_OBSERVABLE` leaves belief unchanged;
- same-key repeats discounted;
- different informative keys can compound;
- probabilities remain normalized and finite.

## 34.8 Global-loop tests

Critical vectorization assertion:

```text
one SAM3 call updates N>1 nodes
```

Additional cases:

- new nodes and existing-node updates happen in same transition;
- action history updates after execution;
- discovery state changes from realized gain;
- replan occurs only after completed transition;
- no per-node hidden SAM3 loop.

## 34.9 Replanning tests

- bank exhausted triggers Qwen when budget available;
- plateau+uncertainty triggers Qwen;
- trigger suppressed when Qwen budget exhausted;
- new bank deduplicates against complete history;
- replanning evidence pack emphasizes new/ambiguous examples.

## 34.10 Cleanup tests

- cleanup skipped when no residual uncertainty;
- large unresolved set does not trigger unbounded local loop;
- batch residual query updates multiple nodes;
- individual verification respects hard cleanup cap;
- Qwen remains scene/cluster level by default.

## 34.11 Logging/replay tests

- replay reconstructs graph exactly from logged observations;
- replay consumes no model calls;
- final budget matches event-derived budget;
- all node origins trace to SAM3 detections;
- semantic keys preserved across replay;
- artifact hashes verify.

## 34.12 Real-model acceptance sequence

Before research sweeps:

1. one image, global bootstrap only;
2. bootstrap + tiled pass;
3. first real Qwen plan;
4. one global positive action;
5. one global confounder action;
6. four-action global loop;
7. one real replan;
8. residual batch cleanup;
9. deterministic replay;
10. five-image mixed-dataset smoke suite;
11. only then full experiments.

---

# 37. Coding constraints for the refactor LLM

A coding model implementing this spec must follow these constraints.

## 35.1 No speculative compatibility layer

Do not create adapters for every V3 runner or policy. V4 compatibility is provided at dataset/evaluation boundaries, not by retaining legacy orchestration.

## 35.2 No giant files

Suggested maximums, subject to reasonable exceptions:

- runner/orchestration modules: <400 lines;
- model wrappers: <350 lines;
- graph/belief modules: <500 lines;
- utility modules: <300 lines.

If a module substantially exceeds this, split by responsibility.

## 35.3 Dependency direction

Allowed direction:

```text
models ----\
sensing ----> pipeline
scene ------/
planning ---/
logging ----/
```

`models` may depend on `sensing` schemas but not `pipeline` or `planning`.

`scene` may depend on `sensing` observation schemas but not model implementations.

`planning` may depend on scene/sensing abstractions but not Ollama/vLLM/SAM3 libraries.

## 35.4 Pure functions where possible

Association scoring, semantic-key normalization, utility computation, belief update, stopping decisions, and replan decisions should be pure/deterministic functions given their inputs and config.

This enables fast tests and exact replay.

## 35.5 No hidden model calls

Only explicit model-adapter methods may perform model inference. A function named `update_belief`, `score_action`, or `associate` must never secretly invoke Qwen or SAM3.

## 35.6 Explicit approximation labels

Names and docs must distinguish:

- exact probability;
- heuristic score;
- proxy information value;
- Qwen prior;
- empirical statistic.

Do not call a heuristic `posterior` or `information_gain` unless the required probability model is actually implemented.

---

# 38. Definition of done for the V4 rewrite

The rewrite is complete when all of the following hold.

## Architecture

- one canonical `Runner` path exists;
- no legacy policy is imported by that path;
- Qwen and SAM3 are accessed only through model adapters;
- graph/state/action/observation objects are explicit and serializable.

## Computational behavior

- initial SAM3 pass always precedes Qwen;
- Qwen operates scene-level, normally 1--3 calls/image;
- global semantic SAM3 passes update many nodes at once;
- local verification is bounded residual cleanup;
- Qwen and SAM3 budgets are independent and enforced.

## Scientific behavior

- discovery and discrimination are represented separately;
- target and confounder prompts use one common evidence model;
- same-semantic-key repeats are correlated/discounted;
- confidence score is not silently treated as class posterior;
- unresolved discovery is not equated with zero.

## Reliability

- fresh-environment unit suite passes;
- end-to-end mock run is deterministic;
- real SAM3/Qwen smoke test completes within configured budgets;
- replay reconstructs final state;
- validator reports no provenance or budget violations.

## Research readiness

- ablation switches exist without separate runners;
- call counts/runtime are logged;
- outputs support count accuracy, discovery recall, classification errors, and computational-efficiency plots;
- V3 baselines remain available externally for comparison.

At this point V4 is not merely a cleaner codebase. It is an implementation whose software abstractions correspond directly to the research formulation: an evolving scene belief, a finite semantic action bank, global prompt-conditioned sensing, feedback-driven replanning, and cost-aware stopping for counting.
