# SAM3-VLM V4

**Zero-shot object counting through active semantic perception with SAM3 and Qwen**

SAM3-VLM V4 is a clean rewrite of the SAM3-VLM research pipeline for zero-shot object counting in difficult visual scenes.

The primary research case is **green citrus counting in dense green foliage**, where the target objects are visually similar to leaves and reliable counting requires both broad discovery and careful discrimination. The architecture is intentionally dataset-independent and should also support tasks such as vehicle counting, PiXMo-Count, blood-cell counting, and other dense counting problems.

V4 is a **greenfield implementation**. The V3 repository may be used as read-only reference for tested low-level components, but the V3 architecture should not be preserved for compatibility.

---

## Core idea

A conventional counting pipeline looks roughly like:

```text
prompt -> detect -> count
```

V4 instead treats counting as a **closed-loop active sensing problem**:

```text
User prompt
    |
    v
Bootstrap SAM3
(global, optionally tiled)
    |
    v
Initial scene graph
    |
    v
Representative crops + sensor state
    |
    v
Qwen scene planner
    |
    v
Semantic action bank
    |
    v
Global SAM3 semantic passes
    |
    v
Scene-wide graph and belief updates
    |
    v
Qwen replanning when useful
    |
    v
Residual targeted cleanup
    |
    v
Final soft count + uncertainty
```

The main architectural change from V3 is that **global semantic sensing is the default**.

One SAM3 pass can simultaneously:

- discover previously unseen objects,
- support or weaken many existing object hypotheses,
- expose confounders,
- update scene-level uncertainty.

Per-object verification is reserved for the small unresolved residue near the end of a run.

---

## Model responsibilities

### SAM3 — visual sensor

SAM3 is the only component allowed to create image-grounded detections.

A SAM3 action may specify:

- text prompt,
- ROI,
- positive/negative exemplars,
- confidence threshold,
- global or tiled execution.

SAM3 returns observations such as masks, boxes, confidence scores, and execution metadata.

SAM3 does **not** choose the next action and does not produce the final count.

### Qwen — semantic planner

Qwen reasons about the current scene and proposes useful semantic sensing actions.

After bootstrap, Qwen may receive:

- the original image,
- the user target concept,
- representative SAM3 crops,
- SAM3 confidence/support information,
- provenance,
- unresolved/confounder examples,
- recent prompt performance,
- a compact scene summary.

Qwen proposes a structured action bank containing:

- **discovery actions** — search for missed target appearance modes,
- **confounder actions** — search for competing explanations,
- **context actions** — identify useful search regions,
- **verification actions** — resolve remaining ambiguity.

Qwen may rank actions semantically, but it does **not** directly create detections or overwrite graph beliefs.

### Numerical controller — active perception

The controller selects which proposed action to execute.

Conceptually:

\[
U_t(x)
=
\alpha_t D_t(x)
+
\beta_t I_t(x)
-
\gamma R_t(x)
-
\lambda C(x)
+
\eta_t Q_t(x),
\]

where:

- \(D_t(x)\): expected discovery gain,
- \(I_t(x)\): expected discrimination/count-information gain,
- \(R_t(x)\): semantic redundancy,
- \(C(x)\): computation cost,
- \(Q_t(x)\): Qwen semantic priority.

Early in a run, discovery should dominate. Later, discrimination and count uncertainty become more important.

---

## Important V4 invariants

1. The user's prompt is executed by SAM3 before the first Qwen planning call.
2. Qwen proposals never become graph detections.
3. Bootstrap detections are **candidates**, not positive labels.
4. SAM3 confidence is not automatically a target-class probability.
5. Global semantic sensing is preferred over per-object sensing.
6. Positive and confounder prompts are both first-class sensing actions.
7. Repeated global/tiled executions of the same semantic prompt are correlated evidence.
8. Qwen is called on meaningful scene-state changes, not once per graph node.
9. Discovery uncertainty and classification uncertainty are separate.
10. Budgets count real SAM3/Qwen executions, not the number of nodes updated.
11. Every graph node and evidence update must have explicit sensor provenance.
12. The final objective is accurate counting under a finite sensing budget.

---

## Mathematical state

At iteration \(t\), the working state is conceptually

\[
B_t=(G_t,\rho_t,U_t,\mathcal S_t,\mathcal A_t,C_t),
\]

with:

- \(G_t\): evolving scene graph,
- \(\rho_t\): semantic beliefs,
- \(U_t\): residual discovery uncertainty,
- \(\mathcal S_t\): semantic sensing history,
- \(\mathcal A_t\): current action bank,
- \(C_t\): consumed computational budget.

The graph is not assumed to contain every true object.

Conceptually,

\[
N^\star
=
N^\star_{\text{represented}}
+
M_t^\star,
\]

where \(M_t^\star\) is the number of still-undiscovered target objects.

This distinction is essential: confidently classifying every existing graph node does not prove that discovery is complete.

The final represented-object estimate uses posterior target mass rather than only threshold-and-count:

\[
\widehat N
=
\sum_i P(Z_i=k^\star\mid H_T),
\]

with registration/discovery uncertainty handled separately.

---

## Research motivation

The motivating green-citrus problem is difficult because green fruit can resemble green leaves extremely closely. The project is fully zero-shot: no task-specific labels are required for the active SAM3-Qwen procedure.

The central hypothesis is not simply that "more prompts improve SAM3." It is:

> A language model can propose semantically diverse sensing experiments, while a prompt-conditioned grounding model executes those experiments on the image. Closed-loop selection of target and confounder queries can improve discovery and discrimination relative to a single prompt or static prompt list, while requiring substantially fewer model calls than per-object verification.

The mathematical framing connects the system to:

- active perception,
- active sequential hypothesis testing,
- sequential experimental design,
- information pursuit,
- zero-shot semantic/attribute recognition,
- submodular discovery and coverage.

---

# Documentation

Two files define the project.

## `docs/V4_DESIGN_SPEC.md`

**Authoritative implementation specification.**

Use this document for:

- architecture,
- state/data schemas,
- module boundaries,
- runner state machine,
- Qwen planner contract,
- action-bank lifecycle,
- association,
- belief/evidence updates,
- replanning,
- stopping,
- cost accounting,
- logging,
- migration,
- tests and acceptance criteria.

If implementation behavior is unclear, follow the design spec.

## `docs/V4_PAPER_TLDR.md`

**Compact scientific overview.**

Use this document for:

- motivation,
- mathematical formulation,
- active-perception/ASHT interpretation,
- scientific claims,
- experiment logic,
- paper-level understanding.

The paper document explains **why** the architecture makes sense. The design spec defines **what the implementation must do**.

---

# Expected repository structure

The exact layout may evolve, but the intended separation is:

```text
sam3-vlm-v4/
|
|-- docs/
|   |-- V4_DESIGN_SPEC.md
|   `-- V4_PAPER_TLDR.md
|
|-- src/
|   `-- sam3_vlm/
|       |-- models/
|       |   |-- sam3.py
|       |   `-- qwen.py
|       |
|       |-- scene/
|       |   |-- graph.py
|       |   |-- node.py
|       |   |-- association.py
|       |   `-- belief.py
|       |
|       |-- sensing/
|       |   |-- action.py
|       |   |-- observation.py
|       |   |-- evidence.py
|       |   `-- tiling.py
|       |
|       |-- planning/
|       |   |-- action_bank.py
|       |   |-- qwen_planner.py
|       |   |-- utility.py
|       |   `-- stopping.py
|       |
|       |-- pipeline/
|       |   |-- bootstrap.py
|       |   |-- runner.py
|       |   `-- cleanup.py
|       |
|       |-- datasets/
|       |-- evaluation/
|       `-- logging/
|
|-- tests/
|-- AGENTS.md
`-- README.md
```

Model wrappers must remain separate from algorithm/controller logic.

---

# Development milestones

Each milestone follows:

```text
Gemini builder
    ->
code + tests + commit
    ->
Claude reviewer
    ->
PASS or CHANGES REQUIRED
    ->
fixes if needed
    ->
next milestone
```

No next milestone starts until the current milestone is approved.

## M0 — Foundation

Build the clean repository foundation:

- Python package structure,
- dependency/configuration strategy,
- test infrastructure,
- basic CI,
- core shared IDs/types,
- architectural module boundaries.

Do not implement the full SAM3/Qwen pipeline yet.

**Exit:** clean imports, green tests, correct package structure.

---

## M1 — Scene graph and registration

Implement:

- graph nodes,
- geometry,
- provenance,
- semantic-belief container,
- cross-pass association,
- duplicate handling,
- merge/split diagnostics,
- graph serialization.

**Exit:** deterministic graph behavior under synthetic multi-pass observations.

---

## M2 — SAM3 sensing and bootstrap

Implement:

- clean SAM3 adapter,
- global sensing,
- tiled sensing,
- observation records,
- coordinate handling,
- initial user-prompt bootstrap,
- optional tiled bootstrap,
- candidate registration,
- bootstrap support statistics,
- representative contact-sheet generation.

**Exit:** an image can be bootstrapped into a stable scene graph and Qwen evidence pack.

---

## M3 — Qwen scene planner

Implement:

- evidence-pack serialization,
- image/contact-sheet input,
- discovery/confounder/context/verification action families,
- semantic keys,
- structured Qwen output,
- output validation,
- retries/failure handling,
- semantic ranking,
- action-bank construction.

Qwen must never directly mutate scene graph state.

**Exit:** Qwen or a mock Qwen reliably produces a valid scene-level action bank.

---

## M4 — Active global multipass loop

Implement the main controller:

- action-bank lifecycle,
- prompt selection,
- global semantic SAM3 execution,
- scene-wide observation projection,
- new-node discovery,
- existing-node updates,
- prompt-performance history,
- redundancy tracking,
- cost accounting,
- global/tiled execution decisions.

**Exit:** mocked end-to-end runs execute several semantic actions against one evolving graph.

---

## M5 — Evidence, beliefs, and count uncertainty

Implement:

- positive evidence,
- confounder evidence,
- strong/weak/absent/not-observable outcomes,
- Qwen semantic priors,
- empirical sensor reliability,
- same-semantic-key correlation handling,
- protection against repeated-evidence overconfidence,
- semantic posteriors,
- soft count,
- count uncertainty.

**Exit:** synthetic cases show correct posterior movement without confidence blow-up.

---

## M6 — Replanning, stopping, and residual cleanup

Implement:

- Qwen replanning triggers,
- updated contact sheets,
- action-bank refresh/extension,
- discovery plateau detection,
- marginal-utility stopping,
- budget stopping,
- count-uncertainty stopping,
- batched residual cleanup,
- individual verification only when still necessary.

**Exit:** difficult mocked scenes progress from discovery to cleanup and terminate predictably.

---

## M7 — Infrastructure and evaluation

Implement:

- compact run logging,
- provenance,
- masks outside JSON,
- replayable state transitions,
- run validation,
- dataset interfaces,
- metrics,
- visualizations/reports,
- experiment configuration,
- failure recovery.

**Exit:** runs can be inspected and replayed without rerunning models.

---

## M8 — Real-model validation

Validate the actual system.

Suggested priority:

1. green citrus,
2. CARPK,
3. PiXMo-Count,
4. blood cells,
5. additional datasets.

Measure:

- count error,
- discovery recall,
- graph registration quality,
- Qwen calls/image,
- SAM3 executions/image,
- global vs tiled contribution,
- confounder-prompt usefulness,
- count uncertainty,
- runtime,
- failure modes.

**Exit:** V4 is stable enough for controlled experiments and ablations.

---

# Builder / reviewer workflow

The development workflow intentionally uses two independent model roles.

### Builder

Recommended:

- Gemini Flash for routine implementation,
- Gemini Pro for architecture/math-heavy milestones.

The builder:

1. reads `AGENTS.md` and the relevant design-spec sections,
2. implements only the current milestone,
3. writes/updates tests,
4. runs the test suite,
5. commits the milestone,
6. stops.

### Reviewer

Recommended:

- Claude Sonnet for routine reviews,
- Claude Opus for core controller/math/final integration.

The reviewer:

1. reads the same design spec,
2. reviews the exact builder commit,
3. inspects implementation and tests,
4. runs tests independently,
5. searches for specification violations and architecture drift,
6. returns either `PASS` or `CHANGES REQUIRED`.

A green test suite alone is not sufficient for approval.

---

# Testing philosophy

Testing is part of the architecture.

### Unit tests

Cover:

- schemas,
- geometry,
- registration,
- semantic keys,
- action-bank validation,
- evidence rules,
- belief updates,
- utility functions,
- stopping,
- count estimation.

### Integration tests

Use mocked Qwen and SAM3 for:

```text
bootstrap
-> Qwen planning
-> global sensing
-> graph update
-> replanning
-> cleanup
-> count
```

### Real-model tests

Keep explicit smoke tests for:

- one global SAM3 pass,
- one tiled SAM3 pass,
- one Qwen planning call,
- one short real end-to-end run.

### Regression tests

Every meaningful implementation or review bug should gain a regression test.

Tests must not be weakened solely to make failing code pass.

---

# Planned research ablations

Important comparisons include:

- user prompt only vs V4,
- fixed prompt bank vs adaptive replanning,
- positive-only vs positive + confounder sensing,
- global-only vs global + tiled execution,
- Qwen ranking vs numerical-controller override,
- independent-repeat fusion vs semantic redundancy handling,
- hard threshold count vs posterior soft count,
- no cleanup vs residual cleanup,
- one Qwen call vs event-driven replanning,
- per-object verification vs vectorized global sensing.

The goal is to identify separately:

1. what improves **accuracy**, and
2. what improves **computational efficiency**.

---

# Status

V4 is currently a greenfield implementation project.

- [ ] **M0** — Foundation
- [ ] **M1** — Scene graph and registration
- [ ] **M2** — SAM3 sensing and bootstrap
- [ ] **M3** — Qwen scene planner
- [ ] **M4** — Active global multipass loop
- [ ] **M5** — Evidence, beliefs, and count uncertainty
- [ ] **M6** — Replanning, stopping, and residual cleanup
- [ ] **M7** — Infrastructure and evaluation
- [ ] **M8** — Real-model validation

---

## License

TBD.

## Citation

TBD.
