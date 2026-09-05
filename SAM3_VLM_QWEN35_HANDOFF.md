# SAM3-VLM V4 — Qwen3.5 Continuation Handoff

Use this document as the initial prompt/context for a new Codex task opened in
the exact V4 project at:

`/Users/emilkielar/Projects/sam3-vlm-2/v4`

The new task must work only in this V4 repository. The sibling V3 code is an
old, error-prone reference and is read-only.

## Assignment for the new task

Continue the SAM3-VLM V4 project from the current clean repository state.
First perform a read-only familiarization pass over the complete V4 codebase.
Do not edit anything until that pass is complete. Then implement the narrowly
defined pending behavior in this handoff, update the authoritative design
documentation, add regression tests, and run all laptop-safe verification.

Do not use Slurm, `sbatch`, or scheduler wrappers. The user runs cluster tests
directly from an interactive GPU shell and wants simple commands.

## Phase 1 — mandatory read-only familiarization

Before changing code:

1. Confirm the repository root and working-tree state with `pwd`,
   `git status --short`, and `git log -5 --oneline`.
2. Read `AGENTS.md` completely.
3. Read `docs/V4_DESIGN_SPEC.md` completely. It is authoritative.
4. Read `README.MD`, `docs/V4_PAPER_TLDR.md`, and
   `docs/M8_CLUSTER_RUNBOOK.md` completely.
5. Inventory and read all source code under `src/sam3_vlm/`, all tests under
   `tests/`, all configuration under `configs/`, and the readiness scripts
   under `scripts/`. Read in sensible batches, but do not rely only on this
   handoff or on filenames.
6. Pay particular attention to:
   - `src/sam3_vlm/models/qwen.py`
   - `src/sam3_vlm/planning/qwen_planner.py`
   - `src/sam3_vlm/planning/action_bank.py`
   - `src/sam3_vlm/planning/replanning.py`
   - `src/sam3_vlm/pipeline/bootstrap.py`
   - `src/sam3_vlm/pipeline/runner.py`
   - `src/sam3_vlm/experiments/m8_smoke.py`
   - `src/sam3_vlm/logging/`
   - all `test_m8_*`, planner, runner, validator, and replay tests.
7. After reading, briefly summarize the architecture and verify that the
   current code matches the state described below. Then begin implementation
   without waiting for another approval unless the repository materially
   contradicts this handoff.

Preserve user changes and do not perform a broad rewrite. The user strongly
prefers a small and comprehensible codebase, but simplification must not remove
scientifically meaningful logging, validation, replay, or experiment variants.

## Repository state at handoff

- Git HEAD: `843b786` (`Fix errors`).
- Working tree was clean before this handoff document was added.
- Laptop-safe suite: `225 passed, 1 skipped`.
- Recent relevant commits:
  - `23479ab Change qwen model`
  - `f8eb366 Fix qwen testcase`
  - `843b786 Fix errors`
- `f8eb366` isolates mocked OpenAI constructor tests from a live exported
  `QWEN_API_KEY=ollama` and separately tests that production honors the
  environment variable.
- `843b786` accepts a single Markdown-fenced JSON object from local Qwen,
  including a mismatched closing fence such as an opening `json` fence and a
  closing `yaml` fence. It does not accept prose or weaken schema/action
  validation. This prevented a needless second model repair call.

## Frozen architecture and scientific policy

Do not change these points unless the user explicitly requests it:

- SAM3 is the only detector and the only component allowed to create grounded
  object candidates.
- Qwen is a semantic experiment planner. It does not count fruit and does not
  overwrite graph beliefs.
- M8 executes information-gain calculations and SAM3 experiments only for
  novel target prompts.
- Qwen may describe confounders in scene analysis/rationale, but confounder
  prompts are not executable M8 actions.
- Each Qwen round admits at most one target `DISCOVERY` action.
- M8 permits up to two Qwen calls and one evidence-driven replan.
- Keep all pilot variants:
  - `A_SAM3_Global`
  - `B_SAM3_Bootstrap`
  - `C_Qwen_OneRound`
  - `D_Qwen_TwoRound`
- A target posterior at or above `0.8` contributes `1.0` to the reported count.
  The stored posterior is unchanged, and the raw posterior sum remains logged.
- Cleanup remains disabled for the current M8 experiment.
- Do not change the posterior threshold or belief update merely to improve one
  difficult image. This dataset is intentionally hard, and missing some fruit
  is acceptable.
- Validator and canonical replay must continue to pass.

## Frozen Qwen deployment

The selected model is the official multimodal Qwen3.5-9B deployment through
Ollama:

- Upstream tag: `qwen3.5:9b-q4_K_M`
- Project alias: `qwen3.5-9b-sam3`
- Modelfile: `configs/ollama_qwen3_5_9b_fast.Modelfile`
- Context: 8192 tokens
- Generation ceiling: 512 tokens
- Temperature: 0.2
- `reasoning_effort`: `none` (non-thinking mode)
- Per-request client timeout: 45 seconds
- OpenAI SDK hidden retries: disabled (`max_retries=0`)
- JSON response mode: enabled
- Ollama runtime guidance: one parallel request, one loaded model, Flash
  Attention when supported.

Do not revert to Qwen2.5, a separate “instruct” model, a large context, or
thinking mode. Do not change Ollama/vLLM infrastructure in this milestone.

Cluster environment:

```bash
export QWEN_BASE_URL="http://127.0.0.1:11434/v1"
export QWEN_MODEL="qwen3.5-9b-sam3"
export QWEN_API_KEY="ollama"
```

## Experiment evidence

### Previous target-only two-round run

```text
final_soft_count:          15.071343731240148
raw_soft_count:            14.24418114934046
count_variance:             3.1809231734838868
committed_target_nodes:     9
node_count:                20
qwen_calls:                 2
sam3_calls:                 6
sam3_tiles:                12
runtime_ms:            120588.844506
qwen_runtime_ms:       117445.58194899946
stop_reason: QWEN_BUDGET
```

### First Qwen3.5-9B M8.3 run

Run ID: `m8_3_2fe4a4ca`

```text
final_soft_count:          14.501705689010466
raw_soft_count:            13.566190387083289
count_variance:             3.924528192648671
committed_target_nodes:     5
node_count:                20
qwen_calls:                 1
sam3_calls:                 4
sam3_tiles:                 4
runtime_ms:             18181.37597500754
qwen_runtime_ms:        16461.62191599433
number_of_replans:          0
discovery_saturated:    false
stop_reason: NO_VALID_ACTIONS
```

Artifact on the cluster:

`/home/ekielar/sam3-vlm-2/runs/qwen35_9b_smoke/M8.3/m8_3_2fe4a4ca/artifacts/qwen/qwen_000001.json`

The persisted Qwen artifact contained:

```text
proposed_actions: []
rejections: []
repair_attempted: false
fallback_used: false
qwen_runtime_ms: 16461.62191599433
```

This is decisive: the model returned a valid, empty action list. The action
bank did not reject anything. The four SAM3 calls were bootstrap calls, so no
Qwen-derived target experiment ran. The unchanged 20-node graph but lower raw
mass, higher variance, and five rather than nine committed nodes reflect less
semantic evidence—not fewer discovered candidates.

Latency improved by about 6.6x end-to-end versus the earlier two-call run. The
model/runtime choice is therefore successful; the pending problem is the
planner contract.

## Pending implementation 1 — unsaturated M8 must not silently abstain

Implement the smallest clear change satisfying all of the following:

1. The numerical controller, not Qwen, owns stopping decisions.
2. In strict/canonical M8, when
   `evidence_pack.discovery_diagnostics["discovery_saturated"]` is false or
   absent, Qwen must return exactly one novel target `DISCOVERY` proposal.
3. That action must continue to satisfy the existing M8 contract:
   - `semantic_key == "target"`
   - `family == "DISCOVERY"`
   - `semantic_prior == {"target": 1.0}`
   - a two- or three-word visible SAM3 grounding prompt
   - `GLOBAL` or `TILED`
   - no ROI/geometry
   - not an exact prompt in `tried_sam3_prompts`.
4. An empty `proposed_actions` list is permitted only when discovery is
   explicitly saturated.
5. Strengthen the real Qwen system/dynamic prompt with direct language such as:
   - Qwen never decides whether the pipeline should stop.
   - If discovery is not saturated, `proposed_actions` must contain exactly one
     novel target experiment even when current candidates look convincing.
   - Only the controller may stop after evaluating sensor evidence and budget.
6. If Qwen still returns an empty list while discovery is unsaturated, do not
   fabricate detections, silently invent an arbitrary action, or hide a model
   call. Preserve a clear `EMPTY_UNSATURATED_PLAN` contract diagnostic in the
   Qwen artifact metadata and terminate interpretably under the existing
   no-valid-action behavior.
7. Do not spend the second Qwen budget call merely as a hidden SDK retry. It is
   intended for an evidence-driven replan after a valid target experiment.
8. Keep the current maximum of one executable action per Qwen round.
9. Update the current M8 policy at the top of `docs/V4_DESIGN_SPEC.md` before or
   alongside implementation so the source of truth and code agree.

Required tests should cover at least:

- the real-Qwen prompt payload explicitly requires one action when discovery
  is not saturated;
- the prompt explicitly permits abstention only when discovery is saturated;
- an unsaturated empty planner output persists
  `EMPTY_UNSATURATED_PLAN` in the Qwen artifact metadata;
- no fake SAM3 action is created for that violation;
- saturated empty output remains permitted;
- generic/non-M8 planner behavior remains frozen;
- existing malformed/fenced JSON and API-key isolation regressions keep
  passing.

Prefer a small helper or small diagnostic field over a new abstraction layer.
Do not broaden the action ontology.

## Pending implementation 2 — artifact-path hardening

A cluster run completed, but the follow-up inspection attempted to open
`/summary.json` and `/artifacts/qwen`. Those leading-root paths proved that the
shell variable `LATEST_M8_RUN` was empty. A likely cause was an unset or empty
`M8_OUTPUT_ROOT`. Currently, passing `--output_dir ""` is accepted and causes
M8.3 to write under a relative `M8.3/` directory, which is easy to lose.

Make this failure mode explicit and simple:

1. Reject an empty or whitespace-only CLI/config output directory with a clear
   error. Do not allow it to silently become the current directory.
2. Resolve/normalize the chosen run output path consistently.
3. At M8.3 start, log the absolute artifact directory.
4. After successful finalization and validation, log the absolute
   `summary.json` path.
5. For pilot completion, log the absolute `pilot_report.json` path.
6. Do not make M8.2 create a fake summary; M8.2 is intentionally only a planner
   smoke test. Document/log that distinction if needed.
7. Add tests for empty-output rejection and successful summary creation/path
   reporting.

## Acceptance criteria

Before handing changes back:

1. `git diff --check` passes.
2. Focused new tests pass.
3. Full `pytest -q` passes.
4. `bash scripts/check_m8_cluster_ready.sh` passes.
5. No real-model network/GPU call is required locally.
6. Summarize exactly which files changed and why.
7. Provide simple interactive-shell commands—no Slurm—to pull and rerun M8.2
   and the same M8.3 image on the GPU cluster.
8. Ask the user to return the new `summary.json` plus the Qwen artifact's
   proposed actions, rejections, repair/fallback flags, and runtime.

Expected cluster validation sequence after the patch:

```bash
cd /home/ekielar/sam3-vlm-2
source .venv/bin/activate
python -m pip install -e .

export QWEN_BASE_URL="http://127.0.0.1:11434/v1"
export QWEN_MODEL="qwen3.5-9b-sam3"
export QWEN_API_KEY="ollama"
export M8_TARGET="green citrus"
export M8_IMAGE="/absolute/path/to/the/same/test/image.jpg"
export M8_OUTPUT_ROOT="$(pwd)/runs/qwen35_9b_smoke"

bash scripts/check_m8_cluster_ready.sh

time python -m sam3_vlm.experiments.m8_smoke \
  --stage M8.2 \
  --require-cuda \
  --target "$M8_TARGET" \
  --qwen-base-url "$QWEN_BASE_URL" \
  --qwen-model "$QWEN_MODEL" \
  --output_dir "$M8_OUTPUT_ROOT"

time python -m sam3_vlm.experiments.m8_smoke \
  --stage M8.3 \
  --require-cuda \
  --image "$M8_IMAGE" \
  --target "$M8_TARGET" \
  --qwen-base-url "$QWEN_BASE_URL" \
  --qwen-model "$QWEN_MODEL" \
  --output_dir "$M8_OUTPUT_ROOT"
```

Do not begin the five-image A/B/C/D pilot until this same-image M8.3 run
actually executes a valid Qwen-derived target action and its validator/replay
passes. Once it does, retain all four pilot variants and proceed with the
existing runbook.
