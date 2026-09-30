# FSCD-147 on a GPU cluster: ten images first

Use the same direct shell workflow as the previous runbooks: connect to your
GPU machine, keep Ollama running in one terminal, and run Python in another.
Assumptions: Linux x86-64, an NVIDIA GPU available to your shell, and Python
3.11 or newer. Substitute your GPU hostname and dataset path.
All five arms use the same ten sampled validation
images: 50 image/arm runs. Sampling uses seed 42 and never ranks by ground truth.

## 1. Transfer the current code from your Mac

The current implementation is uncommitted. A clone/pull of GitHub alone will not
include it. Use a fresh remote directory and copy the local repository contents:

```bash
ssh USER@LOGIN 'mkdir -p ~/sam3-vlm-2-fscd'
rsync -av \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='outputs/' --exclude='runs/' \
  /Users/emilkielar/Projects/sam3-vlm-2/v4/ \
  USER@LOGIN:~/sam3-vlm-2-fscd/
ssh USER@LOGIN
```

Use the fresh directory so previously deleted F files cannot remain in a reused
checkout. Alternatively, commit and push these changes, then clone that commit.

## 2. Set up Python and model access

On the cluster, load its Python/CUDA modules if required by local instructions.
Use a Python executable backed by the cluster's supported CUDA PyTorch build,
or install the appropriate wheel using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
cd ~/sam3-vlm-2-fscd
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,evaluation]' huggingface_hub
python -m pytest -q
```

Accept access to [facebook/sam3](https://huggingface.co/facebook/sam3) with your
Hugging Face account, then authenticate with a read token:

```bash
hf auth login
hf download facebook/sam3
```

Use shared storage for the Hugging Face cache if login and GPU nodes have
different home directories. Set `HF_HOME` before both authentication/downloads
and inference. If compute nodes have no outbound network, download weights on
a node with network access first and make the cache visible to the GPU node.

Install Ollama if it is not already available. For Linux x86-64, this adapts
the [official manual installation](https://docs.ollama.com/linux) to your home
directory; it needs `tar` with zstd support and no sudo:

```bash
mkdir -p "$HOME/.local/ollama"
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama-linux-amd64.tar.zst
tar --zstd -xf /tmp/ollama-linux-amd64.tar.zst -C "$HOME/.local/ollama"
export PATH="$HOME/.local/ollama/bin:$PATH"
ollama --version
```

## 3. Open your normal GPU shell

Connect to the GPU machine as you did for the previous runbooks. If your current
shell already has GPU access, continue there. Run the commands below in that shell:

```bash
cd ~/sam3-vlm-2-fscd
source .venv/bin/activate
export PATH="$HOME/.local/ollama/bin:$PATH"
nvidia-smi
python - <<'PY'
import torch
from transformers import Sam3Model, Sam3Processor
assert torch.cuda.is_available(), "Check GPU access and your CUDA-enabled PyTorch build"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Both terminals should connect to the same GPU machine so that the localhost
Ollama endpoint is reachable from Python. Keep any existing GPU visibility
settings used by your cluster.

## 4. Start Qwen on that GPU node

In terminal 1, start Ollama in the foreground as in the earlier runbooks. If
your existing Ollama server is already running on this GPU machine, keep it
and skip this startup block. The examples use the usual port 11434.

```bash
mkdir -p logs
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

Leave terminal 1 running. In terminal 2 on the same GPU machine, activate the
environment and check readiness; continue once curl succeeds:

```bash
cd ~/sam3-vlm-2-fscd
source .venv/bin/activate
export PATH="$HOME/.local/ollama/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p logs
curl --fail http://127.0.0.1:11434/api/version
ollama pull qwen3.5:9b-q4_K_M
ollama create qwen3.5-9b-sam3 -f configs/ollama_qwen3_5_9b_fast.Modelfile
ollama show --modelfile qwen3.5-9b-sam3
export QWEN_BASE_URL=http://127.0.0.1:11434/v1
export QWEN_MODEL=qwen3.5-9b-sam3
export QWEN_API_KEY=ollama
```

The current checked-in profile uses **65536 context tokens** and a **512-token
output limit**. Recreate the alias even if it already exists so it uses this
profile. Do not silently reduce the context during a comparison. If GPU memory
is exhausted, use a GPU with more memory or a separate Qwen endpoint.
To pre-download Qwen where compute nodes have no network, run `ollama pull`
against your own Ollama server on a permitted node and use the same shared
`OLLAMA_MODELS` directory for both servers.

## 5. Create the random ten-image dataset view

Your original dataset must have this layout (validation first):

```text
/shared/datasets/FSCD-147/
  images_384_VarV2/
  Train_Test_Val_FSC_147.json
  ImageClasses_FSC147.txt
  instances_val.json
  instances_test.json                    # needed later for test evaluation
  annotation_FSC147_384.json              # optional
```

```bash
export FSCD_ROOT=/shared/datasets/FSCD-147
export FSCD_SAMPLE="$PWD/data/fscd_val10_seed42"
export FSCD_RUN="$PWD/outputs/fscd_val10_seed42_ae"
python -m sam3_vlm.experiments.fscd147_smoke prepare \
  "$FSCD_ROOT" "$FSCD_SAMPLE" --split val --count 10 --seed 42
python -m sam3_vlm.experiments.fscd147 run \
  "$FSCD_SAMPLE" "$FSCD_RUN" --split val \
  --config configs/fscd147.json --dry-run
```

Expect `split_images: 10`, `run_images: 10`, and five variants A–E. The helper
creates ten image symlinks, a split containing just those images, filtered
evaluation annotations, and `sample_manifest.json`. It leaves the original
dataset intact. Keep the source images available while using the symlinks.

`--max-images 10` on the full root instead takes the **first** ten images;
it does not sample randomly, and partial full-split runs cannot receive complete
split metrics. Use the prepared root above for a scored random smoke test.

## 6. Run A–E on just those ten images

```bash
python -u -m sam3_vlm.experiments.fscd147 run \
  "$FSCD_SAMPLE" "$FSCD_RUN" --split val \
  --config configs/fscd147.json \
  > logs/fscd_val10.log 2>&1
```

No image limit is needed: the prepared split contains exactly ten images.
The default runs all five arms serially, yielding 50 prediction rows. E retains
its larger search budget and may take substantially longer. A faster optional
first check uses `--arm D` with a different output directory (ten runs).

Check progress from another shell on the same node:

```bash
wc -l "$FSCD_RUN/predictions.jsonl"
ollama ps
nvidia-smi
tail -n 30 logs/fscd_val10.log
```

Each prediction is flushed after an image/arm finishes. The CLI exits nonzero
if any row failed; still run the review/evaluation commands to inspect failures
if predictions were written. Use a new output directory for any rerun. A model
load failure before inference may leave no predictions file.

## 7. Score and inspect before scaling up

```bash
python -m sam3_vlm.experiments.fscd147 evaluate \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val --ap
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
```

Expect `50/50` successful runs in the review result. Metrics are in
`outputs/fscd_val10_seed42_ae/evaluation/metrics.json`. Inspect MAE/RMSE and
failure/completeness status per arm; AP uses bounding boxes with a 1000-detection
limit. These are ten-image diagnostics, not full benchmark results.

From your Mac, download the portable gallery and open `index.html` in a browser:

```bash
rsync -av USER@LOGIN:~/sam3-vlm-2-fscd/outputs/fscd_val10_seed42_ae/review/ \
  ~/Downloads/fscd_val10_review/
```

The gallery shows green ground-truth boxes and red candidate boxes for each
arm, with counts and runtime. Look for missed dense clusters, false positives,
duplicates at tile seams, and fragments replacing whole objects. C–E use soft
counts: the count is the sum of target probabilities and can differ from the
number of red candidate boxes. Ground truth is accessed only during preparation
of evaluation files and post-inference scoring/review, never by the controller.

### Share results without uploading the gallery

Generate a compact report directly from the saved predictions; this never
reruns SAM3/Qwen and does not open image or mask files:

```bash
python -m sam3_vlm.experiments.fscd147_smoke summary \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
```

The command writes `report/summary.md`, `report/summary.json`,
`report/visual_notes.md`, and `report/summary.zip` below the run directory.
The ZIP contains only the three small text files. Share it, or paste
`summary.md`, along with your visual observations. It includes per-arm count
metrics, paired error changes, per-image counts, worst errors, failures,
runtime/calls, tiling decisions, and available Qwen rejection diagnostics.
Full configs and extra numeric diagnostics are in the JSON. Failed and missing
runs remain explicit; they are never counted as zero predictions.

Optionally fill in `visual_notes.md` with missed objects, false positives,
duplicates, fragments, and slow behavior, naming image IDs and affected arms.
Rerun the same command to include those notes in the report and ZIP; your notes
are preserved. Add `--ap` to compute fresh bbox AP, or `--no-artifacts` when
only predictions and evaluation annotations are available. Existing evaluation
files are left intact. Default detail limits are ten images and three worst
images per arm; use `--max-images` and `--top-k` to change report detail limits.
For this command these flags limit report content, not inference.

### Rerun the same ten images after the deduplication fixes

Copy the updated source and config to the cluster using the setup steps above,
and keep the existing Ollama server and prepared sample. Activate the same
Python environment. Choose a fresh output directory so the earlier results
remain available for comparison:

```bash
export FSCD_SAMPLE="$PWD/data/fscd_val10_seed42"
export FSCD_RUN="$PWD/outputs/fscd_val10_seed42_ae_dedup_v2"
python -u -m sam3_vlm.experiments.fscd147 run \
  "$FSCD_SAMPLE" "$FSCD_RUN" --split val \
  --config configs/fscd147.json --no-singular-prompts > logs/fscd_val10_dedup_v2.log 2>&1
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
python -m sam3_vlm.experiments.fscd147_smoke summary \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
```

The new summary records each tiling trigger reason and separates new candidate
target mass from changes to existing candidates, grouped by positive and negative
action families. Candidate counts can improve while soft counts remain low;
compare both. The posterior formula is unchanged. The small-object fallback adds
tile calls to scenes that previously skipped them, so runtime may increase.

### Record suspected annotation gaps separately

Create an optional `annotation_audit.json` mapping image filenames from the
sample manifest to notes. For example, if `7314.jpg` belongs to this sample:

```json
{
  "7314.jpg": {"note": "Visible skateboard side stacks appear to lack annotations."}
}
```

After manually counting the intended target instances, add an integer
`"audited_count"` to that entry. Do not estimate a replacement count from model
predictions. Entries without a manual count are notes only.

```bash
python -m sam3_vlm.experiments.fscd147_smoke summary \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val \
  --annotation-audit annotation_audit.json
```

Official MAE/RMSE and AP continue to use the original annotations. The report
adds a separately labelled count evaluation only for manually audited images,
and leaves metrics unavailable if any of those predictions failed or are missing.
The audit never enters inference and does not modify dataset annotations.

### Singular-prompt comparison on the same sample

After copying the latest source from your Mac, activate the existing environment
and reinstall the editable package to add its pinned `inflect` dependency. Keep
the same Ollama endpoint, model profile, and ten-image sample:

```bash
python -m pip install -e .
export FSCD_SAMPLE="$PWD/data/fscd_val10_seed42"
export FSCD_RUN="$PWD/outputs/fscd_val10_seed42_ae_singular_v1"
python -u -m sam3_vlm.experiments.fscd147 run \
  "$FSCD_SAMPLE" "$FSCD_RUN" --split val \
  --config configs/fscd147.json --singular-prompts \
  > logs/fscd_val10_singular_v1.log 2>&1
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
python -m sam3_vlm.experiments.fscd147_smoke summary \
  "$FSCD_SAMPLE" "$FSCD_RUN/predictions.jsonl" --split val
```

All five arms use singularized executable prompts; original dataset targets stay
in metadata. The summary's executed-query list lets you verify the conversion.
Singularization is enabled in the current deployment configs; the explicit flag
makes the experiment clear. For a fresh matched plural control, repeat with
`--no-singular-prompts`, another unused output directory, and another log file.
Every other policy remains the same. Count agreement alone cannot establish
better mask quality, so inspect the gallery as well as the metrics.

## 8. Test the new policies on the same ten images

Copy the updated source using step 1. Reuse your existing environment, sample,
and Ollama model/endpoint. In the Python terminal on your GPU machine:

```bash
cd ~/sam3-vlm-2-fscd
source .venv/bin/activate
python -m pip install -e .
mkdir -p logs
export FSCD_SAMPLE="$PWD/data/fscd_val10_seed42"
export FSCD_SUITE="$PWD/outputs/fscd_val10_policy_suite_v1"
python -m sam3_vlm.experiments.fscd147_ablation run \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val \
  --config configs/fscd147.json --dry-run
```

The dry run should report 10 images, seven profiles, and **270 runs**. It checks
image paths without loading models or annotations. The suite refuses other split
sizes unless you explicitly supply `--allow-full-split`.

| Profile | Arms | Change relative to singular control |
|---|---|---|
| `plural_control` | A–E | Original plural prompt policy |
| `singular_control` | A–E | Singular noun phrases only |
| `safe_negatives` | C–E | Require a distinct-object semantic assessment before negative sensing |
| `neutral_appearance_misses` | C–E | No negative evidence from misses on Qwen appearance variants |
| `adaptive_e` | E | Stop requesting plans after three consecutive zero-gain target searches |
| `counting_unit` | A–E | Override dataset label `donuts tray` with inference target `donut` |
| `combined` | A–E | All four additional policies together |

The plural/singular pair isolates inflection. Each subsequent isolated profile
is compared with singular control. A/B are omitted from experiments that cannot
affect their execution. All profiles retain their arm's thresholds and hard
budgets, the same models, and controller seed. Qwen sampling may still vary;
repeat promising comparisons before drawing conclusions from ten images.

Launch from the normal GPU shell, with Ollama still running in terminal 1:

```bash
python -u -m sam3_vlm.experiments.fscd147_ablation run \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val \
  --config configs/fscd147.json > logs/fscd_val10_policy_suite_v1.log 2>&1
```

The suite loads SAM3 and the Qwen client once and runs profiles serially. A/B
remain SAM3-only. Every completed image/arm is flushed to its profile's
`predictions.jsonl`; the log prints its success, count and runtime. In another
terminal, follow progress with:

```bash
tail -f ~/sam3-vlm-2-fscd/logs/fscd_val10_policy_suite_v1.log
```

The E policy checks its streak after all queued actions finish. Bootstrap and
negative queries do not count toward the streak; recovering a new active
candidate resets it. The original 100-Qwen/1000-SAM3 ceilings remain. A newly
recovered false positive can also reset this heuristic, so inspect candidate
quality along with runtime. C/D retain their existing stopping behavior.

Negative validation requires Qwen to assess every negative label as a distinct
object, target synonym, subtype, part, container, or uncertain relation, with a
reason. Only distinct objects are eligible. Missing, mismatched, and duplicate
assessments are rejected. A lexical guard also rejects negative labels containing
the target noun. This gate still depends on Qwen's judgment; an incorrect
distinct-object assessment can pass. Assessments and rejections are logged.

Neutral appearance misses apply when a Qwen discovery phrase differs from the
canonical target after singularization. Positive matches still update belief;
neutral misses do not discount later positive evidence. Canonical target queries,
bootstrap, verification and negative-query evidence retain their existing rules.

Each profile saves a `mask_audit.json` per image/arm. It reports mask IoU/IoM
overlap counts and up to twenty pairs with areas, target probabilities and crop
provenance. It does not modify the graph or merge anything. High overlap and
candidate excess are review signals, not proof of duplicates. Missing mask pixels
make the audit incomplete; boxes never substitute for masks.

On completion, inspect `$FSCD_SUITE/comparison.md`. Positive paired error
reduction means better count accuracy; negative runtime change means faster.
Failures and missing images leave complete metrics unavailable. Official
annotations stay unchanged, including in the counting-unit experiment.
The single `$FSCD_SUITE/summary.zip` contains all seven compact summaries,
comparison metrics, per-profile visual-notes templates, and the frozen suite
manifest. It contains no pictures or mask arrays.

Create a gallery for the combined profile, or substitute any other profile name:

```bash
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_SUITE/combined/predictions.jsonl" --split val
```

Add your observations to each profile's `report/visual_notes.md`, then rebuild
the final bundle without running inference again:

```bash
python -m sam3_vlm.experiments.fscd147_ablation report \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val
```

From your Mac, download the bundle (replace `USER@LOGIN`):

```bash
scp USER@LOGIN:~/sam3-vlm-2-fscd/outputs/fscd_val10_policy_suite_v1/summary.zip \
  ~/Downloads/fscd_policy_suite_summary.zip
```

For a smaller run, `--profiles plural_control singular_control combined` runs
150 image/arm combinations. Include the relevant control if you want a paired
comparison. Use a fresh output directory for each suite; there is no resume
support. An interrupted suite can still be reported, with missing runs explicit.
Normal `fscd147 run` keeps the new policy flags disabled; singularization remains
enabled in the deployment config as before.

## 9. Final focused ablation: v3, ten images and 30 runs

After reviewing the v2 results, use `--suite final` for one focused round before
full validation. Reuse **the existing** `data/fscd_val10_seed42` subset so all
three choices see the same ten images. This selects two profiles:

| Output folder | Arms | Policy |
|---|---|---|
| `final_reference` | D, E | Singular prompts and `donuts tray` → `donut` |
| `final_adaptive` | E | Same settings, stop after three consecutive zero-gain Qwen target searches |

Both profiles leave semantic negative validation and neutral appearance misses
disabled. Bootstrap remains full-image, uses the first dataset prompt, and
retains adaptive tiling. Association uses only mask IoU/IoM. Empty **valid binary**
masks are skipped and counted in event logs and summaries; missing or malformed
masks still fail. Empty proposals create no nodes, and their sensing calls,
runtime and searched regions remain accounted for.

Both profiles now request compact JSON, with a 1024-token output limit,
temperature 0 and API sampling seed 42. The context stays at 65536. A seed does
not guarantee identical GPU/model outputs. Compare the fresh D/E controls in
this suite, rather than attributing changes against v2 entirely to stopping.
Malformed JSON retains the existing single budgeted repair attempt. Raw initial
and repair responses, finish reasons, token usage, errors and runtime are saved,
even when the run fails. The compact ZIP includes bounded failure-response
examples; full raw responses stay in the run artifacts.

### Step 1: update the existing cluster checkout from your Mac

The current changes are local. Replace `USER@LOGIN` with the same SSH destination
you used previously. The transfer below preserves cluster datasets, models,
environment and previous outputs.

```bash
rsync -av \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='data/' --exclude='outputs/' \
  --exclude='runs/' --exclude='logs/' \
  /Users/emilkielar/Projects/sam3-vlm-2/v4/ \
  USER@LOGIN:~/sam3-vlm-2-fscd/
```

### Step 2: use your normal GPU shell

Connect to the same GPU machine as previous runs, with the project directory
visible there. No scheduler command is needed in this runbook. In terminal 2:

```bash
cd ~/sam3-vlm-2-fscd
source .venv/bin/activate
python -m pip install -e .
mkdir -p logs
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Keep your existing Hugging Face authentication/cache and GPU visibility settings.

### Step 3: start or reuse Ollama on that same GPU machine

If it is already serving port 11434, leave it running. Otherwise, in terminal 1:

```bash
cd ~/sam3-vlm-2-fscd
export PATH="$HOME/.local/ollama/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

Leave terminal 1 open. Back in terminal 2, create the separate final-run alias
using your existing downloaded Qwen weights:

```bash
export PATH="$HOME/.local/ollama/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
curl --fail http://127.0.0.1:11434/api/version
ollama create qwen3.5-9b-sam3-final -f configs/ollama_qwen3_5_9b_final.Modelfile
ollama show --modelfile qwen3.5-9b-sam3-final
export QWEN_BASE_URL=http://127.0.0.1:11434/v1
export QWEN_MODEL=qwen3.5-9b-sam3-final
export QWEN_API_KEY=ollama
```

Check that the alias shows `num_ctx 65536`, `num_predict 1024`, temperature 0,
and seed 42. Creating this alias reuses model weights; the earlier alias remains
available for historical configurations.

### Step 4: check the dataset and final suite without loading models

```bash
export FSCD_SAMPLE="$PWD/data/fscd_val10_seed42"
export FSCD_SUITE="$PWD/outputs/fscd_val10_policy_suite_v3"
python -m sam3_vlm.experiments.fscd147_ablation run \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val --suite final \
  --config configs/fscd147_final.json --dry-run
```

Expected output:

```json
{
  "images": 10,
  "expected_runs": 30,
  "profiles": {"final_reference": "DE", "final_adaptive": "E"}
}
```

Use a fresh output folder. The suite refuses to overwrite an existing benchmark;
it does not resume an interrupted run. Do not regenerate or replace the subset.

### Step 5: run the 30 image/arm combinations

In terminal 2, with Ollama running:

```bash
python -u -m sam3_vlm.experiments.fscd147_ablation run \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val --suite final \
  --config configs/fscd147_final.json > logs/fscd_val10_policy_suite_v3.log 2>&1
```

The command runs in the foreground and returns when finished. In another shell
on the same machine:

```bash
tail -f ~/sam3-vlm-2-fscd/logs/fscd_val10_policy_suite_v3.log
```

Expect 20 result rows in `final_reference/predictions.jsonl` and ten in
`final_adaptive/predictions.jsonl`. The log prints each image, arm, success, count
and runtime. Keep both inference and Ollama terminals open until completion.

### Step 6: create/recreate the comparison and summary ZIP

The runner builds it automatically on completion. Run this command explicitly
if the archive is missing, or to refresh it after adding visual notes. It reads
saved results and does not run the models again:

```bash
python -m sam3_vlm.experiments.fscd147_ablation report \
  "$FSCD_SAMPLE" "$FSCD_SUITE" --split val
cat "$FSCD_SUITE/comparison.md"
```

Expect `"complete": true` and an archive path ending in
`fscd_val10_policy_suite_v3/summary.zip`. If it says false, the ZIP still contains
available results and failure diagnostics; send it rather than rerunning blind.
The comparison includes reference D → E accuracy/runtime and reference E →
adaptive E paired changes. Positive error reduction is better; negative runtime
change is faster.

### Step 7: inspect predictions and add visual notes

```bash
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_SUITE/final_reference/predictions.jsonl" --split val
python -m sam3_vlm.experiments.fscd147_smoke review \
  "$FSCD_SAMPLE" "$FSCD_SUITE/final_adaptive/predictions.jsonl" --split val
```

From your Mac, optionally download the two galleries:

```bash
scp -r USER@LOGIN:~/sam3-vlm-2-fscd/outputs/fscd_val10_policy_suite_v3/final_reference/review \
  ~/Downloads/fscd_v3_reference_review
scp -r USER@LOGIN:~/sam3-vlm-2-fscd/outputs/fscd_val10_policy_suite_v3/final_adaptive/review \
  ~/Downloads/fscd_v3_adaptive_review
open ~/Downloads/fscd_v3_reference_review/index.html
open ~/Downloads/fscd_v3_adaptive_review/index.html
```

Record misses, noisy detections, duplicates/fragments, dense-scene recovery and
where adaptive E stops too early. On the cluster, edit
`$FSCD_SUITE/final_reference/report/visual_notes.md` and
`$FSCD_SUITE/final_adaptive/report/visual_notes.md`; these files survive report
regeneration. Then repeat step 6 to include your notes in the ZIP.

### Step 8: download the single bundle from your Mac

```bash
scp USER@LOGIN:~/sam3-vlm-2-fscd/outputs/fscd_val10_policy_suite_v3/summary.zip \
  ~/Downloads/fscd_policy_suite_v3_summary.zip
```

Upload `fscd_policy_suite_v3_summary.zip`, with your visual notes in the reports
or chat. It contains nine small Markdown/JSON files, with no images or mask
arrays. If the runs complete without failures and the chosen policy has no
major per-image regressions, proceed to full validation with that policy.

## 10. Run the complete validation split only after reviewing the gallery

After the focused v3 experiment, select and freeze the winning policy before
running full validation. The command below is the earlier A–E baseline workflow;
it does not select a v3 winner or apply the suite's donut override. Use the
**original dataset root** and a fresh output directory for full runs.

```bash
export FSCD_FULL_RUN="$PWD/outputs/fscd_val_ae_v1"
python -u -m sam3_vlm.experiments.fscd147 run \
  "$FSCD_ROOT" "$FSCD_FULL_RUN" --split val \
  --config configs/fscd147.json > logs/fscd_val_full.log 2>&1
python -m sam3_vlm.experiments.fscd147 evaluate \
  "$FSCD_ROOT" "$FSCD_FULL_RUN/predictions.jsonl" --split val --ap
```

Use smoke runtimes to estimate the full run duration; E has no per-image
runtime cap. Keep both the Ollama server and Python shell running. The runner
is serial and has no resume support. An interrupted run is incomplete and
must use a fresh output directory on restart. Use `tmux` if you normally use
it to keep terminal sessions alive across SSH disconnects. Account for
model-load time and substantial mask/event storage.

Tune on validation, then freeze the config. For the final test benchmark,
change both commands to `--split test` and choose another fresh output directory.
When finished, stop your manually started Ollama server with Ctrl-C in terminal
1. Keep an existing shared or managed Ollama service running.
