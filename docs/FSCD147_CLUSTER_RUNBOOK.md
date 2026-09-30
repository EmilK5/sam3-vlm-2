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

## 8. Run the complete validation split only after reviewing the gallery

Use the **original dataset root**, a fresh output directory, and the same config:

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
