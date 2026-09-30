# Supported SAM3-VLM A–E pipeline

This specification is authoritative for the supported A–E counting workflow. The scene graph, semantic action bank, belief updates, Qwen contract, stopping policy, and replay architecture are described in [HISTORICAL_M8_DESIGN_SPEC.md](HISTORICAL_M8_DESIGN_SPEC.md); the requirements below supersede its bootstrap, tiling, and association details. The separate F/MVP controller has been removed.

## Arms and counts

| Arm | Searches | Qwen cap | Reported count |
|---|---|---:|---|
| A | One full-image target search | 0 | Active candidate count |
| B | Target bootstrap, seed refinement, adaptive tiles | 0 | Active candidate count |
| C | B plus one Qwen round | 1 | Sum of target posteriors |
| D | B plus up to two Qwen rounds | 2 | Sum of target posteriors |
| E | B plus exploration until saturation | 100 | Sum of target posteriors |

A/B bootstrap uses threshold 0.20. C/D/E retains the selected D policy: bootstrap threshold 0.25, Qwen target threshold 0.20, confounder threshold 0.50, no exemplars for Qwen target searches, and the existing negative evidence update. E retains its 1000-SAM3 cap and saturation policy. Repairs and corrections consume the ordinary Qwen budget. The fruit CLI remains `python -m sam3_vlm.experiments.m8_smoke --stage pilot --pilot-suite final-ae`; the single-image GUI remains `python -m sam3_vlm.gui.app`.

## Bootstrap

The first model call is a text-only SAM3 search over the entire image, using the first supplied dataset target description. Supported deployment configs now enable `sam3.singularize_prompts`: the executable description is singularized without selecting another label or rewriting the target category. There is no Qwen planning or SAM3 context localization before this call. Bootstrap never selects or locks an ROI. Legacy `locked_context_*` configuration fields are ignored, and the GUI no longer exposes them.

All optional bootstrap refinement and tile calls reuse that same target text. Strong SAM3 seeds (score at least 0.60, at most five) may supply positive pseudoexemplar boxes for one full-image refinement pass. Those boxes come from SAM3 detections, never ground-truth annotations. Only exemplars fully contained in a tile are passed to its sensor crop. A disables both refinement and tiling. B–E preserve the switches for these two features.

## Adaptive tiling

The density calculation and square overlapping tile sizes carry over from the former MVP's [SAM3Count image implementation](https://github.com/Joan947/SAM3Count/blob/main/evaluate_fscd147.py). The decision uses deduplicated nodes from the initial unconditioned target pass, before refinement. Only seeds with SAM3 score at least `adaptive_seed_min_score` (0.50) contribute.

For image area `I`, seed count `N`, and total seed box area `B`, calculate:

```
coverage = B / I
size_score = 1 - min((B / N / I) / 0.1, 1)
density = 0.3 * coverage + 0.5 * min(N / 50, 1) + 0.2 * size_score
```

Density strictly above `adaptive_density_threshold` (0.69) activates the SMALL regime: square tile size `clip(int(max(W/6, H/4)), 97, 1024)`, overlap `int(tile_size * 0.25)`, and a row-major grid. Supported presets also enable `adaptive_enable_fallback`: at least eight deduplicated candidates whose box area is at most 0.003 of image area activate the same SMALL grid, even when few candidates reach the strong-seed score. If there are no strong seeds and insufficient small candidates, a coarse grid uses square size `clip(ceil(0.6 * max(W,H)), 97, 1024)` to search at a different scale. No ground truth is used. A confident sparse scene with larger instances skips both fallbacks. Plans record `trigger_reason` (`density`, `small_objects`, `uncertain_scale`, or `none`), candidate counts, and the original density score.

Edge positions are clamped and deduplicated. Every part of the full image is covered, including places where Stage 1 found no objects. No detection-union ROI or VLM ROI filters the grid. A tile identical to the already searched full image is skipped.

Each adaptive tile is a separate LOCAL sensing action tagged with a `tile_id`. It consumes one actual SAM3 call and one tile budget unit. Bootstrap checks call, tile and runtime limits before requests; exhaustion retains partial artifacts and is an explicit failure. The deployment config allows 128 SAM3 requests and 128 tiles for ordinary runs. These are caps, not guaranteed usage. The historical fixed-grid path is available only when adaptive tiling is disabled.

## Mask association

Every supported A–E configuration enables `association.mask_only` and mask IoM. Within-call suppression, cross-pass matching, containment, and duplicate-risk diagnostics use binary mask IoU and mask IoM only. There is no bounding-box fallback, tile-box association, or box NMS in these runs. Missing, malformed, or empty masks fail explicitly instead of silently changing the association rule. The old box policies remain for historical synthetic fixtures only.

Masks are trimmed to their nonzero bounds and carry integer original-image offsets. Overlap uses aligned pixels across those offsets. Boxes still serve density estimation, display, crops, pseudoexemplars, and COCO box exports. Identical or overlapping boxes do not merge disjoint masks. Within-call suppression uses the lower of its configured threshold and the cross-call registration threshold, independently for IoU and IoM, so duplicates cannot survive simply because they arrived in one call.

IoM containment requires a mask area ratio at most `iom_max_area_ratio` (4), unless the smaller mask touches an interior crop boundary. The real sensor records that evidence before mask trimming; touching the original image edge alone is insufficient. IoU still applies independently. A coarse mask containing at least two comparable smaller masks, with pairwise IoM below 0.1, is rejected in favor of those individual instances. Children must not be crop-clipped, must each be contained at the registration IoM threshold, and must each be more than four times smaller than the parent. A later target discovery can retire an existing coarse parent; negative or verification actions cannot. This is a geometric heuristic, requiring visual validation on nested-object scenes. Retirement is logged for replay. Canonical masks grow only on a single strong association, preventing weak matches from expanding a mask and absorbing neighbors.

Persistent node JSON contains mask area, offset, crop-boundary provenance, and an NPZ artifact pointer; binary arrays stay outside JSON. Mask artifacts include their pixel data, and observation records include offsets. Replay reconstructs graph metadata and beliefs without loading model weights or inlining masks.

## Counting unit and confidence diagnostics

With `sam3.singularize_prompts`, bootstrap, refinement, fixed/adaptive tiles, all Qwen target/negative queries, and optional cleanup use deterministic English noun inflection (`inflect==7.5.0`). For example, `polka dots` becomes `polka dot`, `bottle caps` becomes `bottle cap`, `chicken wings` becomes `chicken wing`, and `leaves` becomes `leaf`. Already singular words such as `glass`, `citrus`, and `lens`, uninflected words, and naturally plural names for one object such as `scissors` and `sunglasses` are protected. Compound modifiers are inflected too: `donuts tray` becomes `donut tray`; morphology does not decide whether that target should instead mean individual donuts.

The original dataset label remains in the run manifest, prediction target, scene state, and Qwen evidence. Raw Qwen proposals remain auditable; normalization precedes prompt validation, duplicate checks, action creation, and semantic history. Singular/plural forms cannot consume separate queries. Observation events and compact summaries record the actual executed SAM3 text. Qwen receives singular-query guidance and the normalized target alongside its original text. Config and CLI `--singular-prompts`/`--no-singular-prompts` permit a controlled comparison on identical samples without changing count, belief, tiling, or budget policies. Historical fixtures retain the opt-out dataclass default.

All Qwen instruction versions preserve the supplied target's individual counting unit. For targets such as polka dots, search variants must describe individual dots, not their supporting sphere or fabric. Parent surfaces containing targets must not serve as negative queries. Vocabulary remains open, with the existing short-prompt and target-action contract; these are planner instructions rather than a hardcoded category allowlist.

Confidence traces record candidate-minus-soft-count gaps, newly recovered target mass, existing-node gains and losses, removed-node mass, and changes grouped by observation relation. The compact FSCD summary aggregates these by action family, separating target recovery from posterior suppression caused by positive misses or negative matches. C/D/E retain their existing posterior formula and soft count; the diagnostics do not recalibrate probabilities from this small sample. E's extended search and budgets remain unchanged.

## FSCD-147 inference and evaluation

Optional summary `--annotation-audit FILE.json` accepts a mapping from split image IDs to a string `note` and optional nonnegative integer `audited_count`. Unknown IDs and malformed entries fail. Audit counts receive separate count metrics over audited images only, with completeness checks; they never replace official ground truth, affect inference, or change official AP/MAE/RMSE. Notes alone do not establish replacement counts.

`python -m sam3_vlm.experiments.fscd147 run DATASET_ROOT OUTPUT_DIR --split val --config configs/fscd147.json` runs all five arms through this same pipeline. `--arm A` through `--arm E` select one arm; `--max-images N` runs a smoke subset. `--dry-run` validates the split, class names and image paths without models. A/B need no Qwen client.

Inference reads only `Train_Test_Val_FSC_147.json`, `ImageClasses_FSC147.txt`, and `images_384_VarV2/`. Each image's class text is its bootstrap prompt. The FSCD config's Qwen scope counts all visible target instances throughout the image, including partial occlusions, without a fruit/tree restriction. Ground-truth exemplar boxes are not used.

The output contains frozen configuration/model metadata, flushed per-image/arm `predictions.jsonl`, errors, candidate boxes/scores, counts, model budgets, and regular run artifacts. A failed image remains explicit and does not stop subsequent images. Use a fresh output directory; existing predictions are never overwritten.

`python -m sam3_vlm.experiments.fscd147 evaluate DATASET_ROOT OUTPUT_DIR/predictions.jsonl --split val` loads `instances_val.json` or `instances_test.json` separately. FSCD box annotations determine true counts. FSC point annotations are optional and remain evaluation-only. An empty annotation list is a valid zero count when the dataset specifies one category.

Evaluation validates frozen split/arm metadata, rejects duplicate/unknown predictions, and reports MAE/RMSE for an arm only when the entire split completed successfully. Errors, missing images, truncated subsets, or runs exhausted by SAM3/tile/runtime/iteration limits suppress complete-split metrics and detection exports. Normal Qwen caps and saturation are valid termination conditions.

Every complete arm exports COCO boxes for all active candidates: A/B use their maximum sensor score; C/D/E use the target posterior. No extra score threshold alters their count rule. Coordinates are scaled to the COCO annotation image dimensions when they differ from inference dimensions. Optional `--ap` uses `pycocotools` to report COCO bbox AP/AP50 with an explicit 1000-detection cap per image; install `pip install -e '.[evaluation]'`. These AP settings are recorded separately from count metrics and may differ from a paper's custom evaluator.

Tune on validation and freeze the config for test. Unit and mocked integration tests verify behavior; they do not establish real-model accuracy or performance gains.
