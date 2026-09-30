# FSCD-147 policy suite v2 analysis

Source ZIP SHA256: `2dd381ccdddcef0079ae10b36f20223ad527427029ba6000bf6f97ac0bc719a7`.

265/270 image/arm runs succeeded. All seven profiles are present. Five run failures explain complete=false.

| Profile | C MAE | D MAE | E MAE | E mean seconds |
|---|---:|---:|---:|---:|
| plural_control | 21.24 | 18.62 | 14.77 | 46.98 |
| singular_control | 12.88 | 12.69 | 15.80 | 51.84 |
| safe_negatives | 16.43 | 15.10 | 15.89 | 47.27 |
| neutral_appearance_misses | 13.91 | 12.56 | Incomplete | 43.78 |
| adaptive_e | — | — | 14.66 | 32.47 |
| counting_unit | 14.16 | 11.09 | 12.50 | 50.84 |
| combined | 14.19 | 15.17 | Incomplete | 26.86 |

## Failures

- singular_control, B, 873.jpg: Empty SAM3 mask.
- neutral_appearance_misses, E, 7314.jpg: Empty SAM3 mask.
- counting_unit, B, 873.jpg: Empty SAM3 mask.
- combined, B, 873.jpg: Empty SAM3 mask.
- combined, E, 3485.jpg: Malformed Qwen response (root cause not established by this compact report).

## Findings

1. Singular prompts improved C MAE from 21.24 to 12.88 (39.4%) and D from 18.62 to 12.69 (31.8%). E worsened from 14.77 to 15.80. This is one ten-image run, with planner sampling variation.
2. Counting-unit correction has direct evidence: the SAM3-only A count on 3775.jpg changed from 1 to 12 with GT 11. All nine other A counts were identical. Counting-unit C/D/E on that image gave 9.88/10.12/11.36. The profile has the best complete D MAE (11.09), but changes on non-donut images cannot be attributed to the donut override.
3. Adaptive E cut mean time from 51.84 to 32.47 seconds (37.4%), with full-sample MAE 15.80 to 14.66. Six images terminated through the new low-marginal-utility policy. This supports retaining adaptive exploration for further evaluation.
4. Neutral appearance misses improved D only slightly (12.69 to 12.56); C worsened. E failed on the skateboard image. On the same nine successful images, experiment MAE is 10.72 versus reference 14.05. This is a partial comparison, not a complete ten-image result. Dense-cap discovery miss losses disappeared in the neutral profile, showing the intended evidence behavior.
5. Safe negatives worsened aggregate C/D and left E nearly unchanged. Nevertheless, on book image 873.jpg D improved from 11.44 to 19.10 with GT 20. The gate is semantically useful but unreliable: Qwen sometimes marks a book spine or the pink dot-bearing sphere as distinct_object while its own reason describes a part or container. The lexical guard catches book spine; synonyms without the target noun can still pass.
6. Combined C/D did not beat singular control. E is incomplete; on the matched nine images its MAE is 12.08 versus 15.99, and runtime is 27.77 versus 53.53 seconds. Combined E overshot 873.jpg (29.99 versus GT 20). Neutral misses and safer negatives do not address excessive or fragmented discoveries.

## Mask audit

All 265 successful runs have complete mask audits and zero final active-mask pairs with IoU >= 0.30. There are 257 high-IoM pair occurrences across runs; all have area ratio > 4, and none of their smaller masks are crop-boundary-clipped. These are intentionally excluded by the current containment dedup rule. Counts repeat the same images across arms/profiles, so 257 is not a count of unique physical duplicates.
The largest overlap IoU among these examples is 0.2434. Book and skateboard pairs include likely whole-object/part ambiguity, requiring visual review. This audit does not detect non-overlapping fragments, false positives or annotation gaps. The dense cap scene has zero high-overlap pairs despite 570+ candidates against GT 435; simply lowering overlap thresholds is not supported by these findings.

## Interpretation limits

Singular control and counting-unit profiles have identical model configurations, with the only semantic override being donuts tray -> donut. Yet several non-donut C/D/E predictions and Qwen proposals differ. The controller seed does not make these planner responses identical. Aggregate changes are measured outcomes, not clean estimates of the isolated policy effect.
No manual visual notes or annotation-audit counts were supplied. The suspected skateboard label gaps remain unresolved. Missing full-run metrics must not be replaced by successful-subset metrics.

## Next work

- Preserve singularization and correct counting-unit labels; evaluate adaptive E further.
- Discard and log valid zero-area SAM3 proposals; continue rejecting missing/malformed masks, with no box fallback.
- Capture full malformed Qwen response and finish_reason; make assessments concise and verify whether the output token cap caused truncation before changing it.
- Validate semantic assessments independently of Qwen self-labels, and inspect the remaining book/skateboard containment pairs.
- Compare evidence policies with fixed recorded sensor observations/planner proposals, or repeat runs with documented model sampling settings. Repeat the failed arms after robustness fixes before selecting a full-dataset configuration.
