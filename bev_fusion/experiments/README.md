# experiments - the cards written since D56 and the result files left beside the deleted cards of bev_fusion

The 12 open cards of this folder (`<folder>/experiment.md`) were deleted on 2026-11-04 (D56). They are not lost: the git tag `archive/cards-before-delete-2026-11-04` (commit 8aa0953) holds every one of them.

```
git show archive/cards-before-delete-2026-11-04:bev_fusion/experiments/<folder>/experiment.md
```

The twelve folders were s1a_lift_pd_vs_p, t7_lane_branch, t9_encoder_2x2, t10_dual_features, t11_motion_field, t13_complete_network, t14_dontcare, t15_motion_joint, t16_full_network, t19_radar_memory, t20_lane_grid_bundle, t21_one_frame_temporal. Older text that names `experiments/<folder>/experiment.md` (ledger, records, reports, `_about` strings of configs, docstrings) points at that tag; it is history and was not rewritten.

Cards written since (each folder holds `experiment.md`: card, detail plan, summary; the ledger rows and the decisions are named in the card):

| folder | card | state |
|---|---|---|
| `t22_student_encoder/` | a small trained image encoder (EfficientViT-B1; ResNet-18 as the control encoder) in place of the three frozen image-side pieces | stage B INCONCLUSIVE, gate C0 INCONCLUSIVE, B2 CUT; the R arm and the seed twin are put back (D68), not started |
| `t23_lane_line_loss/` | tolerance-aware BCE on the lane / road divider presence of the t20 chain | TIE on two seeds (IoU@0.5 unchanged, ranking better) |
| `t24_lane_pull/` | lane branch lifted without the depth weight (lift P) | closed as not run: the go / no-go rule was not met |
| `t25_one_frame_one_scan/` | one camera frame, one radar sweep, the past recursively at feature level (D91) | card and four configs, no ledger row for the arms |
| `t26_one_long_run/` | one long from-scratch run of the D91 network with a trainable image encoder, read against GaussianCaR and CRN | base and reference evaluations and the G2 smoke are in the ledger; the long run waits for the open decision D102 |

What is left here of the deleted cards are the result files that sat beside them:

| folder | files |
|---|---|
| `t7_lane_branch/` | `images/` (pictures, not in git) |
| `t10_dual_features/` | `results_x50.md`, `results_x50_2seeds.md` |
| `t14_dontcare/` | `results/fusion_area_real_gt.txt` (read by `tests/test_fusion_area_real.py`) |
| `t20_lane_grid_bundle/` | `compare_vs_t16_full_s0.txt` (read by `results/build/report_runs.json`) |
| `t21_one_frame_temporal/` | `compare_*.txt`, `memory_*.txt`, `stream_*.txt` (read by `results/build/report_runs.json`) |

What this changes:

- A run that is to feed a verdict needs a new card first (a hypothesis, a prediction and a decision rule written before the run: `run.py --require-card`; hard rule 4). Without one only a diagnostic run is possible (`--no-card`), and it never feeds a verdict.
- The t14 pair (`training/configs/t14_dc_ctl_s0.json`, `t14_dc_fusion_s0.json`) lost its start gates G1 - G5, its prediction, its rule and its kill criterion with the card. The configs and the option `LossConfig.fusion_area_only` stay; do not start the pair without a new card.
- The cards of the finished experiments in `../results/experiments/` are not touched (map of old and new paths: `../results/README.md`).
- `project.json` "experiment_dirs" still lists this folder, so a new card written here is found as before (`../../autoresearch/templates/experiment.md`).
