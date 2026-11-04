# experiments - result files left beside the deleted cards of bev_fusion

The 12 open cards of this folder (`<folder>/experiment.md`) were deleted on 2026-11-04 at Mirko's word ("All 12 cards", D56). They are not lost: the git tag `archive/cards-before-delete-2026-11-04` (commit 8aa0953) holds every one of them.

```
git show archive/cards-before-delete-2026-11-04:bev_fusion/experiments/<folder>/experiment.md
```

The twelve folders were s1a_lift_pd_vs_p, t7_lane_branch, t9_encoder_2x2, t10_dual_features, t11_motion_field, t13_complete_network, t14_dontcare, t15_motion_joint, t16_full_network, t19_radar_memory, t20_lane_grid_bundle, t21_one_frame_temporal. Older text that names `experiments/<folder>/experiment.md` (ledger, records, reports, `_about` strings of configs, docstrings) points at that tag; it is history and was not rewritten.

What is left here are the result files that sat beside the cards:

| folder | files |
|---|---|
| `t7_lane_branch/` | `images/` |
| `t10_dual_features/` | `results_x50.md`, `results_x50_2seeds.md` |
| `t14_dontcare/` | `results/fusion_area_real_gt.txt` (read by `tests/test_fusion_area_real.py`) |
| `t20_lane_grid_bundle/` | `compare_vs_t16_full_s0.txt` (read by `results/build/report_runs.json`) |
| `t21_one_frame_temporal/` | `compare_*.txt`, `memory_*.txt`, `stream_*.txt` (read by `results/build/report_runs.json`) |

What this changes:

- A run that is to feed a verdict needs a new card first (a hypothesis, a prediction and a decision rule written before the run: `run.py --require-card`; hard rule 4). Without one only a diagnostic run is possible (`--no-card`), and it never feeds a verdict.
- The t14 pair (`training/configs/t14_dc_ctl_s0.json`, `t14_dc_fusion_s0.json`) lost its start gates G1 - G5, its prediction, its rule and its kill criterion with the card. The configs and the option `LossConfig.fusion_area_only` stay; do not start the pair without a new card.
- The cards of the finished experiments in `../results/experiments/` are not touched (map of old and new paths: `../results/README.md`).
- `project.json` "experiment_dirs" still lists this folder, so a new card written here is found as before (`../../autoresearch/templates/experiment.md`).
