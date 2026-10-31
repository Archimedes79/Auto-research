# evaluation/lane_grid - metrics for dense (grid) lane heads

Dense heads output a 200 x 200 raster (0.5 m cells, ego frame, row = x, column = y; x = -50 + (row + 0.5) * 0.5 m) with a centre-line probability (or a distance field in metres). The grid metric turns it into points and scores them against dense ground-truth points.

| function (`grid_eval.py`) | what it does |
|---|---|
| `skeleton_points(prob, tau, min_comp=30)` | probability map [N,200,200] -> threshold -> Zhang-Suen thinning (`thin_batch`, GPU) -> drop components < 30 px -> points in metres, \|x\|,\|y\| < 48 m |
| `skeletons(D, thr, blur, low, min_comp, rg)` | distance field in metres -> same, with optional blur, hysteresis (`low`) and a range-dependent threshold (`rg`) |
| `geom_counts(P, G)` | counts for P / R / F1 at 0.5 / 1.0 / 1.5 m (nearest-neighbour within tau, KD-tree) + greedy 1:1 matches (`graph_eval.match_1to1`) |
| `eval_probs(prob, gts, taus)` | threshold sweep on probability maps -> P, R, F1, 1:1 F1 per tau and tolerance |
| `est_eval(D, gts, thr, ...)` | the same on distance fields (counts with `prf`, no 1:1) |
| `load_gt_points(split, part, idx)` | dense exact ground-truth points `~/exp_crn_<split>/feats/<part>_gt_pts_exact.npy` |
| `selftest.py` | spec of `skeletons` with hysteresis and range threshold together (CPU, review bug L3 fixed 2026-10-31: far seeds survive), then re-scores the full validation maps of the dense head `e37_final` and must reproduce the numbers stored in `runs/e37_final/metrics.json` (deviation 0.0 on 873 rows, 2026-10-24) |

The functions are unchanged copies of the code that produced all grid numbers of loops 2-17 (`training/train/dense_head.py`, `field_tools.py`, `heat_align.py`, `centerline_estimator.py`); `dense_head.py` and `field_tools.py` now import them from here.

## Review notes (2026-10-24)

1. Three copies of `thin_batch` and three implementations of the point counts (`CenterlineEstimator.prf` with torch `cdist`, `dense_head.geom_counts` with KD-tree + 1:1, `graph_eval.eval_sample`) existed. They define the same quantity; now one definition each. `prf` (no 1:1) remains for `est_eval` because it is the definition of the loop-1 estimator (`centerline_estimator.py`).
2. The grid metric is topology-blind: a parallel duplicate line costs precision, a missing fork costs nothing. Anything about forks, junctions or sparsity needs the graph (`graph_from_dense.py` -> `../sparse_lanes`).
3. Probability maps use `min_comp = 30`, distance fields `min_comp = 0` by default: sweeps from the two families differ by this alone.
4. Threshold choice is a protocol decision: tau is chosen on VAL (clean) and then fixed for perturbed TEST rows (E57 protocol); do not choose it per condition.
5. Not covered: direction (the heads output it, `graph_eval.dir_acc` scores it on graphs), calibration of the heat map.
