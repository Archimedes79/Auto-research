# evaluation/sparse_lanes - metrics for sparse lane graphs

| file | content |
|---|---|
| `graph_eval.py` | the base: geometry P / R / F1 at 0.5 / 1.0 / 1.5 m (dense points every 0.5 m, \|x\|,\|y\| < 48 m), 1:1-matched F1, lanes found, false length, split / merge P / R / F1 (3 m and 6 m), APLS-like route score, direction accuracy. 75 scripts import it. |
| `sparse_metrics.py` | on top of it: vertices per 100 m (`vert100`), composite score `card()` / `composite()`, frontier scalars, `graph_from_vertices`, and the additions `vert100_dp`, `card_any` (see review). Cut out unchanged from `training/data/p16_common.py`, which re-exports the names. |
| `topo_metrics.py` | added 2026-10-25 for the blind spots of S: `topo_P / topo_R / topo_F1` (directed reachability within 30 m from ground-truth seeds every 5 m, 1:1 matching at 1 m: breaks, cut links, false branches, reversed direction), `dup_frac` (parallel duplicates, fork zones of 4 m excluded), `R@1_near / mid / far` and `P@1_*` (0-20 / 20-40 / > 40 m), `R@1_far_inner / far_ring` and `P@1_*` (the far bin split at max(|x|,|y|) = 45 m, the edge of the inference seed lattice, E266); `paired()` for scene-bootstrap differences; `card(..., extra=True)` adds them. Selftest: `tools/env/_py14.sh topo_metrics.py` |
| `bootstrap.py` | scene-bootstrap intervals (`interval`), paired comparison of two systems with a verdict against the noise band (`paired(variant, control)`: variant - control), pooled metrics from per-frame counts (`frame_records`, `pooled`), `Q` and `len_ratio`. The scene draws and the intervals (also of `topo_metrics.paired`) are the one shared bootstrap `../../../autoresearch/Evaluation/stats.py` (D24, 2026-10-31; the same `RandomState` draws as before, stored numbers reproduced exactly; non-finite draws are dropped and then reported as `n_valid`) |
| `selftest.py` | spec of `graph_from_vertices` (fork, merge, cycles, isolated chain; review bug L1 fixed 2026-10-31, D24: a split vertex with one predecessor now starts its branches), perfect graph scores 1, new == old implementation on 60 ground-truth frames, DP sparsity of the ground truth. Run: `tools/env/_py14.sh sparse_lanes/selftest.py` |

## The contract (one format for every producer)

A graph is `dict(pts=[(n_i, 2) float arrays: polylines in metres, ego frame, x right / y forward, in travel direction], edges=[(i, j): end of piece i continues into piece j], nodes=[list of vertex ids per polyline])`; `nodes` is optional but needed for the vertex count. Prediction and ground truth have the same format (ground truth: `~/exp_gcar_b123/graph_gt_{train,val}.pkl`, extra keys `kind`, `start_open`, `end_open`, `tokens`).

Metrics of a run (`sparse_metrics.card(preds, gts)`): `F1@0.5`, `F1@1.0`, `F1@1.5` (+ P, R), `F1o2o@1.0`, `lanes_found`, `false_len_frac` (FL), `split6_F1`, `merge6_F1` (J = their mean), `vert100`, `Sp = min(1, 14.3 / vert100)`, `dir_acc`, and the composite **S = 0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 - FL)** (plan section 5, `experiment_execution/plans/Loop16_Triage_Plan_EN.md`).

## Which methods can be evaluated this way

| producer | how it gets into the contract | comparable? |
|---|---|---|
| rule graph from a dense head (`graph_from_dense.py`, skeleton or trace) | built as pieces + successor edges, no vertex ids | geometry and J yes; **sparsity only with `vert100_dp` / `card_any`** (`vert100` would count every polyline point: ground truth 208 per 100 m) |
| learned trace network E94, refinement E120 | `nodes` from the vertex bookkeeping | yes, all metrics |
| differentiable builders D1-D8 (loop 16), champion D7 | `graph_from_vertices(xy, edges)` -> pieces + `nodes` | yes, all metrics |
| literature baselines (SAM-Road-style nodes + edges, TopoLogic-style links, LoI single score) | same conversion from their vertices / edges | yes if converted by `graph_from_vertices` |
| route head, assignment, planner consumer | not graphs | **no**: route success, route F, planner L2 are in `p10_route_eval2.py`, `eval_parts.py`, `p14_chain_consumer.py` (kept in `scripts/`, they call this package for the graph part) |
| temporal stability, robustness series | graphs per frame | geometry yes; flicker and series logic in `p17_*` (loop 17, `evaluation/loops/loop17/`), old `stability.py` (removed 2026-11-03, in git history) |

## Review notes (2026-10-24) - read before comparing numbers

1. **Sparsity is only defined with vertex ids.** `vert100` counts unique ids from `nodes`; without `nodes` it falls back to every polyline point, so `Sp` and `S` of a rule graph are not comparable with a vertex graph. Use `card_any` (falls back to `vert100_dp`, Douglas-Peucker 0.3 m, the definition of `evaluation/drivers/oracle_vertices.py`: ground truth 13.2 per 100 m on VAL, oracle value 14.3 on 100 TEST frames).
2. **J depends on the cutting rule.** Splits / merges are the end / start of pieces with >= 2 successors / predecessors; two methods that cut the same graph differently get different J. All learned methods go through `graph_from_vertices` (maximal chains between vertices of in / out degree != 1). Tolerance 6 m is looser than published landmark protocols (0.5-5 m); report the curve (the 3 m values `split_F1` / `merge_F1` are in `graph_eval`).
3. **Route score is sampled** (seeded random source / target pairs); the card uses `with_route=False`; treat route numbers with a seed spread.
4. `lanes_found` = ground-truth lane counted if >= 80 % of its points are within 1 m; FL counts prediction length farther than 1.5 m from any ground-truth point; both use the clip of 48 m.
5. S weights and the 14.3 constant are fixed by the loop-16 plan; S of the VAL subset (250 frames, `p16_common.val_subset`) and of full VAL (873) are different numbers - compare on identical frames.
6. Not covered here: calibration of validity / fork probabilities (open gap, `../../literature/README.md`), planner-level metrics, closed loop.
7. Topo F1 punishes zig-zag graphs (jitter of 0.25 m on every sample: topo F1 0.83, F1@1 1.0) because a zig-zag is longer than the lane it covers; vertex graphs of the builders are not affected. dup_frac counts long unlinked overlaps, also those of a fork that pruning un-linked (champion: 0.018 -> 0.047 with pruning; 0.016 -> 0.049 after the L1 fix of `graph_from_vertices`, 2026-10-31); read it as a check, not as a target.
