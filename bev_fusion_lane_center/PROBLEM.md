# PROBLEM - drive anywhere: lane-centre-line graph from camera + radar, attached to an ADASIS-like SD map (tag prefix `lcr`)

Owned by Mirko. Read by every agent of `../autoresearch/` (method: `../autoresearch/METHOD.md`). Machine-readable paths: `project.json`.

## Goal and constraints
- A planner-ready lane graph without an HD map: lane centre-lines with forks and merges, travel direction, fused over time, assigned to the SD-map route (lane level, road level as fallback), with confidence.
- Very good AND very fast: quality at <= 14.3 vertices per 100 m, ms per frame on one RTX 3060 (idle GPU, batch 1, cached foundation features).
- End-to-end differentiable (D55, 2026-11-04): the forward path from the BEV features to the delivered output, including the map assignment and the classification, is made of soft, differentiable steps (no rule step), and a loss on the output reaches every module. Matching inside the loss (bipartite, one-to-many, Sinkhorn: which prediction is compared with which ground-truth element) is allowed. Hard selection (argmax, thresholds, peak picking) only on the output, and named. Better short lane graphs than false ones.
- Data: nuScenes (non-commercial licence), 255 scenes, scene-disjoint TRAIN 8,414 / VAL 873 / TEST 833 frames; frozen GaussianCaR camera + radar BEV features; simulated ADASIS-like SD map with controlled errors. TEST only once per finished design, with Mirko's OK.
- Compute: one RTX 3060 12 GB, at most 2 GPU jobs at once (3 stall each other), WSL2 + conda env `gcar`.

## The product chain (code in `training/`)
camera + radar -> GaussianCaR BEV features (frozen, cached) -> dense lane heads -> differentiable sparse graph builder D7 (seeds, rays, survival, children, junction head, time recursion) -> assembly (path union, fork pruning, numba) -> temporal memory (hold / learned gate) -> map-conditioned builder + registration / trust to the SD map -> route head along the edges -> planner consumer

## Metrics and protocol
- One entry point: `evaluation/evaluate.py <registry name> [--compare <name>]` (names in `models/registry.json`); protocol in `evaluation/PROTOCOL.md` (splits, seeds, noise band 0.010, failure series, map errors, timing, TEST rule, TEST log `evaluation/TEST_USAGE_LOG.md`).
- Headline: **Q** at <= 14.3 vertices per 100 m, **topo F1** (connectivity), **lanes found**, **R@1 far** (> 40 m), route success (2 seeds), ms per frame. S is kept for the history; FL and dup_frac are checks. Metric response tests: `evaluation/METRIC_RESPONSE.md`.
- Vertex count (D20, 2026-10-30; corrected by D25, 2026-11-01): the budget of 14.3 vertices per 100 m is counted on the per-piece Douglas-Peucker basis of the metric (0.3 m), and every system meets it with and without the output stage `simplify_dp` (champion 11.3, fast_recall 12.2, recall 12.8; E257 `vertex_basis.txt`). The stage (Douglas-Peucker 0.2 m, `assemble(dp_eps=0.2)`, registry names `*_dp`) re-counts the same geometry: it lowers the nodes delivered (node ids: champion 14.8 -> 9.5, fast_recall 15.0 -> 9.9, recall 15.5 -> 10.4 per 100 m), not the budget count, and Q is unchanged by construction. Node-id counts are quoted as "nodes delivered", never against the budget. The stage is off by default, the route head and the map pipeline (`aux=True`) work on the unsimplified vertices, and S of a `*_dp` card is not comparable with S of other cards (it rises through its sparsity term only).

## Current best (VAL 873 frames, `evaluate.py`, 2026-10-25; ms from the loop-18 report)
| system (registry name) | Q | topo F1 | lanes found | R@1 far | J | S | ms per frame |
|---|---|---|---|---|---|---|---|
| `champion` (D7 f1 + fork pruning 0.3) | 0.735 | 0.741 | 0.695 | 0.709 | 0.409 | 0.769 | 24 (17.5 without time recursion) |
| `recall` (rc1 + pruning 0.4) | 0.740 | 0.742 | 0.767 | 0.771 | 0.428 | 0.767 | |
| `fast_recall` (j1rc, 192 / 96) | 0.739 | 0.737 | 0.736 | 0.727 | 0.447 | 0.772 | 20 - 22 |
| `control` (E94 trace network) | 0.737 | 0.763 | 0.800 | 0.811 | 0.408 | 0.725 | 628 |
TEST (once, E230): champion S 0.748, J 0.422, lanes found 0.653 against the control 0.702 / 0.404 / 0.762.

## Known gaps (ranked)
1. Coverage at range: lanes found -0.105 and R@1 far -0.10 against the control; loss weights are spent (rc2 / rc4).
2. Connectivity: topo F1 -0.022 against the control (fork pruning gave +0.021).
3. No planner-level metric that responds (path error at a floor of 0.27 m whatever the graph).
4. Map with 1-3 m error: registration recovers about half of the gap between no map and an exact map.
5. Robustness of the recall variants with cameras off (Q -0.038 for j1rc).

## Problem groups (filed under these ids: notes, cards, summaries; detail in `literature/OVERVIEW.md`)
| id | group | question |
|---|---|---|
| A1 | sensing foundation | which camera + radar BEV features carry lane structure; can heads train on them frozen? |
| A2 | dense lane-centre field | which dense intermediate leads best to a graph? |
| A3 | sparse lane graph | a short, correct graph with forks and merges: growth, set prediction, linking, verification |
| A4 | differentiable selection, training regime | choosing among hypotheses end to end without matching |
| A5 | temporal fusion | fusing detections over time on vertices and edges |
| A6 | assignment to a map graph | attaching the perceived graph to the SD route (lane / road level) |
| A7 | navigation context to the planner | route flag, confidence, what a planner needs |
| A8 | robustness and uncertainty | sensor failure, wrong maps, calibrated confidence |
| A9 | evaluation, data, metrics | fair measurement, splits, leakage, planner-level metrics |

## Foundation topics (researched independently of the problem)
| id | topic | why it may matter here |
|---|---|---|
| F1 | structural building blocks (state-space / recurrent memory, attention variants, deformable / pointer sampling, iterative refinement, gated fusion, set and graph decoders) | far range, temporal fusion, cheaper blocks |
| F2 | training and losses (matching-free set losses, curricula, distillation from a slower teacher, self-training, calibration, schedules) | coverage without false graphs, robustness |
| F3 | mathematical formulations (graphs as point processes or fields, energy-based and optimisation layers, survival / hazard views, equivariance) | a better way to express the lane graph and its uncertainty |

## Problem-specific workflow (deviations from METHOD.md)
- Folders: research = `literature/` (papers.csv -> `tools/literature/catalogue.py all` regenerates CATALOGUE / OVERVIEW); training = `training/`; evaluation = `evaluation/`; experiment summaries = `experiments/E<id>_<slug>/` and `reports/weekly/`; execution state = `HANDOVER.md`, `experiment_execution/PLAN.md`, `experiment_execution/schedule.csv`, `experiment_execution/LEDGER.csv`; scratch = `temp/loopNN/`; data and checkpoints = `data/`, `runs/`, `~/exp_gcar_b123` (WSL).
- Python of the product: `wsl bash tools/env/_py14.sh <script> ...` (from this folder); runs: `wsl python3 ../autoresearch/tools/run.py --id ... -- bash tools/env/_py14.sh <script> ...` (tags the code `lcr/<id>`).
- Every comparison: matched reference on the same frames, paired scene bootstrap, 2 seeds for route heads; an optimisation must reproduce the reference graphs; timing only on an idle GPU.
- Closed questions: `experiment_execution/closed_questions.md`.
