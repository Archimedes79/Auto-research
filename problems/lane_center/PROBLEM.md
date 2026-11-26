# PROBLEM - drive anywhere: lane-centre-line graph from camera + radar, attached to an ADASIS-like SD map (tag prefix `lane_center`; `lcr` before 2026-11-27)

Owned by the project owner. Read by every agent of `../autoresearch/` (method: `../autoresearch/METHOD.md`). Machine-readable paths: `project.json`.

## Goal and constraints
- A planner-ready lane graph without an HD map: lane centre-lines with forks and merges, travel direction, fused over time, assigned to the SD-map route (lane level, road level as fallback), with confidence.
- Very good AND very fast: quality at <= 14.3 vertices per 100 m, ms per frame on one RTX 3060 (idle GPU, batch 1, cached foundation features).
- End-to-end differentiable (D55, 2026-11-04): the forward path from the BEV features to the delivered output, including the map assignment and the classification, is made of soft, differentiable steps (no rule step), and a loss on the output reaches every module. Matching inside the loss (bipartite, one-to-many, Sinkhorn: which prediction is compared with which ground-truth element) is allowed. Hard selection (argmax, thresholds, peak picking) only on the output, and named. Better short lane graphs than false ones.
- Data: nuScenes (non-commercial licence), 255 scenes, scene-disjoint TRAIN 8,414 / VAL 873 / TEST 833 frames; frozen GaussianCaR camera + radar BEV features; simulated ADASIS-like SD map with controlled errors. TEST only once per finished design, with the owner's recorded OK.
- Compute: one RTX 3060 12 GB, at most 2 GPU jobs at once (3 stall each other), WSL2 + conda env `gcar`.

## The product chain (code in `training/`; drawing `training/architecture.svg`)
The system of record is the token champion (D113 - D120): cameras + radar -> frozen GaussianCaR BEV features (cached) -> the frozen lane head B' (pyramid, pixel maps, soft nodes with a soft Matrix-NMS layer) -> its predicted nodes as TOKENS (top-K 512: position, direction, existence, node token) -> ONE token head (three self-attention layers with a relative-geometry bias and an allowed-pair gate, a per-source wiring softmax with a none slot, a seven-way node-type softmax) -> a read-out by four named thresholds -> the lane graph (nodes, links, types). Trained in two stages: the head on the cached tokens, then the head and the node head of B' together (S4). No rule step is in the delivered path besides the named hard items (top-K, the pair gate, the thresholds).
Not in the delivered chain (separate, measured stages): the SD-map route head (E389: a cross-attention from the tokens to the route tokens of the SD map, flags per token), the past at token level (S2 / S2b: TIE), the map block inside the token head (S3: TIE).
Reference systems (in the tag `v1`, not developed): the earlier end-to-end system E324 s and the route-graph builder chain of loops 8 - 18; their numbers below are history.

## Metrics and protocol
- Every number of a network comes from a run of the research system: `research start lane_center E<n>` (steps `predict` + `evaluate`: the live dump, then `evaluation/adapter.py` decodes and scores it with the scorer below); splits, seeds, noise band 0.010, the TEST rule: `problem.json`, `TEST_LOG.md`. The protocol of the old reference systems (`evaluate.py`, `models/registry.json`, `evaluation/PROTOCOL.md`) is in the tag `v1`.
- Headline of the token system (from D113 on): **topo F1** (connectivity) with junction F1, F1@1.0 and lanes found, VAL 873 frames, paired scene bootstrap, noise band 0.010, read by `evaluation/jtoken_eval.py` at the named decode (D115); ms per frame of the whole path on a quiet GPU. The headline of the reference systems below: **Q** at <= 14.3 vertices per 100 m, **topo F1** (connectivity), **lanes found**, **R@1 far** (> 40 m), route success (2 seeds), ms per frame. S is kept for the history; FL and dup_frac are checks. Metric response tests: `evaluation/METRIC_RESPONSE.md`.
- Vertex count (D20, 2026-10-30; corrected by D25, 2026-11-01): the budget of 14.3 vertices per 100 m is counted on the per-piece Douglas-Peucker basis of the metric (0.3 m), and every system meets it with and without the output stage `simplify_dp` (champion 11.3, fast_recall 12.2, recall 12.8; E257 `vertex_basis.txt`). The stage (Douglas-Peucker 0.2 m, `assemble(dp_eps=0.2)`, registry names `*_dp`) re-counts the same geometry: it lowers the nodes delivered (node ids: champion 14.8 -> 9.5, fast_recall 15.0 -> 9.9, recall 15.5 -> 10.4 per 100 m), not the budget count, and Q is unchanged by construction. Node-id counts are quoted as "nodes delivered", never against the budget. The stage is off by default, the route head and the map pipeline (`aux=True`) work on the unsimplified vertices, and S of a `*_dp` card is not comparable with S of other cards (it rises through its sparsity term only).

## Current best (VAL 873 frames; seeds 0 / 1; the final end-to-end report of 2026-11-18 and E372)
| system | topo F1 | junction F1 | F1@1.0 | lanes found | ms per frame (whole path) | parameters |
|---|---|---|---|---|---|---|
| **token champion** (E372 stage 2 = S4, D120; the head with none weight 0.25 at the per-class decode of D115) | 0.6682 / 0.6813 | 0.2538 / 0.2481 | 0.8459 / 0.8476 | 0.7232 / 0.7214 | 16.2 (quiet) | 2.323 M |
| E324 s, end-to-end read-out (reference: five rule steps, three stages) | 0.6912 / 0.6948 | | 0.8777 / 0.8786 | 0.7653 / 0.7648 | 19.7 | 2.741 M |
| plain lane head B (no token head) | 0.6259 / 0.6249 | 0.27 | | | | |
The champion buys simplicity and about 15 % compute for about 0.02 topo F1 against E324 s (+0.042 / +0.056 over plain B); no TEST check of it exists (a TEST check of E324 s, card E343, waits for the owner's recorded OK).
Reference systems of loops 8 - 18 (VAL 873 frames, `evaluate.py`, 2026-10-25; ms from the loop-18 report; kept for comparison):
| system (registry name) | Q | topo F1 | lanes found | R@1 far | J | S | ms per frame |
|---|---|---|---|---|---|---|---|
| `champion` (D7 f1 + fork pruning 0.3) | 0.735 | 0.741 | 0.695 | 0.709 | 0.409 | 0.769 | 24 (17.5 without time recursion) |
| `recall` (rc1 + pruning 0.4) | 0.740 | 0.742 | 0.767 | 0.771 | 0.428 | 0.767 | |
| `fast_recall` (j1rc, 192 / 96) | 0.739 | 0.737 | 0.736 | 0.727 | 0.447 | 0.772 | 20 - 22 |
| `control` (E94 trace network) | 0.737 | 0.763 | 0.800 | 0.811 | 0.408 | 0.725 | 628 |
TEST (once, E230): the old champion S 0.748, J 0.422, lanes found 0.653 against the control 0.702 / 0.404 / 0.762. The topo F1 of the token system is not comparable with the old rows (another graph, another read-out).

## Known gaps (ranked; the numbers are those of the cards)
1. **Junction recall and the train-validation gap of B' tokens:** junction F1 0.25 against 0.27 for the plain head; the delivered graph is precise (0.86) but its recall is low; the lane head is cleaner on TRAIN than on VAL (node recall 0.926 / 0.923 against 0.862 / 0.864), and training noise on the tokens did not close it (E381 TIE).
2. **No assignment to the map in the delivered system:** the token champion reads camera and radar only. E389 built the stage as a separate differentiable head (one cross-attention to the SD route tokens): it reads the map (route F1 per covered node 0.977 / 0.981 against 0.883 / 0.884 for the same head without the map, the geometric rule on the erroneous route 0.951) but its gain over the no-map head, +0.094 / +0.097, missed the bar +0.10 (TIE, NO GAIN); forks +0.14, lane level +0.11, route success@30 m +0.2; it is NOT safe under a wrong map (-0.043 / -0.053 against -0.03) and costs 2 - 3 ms on a busy GPU (not quiet). Open: wrong maps in the training mix, a cached-route variant, a joint fine-tune through S4, a map as a logical road graph (E390).
3. **No feedback:** neither the map assignment nor the time feeds back into perception. The past at token level (S2 E368, S2b E387: at most +0.005, TIE), the recursion at feature level (E354) and the oracle ceiling of a recursive node state (E384: not enough to train for) are closed; one frame and one radar scan is the input (D91). The head to B' feedback IS closed by S4 (the loss reaches the node head).
4. **Rule steps of the better E2E-read-out system:** E324 s is 0.02 topo F1 above the champion with five rule steps (snap wiring, close-node step, candidate selection, keypoint peaks, thresholds); the champion replaced them by learned steps but keeps the named hard items (top-K 512, the allowed-pair gate, four output thresholds).
5. No planner-level metric that responds (path error at a floor of 0.27 m whatever the graph).
6. Map with 1 - 3 m error: registration recovers about half of the gap between no map and an exact map (loops 18 - 19); robustness with cameras off (A8) is measured only on the reference systems.

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
- Code, data, runs and commands: `README.md`; the champion's facts: `champions/lane_center/`; decisions: `DECISIONS.md`.
- Every comparison: matched reference on the same frames, paired scene bootstrap over the 22 VAL scenes (the judge warns below 30), 2 seeds for trained arms; an optimisation must reproduce the reference graphs; timing only on an idle GPU.
- A question answered the same way twice goes into the 'open to attack when' column of its rule in `DESIGN.md`.
