# Week 3 results (27 October - 2 November 2026): Spurmitte on the frozen GaussianCaR features

The short record in git. The extensive report with figures (`Week3_Status_Report_EN.pdf`), the example video (`video/`) and the figures (`figures/`) lie next to this file and are **not in git** (binaries); they are rebuilt by `reports/build/reports/fig_w3_status.py` and `build_w3_status_report.py`, then `print_weekly.ps1`. This record is about the lane-centre system only; the BEV system, the end-to-end set-up, the tools and the disk work are a separate report (assets already collected in `../week3_framework/`). It continues `Week3_Loops18-20` (27 October) and `Status_after_Loop21` (31 October). Numbers come from the metric cards `runs/eval/<name>/<split>.json` and the summaries named in each line; split and denominator are named.

## What it is
Spurmitte reads the **frozen, cached GaussianCaR BEV features** (camera + radar, map-trained; decoder PCA 32 + 7 map logits, 200 x 200 cells of 0.5 m) plus a frozen dense lane head, and builds a sparse lane-centre graph with the learned builder D7 (lattice seeds, rays with survival, children, junction and link heads), followed by rule-step assembly (path union, clean-up, anchoring, relocation, fork pruning, optional Douglas-Peucker) and a route head and planner consumer on stored tokens. 12 of the 24 main-path stages are rule steps. The foundation matters most: map-pretrained GaussianCaR against detection-pretrained CRN lifts F1@1 m from 0.624 to 0.867 on VAL (E27).

## Results
Prototype of record: `recall_sp0_dp` (rc1 + fork pruning 0.4 + spurs kept + Douglas-Peucker 0.2 m), one command on cached outputs: `wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp [--scenes 23 7 --video]`.

| VAL, 873 frames | Q | topo F1 | lanes found | R@1 far | node ids / 100 m | ms per frame |
|---|---|---|---|---|---|---|
| champion | 0.735 | 0.741 | 0.695 | 0.709 | 14.8 | 24 |
| recall | 0.740 | 0.742 | 0.767 | 0.771 | 15.5 | 20.8 (E253, no recursion) |
| fast_recall | 0.739 | 0.737 | 0.736 | 0.727 | 15.0 | 20 - 22 |
| **recall_sp0_dp** | **0.742** | 0.741 | **0.781** | **0.779** | 10.9 | about 21 lane graph, about 27 full chain (estimates) |
| control (E94) | 0.737 | 0.763 | 0.800 | 0.811 | 21.8 | 628 |

TEST (833 frames, the one use, logged first): recall_sp0_dp Q 0.727, lanes found 0.740, R@1 far 0.749, topo F1 0.705; control Q 0.712, 0.762, 0.776, 0.722. Paired: Q +0.015 [+0.000, +0.029], lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012]; the prediction (Q 0.735 - 0.74) hit.

Decided by their written rules this week: **KEEP** E257 (output DP: 14.98 to 9.86 node ids, Q -0.0003), E265 (spurs kept: lanes found +0.0115), E266 (seed ring: R@1 far +0.0152, two rule deviations stated); **TIE** E236 (matched budget), E240 (four seeds: the recall weight is a coverage lever, Q +0.0062 inside the band, lanes found +0.0596), E237 (exact distance map), E243 / E253 (timing); **CUT / KILL** E244 / E255 (Poisson loss), E246, E261 (recall weight 6), E267; **not adopted** E241, E264 (union: precision at range). Gap 1 (coverage at range) is mostly lost in the chain, not in the network; the strong systems at range are blocked by precision.

## Limits of the design
Frozen features: no lane gradient reaches GaussianCaR or the dense head; foundation effect +0.24 F1 and a flat builder fidelity curve from 2 units (E122) mean the features set the level. Rule steps: fork pruning is the largest measured gain (J 0.265 to 0.409, Q 0.701 to 0.735, with vertices 13.5 to 14.8), relocation off costs Q -0.011 / J -0.054, anchoring off Q -0.0004 / J -0.0042 (the first rule to drop), the time recursion costs +6.1 ms.

## Open for Mirko
D4 (promote `recall_sp0_dp`, after route success on spur-kept graphs and idle timing in the next lane loop), D25 (vertex-count basis of the 14.3 budget), D18 (per-metric noise bands); is Douglas-Peucker (D20) the accepted sparsification of the delivered graph, or must the vertex budget come out of the network? Next: lane loop 23, a precision step at range.
