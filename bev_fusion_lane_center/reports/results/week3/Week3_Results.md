# Week 3 results (27 October - 2 November 2026): where we are

The short record in git. The extensive report with figures (`Week3_Status_Report_EN.pdf`), the two example videos (`video/`) and the figures (`figures/`) lie next to this file and are **not in git** (binaries); they are rebuilt by `reports/build/reports/fig_w3_status.py` and `build_w3_status_report.py`, then `print_weekly.ps1`. It continues the earlier `Week3_Loops18-20` (27 October) and `Status_after_Loop21` (31 October). Numbers come from the metric cards `runs/eval/<name>/<split>.json`, `bev_fusion/records/BOARD.md` and the summaries named in each line; split and denominator are named.

## Lane-centre system (Spurmitte)
Prototype of record: `recall_sp0_dp` (rc1 + fork pruning 0.4 + spurs kept + Douglas-Peucker 0.2 m output stage), one command on cached outputs: `wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp [--scenes 23 7 --video]`.

| VAL, 873 frames | Q | topo F1 | lanes found | R@1 far | node ids / 100 m | ms per frame |
|---|---|---|---|---|---|---|
| champion | 0.735 | 0.741 | 0.695 | 0.709 | 14.8 | 24 |
| recall | 0.740 | 0.742 | 0.767 | 0.771 | 15.5 | 20.8 (E253, no recursion) |
| fast_recall | 0.739 | 0.737 | 0.736 | 0.727 | 15.0 | 20 - 22 |
| **recall_sp0_dp** | **0.742** | 0.741 | **0.781** | **0.779** | 10.9 | about 21 lane graph, about 27 full chain (estimates) |
| control (E94) | 0.737 | 0.763 | 0.800 | 0.811 | 21.8 | 628 |

TEST (833 frames, the one use, logged first): recall_sp0_dp Q 0.727, lanes found 0.740, R@1 far 0.749, topo F1 0.705; control Q 0.712, 0.762, 0.776, 0.722. Paired: Q +0.015 [+0.000, +0.029], lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012]; the prediction (Q 0.735 - 0.74) hit.

Decided by their written rules this week: **KEEP** E257 (output DP: 14.98 to 9.86 node ids, Q -0.0003), E265 (spurs kept: lanes found +0.0115), E266 (seed ring: R@1 far +0.0152, two rule deviations stated); **TIE** E236 (matched budget), E240 (four seeds: the recall weight is a coverage lever, Q +0.0062 inside the band, lanes found +0.0596), E237 (exact distance map), E243 / E253 (timing); **CUT / KILL** E244 / E255 (Poisson loss), E246, E261 (recall weight 6), E267; **not adopted** E241, E264 (union: precision at range). Gap 1 (coverage at range) is mostly lost in the chain, not in the network; the strong systems at range are blocked by precision. Open: D4 (promote the prototype), D25 (vertex basis), D18, D19.

## BEV system (camera + radar)
Five-layer, fully differentiable design (14 main-path stages, 0 rule steps). `t10_dual` (both image feature sets) keeps the map gains of the BEVFusion features (drivable +0.090, walkway +0.170, lane divider +0.157 against the base) and loses almost nothing on objects (mAP -0.016, inside the band); it held on its seed twin. The velocity head was dead in 28 of 30 earlier runs; the fix (moving / static logit, regression on moving centres only, decode gate) gives, for B2: velocity skill on moving cars 0.000 to 0.566, vehicle AVE 2.385 to 1.189 m/s, at a cost of front mAP2 0.361 to 0.347 (-0.014 [-0.020, -0.006]); one seed, the seed twin is next. Sources: `bev_fusion/records/{BOARD.md, morning_2026-11-01.md}`.

## Toward an end-to-end lane chain
Lane chain: 24 main-path stages, 12 of them rule steps; BEV system: 0. Prepared and measured, nothing trained: the contract of `Builder.forward` (`training/models/p16_interface.py`), the gradient report (`evaluation/drivers/e2e_gradient_report.py`): all 101 builder tensors get gradient from the builder loss (2 VAL frames, CPU, indicative); the route loss alone through the old joint path reaches the builder strongly through the node tokens (seeds 7.26 against 8.34 for the builder's own, node tokens 20 times) but almost orthogonally (cosines -0.06 to +0.05 apart from the seeds group, 0.51), and gives exactly zero to node positions, validity, link logits, junction and link heads. Card `experiments/E269_soft-graph-tokens-route-loss/` (three critic passes, all REVISE, then restructured): a straight-through proxy with arms a (position path), c (E232's detached control), d (route gradient on the tokens x0.1), two seeds each, about 8 slot-hours; nothing scheduled.

## Work and operations
D16 (run state derived from run records, deployed), D21 (gate hook narrowed to the truth files), D22 - D24 tool fixes, D26 (dense head first, trainable image features), D28 - D30, D32 (layered restructuring in progress), D36 - D38 (coverage rule, fusion area, footprint share). Disk: C: free 4.1 GB at the start, 19.4 GB of old lane run files moved to `D:\Auto-research\lane_archive_2026-11-01` (verified by size, nothing deleted; record in `experiment_execution/history/lane_archive_2026-11-01/`), 55.5 GB free at the last check; golden check (six cached entries) reproduced after the move. Prepared for Mirko, not run: `tools/maintenance/purge_and_compact_2026-11-01.ps1` (delete archive, D5 caches, second wave, compact the WSL image). Main and 25 `lcr/*` run tags were pushed on 26 October; later commits are local.

## Open for Mirko
D4 (promote `recall_sp0_dp`), D25, D18 / D19; run the purge and compaction script when no chat trains; keep or drop the failure-series caches `pert` and `p17` (74.6 GB); E269 build needs a critic code pass (`p18_joint.py` is shared); is Douglas-Peucker (D20) the accepted layer-5 sparsification?
