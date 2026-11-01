# Week 3 results, part 2 (27 October - 2 November 2026): the framework behind Spurmitte

The short record in git for the report that is **not** about the Spurmitte results (those are in `../week3/`). The extensive report (`Week3_Framework_Report_EN.pdf`), its figures and the BEV dashboard video lie next to this file and are not in git; builder: `reports/build/reports/build_w3_framework_report.py`. Nothing was run for this report: numbers come from stored files and the summaries named in each line.

## BEV system (camera + radar)
Five-layer, fully differentiable design (14 main-path stages, 0 rule steps). `t10_dual` (both image feature sets) keeps the map gains of the BEVFusion features (drivable +0.090, walkway +0.170, lane divider +0.157 against the base) and loses almost nothing on objects (mAP -0.016, inside the band); it held on its seed twin. The velocity head was dead in 28 of 30 earlier runs; the fix (moving / static logit, regression on moving centres only, decode gate) gives, for B2: velocity skill on moving cars 0.000 to 0.566, vehicle AVE 2.385 to 1.189 m/s, at a cost of front mAP2 0.361 to 0.347 (-0.014 [-0.020, -0.006]); one seed, the seed twin is next. Sources: `bev_fusion/records/{BOARD.md, morning_2026-11-01.md}`.

## Toward an end-to-end lane chain
Lane chain: 24 main-path stages, 12 of them rule steps; BEV system: 0. Prepared and measured, nothing trained: the contract of `Builder.forward` (`training/models/p16_interface.py`) and the gradient report (`evaluation/drivers/e2e_gradient_report.py`): all 101 builder tensors get gradient from the builder loss (2 VAL frames, CPU, indicative). The route loss alone through the old joint path reaches the builder strongly through the node tokens (seeds 7.26 against 8.34 for the builder's own, node tokens 20 times) but almost orthogonally (cosines -0.06 to +0.05 apart from the seeds group, 0.51), and gives exactly zero to node positions, validity, link logits, junction and link heads. Card `experiments/E269_soft-graph-tokens-route-loss/`: three critic passes (all REVISE), then restructured to a straight-through proxy with arms a (position path), c (E232's detached control), d (route gradient on the tokens x0.1), two seeds each, about 8 slot-hours; nothing scheduled.

## Tools and decisions
D16 (run state derived from run records; 8 defects found by a fresh critic, 7 fixed), D21 (gate hook narrowed to the truth files), D22 - D24 tool fixes, D26 (dense head first, trainable image features), D28 - D30, D32 (layered restructuring in progress), D36 - D38 (coverage rule, fusion area, footprint share), the git rule of 1 November and `keep3`.

## Disk and repository
C: free 4.1 GB at the start; 19.4 GB of old lane run files moved to `D:\Auto-research\lane_archive_2026-11-01` (verified by size, nothing deleted; record in `experiment_execution/history/lane_archive_2026-11-01/`); the six cached golden entries reproduced after the move. Prepared for Mirko, not run: `tools/maintenance/purge_and_compact_2026-11-01.ps1` (delete archive, D5 caches, second wave, compact the 318 GB WSL image). Main and 25 `lcr/*` run tags were pushed on 26 October; later commits are local.

## Open
Run the purge and compaction script when no chat trains; keep or drop the failure-series caches `pert` and `p17` (74.6 GB); D18 / D19; E269 build needs a critic code pass (`p18_joint.py` is shared).
