# E273.p_s1: E273 lane head, seed 1

Run of the experiment index `../E273/README.md` (card `experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md`). Verdict of the experiment: CUT by hand (both seeds, rule 4 of the card).

| | |
|---|---|
| ledger row | `E273.p_s1` (status ok) |
| code | tag `lcr/E273.p_s1`, run commit `8e8d73ff3f` (the launcher's snapshot of the head code, `training/grid/`) |
| checkpoint | `runs/e273_p_s1/model.pt`, hash `0a8e6d5dd8f1` (not in git) |
| lane GT | `autogt/lane-l3`, signature `3576fa1f7e`; front GT of the cached features `c643a7539d` |
| evaluation | VAL 873 frames, frame hash `e04e6704a782`; evaluator commit `5c87b96f7e`, code hash `e67cfd0a9500` |
| scoring card | `runs/e273/e273_p_s1/card.json`, `results.md` (rebuild: `wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s1 --seed 1`) |

## Numbers (VAL 873 frames, from the scoring card)
- lane accuracy of the assignment, map error 1-2 m: p 0.9501, control g 0.8795, p - g +0.0706 [+0.0551, +0.0853]
- lane accuracy, exact map: p 0.9797, g 0.9858, p - g -0.0061 [-0.0144, +0.0023]
- delivered graph (lane suite functions): topo F1 0.595, F1@1.0 0.821, junction F1 0.120, lanes found 0.630, Q 0.630, duplicates 0.192; reference d: 0.741 / 0.878 / 0.273 / 0.781 / 0.742 / 0.077
- nodes: F1 0.850 (P 0.868, R 0.833), median position error 0.275 m; vertices per 100 m 20.5 node ids / 31.5 after Douglas-Peucker 0.3 m; pieces per GT piece 1.60
- time per frame, head only (idle GPU, batch 1, bf16): 23.2 ms = pixel maps 4.3 + nodes 2.8 + links 5.1 + sections 2.7 + map and assignment 7.9

## Files
- `figures/`: one frame per scene (camera views, pixel maps with errors, node graph, ids); `videos/`: the three scenes as mp4; `frames/<scene index>/`: every frame of those scenes; not in git
- `package.json`, `provenance.json`: the result package (versions, hashes, the numbers); fields the generic packager could not read were filled from the run's records
- rebuild pictures: `wsl bash tools/env/_py14.sh qgraph_results.py --run e273_p_s1 --id E273.p_s1 --device cpu --scenes 11 39 23`
