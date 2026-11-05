# E273.p_s0: E273 lane head, seed 0

Run of the experiment index `../E273/README.md` (card `experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md`). Verdict of the experiment: CUT by hand (both seeds, rule 4 of the card).

| | |
|---|---|
| ledger row | `E273.p_s0` (status ok) |
| code | tag `lcr/E273.p_s0`, run commit `ab72b01707` (the launcher's snapshot of the head code, `training/grid/`) |
| checkpoint | `runs/e273_p_s0/model.pt`, hash `cea068319415` (not in git) |
| lane GT | `autogt/lane-l3`, signature `3576fa1f7e`; front GT of the cached features `c643a7539d` |
| evaluation | VAL 873 frames, frame hash `e04e6704a782`; evaluator commit `5c87b96f7e`, code hash `e67cfd0a9500` |
| scoring card | `runs/e273/e273_p_s0/card.json`, `results.md` (rebuild: `wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s0 --seed 0`) |

## Numbers (VAL 873 frames, from the scoring card)
- lane accuracy of the assignment, map error 1-2 m: p 0.9489, control g 0.8767, p - g +0.0722 [+0.0563, +0.0871]
- lane accuracy, exact map: p 0.9799, g 0.9870, p - g -0.0072 [-0.0151, +0.0004]
- delivered graph (lane suite functions): topo F1 0.597, F1@1.0 0.832, junction F1 0.130, lanes found 0.644, Q 0.636, duplicates 0.232; reference d: 0.741 / 0.878 / 0.273 / 0.781 / 0.742 / 0.077
- nodes: F1 0.853 (P 0.873, R 0.833), median position error 0.277 m; vertices per 100 m 18.2 node ids / 28.9 after Douglas-Peucker 0.3 m; pieces per GT piece 1.61
- time per frame, head only (idle GPU, batch 1, bf16): 24.2 ms = pixel maps 4.3 + nodes 2.9 + links 5.5 + sections 2.8 + map and assignment 8.3

## Files
- `figures/`: one frame per scene (camera views, pixel maps with errors, node graph, ids); `videos/`: the three scenes as mp4; `frames/<scene index>/`: every frame of those scenes; not in git
- `package.json`, `provenance.json`: the result package (versions, hashes, the numbers); fields the generic packager could not read were filled from the run's records
- rebuild pictures: `wsl bash tools/env/_py14.sh qgraph_results.py --run e273_p_s0 --id E273.p_s0 --device cpu --scenes 11 39 23`
