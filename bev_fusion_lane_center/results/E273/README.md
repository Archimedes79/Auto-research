# E273: the lane graph from pixels to map ids as one differentiable function

Result folders of experiment E273 (card: `experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md`, summary in its section 10; ledger rows `E273.p_s0`, `E273.p_s1`). One folder per run: `E273.p_s0/`, `E273.p_s1/` (this folder is the index). In git: this README, each run's `README.md`, `package.json`, `provenance.json`; not in git (binaries): `figures/`, `videos/`, `frames/`.

## Question and verdict
Can the chain *pixel maps -> grid-cell nodes -> Hermite links -> soft sections -> assignment of map piece ids* be one differentiable function (no matching rule, hard steps only at the named output), and does its assignment beat the geometric one when the map is wrong, without making the delivered lane graph worse than the best graph so far (`recall_sp0_dp`, "d")?

**Verdict: CUT by hand (rule 4 of the card, both seeds on the same side).** The assignment holds (+0.071 lane accuracy over the geometric control when the map is 1-2 m off, exact-map guard held), the graph does not (topo F1 0.596 and F1@1.0 0.827 against d 0.741 / 0.878: guards broken, junction F1 0.125 against 0.273). The card sends the work back to nodes and links. What was built stays as the base: code, targets, scorer, pictures, trainer with gates.

| VAL 873 frames | p seed 0 | p seed 1 | mean p | g, mean of the seeds (E272 chain + geometric assigner) | d (recall_sp0_dp) |
|---|---|---|---|---|---|
| lane_acc, map error 1-2 m | 0.9489 | 0.9501 | 0.9495 | 0.8781 |  |
| lane_acc, exact map | 0.9799 | 0.9797 | 0.9798 | 0.9864 |  |
| road_acc (road level derived from the lanes), map error | 0.9487 | 0.9468 | 0.9478 | 0.9321 |  |
| p - g, map error | 0.0722 | 0.0706 | 0.0714 |  |  |
| p - g, exact map | -0.0072 | -0.0061 | -0.0066 |  |  |
| topo_F1 | 0.597 | 0.595 | 0.596 |  | 0.741 |
| F1@1.0 | 0.832 | 0.821 | 0.827 |  | 0.878 |
| junction_F1 | 0.130 | 0.120 | 0.125 |  | 0.273 |
| lanes_found | 0.644 | 0.630 | 0.637 |  | 0.781 |
| Q | 0.636 | 0.630 | 0.633 |  | 0.742 |
| dup_frac | 0.232 | 0.192 | 0.212 |  | 0.077 |
| vert100_dp | 28.881 | 31.453 | 30.167 |  |  |
| pieces_per_gt_piece | 1.607 | 1.596 | 1.602 |  |  |
| id_switches_per_100m | 1.398 | 1.372 | 1.385 |  |  |
| node F1 (2 m grid) | 0.8525 | 0.8505 | 0.8515 |  |  |
| node position error, median (m) | 0.2772 | 0.2747 | 0.2759 |  |  |

p - g with the 95 % scene bootstrap (873 VAL frames, 1000 resamples of scenes, paired):
- seed 0: map error +0.0722 [+0.0563, +0.0871]; exact map -0.0072 [-0.0151, +0.0004]
- seed 1: map error +0.0706 [+0.0551, +0.0853]; exact map -0.0061 [-0.0144, +0.0023]

Source of every number: `runs/e273/<run>/card.json` (written by `evaluation/qgraph_run_eval.py score`, rendered in `runs/e273/<run>/results.md`); VAL 873 frames (`data/splits.json["val"]`, frame hash `e04e6704a782`), never TEST. Time per frame (head only, idle GPU, batch 1, bf16, cached features): 24.2 ms seed 0, 23.2 ms seed 1 (`runs/e273/<run>/timing.json`).

## What the network is taught (the targets, in plain words)
Every target is computed from the lane GT (`autogt/lane-l3`) **by location**: nobody matches predictions to GT objects in the loss, a cell or a pair of nodes simply has an answer.
1. **Pixel maps, three resolutions.** 0.5 m: where the lane dividers and the road boundary are. 1 m: where the lane centre lines are, plus six event heat maps (stop, exit, split, merge, start, end). 2 m: road mask and intersection mask (the intersection mask is widened by 2 m so the network sees a margin).
2. **Nodes on a 2 m grid (50 x 50 cells).** A cell that holds a lane centre line gets **one node**: "a node exists here", its offset (where in the cell), its heading, and its type flags (stop, exit, split, merge, start, end, window-in, window-out). The node sits at the GT line's point nearest the cell centre; the end points of a GT piece win their cell. **Inside intersections there are no nodes except at such event markers** (Mirko: nodes in intersections only where there is some form of marker; the connector lines there are not painted, and 0 % of their points lie within 1 m of a painted marker).
3. **Links.** For every pair of nodes: "is B the next node after A on a GT lane?", found by walking along the GT pieces and jumping over the suppressed intersection part, so a link through an intersection is a *long link* (more than 8 m); training also fits the two handle lengths of the Hermite curve to the GT connector (short links use chord / 3). One-to-many: a delivered node within 1.5 m of a GT node counts as that GT node.
4. **Sections (one id per lane piece, new ids only at splits and merges).** For every node: "a new section starts here" (the first node of a GT piece), and nodes of the same GT piece must end in the same section.
5. **Map assignment.** The map is the lane GT of the frame with simulated errors (translation 1-2 m, yaw, thinned shape points, 5 % of the pieces dropped, 5 % lane counts off). For every delivered node the target is the id of the GT piece it came from ("none" if the node has no GT counterpart).

## GT defaults (what the lane GT assumes: the GT chat's defaults, which were kept; each can be changed there)
Road boundary = edge of the drivable area including driveways and parking; a boundary corner is a turn of more than 30 degrees within 2 m; a stop point with several successors only for the intersection branches; pieces shorter than 2 m are dropped as stubs; piece ids change only at splits and merges (`piece` is the same id in every frame, `id` differs only for a part that comes back into the window); the road level is derived from the lanes (the nuScenes road layer is unusable on two of four maps). Front GT `c643a7539d` is not read by the head (the cached GaussianCaR features were made with it).

## Versions (everything that makes the numbers)
| what | version |
|---|---|
| lane GT | `autogt/lane-l3`, signature `3576fa1f7e` (tag `autogt/lane-l3`, commit 4df1032) |
| front GT of the cached features | `c643a7539d` (v8; not read by the head) |
| head code | `training/grid/{pyramid,qnodes,qlinks,sections,qassign,qgraph,qpredict,qtargets,qformat,map_sim,qraster}.py`, data `training/data/lane_gt_data.py`; seed 0 tag `lcr/E273.p_s0` (code commit `ab72b01707`, phases 5-7; phases 2-4 from earlier launches of the same run id, see the card's section 10), seed 1 tag `lcr/E273.p_s1` (code commit `8e8d73ff3f`, all phases) |
| training | `training/train/qgraph_train.py` on the shared `training/trainer.py` (`T.fit`: AdamW, one-cycle, bf16, clip 3, checkpoint and resume); phases 2-7 with 3000 / 6000 / 5000 / 2000 / 1500 / 2000 steps, lr 1e-3 / 1e-3 / 1e-3 / 5e-4 / 1e-3 / 2e-4, batch 4, width 48, map errors random in training (`map_sim` mode "train"), E271 head `runs/e271_a_s<seed>/model.pt` for the 1 m branch; seeds 0 and 1; seed 1 all phases `--no-gates` (by hand, see the card) |
| checkpoints | `runs/e273_p_s0/model.pt` (hash `cea068319415`), `runs/e273_p_s1/model.pt` (hash `0a8e6d5dd8f1`); not in git |
| dataset | nuScenes v1.0-trainval; TRAIN 8414 frames with lane GT, VAL 873 frames (`data/splits.json`, frame hash `e04e6704a782`), TEST not used |
| evaluation code | `evaluation/qgraph_eval.py`, `qgraph_run_eval.py` (+ the lane suite's functions), commit `5c87b96f7e`, hash `e67cfd0a9500`; reference d from `runs/suite/graphs_recall_sp0_dp/quant/graphs_recall_sp0_dp_carry_20261104_181557` |
| pictures / videos | `tools/viz/qgraph_results.py`, `qgraph_view.py`; scenes 0221, 0108, 0092 (VAL) |

## Rebuild
Work from `bev_fusion_lane_center/`. The model files must exist (`runs/<run>/model.pt`).
```
wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s0 --seed 0        # scoring card + results.md, VAL 873 frames (GPU, about 9 min)
wsl bash tools/env/_py14.sh qgraph_run_eval.py verdict --runs e273_p_s0 e273_p_s1    # the card's rule 4
wsl bash tools/env/_py14.sh qgraph_timing.py --run e273_p_s0 --n 60                  # ms per stage (idle GPU)
wsl bash tools/env/_py14.sh qgraph_results.py --run e273_p_s0 --id E273.p_s0 --device cpu --scenes 11 39 23   # figures, videos, frames
```
Training (through the launcher; seed 1 shown): `python3 ../autoresearch/tools/run.py --id E273.p_s1 --seed 1 --gpu --retries 0 --require-card -- bash tools/env/_pyr.sh qgraph_train.py --name e273_p_s1 --seed 1 --workers 3 --no-gates`. The files `eval/metrics.json` and `package.json` of a run were written from its scoring card by a scratch script (`finish_run.py` in the git tag `lcr/e273-scratch`, which keeps the text files of all E273 scratch: launchers, probes, patch scripts, picture and ledger scripts).

## Qualitative
Each run's `videos/` hold three VAL scenes (the six camera views with the lane graph projected, the pixel maps with their errors, the node graph with Hermite links, GT piece ids and delivered ids as colours); `figures/` one frame of each. On plain roads the nodes lie on the lanes with the right ids (colour constant along a lane); every intersection shows a ring of false (red) and missed (blue) connectors, which is the number story above. Start with `E273.p_s0/videos/scene-0221_e273_p_s0.mp4`.
