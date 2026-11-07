# training/grid - the simple grid chain for Spurmitte (E2E only, E271)

Mirko's wish (2026-11-02): a grid output at 1 m that gives centre-lane pixels, then a sparsification, then temporal fusion, then the assignment of the sparse lanes to the map lanes; **E2E differentiable only** (no argmax, no matching, no rule step on the training path), simple, copy and adapt instead of building. Card: `../../experiments/E271_grid-chain-1m-sparse-hold-assign/experiment.md` (stages 1 + 2 and the oracle of the chain; hold, joint assignment and the delivered graph are follow-up cards).

```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m          (x[:39]: the old dense head's channels are not an input)
   |  stride-2 stem, small U-Net (2.05 M parameters, GroupNorm)        head.py
   v
[1] GRID HEAD     6 ch / cell @ 1 m: p (centre-lane logit), heading (cos, sin), sub-cell offset (ox, oy), keep gate s
   v   soft_lines.sparsify          w = sigmoid(p) * ridge(across the line) * sigmoid(s)     weights, no top-k, no threshold on the training path
[2] SPARSE MAP    5 ch / cell: w, cos, sin, ox, oy          loss: coverage + precision + vertex budget (losses.py)
   v   hold.Hold                    warp the last fused map with the ego motion (one grid_sample), gated noisy-or fuse          (E272: trained in the chain)
[3] FUSED MAP
   v   assign.Assigner              softmax over (map tokens + dustbin): distance^2 / sigma^2 + lam (1 - heading . tangent)    (E272: trained in the chain)
[4] ASSIGNMENT    P(road | vertex), lateral offset, lane slot, presence per road
```
The delivered list of vertices is a READ-OUT at inference (`evaluation/vertex_eval.py read_out`: cells with w >= thr that are the 3 x 3 maximum), a named rule step on the output. Where no map exists the output is the sparse vertices only (a named gap).

| file | content | copied / adapted from | test |
|---|---|---|---|
| `head.py` | `GridHead`: stem + 3 down + 3 up blocks, 6 output channels; reads `x[:, :39]` | new, small | `python head.py` (shape; the old-head channels are not read) |
| `targets.py` | `cell_targets`: distance, nearest-point offset and tangent per 1 m cell from the GT lane points; `heat` | new | `python targets.py` |
| `soft_lines.py` | `soft_line_map` (copy), `ridge` / `sparsify` (across-line soft NMS x keep gate), `budget`, `positions`, `vertices` | `bev_fusion/training/extract.py` (`soft_line_map`) | `python soft_lines.py` |
| `losses.py` | `grid_loss`: BCE + Dice on the heat, heading cosine, offset L1, noisy-or coverage of the GT points, precision, vertex budget | new | `python losses.py` |
| `data.py` | `GridData` / `loader`: the 39 frozen channels + GT lane points (VAL through `BuilderData.rows`, never the stored TEST rows) | `training/data/p16_data.py` (paths, rows, normalisation) | `test_grid.py` |
| `hold.py` | `warp` (ego motion, heading and offset rotated), `Hold` (gate, noisy-or) | the idea of `bev_fusion/training/temporal_fusion.py` | `python hold.py` |
| `assign.py` | `map_tokens` (SD map -> tokens), `Assigner` (soft assignment with a dustbin, lateral offset, lane slot), `present` | `training/data/assoc_lib.py` and `sd_lib.py` made soft; registration stays `training/models/p18_regis.py` | `python assign.py` |
| `chain.py` | `GridChain` = head + sparsify + hold + assigner as ONE module, `chain_loss` (per-frame head loss + sparse-map terms on the fused map + assignment loss), G2 gradient test | the parts above | `python chain.py` |
| `chain_data.py` | `ChainData`: T = 3 frames of a scene, the ego step between them (warp checked on real frames), SD map and plausible road labels of the last frame | `data.py`, `training/data/p16_data.py` poses | `ego_step` / `road_labels` used by `chain.py` self-test; warp IoU check in the E272 card |
| `qtargets.py` | E273: nodes on the 2 m grid, links by contraction along the pieces, section starts, `grid_targets` (the 50 x 50 maps of the node head), `decode` / `fit_long_scales` (Hermite curves: the representation oracle), from the lane GT of autogt (`autogt.lane_gt`) | new (card E273 section 3.3) | `python qtargets.py`, `test_qtargets.py` (real frames: lane points 99.9 %, connectors 98.7 % within 1 m; a keyframe without any lane is skipped; on the 200-frame sample of the full l3 store the dropped candidates are 1.04 % of the nodes against the 1 % gate: unchanged by E276, the dense selection is untouched); E276: `node_table(g, sparse=(eps, max_chord))` (the sparse node target), `tier` / `vis` per node and `grid_targets` maps `tier`, `vis`, `ztier` from the l4 labels, `test_tiers_sparse.py` |
| `qformat.py` | E273: THE CONTRACT of what the lane head delivers per frame (nodes, links, assignment, maps) and `oracle_prediction(g)` (a perfect network made from the GT) for the scorer and the pictures | new | `python qformat.py` |
| `map_sim.py` | E273: the map side: the lane GT pieces of a frame with simulated errors (translation, yaw, thinned shape points, dropped pieces, wrong lane counts), tokens every 2 m, relations succ / pred / left / right / opposite | idea of `training/data/sd_lib.py` (degrade_sd) at lane level | `python map_sim.py` |
| `qraster.py` | E273: the pixel targets drawn from the lane GT vectors (0.5 m lines with distance, 1 m event heats and centre points, 2 m road and intersection) | new | `python qraster.py`, `test_qraster.py` |
| `pyramid.py` | E273 stage A: E271's body with three taps (0.5 m lines, 1 m centres + event heats, 2 m road / intersection) and the feature maps of the node grid; `load_e271` | `head.py` | `python pyramid.py` |
| `qnodes.py` | E273 stage B: one node per 2 m cell (existence, flags, offset, direction, section start, long-link scores), neighbourhood attention, refinement by sampling the fine features | new | `python qnodes.py` |
| `qlinks.py` | E273 stage C: the connection matrix (short links in the 8 m neighbourhood, long links between the top-K entries and exits), Hermite curves with learned handle scales, evidence along the curve | new | `python qlinks.py` |
| `sections.py` | E273 stage D: sections (ids) as a soft absorbing walk backwards on the link matrix (a linear solve) and its loss | new | `python sections.py` |
| `qassign.py` | E273 stage E: map encoder (graph transformer over the pieces along succ / pred / left / right / opposite) and the assignment (cross-attention + E272's geometric term + dustbin + soft registration) | `assign.py`, SMERF / SEPT / MAT | `python qassign.py` |
| `qgraph.py` | E273: the whole lane head as one module, the loss of every phase, trainable groups; E274: loss switches `CFG` (event loss weight, existence-positive weights at event / intersection cells, dual-softmax loss over the long links `long_match_loss`; defaults = the E273 loss) | the parts above | `python qgraph.py` (every phase: gradient reaches the trainable groups, none the frozen ones); E276 switches `tier_w`, `vis_w`, `neg_w`, `link_tier_w` (defaults = the E273 loss; `test_tier_loss.py`: defaults bit for bit, each switch moves the gradient as intended) |
| `test_oracle.py` | G0: the GT through the chain must reproduce itself (real VAL frames, degraded copies must fall) | idea of `bev_fusion/evaluation/bev_gtcheck.py` | itself |
| `test_grid.py` | G1: gradient reaches every parameter group, one batch can be overfitted, features equal `BuilderData`'s | - | itself |

Trainers: `../train/grid_train.py` (E271, the head alone), `../train/chain_train.py` (E272, the chain: blocks first with the head frozen, then joint) and `../train/qgraph_train.py` (E273: the lane head phase by phase with a gate after each, `../train/qgates.py`), all on `../trainer.py`; data of E273: `../data/lane_gt_data.py`; scoring: `../../evaluation/vertex_eval.py` and `../../evaluation/chain_eval.py` (hold on / off, trained and start-value assigner), pictures: `../../tools/viz/grid_view.py` (head, vertices, assignment, camera; E271 videos) and `../../tools/viz/assign_view.py` (a registry system's delivered graph against the SD map).
