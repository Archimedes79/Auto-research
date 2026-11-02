# training/grid - the simple grid chain for Spurmitte (design + the parts that exist)

Mirko's wish (2026-11-02): a grid output at 1 m that gives centre-lane pixels, then a sparsification, then temporal fusion, then the assignment of the sparse lanes to the map lanes; **E2E differentiable only** (no argmax, no matching, no rule step on the training path), simple, copy and adapt instead of building.

```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m          (later: the BEV system's trunk state)
   |  stride-2 conv  -> 100 x 100 @ 1 m, small U-Net (~0.5 M parameters)
   v
[1] GRID HEAD     6 ch / cell: p (centre-lane logit), heading (cos, sin), sub-cell offset (ox, oy), keep gate s
   v   soft_lines.sparsify          w = sigmoid(p) * ridge(across the line) * sigmoid(s)     weights, no top-k, no threshold
[2] SPARSE MAP    5 ch / cell: w, cos, sin, ox, oy          vertex budget loss ~ 11 vertices per 100 m
   v   hold.Hold                    warp the last fused map with the ego motion (one grid_sample), gated noisy-or fuse
[3] FUSED MAP     same 5 ch, carried to the next frame
   v   assign.Assigner              softmax over (map tokens + dustbin): distance^2 / sigma^2 + lam (1 - heading . tangent); learned sigma, lam, c_bin
[4] ASSIGNMENT    P(road | vertex), lateral offset, lane slot, presence per road
   -> delivered: the SD map's lanes (its links = the topology), each refined by the weighted vertices pulled to it, with a presence confidence and the route flag
```

Why it is simpler than the current chain (24 stages, 12 rule steps): lane instances, forks and merges are not built by a graph builder and rule clean-up: the **map supplies the topology**, the perception supplies **where each lane really is and whether it exists**. Where no map exists the delivered output is the sparse vertices only (a named gap, not hidden).

| file | content | copied / adapted from | test |
|---|---|---|---|
| `soft_lines.py` | stage 1 + 2: `soft_line_map` (copy), `ridge` / `sparsify` (across-line soft NMS x keep gate), `budget`, `positions`, `vertices` (read-out at inference only) | `bev_fusion/training/extract.py` (`soft_line_map`); the rest new, small | `python soft_lines.py` (ridge thin, gradient to all 6 channels) |
| `hold.py` | stage 3: `warp` (ego motion, heading and offset channels rotated), `Hold` (gate, noisy-or) | the idea of `bev_fusion/training/temporal_fusion.py` (pose step, one bilinear read, conv fusion), without its world memory and window | `python hold.py` (a world line seen from two poses is reproduced, error 0.011) |
| `assign.py` | stage 4: `map_tokens` (SD map -> tokens), `Assigner` (soft assignment with a dustbin, lateral offset, lane slot), `present` | `training/data/assoc_lib.py` (`assign_points`) and `sd_lib.py` made soft; registration stays `training/models/p18_regis.py` (SoftRegister), not copied | `python assign.py` (road accuracy 0.98, distractor to the dustbin 0.81, gradient to every input and scalar) |

Picture and video of the assignment: `tools/viz/assign_view.py` (real VAL frames: the delivered graph of a registry system + its SD map; START values of the assigner, so it shows the idea, not a trained result).

## What is not built (and why)
- the head and its training: card `experiments/E271_grid-chain-1m-sparse-hold-assign/experiment.md` (stages 1 + 2 against the prototype's vertices, plus the oracle of the whole chain; the hold, the joint assignment and the delivered graph are follow-up cards); nothing runs before Mirko's word. Plan: loss on `sparsify` output = soft recall + precision against the GT lines (distance field of the GT, looked up by `grid_sample`, matching-free), heading cosine and offset L1 on GT cells, `budget`; assignment loss = CE of P(road) against the GT road (`assoc_lib.gt_assign`) + slot CE; sequences of 4 frames with `Hold`, detach every N.
- evaluation: nothing to copy - `evaluation/lane_suite.py dense:<run>` already scores a dense head as a picture first and as the graph read from it, with the project's Q / topo numbers (the same evalsuite the BEV system uses). The oracle step (GT lanes through stage 1 -> 4 must reproduce the GT, the idea of `../../../bev_fusion/evaluation/bev_gtcheck.py`) is the first test of the chain.
- known limits to test first: one heading per cell (two lanes crossing in a cell: add a second heading channel), 1 m cells against the 1 m metric tolerance (the sub-cell offset is what saves it), a wrong or missing map.
