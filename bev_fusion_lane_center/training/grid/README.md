# training/grid - the simple grid chain for Spurmitte (E2E only, E271)

Mirko's wish (2026-11-02): a grid output at 1 m that gives centre-lane pixels, then a sparsification, then temporal fusion, then the assignment of the sparse lanes to the map lanes; **E2E differentiable only** (no argmax, no matching, no rule step on the training path), simple, copy and adapt instead of building. Card: `../../experiments/E271_grid-chain-1m-sparse-hold-assign/experiment.md` (stages 1 + 2 and the oracle of the chain; hold, joint assignment and the delivered graph are follow-up cards).

```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m          (x[:39]: the old dense head's channels are not an input)
   |  stride-2 stem, small U-Net (2.05 M parameters, GroupNorm)        head.py
   v
[1] GRID HEAD     6 ch / cell @ 1 m: p (centre-lane logit), heading (cos, sin), sub-cell offset (ox, oy), keep gate s
   v   soft_lines.sparsify          w = sigmoid(p) * ridge(across the line) * sigmoid(s)     weights, no top-k, no threshold on the training path
[2] SPARSE MAP    5 ch / cell: w, cos, sin, ox, oy          loss: coverage + precision + vertex budget (losses.py)
   v   hold.Hold                    warp the last fused map with the ego motion (one grid_sample), gated noisy-or fuse          (follow-up card)
[3] FUSED MAP
   v   assign.Assigner              softmax over (map tokens + dustbin): distance^2 / sigma^2 + lam (1 - heading . tangent)    (follow-up card)
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
| `test_oracle.py` | G0: the GT through the chain must reproduce itself (real VAL frames, degraded copies must fall) | idea of `bev_fusion/evaluation/bev_gtcheck.py` | itself |
| `test_grid.py` | G1: gradient reaches every parameter group, one batch can be overfitted, features equal `BuilderData`'s | - | itself |

Trainer: `../train/grid_train.py` (on `../trainer.py`), scoring: `../../evaluation/vertex_eval.py`, pictures: `../../tools/viz/assign_view.py`.
