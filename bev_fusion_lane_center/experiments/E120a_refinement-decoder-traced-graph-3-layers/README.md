# E120a_refinement-decoder-traced-graph-3-layers

**Refinement decoder on the traced graph (vertices, edges, step tokens of E94 and the 42 BEV channels): 3 layers (or 1, or reference points detached as in MapTR) sample the BEV at the CURRENT vertices, graph-biased self-attention, displacement; point-to-polyline loss on the finished polylines, no matching (1.37 M)**

| | |
|---|---|
| experiment id | E120a - E120c |
| theme | T8 - Differentiable vertex positions (refinement of the traced graph) (loop 15) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP6 vertex positions of the sparse graph (refinement, keep gate) |
| building blocks | BB02 deformable convolution / attention; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB13 relation bias, graph transformer, message passing; BB29 matching-free point-to-polyline loss |
| loop | loop15 - log: [../../experiment_execution/history/indexes/LOOP15_INDEX.md](../../experiment_execution/history/indexes/LOOP15_INDEX.md) |

## Result

TEST F1@0.5 0.661 to 0.694, F1@1 0.851 to 0.863, vertex error 0.54 to 0.43 m; 1 layer 0.861, detached 0.862 (noise 0.002): depth and the gradient between the layers do not matter; the vertices slide 1.8 m along the lane

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K109: map vector queries with hierarchical matching; per-layer refinement  (checked loop 15)
- K43: look forward twice, contrastive denoising  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K115: Chamfer distance as a point-set loss without assignment  (checked loop 15)
- K116: differentiable rasterisation of vectorised maps  (checked loop 15)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
