# E032_sparse-detr-style-laneformer-instead-grid

**Sparse DETR-style LaneFormer (4.3 M, 96 queries, explicit successor matrix) instead of the grid**

| | |
|---|---|
| experiment id | E32 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB15 learned queries with set matching; BB11 self-attention |
| loop | loops02-03 |

## Result

0.058 F1 and 0.178 route F below the grid approach

## Techniques used (register K ids)

- K109: map vector queries with hierarchical matching; per-layer refinement  (checked loop 15)
- K110: vectorised HD map as autoregressive polylines  (checked loops 10, 15)
- K111: DETR: set prediction with bipartite matching  (checked loop 15)
- K10: lane segments + topology head  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `e32_main`
- `code/`: `make_e32.py`, `run_e32_fp1.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
