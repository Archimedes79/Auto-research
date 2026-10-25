# E046_lanegap-path-view-repair-seqgrowgraph-literature

**LaneGAP path view and repair; SeqGrowGraph (autoregressive growth) as a literature case**

| | |
|---|---|
| experiment id | E46 / E47 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP3 graph assembly and junctions (links, connectors, assembler); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB20 generative / sequential heads |
| loop | loop04 |

## Result

only 0.428 of the ground-truth paths recovered; repair +0.001; led to the trace network

## Techniques used (register K ids)

- K12: path-level supervision instead of pieces  (checked)
- K86: lane topology as a chain of graph expansions (autoregressive, quantised)  (checked loops 13, 15)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `make_e46.py`, `make_e47.py`, `run_e46_repair.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
