# E044_learned-split-merge-key-points-node

**Learned split / merge key points (2.7 M) with node-guided assembly; Bezier connectors between key points**

| | |
|---|---|
| experiment id | E44 / E53 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP3 graph assembly and junctions (links, connectors, assembler) |
| building blocks | BB05 peak picking: NMS, top-k, Gumbel top-k; BB19 output heads: class + residual, splines |
| loop | loop04 |

## Result

split / merge F1 at 6 m 0.34 / 0.38 to 0.55 / 0.61 (key points) to 0.71 / 0.64 (connectors); route F -0.025 against the previous system

## Techniques used (register K ids)

- K48: nodes + pairwise edge classifier  (checked)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `make_e44.py`, `make_e53.py`, `run_e44_cfg.sh`, `run_e44_cons.sh`, `run_e44_cut2.sh`, `run_e44_map_cons.sh`, `run_e53.sh`, `run_e53_map.sh`, `run_e53_reg.sh`, `run_e53b.sh`, `run_e53c.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
