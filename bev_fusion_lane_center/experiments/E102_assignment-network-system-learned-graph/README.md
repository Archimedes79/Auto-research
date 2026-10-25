# E102_assignment-network-system-learned-graph

**Assignment network of system A on the learned graph (zero-shot and retrained)**

| | |
|---|---|
| experiment id | E102 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB13 relation bias, graph transformer, message passing |
| loop | loop10 |

## Result

0.042 - 0.048 route success below; oracle 0.913 on both

## Techniques used (register K ids)

- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 9 small files from `e102_ref_rule`, `e102_zs_a`, `e102_zs_a.log`, `e102_zs_or`, `e102_zs_or.log`, `e102b`, `e102b.log`
- `code/`: `_chain_e102.sh`, `_chain_e102b.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
