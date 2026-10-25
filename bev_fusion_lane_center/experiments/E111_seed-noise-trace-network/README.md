# E111_seed-noise-trace-network

**Seed noise of the trace network (second training run)**

| | |
|---|---|
| experiment id | E111 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| loop | loop12 - log: [../../experiment_execution/indexes/LOOP12_INDEX.md](../../experiment_execution/indexes/LOOP12_INDEX.md) |

## Result

F1 differs by 0.001 - 0.002, an order below the gap to the rule graph

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `_apply_e111.py`, `_e111_eval.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
