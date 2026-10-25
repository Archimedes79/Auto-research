# E092_expert-replay-ground-truth-expert-drives

**Expert replay: a ground-truth expert drives the wavefront of the learned extractor**

| | |
|---|---|
| experiment id | E92 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB20 generative / sequential heads |
| loop | loop10 |

## Result

F1 0.99; 25 - 31 % of junctions lost even with perfect decisions (duplicate annotations)

## Techniques used (register K ids)

- K84: graph grown vertex by vertex with a transformer, imitation with corrective expert actions  (checked loops 8, 10)
- K14: iterative graph growth guided by a CNN decision  (checked (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 2 small files from `e92_replay`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
