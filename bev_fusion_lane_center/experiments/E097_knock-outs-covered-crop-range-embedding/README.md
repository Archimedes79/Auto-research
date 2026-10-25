# E097_knock-outs-covered-crop-range-embedding

**Knock-outs: covered crop, range embedding, merge and fork events**

| | |
|---|---|
| experiment id | E97 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB14 pointer attention |
| loop | loop10 |

## Result

the crop of the emitted graph carries the junction topology (split / merge F1 0.35 / 0.46 to 0.19 / 0.21 without)

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 2 small files from `e97_nocov`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
