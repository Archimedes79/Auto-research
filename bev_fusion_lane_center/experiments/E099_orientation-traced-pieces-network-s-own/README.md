# E099_orientation-traced-pieces-network-s-own

**Orientation of traced pieces by the network's own direction head (sign vote)**

| | |
|---|---|
| experiment id | E99 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB19 output heads: class + residual, splines |
| loop | loop10 |

## Result

+0.05 direction, +0.02 route F

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
