# E094b_wrong-direction-states-four-heads-wavefront

**Wrong-direction states and four heads; wavefront order; range-aware thresholds**

| | |
|---|---|
| experiment id | E94b / E96 / E98 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB20 generative / sequential heads |
| loop | loop10 |

## Result

no gain beyond the noise

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
