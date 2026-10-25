# VLA_literature-review-vision-language-action-model

**Literature review (not an experiment): is a vision-language-action model needed?**

| | |
|---|---|
| experiment id | VLA |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |

## Result

language adds about +0.4 driving score, inside run noise; needs a workstation GPU

## Techniques used (register K ids)

- K108: open-loop planning metrics and the ego-status caveat  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
