# E048_learned-registration-map-prior

**Learned registration of the map prior (RAVE-inspired fine matching after a correlation search)**

| | |
|---|---|
| experiment id | E48 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB04 correlation / cost volume + soft-argmax |
| loop | loop04 |

## Result

median error 0.54 to 0.43 m at 1 m

## Techniques used (register K ids)

- K99: differentiable localisation of a BEV raster against a navigation map (pose volume, soft-argmax)  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 31 small files from `e48_align`, `e48_chain2`, `e48_clean_s05_corrfine`, `e48_clean_s05_lalign`, `e48_clean_s1_chain`, `e48_clean_s1_corrfine`, `e48_clean_s1_lalign`, `e48_clean_s2_chain` ...
- `code/`: `make_e48.py`, `run_e48_chain.sh`, `run_e48_eval.sh`, `run_e48_eval2.sh`, `run_e48_sys.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
