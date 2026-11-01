# E107_whole-learned-chain-under-failure-oracle

**Whole learned chain under failure; an oracle flag and a gradual onset**

| | |
|---|---|
| experiment id | E107 / E108 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop11 - log: [../../experiment_execution/history/indexes/LOOP11_INDEX.md](../../experiment_execution/history/indexes/LOOP11_INDEX.md) |

## Result

night route success 0.093 to 0.346; a perfect flag adds +0.07 (blur) and nothing for lost sensors

## Techniques used (register K ids)

- K106: BEV queries pulling from whatever sensors are present, trained for sensor failure  (checked loop 12 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `_e108.sh`, `_e108b.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
