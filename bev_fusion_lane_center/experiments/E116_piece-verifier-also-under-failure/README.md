# E116_piece-verifier-also-under-failure

**Piece verifier (short graphs instead of false lanes), also under failure**

| | |
|---|---|
| experiment id | E116 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP4 keep and verify (precision gates, verifiers, probabilities) |
| building blocks | BB13 relation bias, graph transformer, message passing |
| loop | loop13 - log: [../../experiment_execution/history/indexes/LOOP13_INDEX.md](../../experiment_execution/history/indexes/LOOP13_INDEX.md) |

## Result

+0.043 precision on clean frames; lower F1 in night and blur unless gated or failure-trained

## Techniques used (register K ids)

- K09: learned duplicate removal as a relation module  (checked)
- K107: learning non-maximum suppression  (checked loop 13 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 6 small files from `e116_fail`, `e116_valfail`
- `code/`: `_run_e116c.sh`, `_run_e116c2.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
