# E190_loop-16b-hysteresis-threshold-assembly

**Loop 16b (validation only, after the champion freeze): Hysteresis threshold in the assembly (nodes with validity 0.3 - 0.6 admitted when their parent is, tau 0.7 - 0.9)**

| | |
|---|---|
| experiment id | E190 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB26 tree -> DAG merge and edge split (geometric) |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

lanes found 0.695 -> 0.70 - 0.71, J 0.278 -> 0.257 - 0.271, S 0.745 - 0.748 against 0.749 (VAL250); plain threshold 0.9: S 0.754, J 0.308, lanes found 0.649 (a flat knob, tau stays 0.7): no gain

## Techniques used (register K ids)

- K183: assembly-side fork recovery: admission of nodes down to a lower validity near junction peaks; sibling vertices kept apart near peaks (small   (own idea, tested in loop 16: negative)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
