# E189_loop-16b-inference-time-ensemble-two

**Loop 16b (validation only, after the champion freeze): Inference-time ensemble of two builders (champion f1 + replicate s2, node union)**

| | |
|---|---|
| experiment id | E189 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric) |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

F1@1 0.871 -> 0.885, lanes found 0.695 -> 0.778, J 0.278 -> 0.177, S 0.749 -> 0.734; only confident nodes of s2: S 0.738 - 0.743

## Techniques used (register K ids)

- K183: assembly-side fork recovery: admission of nodes down to a lower validity near junction peaks; sibling vertices kept apart near peaks (small   (own idea, tested in loop 16: negative)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
