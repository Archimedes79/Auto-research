# E188_loop-16b-relaxed-child-selection-near

**Loop 16b (validation only, after the champion freeze): Relaxed child selection near predicted forks at inference (smaller dominance and non-maximum-suppression radii around junction peaks)**

| | |
|---|---|
| experiment id | E188 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric) |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.750, J 0.278 - 0.279 against 0.749 / 0.278; oracle junction heat J 0.282: the second branch of a shallow fork is not in the tree

## Techniques used (register K ids)

- K183: assembly-side fork recovery: admission of nodes down to a lower validity near junction peaks; sibling vertices kept apart near peaks (small   (own idea, tested in loop 16: negative)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
