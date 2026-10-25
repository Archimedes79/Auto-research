# E219_recall-weight-finetune-under-pruning

**Recall-weighted fine-tune of the champion re-opened under fork pruning (weights 3 / 6, range-weighted)**

| | |
|---|---|
| experiment id | E219 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Fast j1rc (recall weight 3 on j1 192 / 96): lanes found 0.671 -> 0.736, J 0.398 -> 0.447, Q 0.729 -> 0.739 for +2 ms (VAL, 2 seeds; equal-budget control: only lanes found and J are the weight's). Slow variants: rc1 (weight 3) + pruning 0.4: Q 0.740, lanes found 0.767, J 0.428; rc2 (weight 6): Q 0.735, lanes found 0.793 (a tie with the control 0.800), precision 0.858; rc4 (extra weight beyond 25 m): Q 0.736, lanes found 0.770. The rules written beforehand (Q >= 0.745 with lanes found >= 0.78) were met by none; rc1 stays the recall variant. Extension: seed 1 of rc1 reproduces it (Q 0.738, lanes found 0.770); the same weight-3 fine-tune on the fast j1 192 / 96 (j1rc) lifts lanes found 0.671 -> 0.736, J 0.398 -> 0.447, Q 0.729 -> 0.739 (paired, all intervals above zero) for +2 ms (19.9 vs 17.3 ms): second seed: Q 0.743, lanes found 0.743, J 0.451, 21.8 ms. An equal-budget control (same fine-tune at recall weight 1, j1eq: Q 0.735, J 0.417, lanes found 0.674) shows that the recall weight buys lanes found (+0.062) and J (+0.030) while the Q gain over j1 is mostly extra fine-tuning. Adopted as the fast recall variant on VAL (two seeds, no TEST). The control of rc1 (f1eq) has lanes found 0.707: rc1 buys +0.060 lanes found and nothing measurable in Q or J. Weight 6 on j1 (j1rc6): lanes found 0.758 but Q 0.736 (j1rc 0.739): j1rc stays. The plain fine-tune at twice the budget (j1eq8, 8 u): Q 0.731, J 0.402, lanes found 0.678 - no further gain, the 4 u control j1eq stays.

## Techniques used (register K ids)

- K184, K190 (see `../../literature/register.md`)

## Data

- `results/rc1.json`
- `results/rc2.json`
- `results/rc4.json`
- `results/rc1s1.json`
- `results/j1rc.json`
- `results/j1rcs1.json`
- `results/j1eq.json`
- `results/j1eq8.json`
- `results/j1eqs1.json`
- `results/f1eq.json`
- `results/j1rc6.json`
- `results/j1p4.json`
- `results/final_boot.json`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
