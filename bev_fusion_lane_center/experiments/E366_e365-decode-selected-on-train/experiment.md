# E366 - the two trained heads of E365 S1 with a decode selected on the TRAIN sample (decode only, no retraining): is the miss of S1 a calibration problem of the decode?

Card of 2026-11-17 (version 2, 01:45 system clock, after the first critic pass (fresh ar-critic, cards: REVISE, 7 findings, taken in: the JSON rule is not the verdict, no second cell, the grid edge, the sample fixed, the verdict aligned with E365, the TRAIN bias said, the two seeds are two heads); before any sweep; written by the Spurmitte chat that Mirko addresses as "Fable", on the lane lead's 01:25 reading: "the head is too conservative, the gap is calibration + capacity"). **Status: draft; the sweep code is being added by a coder (`score --link-thr --keep-thr --no-none-gate` and `sweep` are not yet in `jtoken_eval.py`); nothing run.** E365 S1 (experiments/E365_e2e-token-champion-s1, section 3) delivered precision 0.86 and recall 0.40 - 0.43 with the decode of record (a link when p >= 0.3 and p >= p(none), both ends kept at p_none < 0.5). **The one question:** with the SAME two trained heads (`runs/e365_s1_s0`, `runs/e365_s1_s1`), does a looser decode, selected on the TRAIN sample and applied once to VAL, lift the delivered graph over the gate of E365 (G-topo >= 0.6359 / 0.6349, G-found: F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937)?

## 0. The card (what the tools run)
```json card
{
 "id": "E366",
 "card": "",
 "title": "E365 S1 heads (frozen, two seeds) read out with a decode selected on the TRAIN sample (every third frame of the 873 evenly spaced TRAIN frames: 291 frames, fixed now; link threshold x keep threshold x none gate on / off, 24 cells) and applied once to VAL, against the decode of record of E365 and the gate of E365",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.03, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E365 S1, decode of record: VAL topo F1 0.5486 / 0.5743, F1@1.0 0.7524 / 0.7631, lanes found 0.5001 / 0.5252 (runs/e365_s1_s<S>/token_card_rec.json)"},
  "dec": {"recipe": "the same heads; `jtoken_eval.py sweep` on the TRAIN sample (cards under runs/e365_s1_s<S>/token_sweep_train.json), the cell chosen by the rule below, `jtoken_eval.py score --link-thr T --keep-thr K [--no-none-gate]` once on VAL", "run_ids": ["E366.sweep_s0", "E366.sweep_s1", "E366.val_s0", "E366.val_s1"]}
 },
 "control": "ref",
 "variant": "dec",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 0.0,
 "kill": "a read-out is not finite; no GPU job (CPU, niced, 2 threads)",
 "evaluate": {"split": "val"}
}
```
**The selection rule (written before the sweep, by hand: a decode choice on the TRAIN sample):** the grid = link threshold T in {0.3, 0.2, 0.1, 0.05} x keep threshold K in {0.5, 0.6, 0.7} (a node is kept when p_none < K) x none gate in {on (p >= p(none) required), off}; per cell the TRAIN-sample topo F1 of BOTH seeds; the chosen cell = the cell with the highest MEAN topo F1 over the two seeds among the cells whose mean TRAIN dup_frac is at most 0.05 (a guard: recall must not be bought with duplicates); ties to the cell nearest the decode of record (T, then K, then the gate). The same cell is used for both seeds. The chosen cell is applied ONCE to VAL (873 frames); nothing is chosen on VAL. **The limit, said now:** the TRAIN sample is cleaner than VAL (B' recall 0.926 against 0.862, S1 topo F1 0.61 against 0.55) and the heads were trained on those frames (their probabilities are more confident there), so a decode selected there is biased TIGHTER than the VAL optimum; the VAL result is the honest test, a VAL sweep (in-sample) is reported only as a diagnosis next to it, never chosen from. **The chosen cell is applied ONCE. No second cell is tried, the grid is not extended afterwards, and if the result on VAL is not better than the record (a gain below +0.01 on a seed) the verdict stands and E367 reads its ref at the record decode.** **What will probably happen, said before the sweep:** precision 0.86 against recall 0.40 - 0.47 means topo F1 rises with every loosening, so the rule will probably pick the loosest cell the guard allows (T = 0.05, K = 0.7, none gate off); the guard dup_frac <= 0.05 is measured on TRAIN (dup_frac 0.005 there), VAL duplicates can be higher: the VAL dup_frac is reported next to the result. The two seeds are two different trained heads, not noise draws.
**The verdict (by hand: the gate of E365 section 0, applied to this decode):** PASS-DECODE = G-topo and G-found hold on both seeds (the cost and simplicity gates are those of E365 S1, unchanged: same heads; the thresholds are on the output only, so G-simple holds); MISS-DECODE = G-topo fails on both seeds or G-found fails on both seeds (as E365); else INCONCLUSIVE (reported with the diagnosis; no action beyond the reading of E367's ref). **The JSON rule above is NOT the verdict:** `decide.py` is not used (a gain of +0.011 would read KEEP while the gate needs +0.06 to +0.09); the verdict is by hand: the gate of E365 section 0. The paired scene-bootstrap interval dec - ref is reported per seed. E367 does not take over this cell: it selects the decode PER HEAD by this rule on that head's own TRAIN sample; this card's VAL result of the S1 heads is E367's `ref`.
**Prediction (guesses, with windows):** the chosen cell is looser than the record (T <= 0.2) with probability 85 % (the grid edge, see above); VAL topo F1 +0.03 over the record (window 0 to +0.08), i.e. about 0.58 / 0.60; PASS-DECODE about 5 %, MISS-DECODE about 85 %.

## 1. Card text
- **Statement:** the miss of S1 is largely the conservative decode; **Mechanism:** precision 0.86 against recall 0.40 says the operating point sits far on the precise side; **Not claimed:** that this is the best operating point for another head (E367), anything about TEST; **Cost:** CPU only, 24 cells x 2 seeds x 291 TRAIN frames (fixed above): about 13 minutes per seed, niced, two seeds in parallel.
- **Referee:** the sweep code is reviewed with the E365 read-out code (code critic); the rule is fixed here before the sweep; no card critic planned (a decode-only card).

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| code | `score --link-thr --keep-thr --no-none-gate` and `sweep` (a coder, tests on 3 frames) | Fable / coder | tests green |
| sweep | `jtoken_eval.py sweep --run e365_s1_s<S> --seed S --split train`, CPU, nice, 2 threads, one run.py job per seed | `run.py --id E366.sweep_s<S>` | token_sweep_train.json |
| choose | the rule above, by hand, both seeds' tables | Fable | the cell written in section 3 |
| VAL | `score` with the chosen cell, once per seed | `run.py --id E366.val_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit |

## 3. Summary
(empty until the runs exist)
