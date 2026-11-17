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
  "dec": {"recipe": "the same heads; `jtoken_eval.py dump --split train --every 3` and `dump --split val` (new format: per-candidate handle scales, float32 none probabilities; the E365 dumps were kept as token_*_v1.pkl), `jtoken_eval.py sweep --split train --every 3` (runs/e365_s1_s<S>/token_sweep_train_every3.json), the cell chosen by the rule below, `jtoken_eval.py score --link-thr T --keep-thr K [--no-none-gate]` once on VAL", "run_ids": ["E366.dump_s0", "E366.dump_s1", "E366.sweep_s0", "E366.sweep_s1", "E366.val_s0", "E366.val_s1"]}
 },
 "control": "ref",
 "variant": "dec",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 0.2,
 "kill": "a read-out is not finite; GPU only for the four short dumps (run ids E366.dump_s<S>: the S1 heads re-dumped in the new format, TRAIN every 3 and VAL, a few minutes), the sweep and the scoring on the CPU (niced, 2 threads)",
 "evaluate": {"split": "val"}
}
```
**The selection rule (written before the sweep, by hand: a decode choice on the TRAIN sample):** the grid = link threshold T in {0.3, 0.2, 0.1, 0.05} x keep threshold K in {0.5, 0.6, 0.7} (a node is kept when p_none < K) x none gate in {on (p >= p(none) required), off}; per cell the TRAIN-sample topo F1 of BOTH seeds; the chosen cell = the cell with the highest MEAN topo F1 over the two seeds among the cells whose mean TRAIN dup_frac is at most 0.05 (a guard: recall must not be bought with duplicates); ties to the cell nearest the decode of record (T, then K, then the gate). The same cell is used for both seeds. The chosen cell is applied ONCE to VAL (873 frames); nothing is chosen on VAL. **The limit, said now:** the TRAIN sample is cleaner than VAL (B' recall 0.926 against 0.862, S1 topo F1 0.61 against 0.55) and the heads were trained on those frames (their probabilities are more confident there), so a decode selected there is biased TIGHTER than the VAL optimum; the VAL result is the honest test, a VAL sweep (in-sample) is reported only as a diagnosis next to it, never chosen from. **The chosen cell is applied ONCE. No second cell is tried, the grid is not extended afterwards, and if the result on VAL is not better than the record (a gain below +0.01 on a seed) the verdict stands and E367 reads its ref at the record decode.** **What will probably happen, said before the sweep:** precision 0.86 against recall 0.40 - 0.47 means topo F1 rises with every loosening, so the rule will probably pick the loosest cell the guard allows (T = 0.05, K = 0.7, none gate off); the guard dup_frac <= 0.05 is measured on TRAIN (dup_frac 0.005 there), VAL duplicates can be higher: the VAL dup_frac is reported next to the result. The two seeds are two different trained heads, not noise draws.
**The verdict (by hand: the gate of E365 section 0, applied to this decode):** PASS-DECODE = G-topo and G-found hold on both seeds (the cost and simplicity gates are those of E365 S1, unchanged: same heads; the thresholds are on the output only, so G-simple holds); MISS-DECODE = G-topo fails on both seeds or G-found fails on both seeds (as E365); else INCONCLUSIVE (reported with the diagnosis; no action beyond the reading of E367's ref). **The JSON rule above is NOT the verdict:** `decide.py` is not used (a gain of +0.011 would read KEEP while the gate needs +0.06 to +0.09); the verdict is by hand: the gate of E365 section 0. The paired scene-bootstrap interval dec - ref is reported per seed (`paired_vs_record`: the record decode scored on the SAME re-dump; its difference to the stored E365 record numbers from the v1 dumps is reported, it is the float16 / float32 effect of the dump format). E367 does not take over this cell: it selects the decode PER HEAD by this rule on that head's own TRAIN sample; this card's VAL result of the S1 heads is E367's `ref`.
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
**The chosen cell (written 2026-11-17 02:47 system clock, BEFORE any VAL score of a variant; by the rule of section 0 on the two TRAIN-sample sweep tables `runs/e365_s1_s<S>/token_sweep_train_every3.json`, 291 frames each, runs `E366.sweep_s0 / s1`, scratch `temp/e366/choose.py`): link threshold 0.1, keep threshold 0.6, none gate OFF** (mean TRAIN-sample topo F1 0.7222 (0.7200 / 0.7245), mean dup_frac 0.0435 <= 0.05; precision 0.736 / 0.728, recall 0.705 / 0.721, lanes found 0.780 / 0.785, F1@1.0 0.900 / 0.901; the record cell on the same frames: 0.6228 (0.6191 / 0.6264), precision 0.865 / 0.867, recall 0.482 / 0.490). As said before the sweep, the rule took a loose cell, but not the loosest: (0.05, *, off) and (0.1, 0.7, off) break the duplicate guard (0.079 - 0.106 and 0.0501). This cell is applied ONCE to VAL now (runs `E366.val_s0 / val_s1`).

**Result (2026-11-17 03:10, by hand: the verdict of section 0, the gate of E365 section 0; the chosen cell applied ONCE to VAL, runs `E366.val_s0 / val_s1` on the re-dumps `E366.dump_s0 / s1` (new format); the decode of record scored on the SAME re-dumps gives exactly the stored E365 numbers 0.5486 / 0.5743, so the re-dump changes nothing at the record): PASS-DECODE on the quality conditions (G-topo and G-found hold on BOTH seeds); G-cost holds on the parameters and is OPEN on the milliseconds (to be measured on a quiet host); G-simple holds (the thresholds are on the output only).**
| VAL 873, delivered graph, every section 0 | topo F1 | topo P / R | F1@1.0 | lanes found | junction F1 | dup_frac | vert100_dp |
|---|---|---|---|---|---|---|---|
| floors of the gate, seed 0 / 1 | >= 0.6359 / 0.6349 | | >= 0.8182 / 0.8060 | >= 0.6066 / 0.5937 | | | |
| plain B (stored, own sections, close-node step), seed 0 / 1 | 0.6259 / 0.6249 | 0.670 / 0.587 (s0) | 0.8282 / 0.8160 | 0.6266 / 0.6137 | 0.272 (s0) | 0.235 (s0) | 19.29 (s0) |
| c-rule (the one-line rule on the same tokens), seed 0 / 1 | 0.6265 / 0.6281 | | 0.773 / 0.773 | 0.586 / 0.586 | 0.163 | 0.005 / 0.004 | |
| E365 S1, decode of record, seed 0 / 1 | 0.5486 / 0.5743 | 0.858 / 0.403 and 0.853 / 0.433 | 0.752 / 0.763 | 0.500 / 0.525 | 0.177 / 0.197 | 0.004 | 18.5 / 18.6 |
| **E366: the same heads, the chosen decode (0.1 / 0.6 / gate off), seed 0** | **0.6617** | 0.694 / 0.633 | **0.8359** | **0.6747** | 0.209 | 0.055 | 25.2 |
| **E366, seed 1** | **0.6767** | 0.687 / 0.666 | **0.8422** | **0.6905** | 0.204 | 0.054 | 25.7 |
**Paired scene bootstrap (topo F1, VAL, 1000 scene draws):** against the decode of record +0.113 [+0.090, +0.140] (seed 0) and +0.102 [+0.081, +0.129] (seed 1); against plain B +0.036 [+0.018, +0.052] and +0.052 [+0.032, +0.070]; against B' read with sections 0 and the close-node step +0.046 [+0.028, +0.062] and +0.057 [+0.043, +0.071]: every interval excludes 0. Against c-rule (no paired interval stored): +0.035 / +0.049. The gap share g1 = (E366 - c-rule) / (c-hi - c-rule) = 0.18 / 0.25 (S1: -0.40 / -0.28).
**The gate, per condition:** G-topo 0.6617 / 0.6767 against >= 0.6359 / 0.6349: holds (+0.026 / +0.042 over the floor, outside the band on both seeds); G-found: F1@1.0 0.836 / 0.842 against >= 0.818 / 0.806 and lanes found 0.675 / 0.691 against >= 0.607 / 0.594: holds; G-cost: the heads are those of E365 S1: whole-path parameters 2.32 M against 2.741 M: holds; the milliseconds (19.0 ms against 19.7 ms, measured on a loaded host, not usable) are OPEN: a quiet-host measurement decides (the decode change costs nothing measurable: the same forward, a threshold); G-simple: holds.
**Prediction against the numbers:** the chosen cell looser than the record (85 %): hit (T = 0.1); VAL topo F1 +0.03 over the record (window 0 to +0.08): +0.113 / +0.102 MISSED (far above the window); PASS-DECODE about 5 %: the unlikely branch happened (on quality). The miss of S1 was mostly the conservative decode, not the head: the head's probabilities rank the links well; the operating point of the record decode (p >= 0.3 and p >= p(none), keep at p_none < 0.5) threw away two thirds of the recall.
**What it costs, honestly:** precision falls from 0.86 to 0.69 (more false links), duplicates rise from 0.004 to 0.055 (still far below plain B's 0.235), the vertex density rises (25 against 19 vertices per 100 m), the junction F1 (0.209 / 0.204) stays BELOW plain B's 0.272 and the junction links remain the weak part; the TRAIN-to-VAL drop of the chosen cell is 0.06 / 0.05 (0.720 / 0.725 on the TRAIN sample).
**Caveats:** (1) the cell was selected on the TRAIN sample of the SAME heads (291 frames; the heads were trained on those frames) and applied once to VAL: legitimate by the card, but it is one of 24 cells and the selection used the heads' own training frames; (2) the paired intervals use plain B's stored per-frame records; (3) E365's parity check (cache vs live CPU) failed by its tolerance on 0.1 % of the nodes; (4) G-cost ms open; (5) **a champion is NOT yet declared:** E367's rule asks for a confirmation of the recipe on two fresh head seeds (the S1 recipe retrained with seeds 2 and 3, read at this decode) and the quiet-host timing; both are the next steps (the confirmation needs a `--cache-seed` option so that the head seed differs from the B' cache seed). **What this changes for E367 / E369 / E370:** every arm is read at its own TRAIN-selected decode (the rule of this card), as their cards already say; this result becomes E367's `ref`.
