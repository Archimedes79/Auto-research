# E373 - confirmation of the E367 champion arm by its WRITTEN rule (b_bal25: the E365 head with none weight 0.25) on two fresh head seeds, read at the record decode and at the recipe's TRAIN-selected cell

Card of 2026-11-17, **version 2 at 04:55 (system clock), before any run** (the folder name keeps the v1 name; v1 proposed b_bal5 on the lane lead's tie-break by duplicates; a short fresh ar-critic (cards, REVISE, 5 findings) showed that this departs from the rules written before the runs: E367's champion rule 'the PASSING arm with the highest mean topo F1' and E371's 'an E367 arm that passes its own confirmation with a higher mean'; the lane lead dropped the tie-break at 04:50: "E373 confirms b_bal25 by the rule written BEFORE the runs"). Written by the Spurmitte chat that Mirko addresses as "Fable". **Status: v2, nothing run.** The pattern is E371's (experiments/E371_e366-confirmation-fresh-head-seeds): the recipe frozen, two fresh head seeds on the same B' caches, read once on VAL.

**Why b_bal25:** E367 section 3 (committed before this card): both b arms pass the quality gate on both seeds at their TRAIN-selected decodes; mean VAL topo F1 b_bal25 0.6730, b_bal5 0.6694, the S1 champion at the E366 decode 0.6692 (a three-way tie INSIDE the band 0.010; in-sample: selected among the arms read so far on VAL; a_wide and c_long have not run). By the written rule the arm to confirm is b_bal25. **Forking path, said:** VAL 873 has been read by E365 - E373; the confirmation removes the selection of the head instance, not the reuse of VAL.

**The recipe (frozen now, before any fresh head exists):** the E365 S1 head with `--none-weight 0.25`, 12000 steps, bs 8, lr 1e-3; head seeds 2 (cache 0) and 3 (cache 1); the decode: TWO decodes are read on the same trained heads, chosen by the rule below: (D-rec) the decode of record of E365 (a link where p >= 0.3 and p >= p(none), a node where p_none < 0.5), (D-cell) the cell chosen by the E366 rule over BOTH b_bal25 TRAIN tables (`temp/e366/choose.py e367_b25_s0 e367_b25_s1`, 04:45): link threshold 0.2, keep threshold 0.6, none gate off (mean TRAIN topo F1 0.7183, mean dup 0.0452; one cell for the recipe, chosen on TRAIN).

## 0. The card (what the tools run)
```json card
{
 "id": "E373",
 "card": "",
 "title": "The b_bal25 recipe frozen (the E365 S1 head, none weight 0.25, 12000 steps) retrained with head seeds 2 and 3 on the B' caches 0 and 1, read once on VAL at the record decode and at the recipe cell 0.2 / 0.6 / gate off; against the confirmed S1 champion",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the confirmed S1 champion (E366 + E371, four heads at the decode 0.1 / 0.6 / off); per cache the mean of its two heads: topo F1 0.6618 / 0.6750, F1@1.0 0.8381 / 0.8410, lanes found 0.6798 / 0.6876; the verdict is by hand (decide.py not used)"},
  "cb": {"recipe": "`jtoken_train.py --name e373_cb_s<S> --seed S --head-seed <S + 2> --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4` (cache seed S, head seed S + 2), new-format dumps, `jtoken_eval.py score` at the record decode and at `--link-thr 0.2 --keep-thr 0.6 --no-none-gate`, once each on VAL; nothing re-selected", "run_ids": ["E373.cb_s0", "E373.cb_s1", "E373.read_s0", "E373.read_s1"]}
 },
 "control": "ref",
 "variant": "cb",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 1.5,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200); a run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported, not bit-reproducible); `--timeout` 130 minutes; a crashed read-out is re-run unchanged, never with another cell; the training must end before about 07:30 (the lane lead's night limit)",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run; the JSON rule is NOT the verdict):**
1. **The decode of the recipe (the lane lead's rule of 04:50, the D106 reading 'same cost, simpler = no tuned output thresholds'):** the recipe's decode is D-rec if the okayish gate of E365 section 0 holds on BOTH fresh heads at D-rec (G-topo >= 0.6359 (cache 0) / 0.6349 (cache 1); F1@1.0 >= 0.8182 / 0.8060; lanes found >= 0.6066 / 0.5937); otherwise D-cell if the gate holds on both fresh heads at D-cell; otherwise **NOT CONFIRMED**. A fresh head below a floor by less than the band 0.01 counts as failing ('at the floor'); no third head; no pooled rule.
2. **CONFIRMED** = the gate holds on both fresh heads at the chosen decode (G-cost and G-simple as the S1 champion: the same architecture; D-rec has no tuned threshold, D-cell has three).
3. **Replacement of the S1 champion** (E367 / E371's rule, with the critic's quality-tie condition): b_bal25 replaces the S1 recipe as THE champion when CONFIRMED AND, at the chosen decode, on each cache: topo F1 >= the champion's per-cache mean - 0.01 (>= 0.6518 / 0.6650), F1@1.0 >= 0.8281 / 0.8310, lanes found >= 0.6698 / 0.6776; otherwise the S1 recipe stays the champion and b_bal25 is reported as a confirmed alternative. **Said now:** at D-rec the E367 b_bal25 heads had F1@1.0 0.8193 / 0.8191, below the tie condition 0.8281 / 0.8310: if D-rec is chosen by rule 1, a replacement is unlikely even when the gate holds.
**Reported:** the paired scene-bootstrap interval cb - S1 champion per cache (topo F1) at both decodes; **the seed twin:** cb - E371 cf per cache (the same cache seed and head seed: the none-weight effect at equal initialisation and batch order; the cb train.log must print the same first-batch frame order as cf's, checked); the paired interval against plain B; junction F1, dup_frac, vertices per 100 m, precision / recall against the champion; the TRAIN-to-VAL drop; the degraded read-out at the chosen decode.
**Prediction (guesses, with windows):** at D-rec: topo F1 about 0.650 (0.63 - 0.67) per head, F1@1.0 about 0.819 (0.80 - 0.83): the gate at D-rec holds on both heads about 45 % (F1@1.0 on cache 0 was +0.001 over the floor); at D-cell: topo F1 about 0.670 (0.65 - 0.69), F1@1.0 0.846 (0.83 - 0.86): the gate holds about 85 %; CONFIRMED (at either decode) about 85 %; replacement of the S1 champion about 35 % (needs D-cell chosen AND the tie: likely only if D-rec fails).

## 1. Card text
- **Statement:** the b_bal25 result is a property of its recipe on the B' caches 0 / 1. **Not claimed:** robustness to a new B' seed; that the none weight 0.25 is optimal; anything about S2 / S3 / S4 / TEST. **Cost:** two trainings of about 40 minutes side by side, two dumps, four scores (CPU).
- **Referee:** short fresh ar-critic (cards) on v1: REVISE, 5 findings, taken in (the written rule names b_bal25; a quality-tie condition for the replacement; the forking path; the seed twin; the dup condition dropped as the criterion); v2 changed the arm, the decode rule came from the lane lead before any fresh head.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| train | cb_s0 (cache 0, head seed 2), cb_s1 (cache 1, head seed 3), the supervisor pattern, after the lane chat's S2 pair | `run.py --id E373.cb_s<S> --gpu --require-card` | done flags |
| read-out | dumps (VAL, TRAIN every 3), scores at D-rec and D-cell on VAL, the degraded read-out, the TRAIN sample (diagnosis) | `run.py --id E373.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead (the champion line if it replaces) |

## 3. Summary
**Result (2026-11-17 06:12, by hand: the verdict of section 0 in its written order; runs `E373.cb_s0` (cache 0, head seed 2) and `E373.cb_s1` (cache 1, head seed 3), 12000 steps, none weight 0.25, no crash, no resume (04:52 - 05:39); read-outs `E373.read_s0 / read_s1` (new-format dumps, each decode scored once on VAL); VAL 873, TEST untouched): (1) the record decode FAILS the gate on cache 0 (F1@1.0 0.8162 < 0.8182, 'at the floor'), so the recipe's decode is the cell 0.2 / 0.6 / gate off; (2) at the cell the gate holds on BOTH fresh heads: CONFIRMED; (3) the quality tie with the S1 champion holds on both caches: b_bal25 REPLACES the S1 recipe as THE champion.**
| VAL 873, every section 0 | topo F1 | topo P / R | F1@1.0 | lanes found | junction F1 | dup_frac | vert100_dp |
|---|---|---|---|---|---|---|---|
| floors (cache 0 / 1) | >= 0.6359 / 0.6349 | | >= 0.8182 / 0.8060 | >= 0.6066 / 0.5937 | | | |
| tie conditions (the S1 champion's per-cache mean - 0.01) | >= 0.6518 / 0.6650 | | >= 0.8281 / 0.8310 | >= 0.6698 / 0.6776 | | | |
| E373 fresh heads, D-rec (0.3 / 0.5 / gate on), cache 0 / 1 | 0.6424 / 0.6581 | 0.776 / 0.548 and 0.780 / 0.569 | **0.8162** / 0.8203 | 0.641 / 0.648 | 0.248 / 0.241 | 0.017 / 0.017 | 22.4 / 22.9 |
| **E373 fresh heads, D-cell (0.2 / 0.6 / gate off), cache 0 / 1** | **0.6698 / 0.6769** | 0.680 / 0.660 and 0.681 / 0.673 | **0.8456 / 0.8461** | **0.7274 / 0.7267** | 0.203 / 0.197 | 0.058 / 0.061 | 27.4 / 27.8 |
| E373 D-cell, degraded (17 % of the tokens removed) | 0.5764 / 0.5973 | | 0.811 / 0.816 | 0.622 / 0.633 | | | |
| E373 D-cell, TRAIN sample (diagnosis) | 0.7180 / 0.7227 | | 0.903 / 0.903 | 0.810 / 0.808 | 0.269 / 0.271 | 0.048 / 0.045 | |
| E367 b_bal25 (head seeds 0 / 1), its own cells | 0.6731 / 0.6728 | | 0.849 / 0.845 | 0.741 / 0.725 | 0.194 / 0.197 | 0.056 / 0.061 | 29.1 / 27.9 |
| the S1 champion (E366 + E371, four heads, 0.1 / 0.6 / off), per-cache mean | 0.6618 / 0.6750 | | 0.8381 / 0.8410 | 0.6798 / 0.6876 | 0.207 / 0.209 | 0.057 / 0.058 | 25.2 / 25.5 |
**Per condition:** decode rule: D-rec holds on cache 1 but not on cache 0 (F1@1.0 -0.0020 under the floor; the card counts 'at the floor' as failing) -> D-cell. Gate at D-cell: G-topo +0.034 / +0.042 over the floors, F1@1.0 +0.027 / +0.040, lanes found +0.121 / +0.133: holds on both; G-cost: the same architecture as the S1 champion (2.323 M parameters, the whole path 16.6 - 16.8 ms measured in the quiet window): holds; G-simple: holds (the same three named output thresholds as the S1 champion, different values). **CONFIRMED.** Tie with the S1 champion at D-cell: topo F1 +0.008 / +0.002 over the champion's per-cache mean, F1@1.0 +0.008 / +0.005, lanes found +0.048 / +0.039: every condition holds on both caches: **REPLACES.** Paired against plain B: +0.044 [+0.028, +0.059] (cache 0), +0.052 [+0.034, +0.068] (cache 1).
**What is better and what is not, said plainly:** against the S1 champion the new champion has the same topo F1 (inside the band), a slightly higher F1@1.0 and clearly more lanes found (+0.04 to +0.05), at slightly more vertices (27.4 - 27.8 against 25.2 - 25.5 per 100 m) and the same duplicates (0.058 / 0.061 against 0.057 / 0.058) and junction F1 (0.20 against 0.21, both BELOW plain B's 0.27); precision 0.68 (S1 champion 0.69). The change against the S1 recipe is ONE loss weight (the none target and the none class at 0.25 instead of 1.0) and other output thresholds; nothing is added. At the record decode (no tuned thresholds) the b_bal25 heads are the cleanest operating point of the night (junction F1 0.24 - 0.25, dup 0.017, 22.4 - 22.9 vertices) but miss F1@1.0 on cache 0 by 0.002: the simpler decode does not pass by the written rule.
**Prediction against the numbers:** D-rec topo F1 about 0.650 (0.63 - 0.67): 0.642 / 0.658 hit; D-rec gate on both heads about 45 %: failed (F1@1.0 on cache 0, as feared); D-cell topo F1 about 0.670 (0.65 - 0.69): 0.670 / 0.677 hit; CONFIRMED about 85 %: hit; replacement about 35 %: happened.
**Not done yet (honest):** the seed-twin interval cb - E371 cf per cache and the first-batch frame-order check against cf's train.log (the card lists both; they need a paired score of two runs, a CPU job after this result); the E367 b_bal25 seed-0 head at the recipe cell 0.2 / 0.6 (it was read at its own cell 0.2 / 0.7). **Forking path:** b_bal25 was named by E367's written rule among the arms read on VAL (a three-way tie inside the band); VAL 873 has been read by E365 - E373; the confirmation removes the selection of the head instance, not the reuse of VAL. **Labels:** 'staged, B' frozen' (until S4) and the S1 stage of D113's design.
