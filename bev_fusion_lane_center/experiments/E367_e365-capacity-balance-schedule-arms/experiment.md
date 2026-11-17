# E367 - S1b: arms on the E365 token head after the miss of S1: capacity (wide), loss balance (two none weights), and a longer schedule as a one-seed control

Card of 2026-11-17, **version 2 at 01:40 (system clock) after the first critic pass** (fresh ar-critic, cards: REVISE, 10 findings, taken in: (1) the token-drop arm is REMOVED: the loader's `token_drop` does not re-contract the targets, so a dropped successor turns into a none target, the opposite of what the TRAIN-val gap needs; it comes back as its own arm after the contraction exists; (2) a_wide is now capacity ONLY (same 12000 steps as S1); (3) the decode is selected PER HEAD on that head's TRAIN sample, `ref` is the E366 result; (4) the order of writing is made checkable; (5) the cost of a_wide is timed before its training and expected QUALITY-ONLY; (6) the none weight is a guess and two values are run; (7) the champion needs a confirmation on two fresh seeds; (8) the long-class share is a pre-registered secondary number, the fallback is written; (9) JSON note; (10) timeouts from the speed measured over 1500 steps). Written by the Spurmitte chat that Mirko addresses as "Fable"; the arms follow the lane lead's 01:25 reading of the E365 miss ("too conservative: the none slot dominates, it does not even fit TRAIN, the ceiling is far away: calibration + capacity, not the design") and the diagnosis of E365 section 3. **Status: v2, nothing run.** Design, cache (B' = e292_a, K = 512), loss, gate and controls are those of E365 (experiments/E365_e2e-token-champion-s1, sections 0 - 1) and are NOT repeated; only what differs is written here.

**The question:** which cheap change lifts the E365 head over the okayish gate (G-topo >= 0.6359 / 0.6349; G-found: F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937; both seeds; plus G-cost and G-simple)? Arms, two seeds (0, 1) each unless stated:
- **a_wide (capacity only):** `--d 96 --layers 4 --ffn 192`, 12000 steps (everything else as S1). **G-cost caveat, said now:** the head costs about 1.3 - 1.6 times the S1 head; the whole path of S1 measured 19.0 ms against the limit 19.7 ms on a loaded host, so a_wide is EXPECTED to be QUALITY-ONLY unless the timing of the random-init d 96 / 4-layer head on a quiet host (a 10-minute job before its training) says otherwise; its quality result is still the information (does capacity help at all).
- **b_bal5 (loss balance):** the S1 head with `--none-weight 0.5`, 12000 steps; **b_bal25:** `--none-weight 0.25`. The value 0.5 is a GUESS (about a logit shift of ln 2 on the none class; the ratio R / P = 0.47 / 0.86 of the TRAIN read-out motivates "less than 1", it does not derive 0.5), so two values are run, none tuned on VAL. **What the none weight changes, said before the runs (code critic of the options, 2026-11-17 01:30):** W is an ABSOLUTE down-weight of the empty-G source terms (they are still divided by the number of sources, not by the sum of the weights), and it also weights the none class of the type cross-entropy (normalised by the sum of the weights); so b_bal moves TWO things at once: the none slot of the wiring and, through the type CE, the key weight 1 - p_none of the soft pruning; the reading of the b arms names both. **Expected tie to E366:** the none weight moves the operating point, the same lever as a looser decode; its extra information over E366 is the feature-level effect only.
- **c_long (schedule only, a ONE-SEED control, seed 0):** the S1 head, 24000 steps: the E365 diagnosis (loss plateau 0.42) says more steps alone do not help; one seed suffices to separate schedule from the other arms; no verdict token for this arm.
**The decode of every arm** = the cell chosen by the rule of E366 on THAT head's own TRAIN sample (the same grid, the same 291-frame sample, the same dup_frac guard), written into the arm's sweep card before its VAL read-out; the decode of record and the argmax decode are reported as diagnoses. **`ref`** = E366's VAL result of the S1 heads (decode selected on their TRAIN sample), so that the arm effect is not confounded with a decode shift; the decode of record of S1 is reported next to it.

## 0. The card (what the tools run)
```json card
{
 "id": "E367",
 "card": "",
 "title": "E365 token head arms: a_wide (d 96, 4 layers), b_bal5 / b_bal25 (none weight 0.5 / 0.25), c_long (24000 steps, one seed), decode selected per head on its TRAIN sample, against the S1 heads of E365 read at the decode of E366 and the gate of E365",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.04, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E365 S1 heads at the decode selected by E366 (E366.val_s0 / val_s1; the decode of record: VAL topo F1 0.5486 / 0.5743)"},
  "a_wide": {"recipe": "`jtoken_train.py --name e367_a_s<S> --seed S --ident e292_a --d 96 --layers 4 --ffn 192 --steps 12000 --bs 8 --lr 1e-3 --workers 4`", "run_ids": ["E367.a_s0", "E367.a_s1"]},
  "b_bal5": {"recipe": "`jtoken_train.py --name e367_b5_s<S> --seed S --ident e292_a --none-weight 0.5 --steps 12000 --bs 8 --lr 1e-3 --workers 4`", "run_ids": ["E367.b5_s0", "E367.b5_s1"]},
  "b_bal25": {"recipe": "`jtoken_train.py --name e367_b25_s<S> --seed S --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4`", "run_ids": ["E367.b25_s0", "E367.b25_s1"]},
  "c_long": {"recipe": "`jtoken_train.py --name e367_c_s0 --seed 0 --ident e292_a --steps 24000 --bs 8 --lr 1e-3 --workers 4` (one seed, control)", "run_ids": ["E367.c_s0"]}
 },
 "control": "ref",
 "variant": "a_wide",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 8.0,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 is not below 0.8 x the mean over steps 101 - 200); a run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported, run marked not bit-reproducible); `--timeout` = steps x 0.65 s (0.5 s x 1.3) at the start, then set from the speed measured over the first 1500 steps (a_wide 130 min, b 130 min, c 260 min); two slots at a time",
 "evaluate": {"split": "val"}
}
```
**Recipe note:** `--ffn` is passed only for a_wide (192); the other arms leave it at its default (2 x d); a resume with another `--ffn` spelling is refused by the recipe check (conservative). Token-drop together with a none weight (not used here) would turn a source whose only successor was dropped into an empty-G source that takes W: recorded now, before that combination is ever run. **Start:** b_bal5 pair started 2026-11-17 01:33 (system clock) from the trainer the code critic accepted. **Note on the JSON:** `decide.py` judges only the variant named in it (a_wide); **every verdict of this card is by hand: the gate below**, for all arms; `decide.py` is not used for it.
**The verdict (by hand: the gate of E365 section 0, written before any run, per arm with two seeds):** an arm PASSES when G-topo, G-found, G-cost and G-simple hold on BOTH seeds at its own selected decode (floors per seed as in E365); MISS = G-topo fails on both seeds or G-found fails on both seeds; QUALITY-ONLY = quality holds on both seeds, cost or simplicity fails; else INCONCLUSIVE. **Pre-registered secondary numbers per arm:** the delivered share of the LONG (junction) class (E365: 51 %), the share of the assigned tokens kept, TRAIN-sample topo F1 (S1: 0.610 / 0.625), the paired scene-bootstrap interval to plain B per seed. **Forking paths, said now:** 4 arms x a decode selected per head; the chosen arm is selected among the arms on VAL: its numbers are labelled "selected among 4 arms, in-sample". **A "champion exists" claim needs a confirmation:** the winning recipe is retrained with two other head seeds (a different initialisation and data order on the same B' caches; a `--cache-seed` option is added by a coder before that) and must pass the gate again on VAL; only then the report says "champion". **The champion = the PASSING arm with the highest mean topo F1 over the two seeds.** Rule-step systems never block; E324 s (0.6912 / 0.6948) is reported next to it.
**Order of writing (checkable):** the sweep of a head (CPU, TRAIN sample) is written to its card before that head's VAL score exists (the VAL jobs are launched `--after` the sweep job of the same head); the E366 commit that contains section 3 is named in this card before the first E367 VAL read-out; the cost of a_wide is timed (random init, quiet host) before its training starts.
**Prediction (guesses, with windows; mean topo F1 over both seeds at the selected decode, VAL):** a_wide 0.59 (0.54 - 0.65), b_bal5 0.60 (0.54 - 0.66), b_bal25 0.60 (0.53 - 0.66), c_long (seed 0) 0.56 (0.52 - 0.60); a_wide TRAIN-sample topo F1 above 0.65; the long-class share of b_bal5 / b_bal25 0.60 (0.45 - 0.75); odds: at least one arm PASSES the quality conditions on both seeds about 20 %, but a_wide is expected QUALITY-ONLY, so the chance that some arm is a full PASS is about 12 %; "the schedule alone does not help" (c_long within +/- 0.02 of S1) about 70 %.
**Fallback, written before the runs:** if no arm lifts the long-class share above 0.60, the next card (the next free id; E368 is taken by the S2 card of the lane chat) is a long-class arm (a separate none slot and a weight for the long class, plus token-drop WITH re-contraction of the targets as its own arm); if an arm lifts topo F1 but not over the floors, the next card combines the two best levers; if everything is flat the conclusion is that the design needs the B stage to move (S4, the joint fine-tune), not a bigger head.

## 1. Card text
- **Mechanism:** a_wide tests capacity, b_bal the calibration of the objective, c_long the schedule; together they say which lever is real. **Inputs:** all existing after the coder's options (`--d --layers --ffn --none-weight` are in `jtoken_train.py`; the sweep / selected-decode commands of `jtoken_eval.py` are being added). **Cost:** trainings: a 2 x 12000 steps at about 0.3 s (60 min for the pair), b5 and b25 2 x 12000 at 0.2 s (40 min per pair), c 1 x 24000 (80 min, alongside another run): about 3 hours of two slots; sweeps 6 heads x 13 minutes (CPU), read-outs 6 x 15 minutes.
- **Referee:** fresh `ar-critic` (cards) 2026-11-17: REVISE, 10 findings, taken in as listed in the header; the options are covered by the code critic of the coder's step.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| options | `--d --layers --ffn --none-weight` (done in the trainer), the sweep and the selected-decode flags of `jtoken_eval.py`, `--cache-seed` for the confirmation, tests | coder | tests green, committed |
| timing | the random-init d 96 / 4-layer head: whole-path ms on a quiet host | Fable | json before a_wide starts |
| trainings | pair a, pair b5, pair b25, c_s0, one supervisor family (`temp/e367/sup.sh`), two slots at a time | `run.py --id E367.<arm>_s<S> --gpu --require-card` | done flags |
| sweeps | per head, TRAIN sample (every third of the 873 evenly spaced frames: 291 frames, fixed now), CPU niced | `run.py --id E367.sweep_<arm>_s<S>` | cards |
| read-out | dump, score at the selected cell (+ record + argmax), TRAIN sample, degraded, per arm and seed, `--after` the sweep | `run.py --id E367.read_<arm>_s<S>` | cards |
| decide | the verdict above per arm; the champion rule; the confirmation | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
**Decode choices per head (each written BEFORE that head's VAL score; the rule of E366 on the head's own 291-frame TRAIN sample, `temp/e366/choose.py <run>`):**
- b_bal5, both heads (runs E367.sweep_b5_s0 / s1, 03:35): link 0.2, keep 0.7, none gate off (TRAIN topo F1 0.7185 / 0.7198, dup_frac 0.0275 / 0.0288; the record cell 0.6642 / 0.6727)
- b_bal25 (runs E367.sweep_b25_s0 / s1, 03:45): seed 0 link 0.2, keep 0.7, none gate off (TRAIN topo F1 0.7197, dup 0.046); seed 1 link 0.2, keep 0.6, none gate off (TRAIN 0.7177, dup 0.0456); the record cell 0.6927 / 0.6994

(results follow)
