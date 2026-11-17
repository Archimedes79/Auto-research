# E371 - confirmation of the E366 recipe (the E365 S1 head + the decode selected on the TRAIN sample) on two fresh head seeds

Card of 2026-11-17 (version 1, 03:15 system clock, before any run; written by the Spurmitte chat that Mirko addresses as "Fable"). **Status: draft; the `--head-seed` option is being added by a coder; nothing run.** E366 (experiments/E366_e365-decode-selected-on-train, section 3) passed the quality half of the okayish gate on both seeds (VAL topo F1 0.6617 / 0.6767). Its decode was selected on the TRAIN sample of the SAME two heads, and it is one of 24 cells; E367's champion rule (written before) asks for a confirmation of a passing recipe on two fresh head seeds before "a champion exists" is reported. **The question:** does the FROZEN recipe (the E365 S1 training recipe, unchanged, and the E366 decode, unchanged) pass the gate again with two heads that were not used to choose anything?

## 0. The card (what the tools run)
```json card
{
 "id": "E371",
 "card": "",
 "title": "The E366 recipe frozen (E365 S1 training recipe: d 64, 3 layers, 12000 steps, none weight 1.0, --ident e292_a, K 512; decode: link threshold 0.1, keep threshold 0.6, none gate off) retrained with head seeds 2 and 3 on the B' caches of seeds 0 and 1, read once on VAL",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E366 (head seeds 0 / 1 on the caches 0 / 1): VAL topo F1 0.6617 / 0.6767; the verdict is by hand (decide.py not used)"},
  "cf": {"recipe": "`jtoken_train.py --name e371_cf_s<S> --seed S --head-seed <S + 2> --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4` (S = 0, 1: cache seed S, head seed S + 2), read with `jtoken_eval.py score --link-thr 0.1 --keep-thr 0.6 --no-none-gate` after a new-format dump; nothing re-selected", "run_ids": ["E371.cf_s0", "E371.cf_s1"]}
 },
 "control": "ref",
 "variant": "cf",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 1.0,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200); a run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported); `--timeout` 130 minutes (12000 x 0.5 s x 1.3)",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run):** CONFIRMED = the okayish gate of E365 section 0 holds on BOTH fresh heads at the frozen decode: G-topo >= 0.6359 (the head on cache 0) and >= 0.6349 (cache 1); G-found: F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937; G-cost and G-simple as E366 (the same architecture; the milliseconds from the quiet-host timing of the E365 head). NOT CONFIRMED = G-topo or G-found fails on either fresh head (a stricter reading than a MISS: a confirmation must hold on both). **"A champion exists" (D113) = E366 PASS-DECODE (quality) + E371 CONFIRMED + G-cost confirmed on a quiet host.** Reported: the four heads (seeds 0 - 3) side by side, the paired interval against plain B per head, the TRAIN-sample read-out of the fresh heads at the frozen decode (the TRAIN-to-VAL drop), dup_frac, junction F1, precision / recall.
**Prediction (guesses, with windows):** VAL topo F1 0.665 (0.635 - 0.69) per fresh head; CONFIRMED about 75 % (the margin of E366 over the floors is +0.026 / +0.042; the seed spread of the S1 heads was 0.026 at the record decode).

## 1. Card text
- **Statement:** the E366 result is a property of the recipe, not of the two heads it was selected on. **Not claimed:** that the decode is optimal for other heads; anything about S2 / S3 / S4 / TEST. **Cost:** two trainings of about 40 minutes side by side, two dumps, two scores (CPU).
- **Referee:** a short fresh ar-critic (cards) before the runs.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| option | `--head-seed` (coder; default bit for bit unchanged) | coder | test green, committed |
| train | cf_s0 (cache 0, head seed 2), cf_s1 (cache 1, head seed 3), the supervisor of E367 | `run.py --id E371.cf_s<S> --gpu --require-card` | done flags |
| read-out | dump (TRAIN every 3, VAL), score at the frozen decode (VAL; TRAIN sample as a diagnosis) | `run.py --id E371.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, the champion line to the lead if CONFIRMED |

## 3. Summary
(empty until the runs exist)
