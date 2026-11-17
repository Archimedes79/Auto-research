# E373 - confirmation of the b_bal5 challenger (E367: the E365 head with none weight 0.5 + its TRAIN-selected decode) on two fresh head seeds

Card of 2026-11-17 (version 1, 04:35 system clock, before any run; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 04:30: "b_bal5 ties on topo F1 at clearly fewer duplicates with the same head and one different loss weight, so D106 ('similar performance: the simpler / cheaper one') makes it worth its own confirmation"). **Status: draft; nothing run.** The pattern is E371's (experiments/E371_e366-confirmation-fresh-head-seeds): the recipe frozen, two fresh head seeds on the same B' caches, read once on VAL, nothing re-selected.

**The challenger:** E367 arm b_bal5 (experiments/E367_e365-capacity-balance-schedule-arms, section 3 decode choices): the E365 S1 head with `--none-weight 0.5`, 12000 steps; its decode chosen on each head's own TRAIN sample by the E366 rule = link threshold 0.2, keep threshold 0.7, none gate off on BOTH heads; VAL: topo F1 0.6691 / 0.6696, F1@1.0 0.834 / 0.831, lanes found 0.686 / 0.685, dup_frac 0.032 / 0.035. **The champion it challenges:** the E365 S1 recipe at the E366 decode (0.1 / 0.6 / off), confirmed by E371: VAL topo F1 0.6617 / 0.6767 (E366) and 0.6618 / 0.6733 (E371), dup_frac 0.0548 / 0.0536 (E366) and 0.0588 / 0.0629 (E371).

## 0. The card (what the tools run)
```json card
{
 "id": "E373",
 "card": "",
 "title": "The b_bal5 recipe frozen (the E365 S1 head, none weight 0.5, 12000 steps; decode link 0.2 / keep 0.7 / none gate off) retrained with head seeds 2 and 3 on the B' caches 0 and 1, read once on VAL; against the confirmed S1 champion",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the confirmed S1 champion (E366 / E371, four heads at the decode 0.1 / 0.6 / off); per cache the E366 head: topo F1 0.6617 / 0.6767, dup_frac 0.0548 / 0.0536; the verdict is by hand (decide.py not used)"},
  "cb": {"recipe": "`jtoken_train.py --name e373_cb_s<S> --seed S --head-seed <S + 2> --ident e292_a --none-weight 0.5 --steps 12000 --bs 8 --lr 1e-3 --workers 4` (cache seed S, head seed S + 2), new-format dumps, `jtoken_eval.py score --link-thr 0.2 --keep-thr 0.7 --no-none-gate` once on VAL; nothing re-selected", "run_ids": ["E373.cb_s0", "E373.cb_s1", "E373.read_s0", "E373.read_s1"]}
 },
 "control": "ref",
 "variant": "cb",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 1.5,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200); a run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported, not bit-reproducible); `--timeout` 130 minutes; a crashed read-out is re-run unchanged, never with another cell",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run; the JSON rule is NOT the verdict):** CONFIRMED = the okayish gate of E365 section 0 holds on BOTH fresh heads at the frozen decode (G-topo >= 0.6359 (cache 0) / 0.6349 (cache 1); F1@1.0 >= 0.8182 / 0.8060; lanes found >= 0.6066 / 0.5937; G-cost and G-simple as the S1 champion: the same architecture) AND the dup_frac of each fresh head is not above the S1 champion's on the same cache (0.0548 / 0.0536, the E366 heads); NOT CONFIRMED otherwise (no third head, no pooled rule; 'at the floor' when below a floor by less than the band). **If CONFIRMED, the b_bal5 recipe replaces the S1 recipe as THE champion** (the lane lead's rule, D106: same quality, fewer duplicates, nothing added: the same head, one loss weight changed); if NOT CONFIRMED, the S1 recipe stays the champion. Reported: the paired cb - champion interval per cache (topo F1), the paired interval against plain B per cache, the topo F1 of the four challenger heads against the four champion heads (a tie inside the band is expected), junction F1, precision / recall, vertices per 100 m, the TRAIN-to-VAL drop; the forking path said: b_bal5 was selected among E367's arms on VAL (in-sample), which is why it needs this confirmation.
**Prediction (guesses, with windows):** VAL topo F1 cache 0 about 0.665 (0.64 - 0.69), cache 1 about 0.668 (0.645 - 0.69); dup_frac about 0.034 (0.025 - 0.045) per head; CONFIRMED about 70 % (the dup condition is easy: 0.032 / 0.035 against 0.055 / 0.054; the quality conditions have the margins of E367: +0.033 / +0.035 topo, +0.016 / +0.025 F1@1.0).

## 1. Card text
- **Statement:** the b_bal5 result is a property of its recipe on the B' caches 0 / 1. **Not claimed:** robustness to a new B' seed; anything about S2 / S3 / S4 / TEST. **Cost:** two trainings of about 40 minutes side by side, two dumps, two scores (CPU).
- **Referee:** a short fresh ar-critic (cards) before the runs.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| train | cb_s0 (cache 0, head seed 2), cb_s1 (cache 1, head seed 3), the supervisor pattern, after the quiet-window timing | `run.py --id E373.cb_s<S> --gpu --require-card` | done flags |
| read-out | dumps (VAL, TRAIN every 3), score at the frozen decode (VAL), the record decode (diagnosis), the TRAIN sample (diagnosis) | `run.py --id E373.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead (and the champion line if it replaces) |

## 3. Summary
(empty until the runs exist)
