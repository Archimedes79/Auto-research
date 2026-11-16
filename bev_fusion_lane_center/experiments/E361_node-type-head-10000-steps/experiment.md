# E361 - E359 with 3.3 times the training steps: was the failure of the node-type head and the per-source softmax undertraining? (the same card, ONE parameter changed: 10000 steps instead of 3000)

Card of 2026-11-16 10:25, version 2 of 10:26 after a short critic pass (REVISE: the time plan used a step speed that E359's logs do not show; the change is not exactly one parameter (warm-up, rollback guard, resume); the loss values quoted were mixed and single-batch; the prediction value and the PASS odds were inconsistent). Written by the Spurmitte chat that Mirko addresses as "Fable", on the lane lead's GO (relayed at 10:31 on his clock; D111 covers the node-type-head question; E361 is the same card with one parameter, so it stays inside D111; a PASS proposes nothing automatically: T1 of E355 needs Mirko's word, D105 / D111). **The lane lead named 15000 steps; I use 10000** because the measured speed does not allow 15000 in the window (below): the deviation is mine and is reported to him. **Everything is exactly as in E359** (`experiments/E359_node-type-head-per-source-softmax/experiment.md`, section 0 as committed in 2bc7eb4e and unchanged since: the oracle nodes, features, encoder, type head, the exact likelihood, the decode of record, the controls of E357, the verdict order, the attribution rule, the diagnostics, the code `jtok2*` unchanged and already reviewed) **except the schedule: `--steps 10000`.** **It is not literally one parameter (the critic's reading of `training/trainer.py`):** the learning-rate schedule is a function of (step, total) and is stretched (a warm-up of 5 % = 500 steps instead of 150, a one-cycle cosine down to 2 % of the peak), so the time at the high rate grows with it; the roll-back guard (from step 400: the mean of 100 steps above 1.2 x the best rewinds to the last checkpoint and scales the rate by 0.7) has more chances to fire (E359: none in either log); a `resume` (after a stop or a CUDA "unknown error") re-shuffles the loader, so a run that was interrupted is not reproducible. **Any ROLLBACK, non-finite skip or resume in a run is reported with its result and the run is named.** 3000 steps are 2.85 epochs of the 8 414 TRAIN frames (1 051 steps per epoch), 10000 steps are 9.5 epochs.
E359's result is the reference row: FAIL on both seeds (g1 0.267 / 0.232; topo F1 0.7721 / 0.7626; junction-link F1 0.531 / 0.502; type long_out F1 0.739 / 0.714, recall 0.656 / 0.640; softmax oracle-source top-1 0.779 / 0.765). **Evidence about the loss, corrected:** the logged mean loss falls 0.410 / 0.341 / 0.299 / 0.223 (seed 0) and 0.435 / 0.358 / 0.312 / 0.244 (seed 1) at steps 500 / 1000 / 1400 / 3000, and the last part of that fall is largely the annealing of the rate to 2e-5: "still falling" is weak evidence for undertraining, and this card is the test. **The one question:** does a 3.3 times longer schedule close the gap? **The two numbers to watch** (the lane lead's request): the type head's long_out recall (E359: 0.656 / 0.640) and the exit choice, the oracle-source long top-1 accuracy (0.779 / 0.765).
**Time, from E359's own logs (the critic's measurement):** 3000 steps took about 23 minutes for seed 1 (09:26 - 09:50, about 0.46 s per step overall; clock lines from 0.14 s at step 500 to 0.8 s at step 1000 while the other seed and the other chat's jobs were running), and the idle-host gate measured 0.10 s per step. At 0.28 - 0.46 s per step 10000 steps take 47 - 77 minutes per seed with both seeds side by side; started about 10:26 they are through by about 11:50 at the latest, the read-outs (13 minutes in E359, both seeds in parallel) by 12:10; 15000 steps would need 70 - 115 minutes, which does not fit before the 12:25 cut-off for the next experiment's slots.

## 0. The card (what the tools run)
```json card
{
 "id": "E361",
 "card": "",
 "title": "E359 (oracle nodes of the 2.5 m lattice, node-type head, per-source softmax with a none budget, exact multi-positive likelihood, argmax decode of record at type probability >= 0.5, section-free read-out) with 10000 training steps instead of 3000, two seeds, against E357's stored rules and GT links and E359 as the reference",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.03, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E359 (3000 steps), stored: topo F1 0.7721 / 0.7626, junction-link F1 0.531 / 0.502, g1 0.267 / 0.232; the rules and hi of E357 are the anchors (lo_near 0.7011, hi 0.9666)"},
  "p3": {"recipe": "`jtok2_train.py --name e361_p3_s<S> --seed S --lattice 2.5 --steps 10000 --bs 8 --lr 1e-3 --workers 4` (through `_pyr.sh`, checkpoint every 500 steps), everything else as E359 `p2`; read out with the unchanged `jtok2_eval.py` (dump, score, diag, the degraded read-out)", "run_ids": ["E361.p3_s0", "E361.p3_s1"]}
 },
 "control": "ref",
 "variant": "p3",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 3.0,
 "kill": "a loss is not finite; or the run is not through by 12:00 (the read-outs must be through by 12:25 and the next experiment's slots are needed about 13:00): then it is stopped at the last 500-step checkpoint and reported as 'stopped, not a verdict' (no PASS / FAIL token from a cut run; a cut run also ends at a high learning rate, so its numbers are not a reading of the schedule). E359's no-learning check (mean loss over steps 901 - 1000 below 0.8 x the mean over steps 51 - 100) is DROPPED (the warm-up is 500 steps here, the comparison would be another test); its stored-controls and gate-share checks stand unchanged (same nodes, same cache). Any ROLLBACK / skip / resume is reported",
 "evaluate": {"split": "val"}
}
```
**The verdict, written before the run: the one of E359 section 0 (commit 2bc7eb4e), verbatim** (by hand: oracle-node replay; decode of record = argmax per source and class kept at type probability >= 0.5; G1: g1 = (topo F1 - 0.7011) / (0.9666 - 0.7011) >= 0.5, i.e. topo F1 >= 0.834; G2 link level: junction-link F1 PASS 0.75, FAIL line 0.50; order: FAIL first (both seeds g1 < 0.5, or both seeds G2 < 0.50), then PASS (both seeds g1 >= 0.5 AND G2 >= 0.75), else INCONCLUSIVE; the composition and the strict-bar note, PASS-S, the attribution rule (type head works at long_out F1 >= 0.90 and AUC >= 0.97, softmax works at oracle-source long top-1 >= 0.85, labels "type head failed" / "softmax failed" / "both failed" / "both work but the compound misses the bar" / "neither failed"), the geometry-only null: all as written there). **Added reading of the comparison with E359 (reported, not a token):** the change of g1, topo F1, G2, type recall and top-1 against E359 on each seed; "undertraining" is the label when g1 rises by more than 0.10 on both seeds AND the type recall and top-1 both rise on both seeds; "no resolvable gain from the schedule" when g1 changes by less than +0.03 on both seeds (a fall counts as less than +0.03; E359's own two seeds differ by 0.035, so this is not "no gain"); anything between: "partly". Reported next to it: the mean training loss of the last 100 steps of each run (E359: 0.223 / 0.244), so that a persisting gap can be read as capacity or information rather than schedule only (a train-against-VAL comparison is not run).
**Prediction (guesses, with windows):** topo F1 0.80 (0.77 - 0.85) against E359's 0.77 (+0.03), g1 about 0.37 (0.25 - 0.55); G2 0.60 (0.50 - 0.76); type long_out recall 0.70 (0.62 - 0.80) and F1 0.78 (0.70 - 0.86); oracle-source top-1 0.82 (0.77 - 0.90). By the written order FAIL stays the likeliest outcome (about 65 %), INCONCLUSIVE about 30 %, PASS about 5 % (it needs G2 >= 0.75, the top of the window); "partly" undertraining the likeliest comparison label (about 50 %).

## 1. Card text
- **Statement:** a 3.3 times longer schedule moves the head past the half-gap line or at least shows that the gap closes with training.
- **Mechanism:** E359's head is small (0.10 M parameters) and saw about three epochs; the fall of its loss at the end is mostly the rate annealing, so whether more steps help is open.
- **Inputs:** all existing (E359's code and E357's cache and controls); nothing to build. Cost: two runs of 47 - 77 minutes on the two slots, the read-outs about 15 minutes.
- **Not claimed:** anything about predicted nodes, T1 - T3 of E355, TEST; that a longer schedule is the best change (only the schedule is varied); the first 3000 steps are NOT the same trajectory as E359's (the rate differs; the data order is the same seed, but bf16 and four loader workers are not bit-reproducible).
- **Referee:** short fresh `ar-critic` pass 2026-11-16 (cards mode, read only): REVISE; science fine, time plan and some statements not; taken into this version: the measured speed and the change from 15000 to 10000 steps, the "not exactly one parameter" paragraph (schedule, rollback guard, resume), the corrected loss values, the kill clause, the labels' wording, the prediction value and odds, the dating, the training-loss report; it verified the E359 reference numbers and that nothing in `jtok2_train.py`, `trainer.py` or `jtok2_eval.py` hard-codes 3000.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| train | p3, seeds 0 and 1 side by side on the two slots, `_pyr.sh`, `--workers 4`, `--steps 10000`, from about 10:26 | `run.py --id E361.p3_s<S> --gpu --require-card --split val -- ... jtok2_train.py ...` | done flags; stop at 12:00 at the latest |
| read-out | dump, score, diag, degraded read-out (the unchanged `jtok2_eval.py`) | `run.py --id E361.read_s<S>` | by 12:25 |
| decide | the verdict above | by hand | result section, ledger notes, commit; the E358 slots (lane chat, about 13:00) are not held |

## 3. Summary
(empty until the runs exist)
