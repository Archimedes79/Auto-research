# E362 - E361 with 16000 training steps: does a still longer schedule lift the oracle-node head over the strict bars? (TEXT ONLY: not started, no critic pass yet; starts only on the lane lead's word after E358 is through)

Card of 2026-11-16 (written about 12:50 as text for the next GPU window, on the lane lead's 12:45 message: "you MAY write the E362 card as TEXT ONLY ... no critic, no start; steps about 14000 - 20000 on the then-free slots, the strict link-level G2 bar kept AND the multi-successor decode / source-level numbers reported next to it, the argmax cap named"). Written by the Spurmitte chat that Mirko addresses as "Fable". **Status: draft. Before any start: a fresh `ar-critic` (cards mode), the lead's word, free slots; no start after 15:30 on a weekday window, every run must end before the shutdown / 17:15 rule of the day it is started.** D111 covers the node-type-head question; a PASS proposes nothing automatically (T1 of E355 needs Mirko's word, D105 / D111).

Reference row: E361 (10000 steps), FAIL by the letter on both seeds, by a hair: g1 0.491 / 0.484, topo F1 0.8316 / 0.8297 (half-gap line 0.834), junction-link F1 (G2) 0.687 / 0.688, source-level F1_s 0.772 / 0.772, type long_out recall 0.789 / 0.793 (F1 0.838 / 0.833, AUC 0.982 / 0.980), oracle-source top-1 0.891 / 0.892, loss of the last 100 steps 0.102 / 0.110; E359 (3000 steps): g1 0.267 / 0.232, G2 0.531 / 0.502. **The one question:** does 1.6 times the schedule of E361 (16000 steps) move g1 clearly past 0.5 (not by a hair) and the strict G2 toward 0.75? **What is known in advance, said plainly:** (1) the strict link-level G2 bar 0.75 is capped by the argmax decode: 7 710 GT junction links start at 6 256 source nodes, so recall <= 0.811 and F1 <= about 0.90 x the type precision; G2 0.75 needs an almost perfect type head, so INCONCLUSIVE (g1 >= 0.5 with G2 between 0.50 and 0.75) is the best outcome that is likely; (2) the gain from 3000 to 10000 steps was +0.22 / +0.25 in g1; the loss fell from 0.22 to 0.10 and the half-gap line is within the noise of the quantity, so a further gain is expected to be smaller (diminishing returns); (3) a source-level bar (F1_s) would be chosen AFTER E361 showed F1_s 0.772: if it is reported it is labelled "chosen after E361, informed by it, not independent" and is never the verdict.

## 0. The card (what the tools run)
```json card
{
 "id": "E362",
 "card": "",
 "title": "E361 (oracle nodes of the 2.5 m lattice, node-type head, per-source softmax with a none budget, argmax decode of record at type probability >= 0.5, section-free read-out) with 16000 training steps instead of 10000, two seeds, against E361 as the reference and E357's stored rules and GT links as anchors",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.015, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E361 (10000 steps), stored: topo F1 0.8316 / 0.8297, g1 0.491 / 0.484, junction-link F1 0.687 / 0.688; the rules and hi of E357 are the anchors (lo_near 0.7011, hi 0.9666)"},
  "p4": {"recipe": "`jtok2_train.py --name e362_p4_s<S> --seed S --lattice 2.5 --steps 16000 --bs 8 --lr 1e-3 --workers 4` (through `_pyr.sh`, checkpoint every 500 steps), everything else as E361 `p3`; read out with the unchanged `jtok2_eval.py` (dump, score, diag, the degraded read-out)", "run_ids": ["E362.p4_s0", "E362.p4_s1"]}
 },
 "control": "ref",
 "variant": "p4",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 3.0,
 "kill": "a loss is not finite; or a run is not through by the end time that the lead names at the start (the run.py --timeout is set from the planned step count AT the measured step speed plus 30 %, never shorter: the E361 slip, a 100-minute timeout for a plan of 15000 steps, cut both runs at 12:06; at the contended 0.4 - 0.7 s per step 16000 steps need 107 - 187 minutes per seed, at the idle 0.10 - 0.20 s per step 27 - 54 minutes: the start waits for free slots, the speed is measured in the first 500 steps and the run is stopped by hand at the last 500-step checkpoint if the end time cannot be met): then it is reported as 'stopped, not a verdict'. The stored-controls and gate-share checks of E359 stand unchanged (same nodes, same cache). Any ROLLBACK / skip / resume is reported (a resume re-shuffles the loader)",
 "evaluate": {"split": "val"}
}
```
**The verdict, written before the run: the one of E359 section 0 (commit 2bc7eb4e) and E361 section 0, verbatim, unchanged, the STRICT link-level G2 bar kept** (by hand: oracle-node replay; decode of record = argmax per source and class kept at type probability >= 0.5; G1: g1 = (topo F1 - 0.7011) / (0.9666 - 0.7011) >= 0.5, i.e. topo F1 >= 0.834; G2 link level: junction-link F1 PASS 0.75, FAIL line 0.50; order: FAIL first (both seeds g1 < 0.5, or both seeds G2 < 0.50), then PASS (both seeds g1 >= 0.5 AND G2 >= 0.75), else INCONCLUSIVE; the attribution rule: type head works at long_out F1 >= 0.90 and AUC >= 0.97, softmax works at oracle-source long top-1 >= 0.85). **Reported next to it, never the verdict:** (a) the multi-successor decode (cond03: every candidate with s >= 0.3 of a gated source); (b) the source-level F1_s, labelled "chosen after E361, informed by it, not independent"; (c) the cap: recall <= 0.811 (6 256 / 7 710) and the share of the cap that is reached (recall / 0.811); (d) the change against E361 and E359 on each seed (g1, topo F1, G2, type recall, top-1, training loss of the last 100 steps; the label "still gaining" when g1 rises by more than 0.03 on both seeds, "saturated" when it changes by less than 0.02 on both seeds, otherwise "small gain"); (e) the degraded read-out (17 % of the nodes removed) against the best rule (0.6477). A FAIL here would be a FAIL of a head that is already above every rule: it is read together with (a) - (e), by hand.
**Prediction (guesses, with windows):** topo F1 0.845 (0.82 - 0.87), g1 about 0.54 (0.45 - 0.65); G2 0.72 (0.66 - 0.78); type long_out recall 0.83 (0.77 - 0.88); oracle-source top-1 0.90 (0.87 - 0.93). By the written order INCONCLUSIVE is the likeliest outcome (about 55 %), FAIL about 30 % (g1 under 0.5 on both seeds again), PASS about 15 % (it needs G2 >= 0.75 on both seeds, the top of the window and close to the cap); "saturated" about 40 %, "small gain" 40 %, "still gaining" 20 %.

## 1. Card text
- **Statement:** a 1.6 times longer schedule (16000 steps) moves the oracle-node head clearly past the half-gap line and its junction-link F1 toward 0.75, or shows that the schedule is saturated and the weak output (the type head) is a capacity / information problem.
- **Mechanism:** E361's loss was still falling slowly in its last third (0.22 at 3000 steps, 0.10 at 10000), and the type head (recall 0.79) is the output that fails; the softmax is solved (0.89).
- **Inputs:** all existing (E361's code and E357's cache and controls); nothing to build. Cost: two runs, 27 - 54 minutes on the idle slots (107 - 187 minutes contended), read-outs about 15 minutes.
- **Not claimed:** anything about predicted nodes, T1 - T3 of E355, TEST; that a longer schedule is the best change (only the schedule is varied; the rate schedule stretches with the step count and the data order differs); the strict bar 0.75 as a reachable target (the argmax cap).
- **Referee:** none yet (draft). Before the start a fresh `ar-critic` in cards mode, with the E361 slip (timeout), the cap and the label of the source-level numbers named in the brief.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| gate | free slots (E358 through), GPU free below 2 GB, the lead's word; measure the step speed of the first 500 steps and set `--timeout` = steps x speed x 1.3 | the lead | before the start |
| train | p4, seeds 0 and 1 side by side on the two slots, `_pyr.sh`, `--workers 4`, `--steps 16000` | `run.py --id E362.p4_s<S> --gpu --require-card --split val -- ... jtok2_train.py ...` | done flags |
| read-out | dump, score, diag, degraded read-out (the unchanged `jtok2_eval.py`) | `run.py --id E362.read_s<S>` | within 15 minutes of the training |
| decide | the verdict above and the reported items (a) - (e) | by hand | result section, ledger notes, commit |

## 3. Summary
(empty until the runs exist)
