# E268 TEST check of the loop-22 lane prototype against the control

One file per experiment; it grows from card (P2) to detail plan (P3 start) to summary (P3 end). Report only: nothing is selected or tuned on TEST.

## 0. What the tools run
| field | value |
|---|---|
| id / card / title | E268 / H22.E268 / TEST check of the loop-22 prototype vs the control |
| kind | variant (configuration only; the prototype is a registry entry) |
| deterministic | true (one checkpoint; the paired scene bootstrap of `evaluate.py --compare` gives the interval) |
| metric, direction, band | lanes_found, higher, 0.010 |
| prediction | prototype - control lanes found about -0.03 (VAL: 0.781 vs 0.800), Q about +0.02 (VAL: 0.742 vs the control's ~0.71 range); see below |
| arms | p: the prototype of record (`recall_sp0_dp` unless E267 / E264 replace it by the PLAN rule before the run); c: `control` (stored E94 TEST graphs) |
| evaluate | split test (833 frames, 21 scenes), compare control |

## 1. Card (lead, loop 22, 2026-11-01 05:00)
| | |
|---|---|
| lens | system check (D28 deliverable) |
| group | A3 sparse lane graph; gap 1 |
| sources | E265 (KEEP), E261 (CUT), PLAN.md loop 22 "The prototype"; TEST rows of E230 (loop 18) in `evaluation/TEST_USAGE_LOG.md` |
| status | scheduled (after P.proto; the TEST log row is written before the run) |

- **Statement**: the loop-22 prototype's VAL gain in coverage at range over `recall` (lanes found +0.014, R@1 far +0.008 on VAL 873, VAL-selected) survives on TEST: against the control E94 its lanes-found gap is smaller than `recall`'s TEST gap of E230.rc (0.724 vs 0.762) and its Q stays above the control.
- **Mechanism**: E265 removed a rule that deletes true short geometry at range; nothing was fitted on TEST, so a VAL gain of an assembly setting should transfer unless the 22 VAL scenes over-represent short far stubs.
- **Prediction** (TEST 833 frames): prototype lanes found about 0.735 - 0.74 (E230.rc 0.724 + the VAL gain), Q about 0.73, vertices within the budget (per-piece DP <= 14.3); control lanes found 0.762, Q about 0.70.
- **Decision rule**: report only. The TEST numbers do not choose anything and do not promote anything (champion promotion stays with Mirko, D4). They are labelled TEST, 833 frames, VAL-selected system. If the TEST gain over E230.rc's lanes found is below 0 the summary says the VAL gain did not transfer.
- **Test design**: one run `E268.p`: `evaluate.py <prototype> --split test --test-id E268.p --compare control` (GPU: no cached TEST raw outputs; the network runs fresh once, ~5 - 10 min, idle-ish GPU not needed for numbers). Matched reference: the control on the same 833 frames (paired scene bootstrap). The E230.rc row (rc1 + pruning 0.4, loop-18 pipeline) is the in-sample-free reference for `recall`, named as such (other code version: a reference, not a paired control).
- **Cost**: GPU ~10 min; the one TEST use of loop 22 (D28).
- **Kill criterion**: the TEST log row is missing (the launcher refuses); the prototype changes after the log row (then a new row first).
- **Differentiable path / E2E**: no change on the training path; the delivered graph still has the rule steps path union, clean-up (bubble half), junction anchoring / relocation, fork pruning, output DP (named in the summary).

## 3. Summary (lead, loop 22, 2026-11-01 06:15)
- **Run**: E268.p (`lcr/E268.p`, 06:10 - 06:13, fresh network outputs on the GPU), logged in `evaluation/TEST_USAGE_LOG.md` before the run (grant D28); card `runs/eval/recall_sp0_dp/test.json`. The prototype was fixed on VAL by the PLAN.md rule before the run: E265 KEEP -> S; E261 CUT (no W); E264 not adopted (no U); E266 KEEP (R: lanes found 0.774, below S's 0.781, so S stays); E267 CUT (SR). **Report only.**
- **Numbers** (TEST, 833 frames, 21 scenes; VAL-selected system, nothing tuned on TEST):

| | Q | lanes found | R@1 far | topo F1 | J | false length | dup | vertices / 100 m |
|---|---|---|---|---|---|---|---|---|
| prototype `recall_sp0_dp` | 0.727 | 0.740 | 0.749 | 0.705 | 0.468 | 0.079 | 0.092 | 12.6 (after the output stage) |
| control E94 (same frames) | 0.712 | 0.762 | 0.776 | 0.722 | 0.406 | 0.089 | 0.133 | 22.0 |
| prototype - control [95 %, paired scenes] | +0.015 [+0.000, +0.029] | -0.021 [-0.037, -0.005] | -0.028 [-0.042, -0.012] | -0.017 | +0.061 | | -0.041 | |
| reference E230.rc (rc1 + pruning 0.4, loop-18 pipeline, TEST 2026-10-25) | 0.727 | 0.724 | | | 0.463 | | | 16.6 (node ids) |

- **What it says**: the prediction (lanes found 0.735 - 0.74) is hit; the VAL gain of keeping the spurs transfers to TEST (+0.016 lanes found over the loop-18 TEST number of the same checkpoint and pruning, at the same Q); the coverage gap to the control on TEST shrinks from -0.038 to -0.021 lanes found but is not closed, and R@1 far stays -0.028 behind. The prototype beats the control on Q, junctions (J) and duplicates with 43 % fewer vertices. The E230.rc reference used other code and a node-id vertex count, so it is a reference, not a paired control.
- **Rule steps of the delivered graph (E2E gap)**: path union, clean-up (bubble half; the spur half is now off), junction anchoring / relocation, fork pruning, output Douglas-Peucker.
- **Not checked**: route success on spur-kept graphs (the route head reads the unsimplified vertices); ms per frame re-measured (stored timing of `recall`: lane graph about 21 ms, full chain about 27 ms with the time recursion).
