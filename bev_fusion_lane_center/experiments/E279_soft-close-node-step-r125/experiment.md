# E279 The soft close-node step with radius 1.25 m (E277 with a smaller radius; baseline B, D55 / D58)

Card of 2026-11-08 03:55, written alone (D60: Mirko is away, ask nothing; D57 / D61: this chat has priority). E277 (radius 1.5 m) was CUT by the written guards: the soft step reproduces the hard step of E275 (topo F1 0.621 / 0.624 against 0.626 / 0.625, junction F1 0.263 / 0.268 against 0.271 / 0.249, nothing left for the hard step to remove) but on seed 0 it lost more recall than the guards allow (lanes found -0.039 against the guard -0.030, F1@1.0 -0.0147 against -0.010). The in-sample radius curve of the hard step (E275 section 7) says that a radius of 1.25 m keeps more recall at a smaller gain (seed 0 / 1: topo F1 0.624 / 0.622, junction F1 0.225 / 0.211, lanes found 0.635 / 0.622, F1@1.0 0.830 / 0.819). This card trains the soft step at 1.25 m, nothing else changed: same code, same recipe, same seeds, same rule. GPU only in Mirko's windows. No TEST. Nothing pushed.

**D55 status.** As E277: the thinning of the node row is a differentiable step of the forward path; only the threshold and the greedy order (a comparison without gradient) stay outside.

## 0. The card (what the tools run)
```json card
{
 "id": "E279",
 "card": "",
 "title": "the lane head trained with the soft close-node step at radius 1.25 m (smoothness 0.1 m, 3 passes), read out at e >= 0.5 without a hard step, against the baseline B read out as before and against B with the hard step of E275 (radius 1.5 m)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.025, "vs": "c"},
 "arms": {
  "c": {"recipe": "control B = E273 runs e273_p_s0 / e273_p_s1, read out as before; scoring cards runs/e273/<run>/card.json (no new training)"},
  "h": {"recipe": "B with the hard step of E275 (radius 1.5 m, threshold 0.5): the cards runs/e275/<run>.close1.5_t0.5/card.json (no new training); the bar of the E277 rule"},
  "u": {"recipe": "from the E273 phase-2 checkpoints phases 3 4 5 6 7 as E273 with `--soft-nms 1.25 0.1 3` (E277 used 1.5); scored at e >= 0.5 with no hard step (runs/e279/<run>.plain_t0.5/card.json)"},
  "e277": {"recipe": "the E277 arm (radius 1.5 m): cards runs/e277/e277_u_s<S>.plain_t0.5/card.json; reported against u, not judged"},
  "g": {"recipe": "control of the assignment, scored inside every card"}
 },
 "control": "c",
 "variant": "u",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "junction_F1", "vs": "c", "min_effect": 0.03},
  {"metric": "lane_acc_err", "vs": "g", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.03},
  {"metric": "topo_F1", "vs": "h", "max_loss": 0.01}
 ],
 "budget_gpu_h": 4,
 "kill": "a loss that is not finite; or the gate of phase 3 failing (the gate now reads the damped existence): the run stops (exit 3); or an arm that needs more than 1.5 x the E277 time",
 "evaluate": {"split": "val"}
}
```
**Verdict rule: the rule of E277, unchanged (`qgraph_run_eval.py verdict277 --exp-var e279 ...`, written before the run, section 0 of `experiments/E277_soft-close-node-step/experiment.md`):** KEEP (the soft step replaces the hard read-out) when topo F1 gain over c above +0.02 AND junction F1 gain above +0.03 on both seeds, every guard held (assignment exact-map u - g not below -0.01 and error 1-2 m not below +0.03 on the mean of the seeds, F1@1.0 not below c - 0.01 and lanes found not below c - 0.03 on either seed) and the topo F1 not below the hard step's by more than 0.01 on either seed; TIE when the first part holds and u trails h; CUT when a gain is below its negative band or a guard is broken; otherwise INCONCLUSIVE (also both headlines inside their bands). By hand, VAL 873, final checkpoint, in-sample, no TEST. Reported, not judged: node recall / precision, vertices per 100 m, the comparison with the E277 arm.

**Prediction (a number):** topo F1 0.62 +- 0.01 on both seeds (c + 0.025); junction F1 0.20 or more (the radius curve of the hard step: 0.21 - 0.23); lanes found 0.62 - 0.65 (above E277's 0.604 / 0.619); F1@1.0 within 0.01 of c; vertices per 100 m 20 - 24. Why it could miss: the junction gain may fall under +0.03 (the hard-step curve is 0.095 / 0.091 above c, so little margin) or the trained step may not keep more recall than at 1.5 m (the seed-to-seed spread of lanes found of one recipe is 0.014 - 0.028).

## 1. Design, tests, cost
Code and tests are those of E277 (commit 66cc4084, critic pass done): only the setting differs. Jobs `E279.w_s0` / `w_s1` (training, phases 3-7, about 80 - 110 min with both seeds on the GPU: phase 7 runs at 1.5 s per step when two phase-7 jobs share the card), then `E279.wq_s0` / `wq_s1` (scoring: read-outs plain@0.5 and close:1.5@0.5, about 10 min each); about 4 GPU h in all, the window ends at 07:00. The gate of phase 3 reads the damped existence now (E277's two runs read the raw head).

## 2. Risks
As E277 section 5, plus: the run does not finish before 07:00 (it started at 04:00; if phase 7 is still running at 06:45 the scoring is not done: a partial result is reported as such, nothing is cut short by hand).

## 7. Summary (2026-11-08 06:20; VAL 873, in-sample, no TEST; cards `runs/e279/e279_w_s<k>.plain_t0.5`, ledger rows E279.w_s0 / w_s1 / wq_s0 / wq_s1, page `results/E279/README.md`)
**Verdict: KEEP (the soft step replaces the hard read-out)** (`verdict277`, written by the scheduler at 06:06, `temp/e276/verdicts.log`, the rule of section 0 unchanged): both headline gains hold on both seeds, every guard holds, and the topo F1 is within 0.01 of the hard step on both seeds. The margins on seed 0 are thin (topo F1 gain +0.0210 against the band 0.02; F1@1.0 -0.0097 against the guard -0.010; lanes found -0.026 against -0.030).
| seed 0 / seed 1 | B (c) | B + hard step 1.5 m (h) | E277 soft 1.5 m | E279 soft 1.25 m (u) | u - c |
|---|---|---|---|---|---|
| topo F1 | 0.597 / 0.595 | 0.626 / 0.625 | 0.621 / 0.624 | 0.618 / 0.620 | +0.021 / +0.025 |
| junction F1 | 0.130 / 0.120 | 0.271 / 0.249 | 0.263 / 0.268 | 0.223 / 0.218 | +0.094 / +0.098 |
| F1@1.0 | 0.832 / 0.821 | 0.828 / 0.816 | 0.818 / 0.820 | 0.823 / 0.822 | -0.0097 / +0.0006 |
| lanes found | 0.644 / 0.630 | 0.627 / 0.614 | 0.604 / 0.619 | 0.618 / 0.632 | -0.026 / +0.002 |
| vertices per 100 m (budget 14.3) | 28.9 / 31.5 | 19.3 / 20.1 | 19.1 / 18.8 | 21.4 / 20.9 | |
| node recall / precision | 0.833 / 0.873, 0.833 / 0.868 | 0.793 / 0.915, 0.793 / 0.910 | 0.787 / 0.916, 0.790 / 0.911 | 0.805 / 0.907, 0.804 / 0.901 | |
| assignment p - g exact / error | -0.0072 / +0.0722, -0.0061 / +0.0706 | | | -0.0072 / +0.0734, -0.0059 / +0.0718 | guards held |
- **Against E277 (1.5 m):** the smaller radius gives back recall (lanes found +0.014 / +0.013, F1@1.0 +0.005 / +0.002, node recall 0.805 against 0.787) and pays 0.04 - 0.05 of junction F1 (0.223 / 0.218 against 0.263 / 0.268) and 0.003 - 0.004 of topo F1; vertices per 100 m 21 instead of 19. The hard 1.5 m step on top of this head still drops nodes (the pairs between 1.25 and 1.5 m): topo F1 0.618 / 0.622, junction F1 0.253 / 0.249, lanes found 0.608 / 0.625, vertices 20.0 / 19.5.
- **Predictions (section 0):** topo F1 0.62 +- 0.01 held (0.618 / 0.620); junction F1 0.20 or more held (0.223 / 0.218); lanes found 0.62 - 0.65 held on seed 1 (0.632), missed on seed 0 by 0.002 (0.618); F1@1.0 within 0.01 of c held (-0.0097 / +0.0006); vertices per 100 m 20 - 24 held (21.4 / 20.9); exact-map guard held.
- **What it means.** The thinning of the node row is learned by the network (D55: no rule step is needed between the node head and the delivered nodes for this step); at 1.25 m it passes the guards that the 1.5 m run broke on seed 0, with a smaller junction gain. It is two seeds, in-sample, with a topo gain 0.001 above the band on seed 0: not a headline number. The delivered chain can use either the 1.25 m soft step alone or, as a named extra read-out, the 1.5 m hard step on top (junction F1 higher, lanes found lower); the choice between them is Mirko's, no D-id was opened.
- **Run notes.** Both trainings were hit twice by a host memory failure (kernel page allocation failure order 4 in the DataLoader workers, WSL VM with 188 MB free and 22 GB page cache) at 04:18 and 04:32 and were resumed from their checkpoints (every 500 steps); the page cache was evicted at user level (`temp/e276/reclaim.py`) and a watchdog (`mem_watch.sh`) repeated it. A resumed phase is the same run; the steps after the last checkpoint were redone. Not a property of the method.
