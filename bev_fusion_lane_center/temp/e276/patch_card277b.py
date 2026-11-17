p = "experiments/E277_soft-close-node-step/experiment.md"
s = open(p, encoding="utf-8").read()
a = s.index("## 7. Summary")
new = '''## 7. Summary (2026-11-08 03:50; VAL 873, in-sample, no TEST; cards `runs/e277/e277_u_s<k>.plain_t0.5`, ledger rows E277.u_s0 / u_s1 / uq_s0 / uq_s1, page `results/E277/README.md`)
**Verdict: CUT by the written guards** (`verdict277` by hand, 03:40, `temp/e276/verdicts.log`): seed 0 breaks two guards, F1@1.0 (-0.0147 against -0.010) and lanes found (-0.0394 against -0.030); seed 1 holds every guard. The reasons for the CUT are the guards, not the headline: both gains over B hold on both seeds and the soft step replaces the hard one.
| seed 0 / seed 1 | B (c) | B + hard step (h, E275) | head with the soft step (u) | u - c | h - u (topo F1) |
|---|---|---|---|---|---|
| topo F1 | 0.597 / 0.595 | 0.626 / 0.625 | 0.621 / 0.624 | +0.024 / +0.029 | 0.005 / 0.001 |
| junction F1 | 0.130 / 0.120 | 0.271 / 0.249 | 0.263 / 0.268 | +0.133 / +0.147 | |
| F1@1.0 | 0.832 / 0.821 | 0.828 / 0.816 | 0.818 / 0.820 | **-0.015** / -0.001 | |
| lanes found | 0.644 / 0.630 | 0.627 / 0.614 | 0.604 / 0.619 | **-0.039** / -0.011 | |
| vertices per 100 m (DP; budget 14.3) | 28.9 / 31.5 | 19.3 / 20.1 | 19.1 / 18.8 | | |
| node recall / precision | 0.833 / 0.873, 0.833 / 0.868 | 0.793 / 0.915, 0.793 / 0.910 | 0.787 / 0.916, 0.790 / 0.911 | | |
| long-link precision | 0.167 / 0.191 | 0.172 / 0.194 | 0.172 / 0.186 | | |
| assignment p - g exact / error | -0.0072 / +0.0722, -0.0061 / +0.0706 | -0.0066 / +0.0731, -0.0058 / +0.0712 | -0.0064 / +0.0736, -0.0048 / +0.0715 | guards held | |
- **What the soft step does.** Trained end to end it reproduces the hard step: topo F1 within 0.005 / 0.001, junction F1 0.263 / 0.268 against 0.271 / 0.249, vertices per 100 m 19 against 19 - 20; and the hard step on top of it removes no further node (the two read-outs give identical cards), so the thinning is fully inside the network. The head needs no rule step between the node head and the delivered graph (D55 for this step).
- **What it costs.** Lane recall: lanes found -0.039 / -0.011 and F1@1.0 -0.015 / -0.001 against B (the hard step: -0.017 / -0.016 and -0.004 / -0.005). On seed 1 the soft step is as good as the hard one or better (lanes found 0.619 against 0.614); on seed 0 it loses 0.023 more lanes found than the hard step. Two seeds cannot separate a systematic loss from the seed-to-seed spread of the head (about 0.015 - 0.04 in lanes found between seeds of the same recipe); a third seed would need a third phase-2 checkpoint (about 65 min of training), not available tonight. By the rule written before the run the result is CUT; the more honest word is "the soft step equals the hard step in quality gain and may lose a little more recall"; the guard values were taken over from E275 and whether they are the right ones for a trained step is for Mirko (no D-id opened).
- **Predictions (section 0), all held:** topo F1 0.62 +- 0.01 on both seeds (0.621 / 0.624); junction F1 at least 0.22 (0.263 / 0.268); lanes found 0.60 - 0.64 (0.604 / 0.619); vertices per 100 m 18 - 22 (19.1 / 18.8); exact-map assignment guard held (-0.0064 / -0.0048).
- **Not seen / not checked:** the pictures (`results/E277/figures/frame_*.png`: B | B + hard step | the E277 head | arm t, the same frames) show a node row thinned like the hard step's; the 20-frame look for a false node pulled onto a stronger neighbour (section 5) was not made; the count of nodes at e in [0.2, 0.5) was not made (the head's raw logit is not exported).
- **Open for Mirko:** whether a trained step with the hard-step guards is the right test (the rule says CUT); follow-up E279 (radius 1.25 m, the same rule) started 03:42, see its card.

### 7a. E278: is the joint refinement (phase 7) worth it? (diagnostic, scoring only, never a verdict)
The phase-6 model of B (before phase 7) scored at e >= 0.5, `runs/e278/e273_p_s<k>_p6.plain_t0.5` and `.close1.5_t0.5` (ledger E278.qp_s0 / qp_s1; seed 0 / seed 1; against the final B):
| | p6 | final B | p6 + hard step | final + hard step |
|---|---|---|---|---|
| topo F1 | 0.544 / 0.570 | 0.597 / 0.595 | 0.565 / 0.571 | 0.626 / 0.625 |
| junction F1 | 0.156 / 0.168 | 0.130 / 0.120 | 0.311 / 0.342 | 0.271 / 0.249 |
| lanes found | 0.667 / 0.573 | 0.644 / 0.630 | 0.635 / 0.537 | 0.627 / 0.614 |
| vertices per 100 m | 22.9 / 26.9 | 28.9 / 31.5 | 18.0 / 20.5 | 19.3 / 20.1 |
| duplicates | 0.407 / 0.270 | 0.232 / 0.192 | 0.407 / 0.279 | 0.234 / 0.201 |
| long-link precision | 0.097 / 0.155 | 0.167 / 0.191 | 0.100 / 0.155 | 0.172 / 0.194 |
Reading: the joint phase raises the topo F1 by 0.03 - 0.05 and the long-link precision and lowers the duplicates, and it lowers the junction F1 under the hard step (by 0.04 / 0.09) and raises the vertices per 100 m. It is a trade, not a defect of the recipe: the phase stays. The E275 card's note "phase 7 lowers the junction F1" is true and is paid for by the topology.
'''
s = s[:a] + new
open(p, "w", encoding="utf-8").write(s)
print("ok")
