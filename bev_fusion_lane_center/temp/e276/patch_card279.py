p = "experiments/E279_soft-close-node-step-r125/experiment.md"
s = open(p, encoding="utf-8").read()
a = s.index("## 7. Summary")
new = '''## 7. Summary (2026-11-08 06:20; VAL 873, in-sample, no TEST; cards `runs/e279/e279_w_s<k>.plain_t0.5`, ledger rows E279.w_s0 / w_s1 / wq_s0 / wq_s1, page `results/E279/README.md`)
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
'''
s = s[:a] + new
open(p, "w", encoding="utf-8").write(s)
print("ok")
