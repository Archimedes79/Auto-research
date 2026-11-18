p = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/06_challengers.md"
s = open(p, encoding="utf-8").read()
old = "| E378 | PENDING: the card names its question and rule | PENDING | PENDING | PENDING | E378 |"
new = ("| E378 | does the max-positive wiring loss of E376 replicate on two fresh heads (the loss that sums the positive destinations stays the champion's) | topo F1 at least +0.005 with the interval above 0 on both heads at the recipe cell; junction F1, F1@1.0, lanes found not lower by more than 0 / 0.005 / 0.010 | topo F1 +0.0030 / +0.0000; junction F1 +0.038 / +0.012 (recipe cell); x_s1 lanes found -0.0106 | **NO** (not a TIE either); seed twins not trained; the champion keeps `--pos-loss sum` | E378 |")
assert s.count(old) == 1
s = s.replace(old, new)
k = "**What this ranking does not say.**"
assert s.count(k) == 1
s = s.replace(k, "**The four max-positive heads at the D115 cell (by hand: stored numbers, not a verdict).** Against the champion head of the same cache, the mean of the four max heads (E376 s0 / s1, E378 s0 / s1; SD in brackets) is junction F1 +0.0110 (0.0102), topo F1 +0.0062 (0.0039), lanes found +0.0081 (0.0047); the champion recipe on another head seed (the noise of the same comparison) gives +0.0063 / +0.0014 junction F1 and +0.0030 / -0.0032 topo F1. One fresh head (x_s1: +0.0015 topo F1) shows almost nothing, and the junction gain at the recipe cell (+0.038) was paid in lanes found; a fresh confirmation pair at D115 with a junction bar of +0.015 was estimated at 10 - 15 % to pass and was not run.\n\n" + k)
open(p, "w", encoding="utf-8", newline="\n").write(s)
