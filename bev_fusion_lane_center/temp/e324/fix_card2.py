p = "experiments/E324_e320-duplicates-refined/experiment.md"
s = open(p, encoding="utf-8").read()
a = s.index("**Pre-registered with the reference E316:**"); b = s.index("A hit is within +-0.003 for topo F1")
new = ("**Pre-registered with the reference E316:** E320 at 3000 steps (a) has dup_frac +0.0253 (s0) / +0.0119 (s1) over E316 and topo F1 -0.0095 / -0.0049 below it. If the selectivity of the longer pair stage transfers, half of the excess over E316 falls: dup_frac of s over E316 **+0.013 (s0) / +0.006 (s1)**, "
       "and topo F1 of s over E316 **-0.002 / +0.003** (the sum of E320's a - c and E316's gain against E316's own gain); the false-curve count of s at p >= 0.1 about 5000 (s0, between E316's 4697 and E320's 6201). The additive guess a(s) - c = (a - c) + (E316 - c) leaves the excess over E316 unchanged by construction and would pass the written bounds by the longer training alone, which is why the reading is against E316. "
       "Predictions of the rule quantities against c: topo F1 +0.0051 / +0.0099, dup_frac +0.0115 / +0.0068, junction F1 -0.023 / -0.008 (seed 0 is 0.007 from its guard), F1@1.0 +0.009 / +0.009, vertices -0.3 / -0.3. ")
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s); print("ok")
