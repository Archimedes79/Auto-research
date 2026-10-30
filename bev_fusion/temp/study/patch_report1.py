p = "experiments/t3_improve_loop/report.md"
s = open(p, encoding="utf-8").read()
old = s[s.index("1. **The reference system is robust"):s.index("2. **Both sensors are strong blocks")]
new = """1. **The reference system is robust against block swaps.** Of eleven exchanges, none is BETTER than the reference outside the noise band. Seven tie (gate fusion, CMX fusion,
   ConvGRU, pillars radar encoder, no temporal block, and the two camera-side fixes), four are worse (no depth weight in the lift, random-initialised BEV encoder, radar removed,
   camera removed). The best standard replacement is the pillars radar encoder (mAP2 0.349 against 0.339 and 0.324 for the two reference seeds: a candidate, not a result).
"""
s = s.replace(old, new)
s = s.replace("| zero / shuffled camera collapses `ref_s0` to 0.013 / 0.028 (mAP3), shuffled radar gives 0.151; the long-trained system 0.031 / 0.036 and 0.156 |", "| zero / shuffled camera collapses `ref_s0` to 0.013 / 0.028 (mAP3), shuffled radar gives 0.151 (mAP3); the long-trained system 0.031 / 0.036 and 0.156 (mAP3) |")
open(p, "w", encoding="utf-8", newline="\n").write(s)
