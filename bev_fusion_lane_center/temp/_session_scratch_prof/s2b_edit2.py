p = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/06_challengers.md"
s = open(p, encoding="utf-8").read()
old = "The constraint D91 / D118 stands: one frame and one radar scan per step, the past only recursively at feature level.\n"
new = old + "\nHow it relates to E384: S2b is the token-level form of what E384 asked at the feature level (a recursive state inside the node head of B'). E384 stays closed by the oracle ceiling of its step 0 (chapter 4); S2b asks the same question one level up, where the head already reads the past. If S2b wins its rule, the end-to-end confirmation is the S4 stage 2 over it (the loss reaching the node head with the past block in the head); a win at stage 1 alone is not the champion.\n"
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
