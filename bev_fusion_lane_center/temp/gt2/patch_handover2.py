p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
i = s.index("Results: E286")
j = s.index("## Running unattended")
new = "Results: E286 maps KEEP (kp_f1 0.64 vs 0.39); pair heads E287 / E308 / E305 TIE when the connectors are drawn from the keypoints. **E311 KEEP: connector ends snapped to the delivered pieces, p >= 0.1: topo F1 +0.023..+0.036 over the plain graph, lanes found +0.06..+0.09, junction F1 +0.02..+0.06, duplicates -0.04** (E287 p2, E305, E303; both seeds; cards E309-E311). E310: maneuver coverage (`evaluation/jmaneuver.py`): plain 52.5 %, p>=0.1 61.6 %, perfect pairs 64.3 %; the pair decision is saturated, the ENDS are the cap (52 % of GT pairs have both ends matched) -> E307 / E304 / E312.\n\n"
s = s[:i] + new + s[j:]
open(p, "w", encoding="utf-8").write(s)
print(len(s.encode()))
