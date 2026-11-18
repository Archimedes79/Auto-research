f = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/06_challengers.md"
s = open(f, encoding="utf-8").read()
old = "| E384 | PENDING: the combined optimisation (recipe and K) | PENDING | PENDING | PENDING | E384 |"
new = ("| E384 | a recursive node state in the node head: can the previous keyframe's B' output help this frame's nodes at all (step 0, 851 VAL frames, D118) | "
       "GO bars +0.006 coverage points, +0.005 AP lift (written after the probe numbers were seen) | oracle coverage ceiling net of the identity null +0.0021 .. +0.0037, "
       "pose-minus-identity AP lift +0.0017 / +0.0018, 1 true to 8 false carried-over tokens | CUT by hand: critic step 0 (no code, no GPU hour; card kept ready) | E384 |")
assert s.count(old) == 1
open(f, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
