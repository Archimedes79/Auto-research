p = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/12_winners.md"
s = open(p, encoding="utf-8").read()
a = "| training stages | three, plus the attachment | one (the head; B' is frozen) |"
b = "| training stages | three, plus the attachment | two (D120): stage 1 the head on the frozen cache, stage 2 the head and the node head of B' together (3000 steps) |"
assert s.count(a) == 1
s = s.replace(a, b)
c = "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | champion heads "
assert s.count(c) == 1
s = s.replace(c, "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | **since D120 the stage-2 weights `runs/e372_s4_s0` / `runs/e372_s4_s1` (head + node head of B', `b_state`); PENDING at the build: the export of the winners folder is refreshed to the S4 code path (`jlive_train.py`) and verified**; stage-1 champion heads ")
open(p, "w", encoding="utf-8", newline="\n").write(s)
