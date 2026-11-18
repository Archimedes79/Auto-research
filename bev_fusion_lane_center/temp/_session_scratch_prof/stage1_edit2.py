p = "bev_fusion_lane_center/results/E2E_Report_2026-11-18/parts/07_winners.md"
s = open(p, encoding="utf-8").read()
c = "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | champion heads "
assert s.count(c) == 1
s = s.replace(c, "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | **since D120 the stage-2 weights `runs/e372_s4_s0` / `runs/e372_s4_s1` (head + node head of B', `b_state`); the winners folder below still exports the stage-1 head code (not refreshed at this build)**; stage-1 champion heads ")
open(p, "w", encoding="utf-8", newline="\n").write(s)
j = "bev_fusion_lane_center/results/E2E_Report_2026-11-18/report.json"
t = open(j, encoding="utf-8").read()
a = '  "../../experiments/E2E_champion_scorecard.md",'
if "E2E_cost_table" not in t and t.count(a) == 1:
    t = t.replace(a, a + '\n  "../../experiments/E2E_cost_table.md",')
    open(j, "w", encoding="utf-8", newline="\n").write(t)
print("ok", "E2E_cost_table" in t)
