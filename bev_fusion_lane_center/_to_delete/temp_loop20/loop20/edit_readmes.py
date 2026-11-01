p = "reports/README.md"
t = open(p, encoding="utf-8").read()
a = "| `Weeks1-2_LaneGraph_Presentation_EN.pptx` |"
row = "| `Week3_Loops18-20_Report_EN.pdf` | Week 3 (24 - 27 Oct), loops 18 - 20: seed floor (four paired seeds, Q SD 0.0025), matched vertex budgets, the damaged distance map, map benefit on new error draws, ground-truth offset by city, Poisson set loss CUT at a matched budget (-0.073 Q), D4 candidates by city, the framework test (what worked, what hurt), corrected claims, experiment index (the timing repeat E253 was blocked by another chat's training) |\n"
assert a in t
t = t.replace(a, row + a, 1)
t = t.replace("Both weekly reports end with", "All three weekly reports end with", 1)
open(p, "w", encoding="utf-8").write(t)
p = "reports/build/README.md"
t = open(p, encoding="utf-8").read()
a = "| `reports/` | `build_loop18_report.py`"
row = "| `reports/` | `fig_w3_results.py`, `build_week3_report.py` | the week 3 report (loops 18 - 20): the figures first (they also write `../source/weekly/week3_numbers.json`), then the html; cards from `runs/eval`, E240 / E252 / E255 numbers from result files; PDF with an absolute path in `print_weekly.ps1` |\n"
assert a in t
t = t.replace(a, row + a, 1)
open(p, "w", encoding="utf-8").write(t)
