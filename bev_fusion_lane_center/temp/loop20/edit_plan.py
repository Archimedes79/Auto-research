p = "experiment_execution/PLAN.md"
t = open(p, encoding="utf-8").read()
def rep(a, b):
    global t
    assert a in t, a[:60]
    t = t.replace(a, b, 1)
rep("| CPU 10 min | done: cap NOT supported", "| CPU 10 min | done (E252): cap NOT supported")
rep("| 2 x 20 min GPU | running |", "| 2 x 20 min GPU | done (E240): TIE on four pairs, Q a - c +0.0062 (prediction +0.006 hit), lanes found +0.0596, seed SD Q 0.0030 / 0.0020, paired 0.0043; E241 KEEP as 'worth a 12-pairing replay' (cut test 4 of 4, 0 wrong) |")
rep("| 20 min GPU + grids | queued behind E240.c3; controls' grids done |", "| 20 min GPU + grids | done (E255): CUT, Q a - c -0.0726 at 14.3 vertices (prediction -0.045 missed by -0.028), topo F1 -0.127; K193 closed |")
rep("| 45 min exclusive | last, after the trainings |", "| 45 min exclusive | BLOCKED: the BEV chat's 7-hour training (t1_r4_soa) runs in the window; item E253.t stays planned; replaced by I4b |\n| I4b | E256 D4 candidates by city | the order of the five candidates holds in each city | flip = opposite order by > 0.010 in both cities | CPU 10 min | done (E256): no flip; fast_recall over fast is a Boston-only Q gain, lanes found +0.065 in both |")
rep("| N14 (new) | matched-vertex evaluation had no code, a max over grid points favours the arm with more freedom | `temp/loop20/e255_matched.py` (interpolation at 14.3, stored-card reproduction, held-out frames) |",
    "| N14 (new) | matched-vertex evaluation had no code, a max over grid points favours the arm with more freedom | `temp/loop20/e255_matched.py` (interpolation at 14.3, stored-card reproduction, held-out frames): used in E255, works; promote to `evaluation/drivers/` when used a second time |\n| N15 (new) | `decide.py` has no cross-pairing cut test (E241 was decided by hand) and `schedule.py add` accepts duplicate ids | open (tools chat) |\n| D18 / D19 (new) | per-metric noise bands from four pairs; machine lease between chats | filed in DECISIONS |")
rep("- weekly report week 3 (loops 18 - 20), fresh numbers reviewer, `check_structure.py`, full test suite at the end, handover, shutdown only if no other chat runs a job",
    "- weekly report week 3 (loops 18 - 20): `reports/weekly/Week3_Loops18-20_Report_EN.pdf` built, numbers reviewer run; `check_structure.py` clean; full test suite; handover; shutdown NOT done (the BEV chat's training t1_r4 runs)")
open(p, "w", encoding="utf-8").write(t)
