p = "HANDOVER.md"
t = open(p, encoding="utf-8").read()
def sub(a, b):
    global t
    assert a in t, a[:50]
    t = t.replace(a, b)
sub("loop-20 plan in `history/plans/PLAN_loop20.md`", "loop-20 plan in `experiment_execution/history/plans/PLAN_loop20.md`")
sub("code not yet in `assemble()`, script `temp/loop21/e257_simplify.py`)", "D20 decided: adopted 2026-10-30 as default-off `assemble(dp_eps)` + `simplify_dp`, registry `fast_recall_dp` / `champion_dp` reproduced by evaluate.py, 9 spec tests, code review done, PROBLEM.md / PROTOCOL.md sentences; route / map pipelines (`aux=True`) stay on unsimplified vertices)")
sub("- Open decisions for Mirko: D4 fast variant, D5 purge, D17 (`project.json` loops), D18 (per-metric bands), D19 (machine lease), D20 (vertex budget counting after output simplification).", "- Open decisions for Mirko: D4 fast variant, D5 purge, D17 (`project.json` loops), D18 (per-metric bands), D19 (machine lease).")
sub("2. If Mirko accepts D20: `simplify_dp` as a default-off option of `assemble()` with a spec test, `ar-reviewer` pass, evaluate.py reproduction; check the route head and the map assignment on simplified graphs first.", "2. D20 follow-ups: route success and map assignment on simplified graphs (not possible with `aux=True`: needs a way to carry the route scores to the simplified polylines), a `recall_dp` entry, the ms of the stage on an idle machine.")
open(p, "w", encoding="utf-8").write(t)
