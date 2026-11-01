p = "HANDOVER.md"
t = open(p, encoding="utf-8").read()
a = "- Results video of `recall`"
assert a in t
t = t.replace(a, "- E259 (clean-up ablation): the spur / bubble clean-up stays; timing on this loaded machine (load 4 - 15) is unusable, process CPU time does not fix it (N18: no timing claim above load = physical cores); anchoring and relocation each cost about 25 % of the assembly, the dp stage costs about 1 ms loaded (vectorise only if it stays). E260: 11 dead options and 149 lines removed from `assemble()` (31 -> 24 parameters), graphs identical (hashes `temp/loop21/graph_hash.py`), reviewed.\n" + a, 1)
t = t.replace("1. Idle-window repeat of the timing with `fast_recall_na` (E253 driver, new entry) once the BEV lane pauses.", "1. Idle-window repeat of the timing (E253 driver) with `fast_recall_na`, `*_dp` and the stage ranking of E259, once the BEV lane pauses; then identical-graph speed-ups of `relocate_forks` / `anchor_junctions_fast` (about 25 % of the assembly each), only with a valid timing.")
open(p, "w", encoding="utf-8").write(t)
