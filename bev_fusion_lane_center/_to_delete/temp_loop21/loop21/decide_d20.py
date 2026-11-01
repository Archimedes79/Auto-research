p = "experiment_execution/DECISIONS.md"
lines = open(p, encoding="utf-8").read().split("\n")
out, moved = [], False
for ln in lines:
    if ln.startswith("| D20 |"):
        continue
    out.append(ln)
    if ln.startswith("|---|---|---|") and any(l.startswith("## Decided") for l in out[-4:]):
        out.append("| 2026-10-30 | D20 (Mirko, in chat: \"Yes, adopt (b)\", then \"go\" after an explanation of the steps): the vertex budget is quoted after the output simplification (`simplify_dp`, tolerance 0.2 m), the node-id count stays next to it in every table; the stage is default off in `assemble()` | `assemble(dp_eps=0.0)` plus `simplify_dp` in `training/pipeline/p16_infer.py`, spec tests `evaluation/tests/test_dp_simplify.py`, registry entries `fast_recall_dp` / `champion_dp` (reproduced by `evaluate.py`: fast_recall_dp Q 0.739, 9.86 vertices), PROBLEM.md sentence under a D20 grant; route head and map pipeline use `aux=True` (unsimplified vertices) and are unchanged; E257 |")
        moved = True
assert moved
open(p, "w", encoding="utf-8").write("\n".join(out))
