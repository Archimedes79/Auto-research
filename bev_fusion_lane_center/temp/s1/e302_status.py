p = "experiments/E302_capacitated-pair-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
anchor = "\n\nCard of 2026-11-11 (plan card, behind E287)."
assert s.count(anchor) == 1
s = s.replace(anchor, "\n\n**Status (2026-11-13): code removed, never run.** The `capacity` option (the degree heads, the log-domain Sinkhorn plan with a dustbin, `degree_classes`, `--capacity`) was removed from `jends.py`, `jends_train.py`, `jends_eval.py` and `jmodel_cost.py` as part of the code simplification the user asked for (D84 point 4, D87 / D88: simplify; the options of cut or never-run experiments go after a critic pass, history keeps them). It was never run beyond a smoke run (`runs/e302_smoke_s0`, tag `lcr/E302.smoke_s0`); the code stays in git history (added in a61ef288) and in that tag. The card below is the record of the idea." + anchor, 1)
open(p, "w", encoding="utf-8").write(s); print("ok")
