p = "bev_fusion/evaluation/README.md"
s = open(p, encoding="utf-8").read()
old = "`bev_eval.py` adds no logic to them (it calls `bev_adapter.build`, `bev_suite._task_only`, `bev_suite.contract_card`)."
assert s.count(old) == 1
new = old + (" A network older than D48 (the three of `records/keep3.json`: t8 / t10 / t11; their snapshot has no `amp_context` / `gap_steps`) cannot run without ground truth: "
             "`Evaluation.run` / `Runner.predict` refuse it with 'too old for the run without ground truth' (`Evaluation.check` says so); evaluate it with `bev_suite.py eval` or `evaluate.py`, "
             "which read the ground truth with it (the golden check does exactly that on `t1_r4_soa`).")
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
print("ok")
