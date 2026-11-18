import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jmodel_cost.py")
s = open(p, encoding="utf-8").read()
old = "The ms per frame of the same blocks come from `temp/jr/timing_junction.py` on an idle GPU (batch 1, bf16); B's own stage times are in `runs/e273/<run>/timing.json`."
new = "The ms per frame of the same blocks come from `jtiming.py` (the successor of the scratch `temp/jr/timing_junction.py` of E330) on an idle GPU (batch 1, bf16); B's own stage times are in `runs/e273/<run>/timing.json`."
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
