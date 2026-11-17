import re
p = "training/README.md"
s = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(s) if l.startswith("| `grid/` |"))
row = ("| `lanegraph/` | the champion code of the system of record E324 s (D94): a clean re-implementation of the inference path "
       "(pyramid, B's node / link head and sections, junction maps, pair head with the order rule, attachment, the named hard read-out steps) "
       "that loads `runs/e273_p_s<S>` + `runs/e324_s_s<S>` unchanged and is bit-identical to the original modules on CPU; modules flat with the "
       "prefix `lg_`; see its [README](lanegraph/README.md) | `lg_model.py` (LaneGraphNet), `lg_readout.py`, `scripts/lg_infer.py`, "
       "`scripts/lg_eval.py`, `scripts/lg_timing.py`, `tests/run_lg_tests.py` |")
assert not any(l.startswith("| `lanegraph/` |") for l in s)
s.insert(i + 1, row)
open(p, "w", encoding="utf-8").write("\n".join(s))
q = "tools/maintenance/run_all_tests.sh"
t = open(q, encoding="utf-8").read().split("\n")
j = max(k for k, l in enumerate(t) if l.startswith("run "))
line = 'run "lanegraph champion code == original modules (exact, 24 frames x 2 seeds, read-out fuzz, golden digests; CPU, about 2 minutes)" lanegraph/tests/run_lg_tests.py'
assert line not in t
t.insert(j + 1, line)
open(q, "w", encoding="utf-8", newline="\n").write("\n".join(t))
print("ok", i + 2, j + 2)
