p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = "order of building: twin + `wire_choice` path first, pilot, TRAIN features, head."
assert s.count(a) == 1
b = ("order of building: twin + `wire_choice` path first (DONE 2026-11-13: `training/grid/jwire.py`, `add_connectors(wire_choice=)`, `jends_eval.py score --snap --wire twin --tag twin` reproduces the E312 cards on both seeds exactly, `jtest_eval.py verify` seed 0 all reproduced; code critic pending/answered in the commit), "
     "then pilot, TRAIN features (`jwire_data.py`), head (GPU window).")
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("ok")
