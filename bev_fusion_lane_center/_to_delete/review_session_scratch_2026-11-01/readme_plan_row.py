import sys
old = "`ledger.py`, `schedule.py`, `decide.py`"
new = "`ledger.py`, `schedule.py`, `plan.py` (status of the plan nodes from the schedule and the ledger), `decide.py`"
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, (p, s.count(old))
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", p)
