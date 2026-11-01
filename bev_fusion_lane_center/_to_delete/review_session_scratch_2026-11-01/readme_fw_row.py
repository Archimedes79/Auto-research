import sys
old = "proposal: experiment ideas in one planner file, one file per experiment and one list (card as a block inside `experiment.md`, status derived from the ledger); not decided |"
new = "built 2026-11-01: the card is a block inside `experiment.md` (`experiment.py migrate`), the status of plan nodes comes from `plan.py status`; what was left out and why |"
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, (p, s.count(old))
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", p)
