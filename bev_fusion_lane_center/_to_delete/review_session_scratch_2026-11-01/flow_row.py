import sys
old = "`loop.md` (P3 + P4, pasted after `/loop`) |"
new = "`loop.md` (P3 + P4, pasted after `/loop`), `experiment_flow.md` (one experiment in six steps: plan card, coding, experiment, training, evaluation, store to the result folder) |"
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, (p, s.count(old))
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", p)
