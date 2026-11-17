import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
a = "Results live in the cards (`experiments/E<id>_*/experiment.md`), `experiment_execution/LEDGER.md` and `decisions/`."
assert a in s
s = s.replace(a, "Results: cards (`experiments/E<id>_*/experiment.md`), `LEDGER.md`, `decisions/`.", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
