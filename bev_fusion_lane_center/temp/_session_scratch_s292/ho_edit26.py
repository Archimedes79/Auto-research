import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("B measured (card section 5: launch bound, 11 syncs, E341 saves -1.1 ms in the maps head).", "B measured (card section 5: launch bound, E341 -1.1 ms maps head).")
rep("Results: cards (`experiments/E<id>_*/experiment.md`), `LEDGER.md`, `decisions/`.", "Results: cards, `LEDGER.md`, `decisions/`.")
rep(" **E334** NOT adopted (no re-run).", "")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
