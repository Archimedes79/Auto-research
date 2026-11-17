import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(card section 4: dup_frac PRIMARY s1 +0.0109 vs +0.01, topo F1 in the band, sign flips between seeds)", "(card section 4: one guard by 0.0009)")
rep(" **E334** NOT adopted (card section 2, no re-run).", " **E334** NOT adopted (no re-run).")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
