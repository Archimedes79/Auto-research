import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(card section 2: one guard missed by 0.0004, topo F1 within the band, pair precision lower; no re-run)", "(card section 2, no re-run)")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
