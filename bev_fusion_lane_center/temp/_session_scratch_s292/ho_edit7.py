import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(learned wiring head, +0.0023 / +0.0032, trained on stored records, output still an argmax; next card: the head on E324 s / c2)", "(learned wiring head, +0.0023 / +0.0032, output still an argmax; transfer = card E350)")
rep("(unsnapped s0 +0.0000, s1 -0.0080, 0.002 inside the band; saves 1.477 GMACs)", "(saves 1.477 GMACs)")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
