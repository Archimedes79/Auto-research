import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("the lead / Mirko decide (kill, or a recorded amendment; card section 2); log row drafted.", "HOLD: Mirko decides (a kill / b tolerance 1e-2, recorded); until his word no row, no grant, no TEST (card section 2); row text drafted.")
rep("1. E343: decision on the failed check, then log row + ONE TEST run;", "1. E343 HOLD (Mirko);")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
