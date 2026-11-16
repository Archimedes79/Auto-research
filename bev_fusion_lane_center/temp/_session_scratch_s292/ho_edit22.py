import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("transfer = card E350)", "transfer = card E350). **E348** NOT adopted (card section 4: dup_frac PRIMARY s1 +0.0109 vs +0.01, topo F1 in the band, sign flips between seeds)")
rep("; transfer = card E348 (smoke ok), chain day24 queued.", ".")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
