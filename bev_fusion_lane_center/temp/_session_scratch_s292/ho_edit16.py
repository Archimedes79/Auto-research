import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("D96 holds lifted. D97 old loop 8-18 chain removed in D89's order.", "D96 holds lifted. D97 old loop 8-18 chain removal.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
