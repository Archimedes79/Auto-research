import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("Forward 17.09 ms (pyramid 4.55, maps 2.82, pair head 9.72). ", "Forward 17.09 ms. ")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
