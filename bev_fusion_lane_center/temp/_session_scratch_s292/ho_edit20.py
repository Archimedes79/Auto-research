import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("log row drafted (temp/_session_scratch_s292/e343_row.txt).", "log row drafted.")
rep("(vertices/100 m 4.9e-3 vs 3e-3; stack exact, residual = B's CPU dump)", "(vertices/100 m 4.9e-3 vs 3e-3)")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
