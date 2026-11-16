import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(card section 4: snapped +0.0077 / +0.0098, end-geometry read-out +0.0199 / +0.0212)", "(card section 4)")
rep("D86 weights: detection up, SIMPLE, LITTLE compute, E2E differentiable; the lane chats", "D86: detection up, SIMPLE, LITTLE compute, E2E; the lane chats")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
