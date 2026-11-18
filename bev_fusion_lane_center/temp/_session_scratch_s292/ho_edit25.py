import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(card section 5: B's 2 m code passes, 1 m buys nothing, chain code rejected 57 - 60 %)", "(section 5: 2 m passes, 1 m no gain, chain code rejected)")
rep("**E344** plan; E346 / E347 = plans until Mirko says build.", "**E344** plan; E347 = plan.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
