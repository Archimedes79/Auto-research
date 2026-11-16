import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("2. E346 (CPU, scratch build running; no frozen file);", "2. **E346 DONE** (card section 5: B's 2 m code passes, 1 m buys nothing, chain code rejected 57 - 60 %); E347 staged build (coder);")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
