import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("2. **E346 DONE** (section 5: 2 m passes, 1 m no gain, chain code rejected); E347 staged build (coder);", "2. E351 card (CUDA graph of the launch-bound forward, critic running); E347 staged build (coder); E346 done;")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
