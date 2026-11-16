import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("2. E350 phase A committed (10518e70), phase B after E343's TEST; E345 results; D97; dead code; E318 / E319 / E348 results; E349.",
    "2. E346 (CPU, scratch build running; no frozen file); E318 / E348 results; 3. WAIT for E343's TEST end (freeze): E349, E350 phase B, E347 (staged copies in temp/e347, install after), D97, dead code; E345 results.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
