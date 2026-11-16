import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
a = "E350 card (E325 head on E324 s)."
assert a in s
s = s.replace(a, "E350 (E325 head on E324 s): card v1 written, critic running, then build.", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
