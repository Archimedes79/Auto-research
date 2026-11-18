import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
a = "(card section 2: one guard missed by 0.0004, topo F1 -0.001 .. -0.005, pair precision lower; no re-run)"
assert a in s
s = s.replace(a, "(card section 2: one guard missed by 0.0004, topo F1 within the band, pair precision lower; no re-run)", 1)
a = "**E342** A installed (27fee60b), B (idle-GPU profile) open."
assert a in s
s = s.replace(a, "**E342** A installed, B open.", 1)
a = "E350 (E325 head on E324 s): card v1 written, critic running, then build."
s = s.replace(a, "E350 (E325 head on E324 s): card v1, critic running.", 1)
a = " (automatic operation, D86 - D100)"
assert a in s
s = s.replace(a, " (auto operation, D86 - D100)", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
