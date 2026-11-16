import io
p = "experiments/E341_evidence-without-paint-branch/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "less than the 1.3 ms estimated, because the forward is launch bound and the removed branch is mostly arithmetic."
assert s.count(a) == 1
s = s.replace(a, "less than the 1.3 ms estimated (a reading, not tested: the forward is launch bound and the removed branch is mostly arithmetic).")
io.open(p, "w", encoding="utf-8", newline="").write(s)
p = "HANDOVER.md"
t = io.open(p, encoding="utf-8").read()
b = "**E342** A installed, B open."
assert b in t
t = t.replace(b, "**E342** A installed, B measured (card section 5: launch bound, 11 syncs, E341 saves -1.1 ms in the maps head).")
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
