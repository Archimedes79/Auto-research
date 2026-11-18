import io
p = "temp/report/own_lane_note.md"
s = io.open(p, encoding="utf-8").read()
a = ": the far range stays about 0.17 below the near range's quality of the same graphs; no system closes it."
assert a in s
s = s.replace(a, "; the far range gains about 0.07 - 0.08 recall from B to the junction systems, but the numbers stay below 0.77 (a near-range comparison was not made here).")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
p = "HANDOVER.md"
t = io.open(p, encoding="utf-8").read()
b = "E291 v2 critic REVISE (v3 to write); report analyses in temp/report/ (own_lane);"
assert b in t
t = t.replace(b, "E291 v2 critic REVISE; report analyses temp/report/;", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
