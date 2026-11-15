p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = "E292 layer arm a: large price"
assert s.count(a) == 1
s = s.replace(a, "**Stack E338** (E324 s heads on c2's graph, no retraining): 0.7066 / 0.7016 = +0.0242 / +0.0152 over E312, additive to the dot (all pre-registered numbers hit), TIE by its own rule (seed 1 D +0.0006: c2 adds little there). E292 layer arm a: large price")
b = "E338 stack check (E324 s heads on c2's graph; card written, critic pending, then 6 scorings, rule written before the numbers); "
assert s.count(b) == 1
s = s.replace(b, "")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(len(s.encode("utf-8")))
