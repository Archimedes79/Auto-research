p = "experiments/E317_connector-geometry-loss/experiment.md"
s = open(p, encoding="utf-8").read()
a = "**Token by the rule written above: TIE, kept**"
assert s.count(a) == 1
b = "**Which read-out is E317's primary one:** the SNAPPED read-out (`--snap --thr 0.1`) is the read-out of record and the one the rule judges (the E312 / E324 s comparison column); the unsnapped read-out is the differentiable one that is reported (D85); the snap-nogeom read-out is a DIAGNOSTIC that isolates the end geometry and is compared only with u' (E312 with the same wiring, 0.6706 / 0.6760), never with a snapped number of another card: its +0.0199 / +0.0212 and the snapped +0.0077 / +0.0098 are not comparable (a different baseline). **Token by the rule written above: TIE, kept**"
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
