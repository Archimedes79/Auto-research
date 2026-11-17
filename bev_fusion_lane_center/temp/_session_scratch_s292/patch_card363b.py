import io
p = "experiments/E363_e2e-stack-on-faithful-graphs/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "`temp/e363/fixture.py` (the dump comparison of stop rule (iii))"
assert s.count(a) == 1, s.count(a)
s = s.replace(a, "`temp/e363/surgery_check.py` (the array checks of the stop rules (ii) - (v): the fixture on both seeds bit for bit, the node tensors, the provenance; it REPLACED the first `fixture.py`, atol 2e-3, seed 0 only), `temp/e363/faithful.py` (the layer-faithful dump of arm a and the node check of the stop rule (vi))")
s = s.replace("stop rules (i) - (v), the bounds", "stop rules (i) - (vii), the bounds")
io.open(p, "w", encoding="utf-8", newline="").write(s)
p2 = "temp/e363/verdict_e363.py"
t = io.open(p2, encoding="utf-8").read()
t = t.replace("temp/e363/fixture_result.json (written by fixture.py)", "temp/e363/surgery_check.json (surgery_check.py) and temp/e363/nodecheck_{a,c2}_s<S>.json (faithful.py check)")
io.open(p2, "w", encoding="utf-8").write(t)
print("ok")
