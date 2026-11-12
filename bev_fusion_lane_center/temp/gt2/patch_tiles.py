p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = '    t.append(("good", "TIE", "Matrix-NMS = the hard close-node step", "differentiable: R 1.75, sigma 0.1"))'
assert old in s
s = s.replace(old, '    t.append(("good", "+0.030", "topo F1 over the plain graph (E305, snapped connector ends, p >= 0.1, seed 0; seed 1 +0.029)", "the same pairs drawn from the keypoints at p >= 0.5: +0.007 (E311)"))\n' + old, 1)
open(p, "w", encoding="utf-8").write(s)
print("tiles ok")
