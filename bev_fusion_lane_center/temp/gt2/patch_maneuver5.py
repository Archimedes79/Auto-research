p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()
old = "tuple(float(x) for x in sc_all[ip])"
assert old in s
s = s.replace(old, "tuple(float(x) if np.isfinite(x) else 1.0 for x in sc_all[ip])")
open(p, "w", encoding="utf-8").write(s)
