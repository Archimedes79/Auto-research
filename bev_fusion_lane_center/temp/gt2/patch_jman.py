p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()
old = "    arc = np.stack([-15 + 15 * np.cos(th), -15 + 15 * np.sin(th)], 1)\n    arc = np.stack("
assert old in s
s = s.replace(old, "    arc = np.stack(", 1)
open(p, "w", encoding="utf-8").write(s)
