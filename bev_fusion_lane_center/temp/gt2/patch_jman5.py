p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()
old = "    crowd = [line(1.2)] + [line(0.05 * k, rev=True) for k in range(1, 9)] + [np.stack([np.zeros(41), np.linspace(-10, 10, 41)], 1)]"
assert old in s
s = s.replace(old, "    crowd = [line(0.05 * k, rev=True) for k in range(1, 9)] + [np.stack([np.zeros(41), np.linspace(-10, 10, 41)], 1)] + [line(1.2)]          # the right line comes LAST in the point order", 1)
open(p, "w", encoding="utf-8").write(s)
