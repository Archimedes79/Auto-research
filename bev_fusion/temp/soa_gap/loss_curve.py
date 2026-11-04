import json, sys
import numpy as np
raw = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in raw if "group.areas" in r and "step" in r]
val = [r for r in raw if "val_total" in r or any(k.startswith("val") for k in r)]
steps = [r["step"] for r in rows]
print(sys.argv[1], "train rows", len(rows), "steps", min(steps), "-", max(steps), "| other keys in non-train rows:", sorted({k for r in raw if "group.areas" not in r for k in r})[:12])
last = max(steps)
wins = [(1, 200), (1000, 1400), (2000, 2400), (3000, 3400), (4000, 4400), (5000, 5400), (6000, 6400), (last - 400, last)]
print("%-12s" % "steps", *["%9s" % g for g in ("areas", "lines", "objects", "motion", "total")])
for a, b in wins:
    sel = [r for r in rows if a <= r["step"] <= b]
    if sel:
        print("%-12s" % f"{a}-{b}", *["%9.3f" % np.mean([r[k] for r in sel]) for k in ("group.areas", "group.lines", "group.objects", "group.motion", "total")])
