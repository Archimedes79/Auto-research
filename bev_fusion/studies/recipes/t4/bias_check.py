"""One-off (t4_soa_front): signed centre offset (det - GT) of confident detections matched to the nearest GT within 2 m, per system."""
import math
import pickle
import sys

import numpy as np

for lab, p in (a.split("=", 1) for a in sys.argv[1:]):
    r = pickle.load(open(p, "rb"))["det"]
    d = {"vehicle": [], "pedestrian": []}
    for s in r.values():
        for dets, gt in s[0.0]:
            for b in dets:
                if b["score"] < 0.3 or b["group"] not in d or not (0 < b["cx"] < 50):
                    continue
                g = min((g for g in gt if g["group"] == b["group"]), key=lambda g: math.hypot(g["cx"] - b["cx"], g["cy"] - b["cy"]), default=None)
                if g and math.hypot(g["cx"] - b["cx"], g["cy"] - b["cy"]) < 2.0:
                    d[b["group"]].append((b["cx"] - g["cx"], b["cy"] - g["cy"]))
    for c, v in d.items():
        v = np.array(v)
        print(lab, c, "n", len(v), "median dx %.2f dy %.2f | mean dx %.2f dy %.2f | MAE %.2f" % (np.median(v[:, 0]), np.median(v[:, 1]), v[:, 0].mean(), v[:, 1].mean(), np.hypot(v[:, 0], v[:, 1]).mean()))
