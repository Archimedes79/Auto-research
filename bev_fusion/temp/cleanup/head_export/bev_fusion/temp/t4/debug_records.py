"""One-off (t4_soa_front): print detections vs GT of the first frames of a records file (sanity)."""
import pickle
import sys
import math
r = pickle.load(open(sys.argv[1], "rb"))["det"]
n = 0
for scene, d in r.items():
    for dets, gt in d[0.0]:
        gt = [g for g in gt if 0 < g["cx"] < 55 and abs(math.atan2(g["cy"], g["cx"])) < 0.56]
        print(scene, "dets", len(dets), "gt in front", len(gt))
        for g in gt[:6]:
            near = min(dets, key=lambda b: math.hypot(b["cx"] - g["cx"], b["cy"] - g["cy"]), default=None)
            print("  gt", g["group"], round(g["cx"], 1), round(g["cy"], 1), "| nearest det", None if near is None else
                  (near["group"], round(near["cx"], 1), round(near["cy"], 1), round(near["score"], 2)))
        print("  top dets", [(b["group"], b["cls"], round(b["cx"], 1), round(b["cy"], 1), round(b["score"], 2)) for b in dets[:5]])
        n += 1
    if n >= 4:
        break
