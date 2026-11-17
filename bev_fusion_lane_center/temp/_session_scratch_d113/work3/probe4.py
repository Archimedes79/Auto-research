import os, sys
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tokcache as LC
c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
lane_gt, _ = LD.lane_store()
n = 0
for i in range(0, 873, 30):
    g = lane_gt.load(c.tokens[i])
    for p in g["pieces"]:
        if any(s.get("kind") == "connector" for s in p["segments"]) or p["kind"] != "lane":
            print(i, p["id"], p["kind"], len(p["pts"]), p["segments"][:4], {k: (type(v).__name__) for k, v in p["segments"][0].items()} if p["segments"] else None)
            n += 1
            break
    if n >= 4:
        break
g = lane_gt.load(c.tokens[300])
print(sorted(g["pieces"][0].keys()))
for p in g["pieces"][:6]:
    print(p["id"], p["kind"], len(p["pts"]), len(p["segments"]), p["segments"][:3])
import collections
kinds = collections.Counter()
keys = collections.Counter()
for i in range(0, 873, 20):
    g = lane_gt.load(c.tokens[i])
    for p in g["pieces"]:
        for s in p["segments"]:
            kinds[(s.get("kind"), bool(s.get("in_intersection")))] += 1
            keys[tuple(sorted(s.keys()))] += 1
print(kinds); print(keys)
