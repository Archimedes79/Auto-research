"""scratch: why does a stop / exit event have no intersection polygon within 4 m"""
import os, sys
import numpy as np, shapely
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(120)
rows = []
for k, g in enumerate(fr):
    inter = QR._polygons(g["areas"]["intersection"], 0.0)
    un = shapely.union_all(inter) if inter else None
    road = shapely.union_all(QR._polygons(g["areas"]["road"], 0.0)) if g["areas"]["road"] else None
    pid = {p["id"]: p for p in g["pieces"]}
    for e in g["events"]:
        if e["type"] not in ("stop", "exit"):
            continue
        pos = np.asarray(e["pos"], float)
        pt = shapely.Point(*pos)
        d = float(un.distance(pt)) if un is not None else np.inf
        dr = float(road.distance(pt)) if road is not None else np.inf
        border = 50 - np.abs(pos).max()
        # the connector side: out of a stop / in of an exit; first point of that piece inside any intersection polygon?
        side = e["out"] if e["type"] == "stop" else e["in"]
        con = [pid[i] for i in side if i in pid]
        cons_in = [any(s["in_intersection"] for s in c["segments"]) for c in con]
        rows.append((k, e["type"], d, dr, border, len(con), sum(cons_in), e.get("partial"), e.get("n_succ", e.get("n_pred")), len(e["out"]) if e["type"] == "stop" else len(e["in"])))
R = np.array([(r[2], r[3], r[4]) for r in rows])
print("events", len(rows), "dist poly>4:", int((R[:, 0] > 4).sum()), "road dist > 1:", int((R[:, 1] > 1).sum()))
bad = [r for r in rows if r[2] > 4]
print("bad: share near window border (<5 m):", np.mean([r[4] < 5 for r in bad]), " by type:", {t: sum(1 for r in bad if r[1] == t) for t in ("stop", "exit")}, "of", {t: sum(1 for r in rows if r[1] == t) for t in ("stop", "exit")})
print("bad: partial flag:", {str(p): sum(1 for r in bad if r[7] == p) for p in (True, False, None)}, " connector in_intersection among the connected pieces:", np.mean([r[6] > 0 for r in bad]), "n_con==0:", np.mean([r[5] == 0 for r in bad]))
print("bad: road distance (m) pct 50/90:", np.percentile([r[3] for r in bad], [50, 90]), "dist to intersection pct 10/50/90:", np.percentile([r[2] for r in bad], [10, 50, 90]))
print("bad: border distance pct 10/50/90:", np.percentile([r[4] for r in bad], [10, 50, 90]))
print("all  : dist to intersection pct 50/80/90/99:", np.percentile(R[:, 0], [50, 80, 90, 99]))
print("good (<=4) with in_intersection connectors:", np.mean([r[6] > 0 for r in rows if r[2] <= 4]))
from collections import Counter
print("bad frames:", Counter(r[0] for r in bad).most_common(6))
print([ (r[0], r[1], round(r[2], 1), round(r[4], 1), r[5], r[6], r[7], r[8], r[9]) for r in bad[:25]])
