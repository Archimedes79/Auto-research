"""Where is the detection AP lost? (2026-10-30, CPU, from the stored per-frame detections + GT of an evaluation run.)
Every number is computed with evaluation/metrics.DetMeter on (modified) copies of the stored detections, so the baseline row must
reproduce the stored metrics.json (layer-1 check). Oracles are applied in order (cumulative), TIDE style:
  base        the run as evaluated (AP over centre distances 0.5 / 1 / 2 / 4 m, mean = the headline per class)
  + loc       every detection whose greedy nearest same-group GT lies within 4 m (one detection per GT, by score) is moved onto that GT:
              the cost of centre precision (the metric asks 0.5 m at the tightest threshold)
  + dup/bg    detections left that match no GT within 4 m (background false positives, duplicates) are removed
  + misses    a detection with score 1 is added for every GT not matched within 4 m: what is left is the ranking error only (= 1.0)
Plus: recall of GT at the operating point (score >= 0.3, matched within 2 m) by range bin, size and motion.
Usage (bev_fusion folder): python temp/soa_gap/det_breakdown.py runs/<run>/eval/scene_records.pkl [dt]"""
import math
import pickle
import sys

import numpy as np

sys.path.insert(0, "evaluation")
from metrics import DetMeter, RANGE_BINS, in_front  # noqa: E402

path = sys.argv[1]
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
rec = pickle.load(open(path, "rb"))["det"]
frames = [fr for sc in rec.values() for fr in sc[DT]]
print(f"{path}: {len(rec)} scenes, {len(frames)} frames at dt {DT:+.1f}")
GROUPS = ("vehicle", "pedestrian", "two_wheeler")


def run_meter(framelist):
    m = DetMeter(classes=list(GROUPS))
    for dets, gts in framelist:
        m.add(dets, gts, dt=DT)
    r = m.result()
    return r


def ap_table(r, cls):
    """{thr: AP} for a class from DetMeter.result() (flat or nested: use the nested structure)"""
    node = r[round(DT, 3)] if round(DT, 3) in r else r[DT]
    c = node[cls]["all"]
    return {t: c["ap@%g" % t] for t in (0.5, 1.0, 2.0, 4.0)}


def snap(dets, gts, cls, radius=4.0):
    """greedy by score: move a detection onto its nearest unclaimed same-group GT within radius"""
    gs = [g for g in gts if g["group"] == cls and in_front(g)]
    used, out, moved = set(), [], 0
    for d in sorted([d for d in dets if d["group"] == cls], key=lambda d: -d["score"]):
        best, bd = None, radius
        for j, g in enumerate(gs):
            if j in used:
                continue
            dist = math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"])
            if dist < bd:
                best, bd = j, dist
        if best is None:
            out.append((d, None))
        else:
            used.add(best)
            g = gs[best]
            out.append((dict(d, cx=g["cx"], cy=g["cy"]), best))
            moved += 1
    return out, used, gs


def build(mode):
    out = []
    for dets, gts in frames:
        nd = []
        for cls in GROUPS:
            sn, used, gs = snap(dets, gts, cls)
            orig = sorted([d for d in dets if d["group"] == cls], key=lambda d: -d["score"])
            for (d_snapped, j), d0 in zip(sn, orig):
                if mode == "base":
                    nd.append(d0)
                elif mode == "loc":
                    nd.append(d_snapped if j is not None else d0)
                elif mode in ("locbg", "all"):
                    if j is not None:
                        nd.append(d_snapped)
            if mode == "all":
                for k, g in enumerate(gs):
                    if k not in used:
                        nd.append(dict(g, score=1.0, group=cls))
        out.append((nd, gts))
    return out


res = {}
for mode in ("base", "loc", "locbg", "all"):
    res[mode] = run_meter(build(mode) if mode != "base" else frames)
print()
print("%-12s" % "AP / class", *["%-30s" % m for m in ("base", "+ loc", "+ dup/bg removed", "+ misses added")])
for cls in GROUPS:
    row = []
    for mode in ("base", "loc", "locbg", "all"):
        t = ap_table(res[mode], cls)
        row.append("mean %.3f (%.3f/%.3f/%.3f/%.3f)" % (np.mean(list(t.values())), *t.values()))
    print("%-12s" % cls, *["%-30s" % r for r in row])

# recall at the operating point by range / size / motion
print("\nGT recall at the operating point (score >= 0.3; a hit = a same-group detection within 2 m; 'any' = within 4 m, any score >= 0.05)")
rows = {}
for dets, gts in frames:
    for g in gts:
        if not in_front(g) or g["group"] not in GROUPS:
            continue
        ds = [d for d in dets if d["group"] == g["group"]]
        dd = [(math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"]), d["score"]) for d in ds]
        hit = any(dist < 2.0 and s >= 0.3 for dist, s in dd)
        anyd = any(dist < 4.0 for dist, s in dd)
        low = any(dist < 2.0 and 0.05 <= s < 0.3 for dist, s in dd)
        b = next(i for i, (lo, hi) in enumerate(RANGE_BINS) if lo <= g["cx"] < hi)
        speed = math.hypot(g["vx"], g["vy"])
        size = g.get("length", 0.0) if g["group"] != "pedestrian" else 0.0
        keys = [(g["group"], "range %d-%d m" % RANGE_BINS[b]), (g["group"], "moving (> 1 m/s)" if speed > 1 else "static (<= 1 m/s)")]
        if g["group"] == "vehicle":
            keys.append((g["group"], "length < 4 m" if size < 4 else ("length 4-6 m" if size < 6 else "length >= 6 m")))
        for k in keys:
            r = rows.setdefault(k, [0, 0, 0, 0])
            r[0] += 1
            r[1] += hit
            r[2] += (low and not hit)
            r[3] += anyd
for k in sorted(rows):
    n, h, lo, a = rows[k]
    print("  %-11s %-20s n=%5d  recall@0.3 %.3f   only a low-score (0.05-0.3) hit within 2 m: %.3f   any detection within 4 m: %.3f" % (k[0], k[1], n, h / n, lo / n, a / n))

# false positives at the operating point
fp = {"duplicate of a matched GT / localisation (within 4 m of a GT)": 0, "background (no GT within 4 m)": 0}
nfp = ntp = 0
for dets, gts in frames:
    gs = [g for g in gts if in_front(g)]
    for d in dets:
        if d["score"] < 0.3 or d["group"] not in GROUPS or not in_front(d):
            continue
        near = [math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"]) for g in gs if g["group"] == d["group"]]
        if near and min(near) < 2.0:
            ntp += 1
        else:
            nfp += 1
            fp["duplicate of a matched GT / localisation (within 4 m of a GT)" if near and min(near) < 4.0 else "background (no GT within 4 m)"] += 1
print("\ndetections with score >= 0.3 in the front field of view: %d within 2 m of a same-group GT, %d not (%s)" % (
    ntp, nfp, "; ".join("%s: %d" % kv for kv in fp.items())))
