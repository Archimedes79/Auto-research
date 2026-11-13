"""scratch (E282 critic pass, read-only): what do trivial thin rows score on the card's zone metrics? (GT only, no model)
rows: sparse GT (must be 1 / 1 / 0), every 2nd / 3rd dense node along a piece (random phase), greedy NMS of the dense row at 3 m and 4 m (random order), and the same with anchors forced.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh null_rows.py
"""
import os, sys
from collections import Counter
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qtargets as QT
rng = np.random.RandomState(0)
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
acc = {}
def add(name, key, v):
    acc.setdefault(name, Counter())[key] += v
for i in range(0, len(ds), 6):
    k = ds.ks[i]; row = int(ds.bd.rows[k]); token = ds.tokens[row]
    g = ds.lg.load(token)
    nd = QT.node_table(g); ns = QT.node_table(g, sparse=(0.3, 7.5))
    gd, gs = nd["gid"], ns["gid"]; pieces = nd["piece"]; off = nd["off"]
    anchors = set()
    for p in np.unique(pieces):
        ids = np.where(pieces == p)[0]
        anchors.add(int(gd[ids[0]])); anchors.add(int(gd[ids[-1]]))
    anchors |= set(int(x) for x in gd[nd["ev"]])
    zone_of = {}
    for p in np.unique(pieces):
        ids = [int(x) for x in gd[pieces == p]]
        sp = sorted(int(x) for x in gs if off[p] <= x < off[p + 1])
        for gid in ids:
            d = min((abs(gid - s), s) for s in sp); s = d[1]
            zone_of[gid] = None if (gid != s and s in anchors) else s
    L = sum(float(np.linalg.norm(np.diff(np.asarray(p["pts"]), axis=0), axis=1).sum()) for p in g["pieces"])
    pos = {int(a): nd["pos"][m] for m, a in enumerate(gd)}
    def score(name, delivered):
        cnt = Counter(zone_of[v] for v in delivered if zone_of[v] is not None)
        off_n = sum(1 for v in delivered if zone_of[v] is None)
        zones = set(v for v in zone_of.values() if v is not None)
        fixed = [z for z in zones if z in anchors]; free = [z for z in zones if z not in anchors]
        add(name, "zones", len(zones)); add(name, "rec", sum(1 for z in zones if cnt[z] >= 1)); add(name, "exact", sum(1 for z in zones if cnt[z] == 1))
        add(name, "fixed", len(fixed)); add(name, "fixed_rec", sum(1 for z in fixed if cnt[z] >= 1))
        add(name, "free", len(free)); add(name, "free_rec", sum(1 for z in free if cnt[z] >= 1)); add(name, "free_exact", sum(1 for z in free if cnt[z] == 1))
        add(name, "deliv", len(delivered)); add(name, "off", off_n); add(name, "len", L)
    score("sparse_GT", [int(x) for x in gs])
    for kk in (2, 3):
        dl = []
        for p in np.unique(pieces):
            ids = [int(x) for x in gd[pieces == p]]
            ph = rng.randint(kk)
            dl += ids[ph::kk]
        score(f"every_{kk}nd_dense", dl)
    for r in (3.0, 4.0):
        order = list(gd); rng.shuffle(order)
        kept = []
        for v in order:
            if all(np.linalg.norm(pos[int(v)] - pos[u]) >= r for u in kept):
                kept.append(int(v))
        score(f"nms_{r}m_random_order", kept)
        # anchors first (events + piece ends always delivered), then the rest greedily
        kept = []
        for v in sorted(anchors):
            kept.append(v)
        for v in order:
            if int(v) in anchors: continue
            if all(np.linalg.norm(pos[int(v)] - pos[u]) >= r for u in kept):
                kept.append(int(v))
        score(f"nms_{r}m_anchors_first", kept)
    score("dense_all", [int(x) for x in gd])
for name, c in acc.items():
    print(f"{name:26s} per100m {100*c['deliv']/c['len']:5.1f} zone_recall {c['rec']/c['zones']:.3f} zone_exact {c['exact']/c['zones']:.3f} off_zone {c['off']/max(c['deliv'],1):.3f} | fixed_rec {c['fixed_rec']/c['fixed']:.3f} free_rec {c['free_rec']/c['free']:.3f} free_exact {c['free_exact']/c['free']:.3f}")
