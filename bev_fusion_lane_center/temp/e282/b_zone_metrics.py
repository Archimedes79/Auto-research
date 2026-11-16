"""scratch (E282 critic pass, read-only): the card's zone metrics for the stored dense baseline B (E275 dump, VAL, every 6th frame) at e >= 0.5 and with the hard close-node step of E275 at 1.5 / 1.75 / 2.0 m.
This is the reference that the card does not have: what an existing thin row scores on zone_recall / zone_exact / off_zone.
"""
import os, sys, pickle
from collections import Counter
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qtargets as QT
import qgraph_readout as RO
seed = sys.argv[1] if len(sys.argv) > 1 else "0"
dump = pickle.load(open(os.path.join(ROOT, "temp", "e275", f"pred_e273_p_s{seed}.pkl"), "rb"))
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
acc = {}
def add(name, key, v):
    acc.setdefault(name, Counter())[key] += v
for i in range(0, len(dump), 6):
    d = dump[i]
    token = ds.tokens[d["row"]]
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
            dd = min((abs(gid - s), s) for s in sp); s = dd[1]
            zone_of[gid] = None if (gid != s and s in anchors) else s
    cell_zone = {tuple(c): zone_of[int(a)] for a, c in zip(gd, nd["cell"])}
    zones = set(v for v in zone_of.values() if v is not None)
    fixed = [z for z in zones if z in anchors]; free = [z for z in zones if z not in anchors]
    L = sum(float(np.linalg.norm(np.diff(np.asarray(p["pts"]), axis=0), axis=1).sum()) for p in g["pieces"])
    sel = np.flatnonzero(d["e"] >= 0.5)
    pos, e = d["pos"][sel].astype(float), d["e"][sel].astype(float)
    cells = d["cell"][sel]
    variants = {"B plain e>=0.5": np.ones(len(sel), bool)}
    for R in (1.5, 1.75, 2.0):
        variants[f"B + hard step {R} m"] = RO.suppress_close_nodes(pos, e, R, 0.5)
    for name, keep in variants.items():
        deliv = [tuple(c) for c in cells[keep]]
        cnt = Counter(cell_zone.get(c, None) for c in deliv)
        off_n = sum(1 for c in deliv if cell_zone.get(c, None) is None)
        add(name, "zones", len(zones)); add(name, "rec", sum(1 for z in zones if cnt[z] >= 1)); add(name, "exact", sum(1 for z in zones if cnt[z] == 1))
        add(name, "fixed", len(fixed)); add(name, "fixed_rec", sum(1 for z in fixed if cnt[z] >= 1))
        add(name, "free", len(free)); add(name, "free_rec", sum(1 for z in free if cnt[z] >= 1)); add(name, "free_exact", sum(1 for z in free if cnt[z] == 1))
        add(name, "deliv", len(deliv)); add(name, "off", off_n); add(name, "len", L)
for name, c in acc.items():
    print(f"{name:22s} per100m {100*c['deliv']/c['len']:5.1f} zone_recall {c['rec']/c['zones']:.3f} zone_exact {c['exact']/c['zones']:.3f} off_zone {c['off']/max(c['deliv'],1):.3f} | fixed_rec {c['fixed_rec']/c['fixed']:.3f} free_rec {c['free_rec']/c['free']:.3f} free_exact {c['free_exact']/c['free']:.3f}")
# sanity: share of delivered B nodes (e >= 0.5) whose slot cell holds a dense GT node / whose position is within 2 m of a dense GT node
from scipy.spatial import cKDTree
tot = Counter()
for i in range(0, len(dump), 6):
    d = dump[i]
    g = ds.lg.load(ds.tokens[d["row"]])
    nd = QT.node_table(g)
    cells = set(map(tuple, nd["cell"]))
    sel = np.flatnonzero(d["e"] >= 0.5)
    tr = cKDTree(nd["pos"])
    for j in sel:
        tot["n"] += 1
        tot["in_gt_cell"] += int(tuple(d["cell"][j]) in cells)
        tot["within2m"] += int(tr.query(d["pos"][j])[0] <= 2.0)
        tot["within1m"] += int(tr.query(d["pos"][j])[0] <= 1.0)
print("B delivered nodes: slot cell holds a dense GT node", tot["in_gt_cell"] / tot["n"], " within 2 m of a dense GT node", tot["within2m"] / tot["n"], " within 1 m", tot["within1m"] / tot["n"])
