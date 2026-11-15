"""scratch (E282 critic pass, read-only): the card's zone metrics on the E276 arm s checkpoint (VAL, CPU), read at e >= 0.5 / 0.4 / 0.3 and with a greedy close-node step (rule, E275 style) on top."""
import os, sys
from collections import Counter
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qgraph as QG
import qtargets as QT
import qgraph_readout as RO
step = int(sys.argv[1]) if len(sys.argv) > 1 else 24
LD.QData.DEFAULT_SPARSE = (0.3, 7.5)
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
model = QG.QGraph(width=48)
model.load_state_dict(torch.load(os.path.join(ROOT, "runs", "e276_s_s0_p3", "model.pt"), map_location="cpu")["state"])
model.eval()
acc = {}
def add(name, key, v):
    acc.setdefault(name, Counter())[key] += v
cc = QT.centre_of(np.arange(50))
for i in range(0, len(ds), step):
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
            dd = min((abs(gid - s), s) for s in sp); s = dd[1]
            zone_of[gid] = None if (gid != s and s in anchors) else s
    cell_zone = {tuple(c): zone_of[int(a)] for a, c in zip(gd, nd["cell"])}
    zones = set(v for v in zone_of.values() if v is not None)
    fixed = [z for z in zones if z in anchors]; free = [z for z in zones if z not in anchors]
    L = sum(float(np.linalg.norm(np.diff(np.asarray(p["pts"]), axis=0), axis=1).sum()) for p in g["pieces"])
    x = torch.from_numpy(np.asarray(ds[i]["x"], np.float32))[None]
    with torch.no_grad():
        no = model.node_stage(model.pyr(x))
    e = no["e"][0].sigmoid().numpy(); pos = no["pos"][0].permute(1, 2, 0).numpy()
    for thr in (0.5, 0.4, 0.3):
        ii, jj = np.nonzero(e >= thr)
        P = pos[ii, jj].astype(float); E = e[ii, jj].astype(float)
        variants = {f"s e>={thr}": np.ones(len(ii), bool)}
        if thr == 0.3:
            for R in (2.5, 3.0):
                variants[f"s e>={thr} + greedy step {R} m"] = RO.suppress_close_nodes(P, E, R, thr) if len(ii) else np.zeros(0, bool)
        for name, keep in variants.items():
            deliv = [(int(a), int(b)) for a, b in zip(ii[keep], jj[keep])]
            cnt = Counter(cell_zone.get(c, None) for c in deliv)
            add(name, "zones", len(zones)); add(name, "rec", sum(1 for z in zones if cnt[z] >= 1)); add(name, "exact", sum(1 for z in zones if cnt[z] == 1))
            add(name, "fixed", len(fixed)); add(name, "fixed_rec", sum(1 for z in fixed if cnt[z] >= 1))
            add(name, "free", len(free)); add(name, "free_rec", sum(1 for z in free if cnt[z] >= 1)); add(name, "free_exact", sum(1 for z in free if cnt[z] == 1))
            add(name, "deliv", len(deliv)); add(name, "off", sum(1 for c in deliv if cell_zone.get(c, None) is None)); add(name, "len", L)
for name, c in acc.items():
    print(f"{name:34s} per100m {100*c['deliv']/c['len']:5.1f} zone_recall {c['rec']/c['zones']:.3f} zone_exact {c['exact']/c['zones']:.3f} off_zone {c['off']/max(c['deliv'],1):.3f} | fixed_rec {c['fixed_rec']/c['fixed']:.3f} free_rec {c['free_rec']/c['free']:.3f} free_exact {c['free_exact']/c['free']:.3f}")
