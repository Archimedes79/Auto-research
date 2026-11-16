import os, sys, time, numpy as np, torch
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD, qtargets as QT, qzones as QZ, qgraph as QG
S = (0.3, 7.5)
ds = LD.QData("val_clean", pixels=False, map_mode="exact", sparse=S, zone=True)
sel = list(range(0, len(ds), 8))[:110]
sizes, nfix = [], 0
t_nt = t_z = 0.0
for i in sel:
    g = ds.lg.load(ds.tokens[int(ds.bd.rows[ds.ks[i]])])
    t0 = time.time(); QT.node_table(g, sparse=S); t_nt += time.time() - t0
    t0 = time.time(); zs = QZ.zones(g, S); QZ.zone_maps(zs); t_z += time.time() - t0
    for z in range(zs["n_zones"]):
        if zs["kind"][z] == 1: sizes.append(len(zs["members"][z]))
        else: nfix += 1
sizes = np.array(sizes)
print("frames", len(sel), "free zones", len(sizes), "fixed", nfix, "size dist 1/2/3/4/5+", [int((sizes == k).sum()) for k in (1, 2, 3, 4)], int((sizes >= 5).sum()), "mean", sizes.mean().round(2))
print("random-uniform winner off-GT share over free zones:", float(np.mean(1 - 1.0 / sizes)).__round__(3))
print("time per frame: node_table(sparse) %.3f s ; zones+zone_maps %.3f s" % (t_nt / len(sel), t_z / len(sel)))
# real batch: random logits, winners == zones
items = [ds[i] for i in sel[:4]]
b = LD.collate(items)
e = torch.randn(4, 50, 50) * 0.5
ex, info = QG.zone_targets(e, b, 0.05)
print("winners", int(ex.sum()), "zones", [int(it['exists'].sum()) for it in items], "sum", sum(int(it['exists'].sum()) for it in items), "win_off_gt", float(info['win_off_gt']))
# winner cell is in its zone and ex is zero outside zones
assert float(ex[b["zone"] < 0].sum()) == 0
# per-item loading time with / without zone
ds0 = LD.QData("val_clean", pixels=False, map_mode="exact", sparse=S, zone=False)
t0 = time.time(); [ds0[i] for i in sel[:20]]; a = (time.time() - t0) / 20
t0 = time.time(); [ds[i] for i in sel[:20]]; c = (time.time() - t0) / 20
print("QData item: %.3f s without zones, %.3f s with zones" % (a, c))
