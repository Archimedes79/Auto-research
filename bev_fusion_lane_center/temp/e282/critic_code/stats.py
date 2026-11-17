import os, sys, numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qtargets as QT, qzones as QZ
S = (0.3, 7.5)
ds = LD.QData("val_clean", pixels=False, map_mode="exact", sparse=S, zone=True)
n = len(ds)
sel = list(range(0, n, 8))
free_u = free_tot = anchors_u = anchors_tot = 0
free_s = free_f = 0
empty = 0
for i in sel[:110]:
    k = ds.ks[i]; token = ds.tokens[int(ds.bd.rows[k])]
    g = ds.lg.load(token)
    zs = QZ.zones(g, S)
    nt = QT.node_table(g, sparse=S)
    nd = zs["nd"]
    if zs["n_zones"] == 0:
        empty += 1; continue
    gd = nd["gid"]
    # sparse node m -> zone kind via gid
    gs = nt["gid"]
    kind_of = {int(gd[v]): zs["kind"][z] for z, v in enumerate(zs["gt_node"])}
    t = QT.grid_targets(nt)
    for m, gid in enumerate(gs):
        kd = kind_of[int(gid)]
        c = tuple(nt["cell"][m])
        u = t["uout"][c] + t["uin"][c]
        s = t["sstart"][c]
        f = t["flags"][:, c[0], c[1]].sum()
        if kd == 1:
            free_tot += 1; free_u += int(u > 0); free_s += int(s > 0); free_f += int(f > 0)
        else:
            anchors_tot += 1; anchors_u += int(u > 0)
print("frames", len(sel[:110]), "empty", empty, "free zones", free_tot, "free with uout/uin", free_u, "free with sstart", free_s, "free with flags", free_f, "anchors", anchors_tot, "anchors with u", anchors_u)
