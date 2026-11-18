import os, sys
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_tokcache as LC
for name in ("be292_a_0_K512_9ad1d4ce30_v1/val", "be292_a_0_K512_9ad1d4ce30_v1/train"):
    c = LC.TokCache(os.path.join(LC.CACHE, name))
    f = c.f
    n = len(c)
    print(name, n, "done", c.done)
    print(" n_e50_in==n_deliv:", bool((np.array(f["n_e50_in"]) == np.array(f["n_deliv"])).all()))
    ep = np.array(f["e_p"]); va = np.array(f["valid"])
    print(" valid == e_p>0.05:", bool((va == (ep > 0.05)).all()), " e_p sorted desc:", bool((np.diff(ep, axis=1) <= 0).all()))
    cell = np.array(f["cell"]).astype(int); pos = np.array(f["pos"]); off = np.array(f["off"])
    cx = -50 + (cell // 50 + 0.5) * 2.0; cy = -50 + (cell % 50 + 0.5) * 2.0
    err = np.abs(pos - np.stack([cx, cy], -1) - off).max()
    print(" |pos - (cell centre + off)| max:", err)
    gid = np.array(f["gt_id"]); gt_type = np.array(f["gt_type"])
    print(" gt_type>0 iff gt_id>=0:", bool(((gt_type > 0) == (gid >= 0)).all()), " gt_id on invalid tokens:", int(((gid >= 0) & ~va).sum()))
    dup = 0
    for i in range(n):
        g = gid[i][gid[i] >= 0]; dup += len(g) - len(set(g.tolist()))
    print(" duplicate GT assignment:", dup)
    yp = np.array(f["ypair"]).astype(int); ok = yp[..., 0] >= 0
    bad = 0
    for i in range(n):
        k = yp[i][ok[i]]
        bad += int((~va[i][k[:, 0]]).sum() + (~va[i][k[:, 1]]).sum() + (k[:, 0] == k[:, 1]).sum() + (gid[i][k[:, 0]] < 0).sum())
    print(" pair endpoint invalid / self pair / source unassigned:", bad)
