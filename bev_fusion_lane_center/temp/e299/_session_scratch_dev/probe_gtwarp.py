import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(HERE, "training", "data"))
for s in ("training/grid", "training/data", "training/data/prep", "evaluation", ".."):
    sys.path.append(os.path.join(ROOT, s))
import lane_gt_data as LD
torch.set_num_threads(2)
ds = LD.QData("val_clean", pixels=False, prev=True, ends=True, map_mode="exact")
z = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
hit = tot = 0; hit_wrong = 0
for i in range(1, 120):
    row = int(ds.bd.rows[ds.ks[i]]); prow = int(z["prev"][row])
    if prow < 0 or prow not in ds.bd.row2k: continue
    cur, prv = ds[i], ds[ds.ks.index(ds.bd.row2k[prow])]
    R, t = LD.rel_pose(z["R"][row], z["t"][row], z["R"][prow], z["t"][prow])
    for key in ("ent", "ext"):
        P = prv[key].numpy(); C = cur[key].numpy()
        if not len(P): continue
        pos = P[:, :2] @ R.T + t; d = P[:, 2:] @ R.T
        # in view of both frames: inside |x|<35, 0<y<35 of current frame
        ok = (np.abs(pos[:, 0]) < 35) & (pos[:, 1] > -10) & (pos[:, 1] < 35)
        for p, dd in zip(pos[ok], d[ok]):
            tot += 1
            if len(C):
                dist = np.linalg.norm(C[:, :2] - p, axis=1); ang = C[:, 2:] @ dd
                hit += bool(((dist < 1.0) & (ang > np.cos(np.radians(30)))).any())
        pw = P[:, :2]  # no warp
        for p in pw[ok]:
            if len(C): hit_wrong += bool((np.linalg.norm(C[:, :2] - p, axis=1) < 1.0).any())
print("ends in view", tot, "hit", hit, hit / max(tot, 1), "wrong-dir hit", hit_wrong / max(tot, 1))
