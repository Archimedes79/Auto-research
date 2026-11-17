import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_seq_data as LS
import lane_tokcache as LC
import p16_data as PD

lane_gt, d = LD.lane_store()
pose = dict(np.load(f"{PD.D}/poses_val.npz"))
c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
row2i = {int(r): i for i, r in enumerate(c.rows)}
pairs = [(row2i[int(pose["prev"][r])], i) for i, r in enumerate(c.rows) if int(pose["prev"][r]) in row2i]
print("val pairs", len(pairs), "starts", len(c.rows) - len(pairs))
rng = np.random.default_rng(0)
sel = rng.choice(len(pairs), 120, replace=False)


def pts(token):
    g = lane_gt.load(token)
    return np.concatenate([np.asarray(p["pts"], float) for p in g["pieces"]]) if g["pieces"] else np.zeros((0, 2))


def stats(tag, x):
    x = np.asarray(x)
    print(f"  {tag:<26} n {len(x):6d} median {np.median(x):.3f} p90 {np.percentile(x, 90):.3f} p99 {np.percentile(x, 99):.3f}")


out = {k: [] for k in ("warp", "identity", "inverse", "transposed")}
tokn = {k: [] for k in ("warp", "identity")}
for s in sel:
    ip, ic = pairs[s]
    rp, rc = int(c.rows[ip]), int(c.rows[ic])
    Rm, tv = LS.rel_pose(pose["R"][rc], pose["t"][rc], pose["R"][rp], pose["t"][rp])
    P0, P1 = pts(c.tokens[ip]), pts(c.tokens[ic])
    if len(P0) == 0 or len(P1) == 0:
        continue
    tree = cKDTree(P1)
    for name, W in (("warp", P0 @ Rm.T + tv), ("identity", P0), ("inverse", P0 @ Rm + (-(Rm.T @ tv))), ("transposed", P0 @ Rm + tv)):
        inside = (np.abs(W) < 45).all(1)
        d_, _ = tree.query(W[inside])
        out[name].append(d_[d_ < 6.0])
    # B' tokens with e >= 0.5
    for name, rm, tm in (("warp", Rm, tv), ("identity", np.eye(2, dtype=np.float32), np.zeros(2, np.float32))):
        a = c.f["pos"][ip][(c.f["valid"][ip]) & (c.f["e_p"][ip] >= 0.5)].astype(float) @ rm.T + tm
        b = c.f["pos"][ic][(c.f["valid"][ic]) & (c.f["e_p"][ic] >= 0.5)].astype(float)
        a = a[(np.abs(a) < 45).all(1)]
        dd, _ = cKDTree(b).query(a)
        tokn[name].append(dd[dd < 6.0])
print("GT lane points of t-1 (warped) -> nearest GT point of t, those inside 45 m and within 6 m of one:")
for k, v in out.items():
    stats(k, np.concatenate(v))
print("B' tokens e >= 0.5 of t-1 (warped) -> nearest B' token e >= 0.5 of t:")
for k, v in tokn.items():
    stats(k, np.concatenate(v))
