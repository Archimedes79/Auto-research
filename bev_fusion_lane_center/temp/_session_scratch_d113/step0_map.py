import os
import sys

import numpy as np
from scipy.spatial import cKDTree

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tok_data as LT
import lane_tokcache as LC
import map_sim as MS

mode = sys.argv[1] if len(sys.argv) > 1 else "error"
seed = "0"
c = LC.TokCache(os.path.join(R, f"runs/e365_cache/be292_a_{seed}_K512_9ad1d4ce30_v1/val"))
ds = LT.TokData("val", res=2.0, feats=False)
lane_gt, _ = LD.lane_store()
RAD, COS = 1.5, 0.5
acc = {}


def add(k, v):
    acc[k] = acc.get(k, 0) + v


dists = {k: [] for k in ("novel", "dup", "false", "deliv_true", "deliv_false")}
for i in range(len(c)):
    tok = c.tokens[i]
    g = lane_gt.load(tok)
    mp = MS.make_map(g, mode, rng=np.random.RandomState(0) if mode == "train" else None, token=tok)
    if len(mp["pos"]) == 0:
        add("nomap", 1)
        continue
    tree = cKDTree(mp["pos"])
    t = ds.table(i)
    gpos, gdir = t["pos"], t["dirv"]
    pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
    deliv = valid & (ep >= 0.5)
    weak = valid & ~deliv
    if len(gpos) == 0:
        continue
    near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
    found = (near & deliv[:, None]).any(0)
    miss = ~found
    d, j = tree.query(pos)
    tan = mp["tan"][j]
    align = np.abs((dirv * tan).sum(1))                                   # direction of the token against the map tangent (either way: a road is two-way)
    for name, sel in (("novel", weak & (near & miss[None]).any(1)), ("dup", weak & near.any(1) & ~(near & miss[None]).any(1)), ("false", weak & ~near.any(1)),
                      ("deliv_true", deliv & near.any(1)), ("deliv_false", deliv & ~near.any(1))):
        dists[name].append(np.stack([d[sel], align[sel]], 1))
print("map mode", mode, "frames", len(c), "without map", acc.get("nomap", 0))
print("token class | n | median distance to the nearest map sample (m) | within 3 m | within 3 m and |cos| >= 0.8 with the map tangent")
for k, v in dists.items():
    a = np.concatenate(v)
    print("  %-12s %7d | %.2f | %.3f | %.3f" % (k, len(a), np.median(a[:, 0]), (a[:, 0] <= 3).mean(), ((a[:, 0] <= 3) & (a[:, 1] >= 0.8)).mean()))
