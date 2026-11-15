"""Geometry check 2: only cells where both teachers are informative (entropy < 3.3 nats), mode depth, MAE of log mode-depth against an extra shift of H8 cells."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
sys.path.insert(0, "training")
from feature_cache import FeatureStore
import splits
from lane_branch import image_path
s16, s8 = FeatureStore("data/cache/cam_feats"), FeatureStore("data/cache/cam_feats_bevfusion")
shard_scene = [os.path.basename(p)[:-4] for p in s16.shards]
sp = splits.load("large")
toks = sorted(t for t, (sid, _) in s16.index.items() if shard_scene[sid] in set(sp["val"]) and os.path.exists(image_path("data/cache/img_front", t)))
toks = toks[::max(1, len(toks) // 400)][:400]
c16 = 1.25 + 0.5 * torch.arange(118); c8 = 1.0 + 0.5 * torch.arange(118)
def stats(st, c):
    P = torch.stack([st.get(t)[0].float() for t in toks])
    ent = -(P.clamp_min(1e-8) * P.clamp_min(1e-8).log()).sum(1)
    mode = c[P.argmax(1)]
    return mode.log(), ent
L16, H16 = stats(s16, c16); L8, H8 = stats(s8, c8)
print("H16 entropy per row", [round(float(x), 2) for x in H16.mean((0, 2))])
print("H8 entropy per row ", [round(float(x), 2) for x in H8.mean((0, 2))])
yy, xx = torch.meshgrid(16 * torch.arange(16) + 7.5, 16 * torch.arange(44) + 7.5, indexing="ij")
def nearest(M, j, i, dj=0., di=0.):
    jj = (j + dj).round().clamp(0, 87).long(); ii = (i + di).round().clamp(0, 31).long()
    return M[:, ii, jj]
def score(j, i, mask, dj=0., di=0., rows=range(6, 12)):
    s, e = nearest(L8, j, i, dj, di), nearest(H8, j, i, dj, di)
    sel = mask[None] & (H16 < 3.3) & (e < 3.3)
    sel = sel & torch.isin(torch.arange(16).view(1, 16, 1).expand_as(sel), torch.tensor(list(rows)))
    d = (s - L16)[sel]
    return round(float(d.abs().mean()), 4), round(float(d.mean()), 4), int(sel.sum())
r = 0.44 / 0.48
xb, yb = xx / r - 32, (yy + 140) / r - 176
j, i = xb * 87 / 703, yb * 31 / 255
mask = (j >= 0) & (j <= 87) & (i >= 0) & (i <= 31)
print("claimed (0,0): (MAE, bias, n)", score(j, i, mask))
for name, dj, di in [("dj=-2", -2, 0), ("dj=-1", -1, 0), ("dj=+1", 1, 0), ("dj=+2", 2, 0), ("di=-2", 0, -2), ("di=-1", 0, -1), ("di=+1", 0, 1), ("di=+2", 0, 2), ("di=+3", 0, 3), ("di=-3", 0, -3)]:
    print(name, score(j, i, mask, dj, di))
jid, iid = (xx / 8 - 3.5 / 8), (yy / 8 - 3.5 / 8)
print("identity window:", score(jid, iid, torch.ones_like(mask)))
r2 = 0.48 / 0.44
xb2, yb2 = xx / r2 - 32, (yy + 140) / r2 - 176
j2, i2 = xb2 * 87 / 703, yb2 * 31 / 255
print("inverted ratio:", score(j2, i2, (j2 >= 0) & (j2 <= 87) & (i2 >= 0) & (i2 <= 31)))
# x only: column-wise check (does the horizontal map hold?): same with columns restricted to left / right thirds, claimed vs dj=+-1
for nm, cols in (("left", range(0, 15)), ("mid", range(15, 30)), ("right", range(30, 44))):
    m = mask.clone(); m[:, [c for c in range(44) if c not in cols]] = False
    print(nm, {dj: score(j, i, m, dj, 0)[0] for dj in (-2, -1, 0, 1, 2)})
