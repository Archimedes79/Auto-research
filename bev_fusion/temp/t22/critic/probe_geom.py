"""Empirical check of the H8 <-> JPEG-window mapping on the two TEACHER caches (no student): E[depth] of H16 (DualBEV window, cell centres) against E[depth] of H8 (BEVFusion
frustum cells) sampled at the position the claimed transform gives. Error as a function of an extra shift (cells of H8) and for wrong alternatives."""
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
toks = toks[::max(1, len(toks) // 300)][:300]
c16 = 1.25 + 0.5 * torch.arange(118); c8 = 1.0 + 0.5 * torch.arange(118)
E16 = torch.stack([(s16.get(t)[0].float() * c16.view(118, 1, 1)).sum(0) for t in toks])   # (N,16,44)
E8 = torch.stack([(s8.get(t)[0].float() * c8.view(118, 1, 1)).sum(0) for t in toks])      # (N,32,88)
L16, L8 = E16.log(), E8.log()
yy, xx = torch.meshgrid(16 * torch.arange(16) + 7.5, 16 * torch.arange(44) + 7.5, indexing="ij")   # DualBEV-window pixels of H16 cells
def sample(j, i, dj=0.0, di=0.0):
    """L8 at fractional frustum index (i, j) (+ shift in cells), border clamp; returns (N,16,44) and the in-window mask"""
    jj, ii = j + dj, i + di
    g = torch.stack([2 * jj / 87 - 1, 2 * ii / 31 - 1], -1)[None].expand(len(toks), -1, -1, -1)
    return F.grid_sample(L8[:, None], g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
def report(name, j, i, mask, dj=0.0, di=0.0):
    s = sample(j, i, dj, di)
    # use only rows that are not sky: rows 6..15 of H16 (below the horizon) and in-window cells
    m = mask & (torch.arange(16).view(16, 1) >= 7)
    d = (s - L16)[:, m]
    return float(d.abs().mean()), float(np.corrcoef(s[:, m].flatten().numpy(), L16[:, m].flatten().numpy())[0, 1])
# claimed: x_b = x_d / (0.44/0.48) - 32 ; y_b = (y_d + 140) / (0.44/0.48) - 176 ; j = x_b * 87/703 ; i = y_b * 31/255
r = 0.44 / 0.48
xb, yb = xx / r - 32, (yy + 140) / r - 176
j, i = xb * 87 / 703, yb * 31 / 255
mask = (j >= 0) & (j <= 87) & (i >= 0) & (i <= 31)
print("claimed mapping, cells inside the BEVFusion window:", int(mask.sum()), "of", mask.numel())
res = {}
for di in (-2, -1, -0.5, 0, 0.5, 1, 2):
    for dj in (-2, -1, -0.5, 0, 0.5, 1, 2):
        res[(di, dj)] = report("", j, i, mask, dj, di)
best = min(res, key=lambda k: res[k][0])
print("MAE(log depth) at shift (di, dj) in H8 cells; claimed (0,0):", res[(0, 0)], " best:", best, res[best])
print("row of shifts di=0:", {dj: round(res[(0, dj)][0], 4) for dj in (-2, -1, -0.5, 0, 0.5, 1, 2)})
print("col of shifts dj=0:", {di: round(res[(di, 0)][0], 4) for di in (-2, -1, -0.5, 0, 0.5, 1, 2)})
# wrong alternatives: identity (cell k of H16 -> cell 2k+0.5 of H8, i.e. same window), swapped roles of the two scales
jid, iid = (xx / 8 - 3.5 / 8), (yy / 8 - 3.5 / 8)           # same pixel in the SAME window, native stride-8 centres
print("identity window (BEVFusion window == JPEG window):", report("", jid, iid, torch.ones_like(mask)))
r2 = 0.48 / 0.44                                              # inverted direction
xb2, yb2 = xx / r2 - 32, (yy + 140) / r2 - 176
print("inverted ratio:", report("", xb2 * 87 / 703, yb2 * 31 / 255, (xb2 * 87 / 703 >= 0) & (xb2 * 87 / 703 <= 87) & (yb2 * 31 / 255 >= 0) & (yb2 * 31 / 255 <= 31)))
# native-centre variant (cell centre 8k+3.5 in the BEVFusion window as the teacher's content location)
jc, ic = (xb - 3.5) / 8, (yb - 3.5) / 8
print("content at 8k+3.5 (RF-centre) instead of the frustum position:", report("", jc, ic, mask))
