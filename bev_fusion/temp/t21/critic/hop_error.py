"""critic: error of the recurrent hop (previous keyframe -> oldest frame of this window) built by model._step_to_oldest
(dt = gap dt - sum dt of the oldest frame's chain, v and omega = gap mean) against the exact chain of per-sweep poses. nuScenes mini, JSON only."""
import math, os, sys
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, ROOT + "/training")
sys.path.insert(0, ROOT)
from nuscenes_dataset import NuScenesGT
from lift_pd import Grid
import splat as S
import train as T

gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
grid = Grid()
pts = S.grid_points(grid, dtype=torch.float64)[None]
errs, errs_all, om_diff, nsw = [], [], [], []
for s in gt.nusc.sample:
    if not (s["prev"] and s["next"]):
        continue
    track = gt.camera_track(s["token"])
    cur = gt.cam(s["token"])
    i0 = [t["token"] for t in track].index(cur["token"])
    prev = gt.cam(s["prev"])
    ip = [t["token"] for t in track].index(prev["token"])
    for stride, ncam in ((2, 3), (4, 2)):
        io = i0 - stride * (ncam - 1)                              # the oldest frame of this window
        if io < ip:
            continue
        gap = gt.ego_steps(prev["timestamp"], cur["timestamp"], track)           # K' -> K (what clip_step composes)
        oldest = gt.ego_steps(track[io]["timestamp"], cur["timestamp"], track)   # oldest -> K (steps[-1])
        T_, v, w = T.gap_ego_motion(gap)
        used = sum(x[0] for x in oldest)
        hop_exact = gt.ego_steps(prev["timestamp"], track[io]["timestamp"], track) if io > ip else []
        if not hop_exact:
            continue
        approx = (T_ - used, v, w)
        e = (S.chain_positions(pts, [tuple(float(a) for a in h) for h in hop_exact]) - S.splat_positions(pts, *approx)).norm(dim=-1)
        errs.append(float(e.max()))
        errs_all.append(float(e.mean()))
        w_hop = sum(h[0] * h[2] for h in hop_exact) / sum(h[0] for h in hop_exact)
        om_diff.append(abs(w_hop - w))
        nsw.append((stride, len(gap), len(hop_exact), round(T_ - used, 3), round(sum(h[0] for h in hop_exact), 3)))
errs, errs_all, om_diff = map(np.array, (errs, errs_all, om_diff))
n = len(errs)
st = np.array([x[0] for x in nsw])
for stride in (2, 4):
    m = st == stride
    print(f"stride {stride}: n={m.sum()}  max-over-grid position error [m]: median {np.median(errs[m]):.3f} p95 {np.percentile(errs[m], 95):.3f} max {errs[m].max():.3f};"
          f" mean-over-grid median {np.median(errs_all[m]):.3f}; |w_hop - w_gap| p95 {np.percentile(om_diff[m], 95):.3f} rad/s max {om_diff[m].max():.3f}")
print("sweeps per gap / hop (stride 2):", sorted({(x[1], x[2]) for x in nsw if x[0] == 2}))
print("hop dt: true vs formula (stride 2) max abs diff:", max(abs(x[3] - x[4]) for x in nsw if x[0] == 2))
