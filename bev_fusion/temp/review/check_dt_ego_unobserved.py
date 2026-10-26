"""Review check (2026-10-26) on nuScenes mini: numbers behind three review findings.

1. keyframe gap: actual dt of the next / previous keyframe (camera timestamps) against the nominal +-0.5 s the heads decode
2. ego motion: a static point moved to the next / previous keyframe by (a) the exact pose transform, (b) splat_positions
   with the NOMINAL dt and the constant (v, omega) of the last camera step (what model.forward does today), (c) the same
   with the true dt and the true mean (v, omega) over the gap (ego_speed_yawrate of the two poses): position error in m
   at (20, 0), (40, 0), (40, 10) and (55, -10)
3. unobserved boxes: share of annotations of the evaluated groups with num_lidar_pts + num_radar_pts == 0
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/review/check_dt_ego_unobserved.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
from gt_bev import NuScenesGT, category_group, pose_matrix  # noqa: E402
from splat import ego_speed_yawrate, splat_positions  # noqa: E402

gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
nusc = gt.nusc
samples = [s for s in nusc.sample if s["prev"] and s["next"]]
pts = torch.tensor([[[20.0, 0.0], [40.0, 0.0], [40.0, 10.0], [55.0, -10.0]]], dtype=torch.float64)
dts, e_nom, e_true = [], {1: [], -1: []}, {1: [], -1: []}
for s in samples:
    track = gt.camera_track(s["token"])
    times = [t["timestamp"] for t in track]
    cur = gt.cam(s["token"])
    i0 = times.index(cur["timestamp"])
    last = gt.ego_steps(track[i0 - 1]["timestamp"], cur["timestamp"], track) if i0 > 0 else None
    if not last:
        continue
    v_last, w_last = last[-1][1], last[-1][2]
    Tc = gt.ego_pose(cur)
    for sign, key in ((1, "next"), (-1, "prev")):
        other = gt.cam(s[key])
        dt = (other["timestamp"] - cur["timestamp"]) * 1e-6
        dts.append(dt)
        To = gt.ego_pose(other)
        rel = np.linalg.inv(Tc) @ To                         # other ego pose in the current ego frame
        exact = np.linalg.inv(rel) @ np.stack([pts[0, :, 0].numpy(), pts[0, :, 1].numpy(), np.zeros(4), np.ones(4)])
        nom = splat_positions(pts, 0.5 * sign, v_last, w_last)[0].numpy()
        v_t, w_t = ego_speed_yawrate(torch.tensor(Tc), torch.tensor(To), dt)
        tru = splat_positions(pts, dt, float(v_t), float(w_t))[0].numpy()
        e_nom[sign].append(np.hypot(nom[:, 0] - exact[0], nom[:, 1] - exact[1]))
        e_true[sign].append(np.hypot(tru[:, 0] - exact[0], tru[:, 1] - exact[1]))
dts = np.array(dts)
print(f"samples {len(samples)}; keyframe gaps (s): min {np.abs(dts).min():.3f} median {np.median(np.abs(dts)):.3f} "
      f"max {np.abs(dts).max():.3f}; share outside 0.45..0.55 s: {(np.abs(np.abs(dts) - 0.5) > 0.05).mean():.2f}")
for sign, name in ((1, "next (+0.5 s)"), (-1, "prev (-0.5 s)")):
    a, b = np.array(e_nom[sign]), np.array(e_true[sign])
    print(f"{name}: static-point error (m) at (20,0) (40,0) (40,10) (55,-10)")
    print("   nominal dt + last-step (v, omega) : mean " + " ".join(f"{x:.2f}" for x in a.mean(0)) +
          " | 95th pct " + " ".join(f"{x:.2f}" for x in np.percentile(a, 95, 0)))
    print("   true dt + true mean (v, omega)    : mean " + " ".join(f"{x:.2f}" for x in b.mean(0)) +
          " | 95th pct " + " ".join(f"{x:.2f}" for x in np.percentile(b, 95, 0)))

n_all = n_unobs = 0
by_grp = {}
for s in nusc.sample:
    for atok in s["anns"]:
        a = nusc.get("sample_annotation", atok)
        g = category_group(a["category_name"])
        if g is None:
            continue
        n_all += 1
        un = a["num_lidar_pts"] + a["num_radar_pts"] == 0
        n_unobs += un
        t = by_grp.setdefault(g, [0, 0])
        t[0] += 1
        t[1] += un
print(f"annotations of the evaluated groups: {n_all}; without any lidar / radar point: {n_unobs} ({n_unobs / n_all:.1%}); "
      + ", ".join(f"{g}: {u}/{n}" for g, (n, u) in by_grp.items()))
