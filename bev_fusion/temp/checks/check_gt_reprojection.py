"""GT check of plan 9.3: do the projected divider lines sit on the painted markings?

For keyframes of nuScenes mini: the GT divider cells (lane and road dividers) at 6-30 m are projected into the front
image at their position and shifted sideways (ego y) by -1.5..+1.5 m; the mean brightness (paint is brighter than
asphalt) along the projected points is measured per shift. If the GT and the transforms are right, the profile peaks
at 0 (a constant offset would show as a shifted peak, a wrong frame or time as no peak at all).
The shift is applied to the BEV positions in the ego frame (y), the lines are sampled densely along their cells.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_gt_reprojection.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from lift_pd import Grid  # noqa: E402
from view_training_samples import project  # noqa: E402

G = Grid(res=0.25)                                   # finer cells: less quantisation of the line positions
SHIFTS = np.arange(-1.5, 1.51, 0.25)
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")
gx, gy = cell_centres(G)
profiles = []
for sc in src.nusc.scene:
    tok = sc["first_sample_token"]
    for k in range(40):
        s = src.nusc.get("sample", tok)
        if k % 8 == 4:
            gt = src.keyframe_gt(tok, G)
            img = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("L"), np.float32)
            H, W = img.shape
            m = ((gt["lane_divider"] > 0.5) | (gt["road_divider"] > 0.5)) & (gx > 6) & (gx < 20)
            if m.sum() < 20:
                tok = s["next"] or tok
                continue
            prof = []
            def bright(dy):
                xyz = np.stack([gx[m], gy[m] + dy, np.zeros(m.sum())], 1)
                uv, z = project(gt, xyz)
                ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
                val = np.full(len(z), np.nan)
                val[ok] = img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
                return val
            for dy in SHIFTS:                                # local contrast: line minus 0.4 m to either side
                c = bright(dy) - 0.5 * (bright(dy - 0.4) + bright(dy + 0.4))
                prof.append(np.nanmean(c) if np.isfinite(c).sum() > 10 else np.nan)
            prof = np.array(prof)
            if np.isfinite(prof).all():
                profiles.append((sc["name"], k, prof - np.nanmedian(prof)))
        if not s["next"]:
            break
        tok = s["next"]

print("shift (m):        " + " ".join(f"{d:+5.2f}" for d in SHIFTS))
peaks = []
for name, k, p in profiles:
    peaks.append(SHIFTS[int(np.argmax(p))])
    print(f"{name} kf {k:2d}:  " + " ".join(f"{v:+5.1f}" for v in p) + f"   peak at {SHIFTS[int(np.argmax(p))]:+.2f} m")
mean_prof = np.mean([p for _, _, p in profiles], 0)
print("mean profile:     " + " ".join(f"{v:+5.1f}" for v in mean_prof))
print(f"frames {len(profiles)}; peak of the mean profile at {SHIFTS[int(np.argmax(mean_prof))]:+.2f} m; "
      f"per-frame peaks: median {np.median(peaks):+.2f} m, within +-0.25 m: {np.mean(np.abs(peaks) <= 0.25):.0%}")
