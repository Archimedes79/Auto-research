"""One-off: run the simple radar block (training/radar_bev.py) on a real nuScenes keyframe and draw what it sees.

Adapter (nuScenes -> detections): the five radars of one keyframe (current sweep only, dt = 0, no filters), points in the sensor
frame -> range r, azimuth phi; sensor pose from calibrated_sensor (translation, yaw of the quaternion); Doppler vr = the
ego-compensated velocity (vx_comp, vy_comp) projected on the ray; rcs as given.
Output: figure with the front camera and the BEV channels (Doppler as a circular colour, distance, direction) + printed statistics.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_bev_nuscenes_check.py [sample index]
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
import radar_bev as rb  # noqa: E402
from lift_pd import Grid  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.data_classes import RadarPointCloud  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]


def frame_detections(nusc, sample):
    RadarPointCloud.disable_filters()
    cols = {k: [] for k in ["x", "y", "r", "psi", "vr", "rcs", "dt", "sid"]}
    for sid, ch in enumerate(RADARS):
        sd = nusc.get("sample_data", sample["data"][ch])
        cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        P = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"])).points
        r = torch.tensor(np.hypot(P[0], P[1]), dtype=torch.float32)
        phi = torch.tensor(np.arctan2(P[1], P[0]), dtype=torch.float32)
        from pyquaternion import Quaternion
        yaw = Quaternion(cal["rotation"]).yaw_pitch_roll[0]
        sx, sy = cal["translation"][0], cal["translation"][1]
        x, y, psi = rb.to_vehicle(r, phi, sx, sy, yaw)
        vr = torch.tensor(P[8] * np.cos(P[1] * 0 + np.arctan2(P[1], P[0])) + P[9] * np.sin(np.arctan2(P[1], P[0])), dtype=torch.float32)
        for k, v in [("x", x), ("y", y), ("r", r), ("psi", psi), ("vr", vr), ("rcs", torch.tensor(P[5], dtype=torch.float32)),
                     ("dt", torch.zeros_like(r)), ("sid", torch.full_like(r, sid))]:
            cols[k].append(v)
    det = {k: torch.cat(v) for k, v in cols.items()}
    det["b"] = torch.zeros(len(det["x"]), dtype=torch.long)
    return det


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    nusc = NuScenes(version="v1.0-mini", dataroot=os.path.expanduser("~/data/nuscenes"), verbose=False)
    sample = nusc.sample[idx]
    det = frame_detections(nusc, sample)
    grid = Grid()
    out = rb.rasterize(det, 1, grid)
    inside = (det["x"] >= 0) & (det["x"] < 60) & (det["y"] >= -26) & (det["y"] < 26)
    cnt = out[0, 0]
    print(f"sample {idx}: {len(det['x'])} detections of 5 radars, {int(inside.sum())} inside the grid (x 0..60 m, |y| < 26 m); "
          f"{int((cnt > 0).sum())} occupied cells, {int((cnt > 1).sum())} with more than one detection (max {int(cnt.max())})")
    print(f"vr (compensated) 1/50/99 %: {np.quantile(det['vr'].numpy(), [0.01, 0.5, 0.99]).round(1)} m/s; "
          f"Doppler code magnitude per cell (1 = a single phase): mean {out[0, 4:6].norm(dim=0)[cnt > 0].mean():.2f}")
    # the raster must agree with the detections: every inside detection is in the cell it resides in
    ix = ((det["x"][inside] - grid.x_min) / grid.res).floor().long()
    iy = ((det["y"][inside] - grid.y_min) / grid.res).floor().long()
    assert (cnt[ix, iy] > 0).all(), "a detection is not in its cell"
    assert int(cnt.sum()) == int(inside.sum()), "count channel != detections inside the grid"
    # figure
    xs, ys = torch.nonzero(cnt > 0, as_tuple=True)
    cx, cy = grid.x_min + (xs.float() + 0.5) * grid.res, grid.y_min + (ys.float() + 0.5) * grid.res
    vr_mean = torch.atan2(out[0, 4, xs, ys], out[0, 5, xs, ys])         # phase of the k = 1 Doppler code
    dist = out[0, 1, xs, ys] * rb.RANGE_SCALE
    psi = torch.atan2(out[0, 3, xs, ys], out[0, 2, xs, ys])
    fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))
    img = Image.open(os.path.join(nusc.dataroot, nusc.get("sample_data", sample["data"]["CAM_FRONT"])["filename"]))
    ax[0, 0].imshow(img)
    ax[0, 0].set_title("front camera")
    ax[0, 0].axis("off")
    for a, c, title, cmap, vmin, vmax in [(ax[0, 1], vr_mean, "Doppler as a circular value (phase of k = 1)", "twilight", -math.pi, math.pi),
                                          (ax[1, 0], dist, "distance from the sensor (m)", "viridis", 0, 60),
                                          (ax[1, 1], psi, "direction from the sensor in the vehicle frame (circular)", "twilight", -math.pi, math.pi)]:
        s = a.scatter(cy, cx, c=c, s=22, marker="s", cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_xlim(26, -26)
        a.set_ylim(0, 60)
        a.set_xlabel("y left (m)")
        a.set_ylabel("x forward (m)")
        a.set_title(title)
        a.set_aspect("equal")
        a.plot(0, 0, "k^")
        fig.colorbar(s, ax=a, shrink=0.7)
    fig.suptitle(f"nuScenes mini sample {idx}: 5 radars, one cell = 0.5 m, {int((cnt > 0).sum())} occupied cells", y=0.98)
    os.makedirs("temp/radar/figs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("temp/radar/figs/radar_bev_check.png", dpi=110)
    print("figure: temp/radar/figs/radar_bev_check.png")
