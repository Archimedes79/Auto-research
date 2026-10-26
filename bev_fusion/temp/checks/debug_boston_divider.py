"""Debug of the Boston road divider (scene-0103): timing, wrong annotation, or something else? (Mirko, 2026-10-26)

1. Over the CAM_FRONT frames of scene-0103 (keyframes and 12 Hz sweeps, every 2nd frame): per frame the lateral offset
   dy that best puts each map line element on the paint (contrast: paint minus 0.4 m to either side, 6-25 m ahead,
   pose and calibration as given), with the frame's ego speed and yaw rate. Timing would make the road divider's dy
   change with speed / yaw rate and move the lane divider alike; a wrong map line gives a dy that is constant in the
   world while the other lines stay at 0.
2. Map-internal: the road divider against the edges of the lane polygons next to it (distance from the divider to
   the nearest lane-polygon boundary, sampled along the divider) - do the map's own layers agree with each other?
3. A picture: dy per frame for each line element and the speed.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/debug_boston_divider.py <out png>
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from check_gt_pitch import bilinear, dense  # noqa: E402  (pure helpers; that script runs only under __main__)
from gt_bev import NuScenesGT, pose_matrix  # noqa: E402

DY = np.round(np.arange(-1.0, 1.001, 0.05), 3)
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")
nusc = src.nusc
scene = next(s for s in nusc.scene if s["name"] == "scene-0103")
nmap = src.map_for(scene["token"])
first = nusc.get("sample", scene["first_sample_token"])
track = src.camera_track(first["token"])
cs = nusc.get("calibrated_sensor", track[0]["calibrated_sensor_token"])
K = np.array(cs["camera_intrinsic"])
T_c2e = pose_matrix(cs)
T_e2c = np.linalg.inv(T_c2e)


def project(xyz):
    P = np.c_[xyz, np.ones(len(xyz))] @ T_e2c.T
    z = P[:, 2]
    return np.stack([K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1), z


def contrast(img, xy, dy):
    def b(d):
        uv, z = project(np.c_[xy[:, 0], xy[:, 1] + d, np.zeros(len(xy))])
        v = bilinear(img, uv)
        v[z <= 1] = np.nan
        return v
    return np.nanmean(b(dy) - 0.5 * (b(dy - 0.4) + b(dy + 0.4)))


# the line elements near the scene's path (global geometry, tokens kept)
xs = [src.ego_pose(sd)[0, 3] for sd in track]
ys = [src.ego_pose(sd)[1, 3] for sd in track]
box = (min(xs) - 40, min(ys) - 40, max(xs) + 40, max(ys) + 40)
recs = nmap.get_records_in_patch(box, ["lane_divider", "road_divider", "lane", "lane_connector"], mode="intersect")
elements = []
for lay in ("road_divider", "lane_divider"):
    for tok in recs[lay]:
        line = nmap.extract_line(nmap.get(lay, tok)["line_token"])
        if not line.is_empty:
            elements.append((lay, tok, np.asarray(line.coords)[:, :2]))

series = {}
for i, sd in enumerate(track):
    if i % 2:
        continue
    T_e2g = src.ego_pose(sd)
    T_g2e = np.linalg.inv(T_e2g)
    img = np.asarray(Image.open(os.path.join(root, sd["filename"])).convert("L"), np.float32)
    if i + 1 < len(track):
        rel = np.linalg.inv(T_e2g) @ src.ego_pose(track[i + 1])
        dt = (track[i + 1]["timestamp"] - sd["timestamp"]) * 1e-6
        v, w = rel[0, 3] / dt, math.atan2(rel[1, 0], rel[0, 0]) / dt
    for lay, tok, g in elements:
        e = (np.c_[g, np.zeros(len(g)), np.ones(len(g))] @ T_g2e.T)[:, :2]
        d = dense([e])
        d = d[(d[:, 0] > 6) & (d[:, 0] < 25) & (np.abs(d[:, 1]) < 10)] if len(d) else d
        if len(d) < 40:
            continue
        s = np.array([contrast(img, d, dy) for dy in DY])
        if not np.isfinite(s).any():
            continue
        j = int(np.nanargmax(s))
        peak_clear = s[j] - np.nanmedian(s) > 4.0
        series.setdefault((lay, tok), []).append(((sd["timestamp"] - track[0]["timestamp"]) * 1e-6, DY[j], v, w, peak_clear))

print("line element (layer, token) | frames | clear peaks | dy median (m) | dy IQR (m) | corr(dy, speed) | corr(dy, yaw rate)")
for (lay, tok), rows in series.items():
    R = np.array([r[:4] for r in rows if r[4]])
    if len(R) < 5:
        print(f"{lay:12s} {tok[:8]} | {len(rows):3d} | {len(R):3d} | (too few clear peaks)")
        continue
    c_v = np.corrcoef(R[:, 1], R[:, 2])[0, 1] if R[:, 2].std() > 1e-6 and R[:, 1].std() > 1e-6 else float("nan")
    c_w = np.corrcoef(R[:, 1], R[:, 3])[0, 1] if R[:, 3].std() > 1e-6 and R[:, 1].std() > 1e-6 else float("nan")
    print(f"{lay:12s} {tok[:8]} | {len(rows):3d} | {len(R):3d} | {np.median(R[:, 1]):+.2f} | "
          f"{np.percentile(R[:, 1], 25):+.2f}..{np.percentile(R[:, 1], 75):+.2f} | {c_v:+.2f} | {c_w:+.2f}")

# map-internal: distance from each road divider to the nearest lane / lane-connector polygon boundary
from shapely.geometry import LineString, Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
lane_edges = []
for lay in ("lane", "lane_connector"):
    for tok in recs[lay]:
        poly = nmap.extract_polygon(nmap.get(lay, tok)["polygon_token"])
        if not poly.is_empty:
            lane_edges.append(poly.exterior)
edges = unary_union(lane_edges)
print("\nmap-internal: road divider vs nearest lane-polygon boundary (m), sampled every 1 m along the divider")
for lay, tok, g in elements:
    if lay != "road_divider":
        continue
    ls = LineString(g)
    d = [edges.distance(ls.interpolate(s)) for s in np.arange(0, ls.length, 1.0)]
    print(f"road_divider {tok[:8]}: length {ls.length:5.1f} m, distance to lane edges median {np.median(d):.2f} m, "
          f"max {np.max(d):.2f} m")

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for (lay, tok), rows in series.items():
    R = np.array([r[:4] for r in rows if r[4]])
    if len(R) >= 5:
        ax[0].plot(R[:, 0], R[:, 1], "o-" if lay == "road_divider" else "s--", markersize=3,
                   label=f"{lay} {tok[:6]}", color=(1, 0.45, 0) if lay == "road_divider" else None)
ax[0].axhline(0, color="k", linewidth=0.5)
ax[0].set_ylabel("best lateral shift dy (m)")
ax[0].legend(fontsize=7, ncol=3)
ax[0].set_title("scene-0103: per frame, the lateral shift that puts each map line on the paint (clear peaks only)")
allr = [r for rows in series.values() for r in rows]
tt = sorted({r[0] for r in allr})
vv = [next(r[2] for r in allr if r[0] == t) for t in tt]
ax[1].plot(tt, vv, color="k")
ax[1].set_ylabel("ego speed (m/s)")
ax[1].set_xlabel("time in scene (s)")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
print("saved", sys.argv[1])
