"""Automatic consistency checks of the generated GT (training/gt_bev.py) over every nuScenes mini keyframe (2026-10-26).

Mirko: "generate GT automatically from nuScenes, debug remaining mistakes". Each check is a physical or bookkeeping
fact the GT must obey; the script counts violations and lists the worst frames (to look at with view_gt_samples.py).
Objects:
  O1 box bottom on the road: z of the box bottom in the ego frame (|z| > 0.5 m within 40 m = height / pose problem)
  O2 moving vehicles (> 1 m/s) stand on the drivable area (centre cell coverage > 0)
  O3 no two vehicle footprints overlap (IoU > 0.10 = duplicate or wrong annotation)
  O4 boxes without lidar and radar points (now ignored) - how many, how far
  O5 our occlusion mask vs the annotators' visibility (80-100 % visible but centre occluded by our ray casting)
  O6 timing: the camera is ~35 ms before the annotation time - displacement |v| * |dt| of moving boxes
  O7 instance layer points at the box of its own group (the bug fixed on 2026-10-26)
  O8 stationary rule: stationary boxes keep their world pose between keyframes
Map:
  M1 the ego vehicle stands on the drivable area (cells x 0..3 m, |y| < 1 m)
  M2 centre-line cells lie on the drivable area
  M3 lane-divider cells lie on the drivable area (a divider off the road = a map / registration error)
  M4 centre-line direction: unit length, and the ego lane's direction is close to the ego heading while driving
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_gt_auto.py
"""
import collections
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, "training")
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from lift_pd import Grid  # noqa: E402

G = Grid()
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")
nusc = src.nusc
gx, gy = cell_centres(G)


def cell(x, y):
    i, j = int((x - G.x_min) / G.res), int((y - G.y_min) / G.res)
    return (i, j) if 0 <= i < G.nx and 0 <= j < G.ny else None


stats = collections.defaultdict(list)
worst = collections.defaultdict(list)          # check -> (value, scene, keyframe index, sample token, note)
prev_world = {}
for sc in nusc.scene:
    tok, k = sc["first_sample_token"], 0
    while tok:
        gt = src.keyframe_gt(tok, G)
        m = gt["meta"]
        sd = src.cam(tok)
        ann_t = nusc.get("sample", tok)["timestamp"]
        dt = (sd["timestamp"] - ann_t) * 1e-6
        stats["dt_cam_minus_ann"].append(dt)
        da = gt["drivable_area"]
        for bi, b in enumerate(m["boxes"]):
            x, y, z = b["center"]
            if not (0 < x < 60 and abs(y) < 26):
                continue
            if not b["observed"]:
                stats["O4_unobserved_range"].append(math.hypot(x, y))
                continue
            stats["O4_observed"].append(1)
            zb = z - b["size"][2] / 2
            if x < 40:
                stats["O1_zbottom"].append(zb)
                if abs(zb) > 0.5:
                    worst["O1"].append((abs(zb), sc["name"], k, tok, f"{b['group']} at x {x:.0f} y {y:.0f}: bottom z {zb:+.2f} m"))
            sp = math.hypot(*b["vel"])
            c = cell(x, y)
            if b["group"] == "vehicle" and sp > 1 and c:
                on = da[c] > 0
                stats["O2_moving_on_road"].append(on)
                if not on:
                    worst["O2"].append((sp, sc["name"], k, tok, f"vehicle {sp:.1f} m/s at x {x:.0f} y {y:.0f} off the drivable area"))
            if sp > 1:
                stats["O6_shift"].append(sp * abs(dt))
            vis = b["visibility"]
            if c and gt["camera"][c] > 0:
                stats[f"O5_occl_vis{vis}"].append(gt["occluded"][c] > 0)
                if vis == "4" and gt["occluded"][c] > 0:
                    worst["O5"].append((1, sc["name"], k, tok, f"{b['group']} at x {x:.0f} y {y:.0f} visibility 80-100 % but behind a box"))
            if b["stationary"]:
                if b["instance"] in prev_world:
                    stats["O8_stationary_move"].append(np.linalg.norm(prev_world[b["instance"]] - b["world"]))
                prev_world[b["instance"]] = b["world"]
        vb = [(i, Polygon(b["corners"])) for i, b in enumerate(m["boxes"]) if b["group"] == "vehicle" and b["observed"]]
        for a in range(len(vb)):
            for c2 in range(a + 1, len(vb)):
                pa, pb = vb[a][1], vb[c2][1]
                if pa.intersects(pb):
                    iou = pa.intersection(pb).area / max(pa.union(pb).area, 1e-6)
                    if iou > 0.10:
                        ba, bb = m["boxes"][vb[a][0]], m["boxes"][vb[c2][0]]
                        worst["O3"].append((iou, sc["name"], k, tok, f"vehicles at ({ba['center'][0]:.0f},{ba['center'][1]:.0f}) and "
                                            f"({bb['center'][0]:.0f},{bb['center'][1]:.0f}) overlap, IoU {iou:.2f}"))
        stats["O3_frames"].append(1)
        ii = gt["instance"]
        for i in np.unique(ii[ii >= 0]):
            stats["O7_ok"].append(gt[m["boxes"][i]["group"]][ii == i].min() > 0)
        # map
        ego = da[0:6, 50:54]
        stats["M1_ego_on_road"].append(ego.mean())
        if ego.mean() < 0.5:
            worst["M1"].append((1 - ego.mean(), sc["name"], k, tok, f"ego cells drivable {ego.mean():.2f}"))
        cl = gt["centreline"] > 0
        if cl.any():
            stats["M2_cl_on_road"].append((da[cl] > 0).mean())
            if (da[cl] > 0).mean() < 0.9:
                worst["M2"].append((1 - (da[cl] > 0).mean(), sc["name"], k, tok, f"centre-line cells on drivable {(da[cl] > 0).mean():.2f}"))
            n = np.hypot(*gt["centreline_dir"][:, cl])
            stats["M4_dir_norm"].extend(n.tolist())
        ld = gt["lane_divider"] > 0
        if ld.any():
            stats["M3_ld_on_road"].append((da[ld] > 0).mean())
            if (da[ld] > 0).mean() < 0.9:
                worst["M3"].append((1 - (da[ld] > 0).mean(), sc["name"], k, tok, f"lane-divider cells on drivable {(da[ld] > 0).mean():.2f}"))
        # ego lane direction: centre-line cells within 1.5 m lateral, 3-10 m ahead, while driving
        sel = cl & (gx > 3) & (gx < 10) & (np.abs(gy) < 1.5)
        if sel.any():
            ang = np.degrees(np.abs(np.arctan2(gt["centreline_dir"][1][sel], gt["centreline_dir"][0][sel])))
            stats["M4_ego_lane_angle"].append(float(np.median(ang)))
            if np.median(ang) > 45:
                worst["M4"].append((float(np.median(ang)), sc["name"], k, tok, f"ego-lane centre-line {np.median(ang):.0f} deg from the ego heading"))
        tok, k = nusc.get("sample", tok)["next"], k + 1
    prev_world = {}
    print(f"{sc['name']}: {k} keyframes", flush=True)


def rate(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1%} of {len(v)}"


print("\n=== results over all mini keyframes ===")
dt = np.array(stats["dt_cam_minus_ann"]) * 1e3
print(f"camera minus annotation time: median {np.median(dt):+.1f} ms, range {dt.min():+.1f}..{dt.max():+.1f} ms")
z = np.array(stats["O1_zbottom"])
print(f"O1 box bottom z (ego, < 40 m): median {np.median(z):+.2f} m, IQR {np.percentile(z, 25):+.2f}..{np.percentile(z, 75):+.2f}, "
      f"|z| > 0.5 m: {np.mean(np.abs(z) > 0.5):.1%} of {len(z)}")
print(f"O2 moving vehicles on the drivable area: {rate(stats['O2_moving_on_road'])}")
print(f"O3 overlapping vehicle pairs (IoU > 0.10): {len(worst['O3'])} in {len(stats['O3_frames'])} frames")
ur = np.array(stats["O4_unobserved_range"])
print(f"O4 boxes without lidar / radar points in the grid: {len(ur)} vs {len(stats['O4_observed'])} observed; "
      f"range median {np.median(ur) if len(ur) else float('nan'):.0f} m -> ignore mask")
for v in "1234":
    if stats[f"O5_occl_vis{v}"]:
        print(f"O5 visibility {v} ({['0-40', '40-60', '60-80', '80-100'][int(v) - 1]} %): centre in our occlusion mask {rate(stats[f'O5_occl_vis{v}'])}")
sh = np.array(stats["O6_shift"])
print(f"O6 moving boxes: displacement over the camera-annotation gap median {np.median(sh) * 100:.0f} cm, "
      f"95 % {np.percentile(sh, 95) * 100:.0f} cm, max {sh.max() * 100:.0f} cm (n = {len(sh)})")
print(f"O7 instance layer consistent: {rate(stats['O7_ok'])}")
mv = np.array(stats["O8_stationary_move"])
print(f"O8 stationary boxes, world motion between keyframes: max {mv.max() * 100 if len(mv) else 0:.2f} cm (n = {len(mv)})")
print(f"M1 ego on the drivable area (mean coverage of the ego cells): median {np.median(stats['M1_ego_on_road']):.2f}, "
      f"< 0.5 in {np.mean(np.array(stats['M1_ego_on_road']) < 0.5):.1%} of frames")
print(f"M2 centre-line cells on the drivable area: median {np.median(stats['M2_cl_on_road']):.1%}")
print(f"M3 lane-divider cells on the drivable area: median {np.median(stats['M3_ld_on_road']):.1%}")
dn = np.array(stats["M4_dir_norm"])
print(f"M4 centre-line direction norm: median {np.median(dn):.3f}, < 0.9 in {np.mean(dn < 0.9):.1%}; "
      f"ego-lane angle to the heading median {np.median(stats['M4_ego_lane_angle']):.1f} deg, > 45 deg in "
      f"{np.mean(np.array(stats['M4_ego_lane_angle']) > 45):.1%} of {len(stats['M4_ego_lane_angle'])}")
print("\nworst cases per check:")
for chk in sorted(worst):
    items = sorted(worst[chk], key=lambda r: -r[0])
    print(f"{chk}: {len(items)}")
    for r in items[:6]:
        print(f"   {r[1]} kf {r[2]:2d} {r[3][:8]}: {r[4]}")
