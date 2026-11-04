"""Automatic consistency checks of the generated GT over keyframes: python -m autogt check [--stride N] [--holdout]

Each check is a physical or bookkeeping fact the GT must obey; the report counts violations and lists the worst frames.
Objects
  O1 box bottom on the ground: z of the box bottom in the ego frame (|z| > 0.5 m within 40 m = slope / overpass, a ground-plane
     assumption of the camera lift, not a GT error)
  O2 moving vehicles (> 1 m/s) stand on the drivable area
  O3 no two observed vehicle footprints overlap (IoU > 0.10)
  O4 boxes without lidar / radar points (ignored): share and range
  O5 our occlusion mask against the annotators' visibility (should fall monotonically with visibility)
  O6 timing: camera time minus annotation time, and what it does to moving boxes (v * dt)
  O7 instance layer points at an observed box of the group whose layer covers the cell
  O8 stationary boxes keep their world pose between keyframes
Map
  M1 the ego car stands on the drivable area; M2 centre-line cells and M3 lane-divider cells lie on the drivable area
Velocity / flow
  V1 flow = flow_ego + velocity everywhere; velocity is 0 outside observed boxes; weight 0 on ignored boxes
  V2 flags of the velocity estimate over all boxes (radar_conflict, radar_override, stationary_but_moving, short_track)
  V3 radar field of view: share of all radar detections inside +-60 deg of the sensor axis (the assumption of the radar mask)
--holdout: leave-one-radar-out test of the radar correction: each radar is scored on boxes corrected with the OTHER four
  (radial error against the held-out Doppler: RMSE / median / within 0.5 m/s; track only vs corrected vs zero velocity)
"""
import collections
import math
import os

import numpy as np
from shapely.geometry import Polygon

from . import flow as flowmod
from .nuscenes_adapter import RADARS, NuScenesScene
from .suite import AutoGT


def _cell(grid, x, y):
    i, j = int((x - grid.x_min) / grid.res), int((y - grid.y_min) / grid.res)
    return (i, j) if 0 <= i < grid.nx and 0 <= j < grid.ny else None


def run(dataroot, version="v1.0-mini", scenes=None, stride=1, holdout=False):
    gt = AutoGT(dataroot, version)
    g = gt.grid
    st, worst, prev_world, flags = collections.defaultdict(list), collections.defaultdict(list), {}, collections.Counter()
    for name in (scenes or gt.scenes()):
        prev_world = {}
        for k, tok in enumerate(gt.keyframes(name)):
            if k % stride:
                continue
            gk = gt.keyframe(tok)
            m, da = gk["meta"], gk["drivable_area"]
            sample = gt.nusc.get("sample", tok)
            st["dt_ms"].append((m["timestamp"] - sample["timestamp"]) * 1e-3)
            for bi, b in enumerate(m["boxes"]):
                x, y, z = b["center"]
                flags[b["vel_flag"]] += 1
                if not (0 < x < 60 and abs(y) < 26):
                    continue
                if not b["observed"]:
                    st["O4_unobs_range"].append(math.hypot(x, y))
                    continue
                st["O4_obs"].append(1)
                zb = z - b["size"][2] / 2
                if x < 40:
                    st["O1_z"].append(zb)
                sp = math.hypot(*b["vel"])
                c = _cell(g, x, y)
                if b["group"] == "vehicle" and sp > 1 and c:
                    st["O2"].append(da[c] > 0)
                    if da[c] == 0:
                        worst["O2"].append((sp, name, k, f"vehicle {sp:.1f} m/s at x {x:.0f} y {y:.0f} off the drivable area"))
                if sp > 1:
                    st["O6_shift_cm"].append(sp * abs(st["dt_ms"][-1]) * 0.1)
                if c and gk["camera"][c] > 0:
                    st[f"O5_vis{b['visibility']}"].append(gk["occluded"][c] > 0)
                if b["stationary"]:
                    if b["instance"] in prev_world:
                        st["O8_move_cm"].append(np.linalg.norm(prev_world[b["instance"]] - b["world"]) * 100)
                    prev_world[b["instance"]] = b["world"]
            vb = [(i, Polygon(b["corners"])) for i, b in enumerate(m["boxes"]) if b["group"] == "vehicle" and b["observed"]]
            for a in range(len(vb)):
                for c2 in range(a + 1, len(vb)):
                    pa, pb = vb[a][1], vb[c2][1]
                    if pa.intersects(pb) and pa.intersection(pb).area / max(pa.union(pb).area, 1e-6) > 0.10:
                        st["O3"].append(1)
            ii = gk["instance"]
            for i in np.unique(ii[ii >= 0]):
                st["O7"].append(m["boxes"][i]["observed"] and gk[m["boxes"][i]["group"]][ii == i].min() > 0)
            st["M1"].append(float(da[0:6, 50:54].mean()))
            if (gk["centreline"] > 0).any():
                st["M2"].append(float((da[gk["centreline"] > 0] > 0).mean()))
            if (gk["lane_divider"] > 0).any():
                st["M3"].append(float((da[gk["lane_divider"] > 0] > 0).mean()))
            st["V1_flow"].append(float(np.abs(gk["flow"] - gk["flow_ego"] - gk["velocity"]).max()))
            obj = gk["vehicle"] + gk["pedestrian"] + gk["two_wheeler"]
            st["V1_vel_outside"].append(float(np.abs(gk["velocity"][:, obj == 0]).max()) if (obj == 0).any() else 0.0)
            st["V1_w"].append(bool((gk["flow_weight"][gk["ignore"] > 0.9] == 0).all()))
        print(name, flush=True)
    rate = lambda v: f"{np.mean(v):.1%} of {len(v)}"  # noqa: E731
    dt = np.array(st["dt_ms"])
    print("\n=== GT checks ===")
    print(f"keyframes {len(dt)}; camera minus annotation time: median {np.median(dt):+.1f} ms, range {dt.min():+.1f}..{dt.max():+.1f}")
    z = np.array(st["O1_z"])
    print(f"O1 box bottom z (< 40 m): median {np.median(z):+.2f} m, |z| > 0.5 m in {np.mean(np.abs(z) > 0.5):.1%} of {len(z)}")
    print(f"O2 moving vehicles on the drivable area: {rate(st['O2'])}")
    print(f"O3 overlapping observed vehicle pairs (IoU > 0.10): {len(st['O3'])}")
    ur = np.array(st["O4_unobs_range"])
    print(f"O4 boxes without lidar / radar points: {len(ur)} vs {len(st['O4_obs'])} observed (median range {np.median(ur) if len(ur) else float('nan'):.0f} m)")
    print("O5 centre cell in our occlusion mask by annotator visibility 0-40 / 40-60 / 60-80 / 80-100 %: " +
          " / ".join(f"{np.mean(st[f'O5_vis{v}']):.0%}" if st[f"O5_vis{v}"] else "-" for v in "1234"))
    sh = np.array(st["O6_shift_cm"])
    print(f"O6 moving boxes: v * (camera - annotation time) median {np.median(sh):.0f} cm, 95 % {np.percentile(sh, 95):.0f} cm, max {sh.max():.0f} cm")
    print(f"O7 instance layer consistent: {rate(st['O7'])}")
    print(f"O8 stationary boxes, world motion between keyframes: max {max(st['O8_move_cm'], default=0):.2f} cm (n = {len(st['O8_move_cm'])})")
    print(f"M1 ego on the road: median {np.median(st['M1']):.2f}, < 0.5 in {np.mean(np.array(st['M1']) < 0.5):.1%}; "
          f"M2 centre-line cells on the road {np.median(st['M2']):.1%}; M3 lane-divider cells on the road {np.median(st['M3']):.1%}")
    print(f"V1 flow = flow_ego + velocity (max abs difference {max(st['V1_flow']):.1e}); velocity outside boxes (max {max(st['V1_vel_outside']):.1e}); "
          f"weight 0 on ignored boxes: {rate(st['V1_w'])}")
    print("V2 velocity flags over all boxes:", dict(flags))
    print(f"V3 radar field of view: {radar_fov(gt, scenes, stride):.2%} of all detections lie within +-60 deg of the sensor axis")
    if holdout:
        leave_one_radar_out(gt, scenes, stride)
    return st


def radar_fov(gt, scenes=None, stride=1):
    from nuscenes.utils.data_classes import RadarPointCloud
    inside = total = 0
    for name in (scenes or gt.scenes())[:5]:
        for k, tok in enumerate(gt.keyframes(name)):
            if k % max(stride, 4):
                continue
            for ch, sdt in gt.nusc.get("sample", tok)["data"].items():
                if ch.startswith("RADAR"):
                    sd = gt.nusc.get("sample_data", sdt)
                    pts = RadarPointCloud.from_file(os.path.join(gt.dataroot, sd["filename"])).points
                    ang = np.abs(np.degrees(np.arctan2(pts[1], pts[0])))
                    inside += int((ang <= 60).sum())
                    total += pts.shape[1]
    return inside / max(total, 1)


def leave_one_radar_out(gt, scenes=None, stride=1):
    res = collections.defaultdict(list)
    per_pair = []
    fl = collections.Counter()
    for name in (scenes or gt.scenes()):
        sc = NuScenesScene(gt.nusc, gt.dataroot, name)
        by_id = {tr.id: tr for tr in sc.tracks}
        for k, tok in enumerate(gt.keyframes(name)):
            if k % stride:
                continue
            t = sc.time_of(tok, "annotation")
            frames = sc.radar(t)
            prior = {o.id: o for o in flowmod.estimate_flow(sc.tracks, sc.ego, t, None).objects}
            for R in RADARS:
                held = [f for f in frames if f.sensor == R]
                rest = [f for f in frames if f.sensor != R]
                if not held:
                    continue
                fused = {o.id: o for o in flowmod.estimate_flow(sc.tracks, sc.ego, t, None, rest).objects}
                for i, op in prior.items():
                    if op.stationary or not op.observed or not np.isfinite(op.v).all():
                        continue
                    rh, d, _ = flowmod.radar_points_in_box(op, by_id[i], held, 0.5)
                    of = fused[i]
                    if len(d) == 0 or max(op.speed, of.speed) < 1.0:
                        continue
                    fl[of.flag] += 1
                    res["zero"].extend((-d).tolist())
                    res["track"].extend((rh @ op.v - d).tolist())
                    res["radar-corrected"].extend((rh @ of.v - d).tolist())
                    per_pair.append((float(np.sqrt(np.mean((rh @ op.v - d) ** 2))), float(np.sqrt(np.mean((rh @ of.v - d) ** 2)))))
        print(name, "(holdout)", flush=True)
    print(f"\nleave-one-radar-out: {len(res['zero'])} held-out detections in moving observed boxes ({len(per_pair)} box-radar pairs)")
    print("estimate | RMSE (m/s) | median |err| | within 0.5 m/s | share > 2 m/s")
    for k in ("zero", "track", "radar-corrected"):
        e = np.abs(np.array(res[k]))
        print(f"{k:15s} | {np.sqrt((e ** 2).mean()):.3f} | {np.median(e):.3f} | {np.mean(e < 0.5):.1%} | {np.mean(e > 2):.1%}")
    pp = np.array(per_pair)
    diff = pp[:, 1] - pp[:, 0]
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    print(f"per box-radar pair: corrected better {np.mean(diff < -1e-9):.1%}, equal {np.mean(np.abs(diff) <= 1e-9):.1%}, "
          f"worse {np.mean(diff > 1e-9):.1%}; mean RMSE change {diff.mean():+.3f} m/s (bootstrap {np.percentile(boot, 2.5):+.3f} .. "
          f"{np.percentile(boot, 97.5):+.3f}; pairs of one scene are correlated: optimistic); flags {dict(fl)}")
