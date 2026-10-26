"""Map-image alignment per nuScenes location, and the root cause: map element, localisation, or convention? (2026-10-26)

nuScenes trainval (the 510 scenes with images on disk), per location (boston-seaport, singapore-onenorth,
singapore-queenstown, singapore-hollandvillage) up to 15 scenes x 4 keyframes. Per keyframe and map line element
(lane_divider, road_divider; 6-25 m ahead, |y| < 10 m): the lateral shift dy in -1..+1 m that best puts the element on
the paint (contrast: paint minus 0.4 m to either side, bilinear sampling, pose and calibration as given); clear peaks
only (peak minus median contrast > 4 grey levels).
Root cause signatures:
- map element error: the same element shows the same dy from different logs (different days / drives)
- localisation / pose: dy differs between logs for the same element, and all elements of a frame move together
- convention: a systematic dy of one layer in every location (e.g. a divider drawn at the edge of a double line)
Low memory: the big tables (sample_data, ego_pose) are scanned line by line and only the CAM_FRONT keyframes and their
poses are kept (the full devkit needs > 8 GB, more than is free while other jobs run).
Run from the bev_fusion folder:
    wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_map_alignment_locations.py
"""
import collections
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from check_gt_pitch import bilinear, dense  # noqa: E402
from gt_bev import pose_matrix  # noqa: E402

DY = np.round(np.arange(-1.0, 1.001, 0.05), 3)
root = os.path.expanduser("~/data/nuscenes_full")
meta = os.path.join(root, "v1.0-trainval")


def scan(name, keep):
    """records of a pretty-printed nuScenes table for which keep(record text) is true, without loading the table"""
    out, buf = [], []
    with open(os.path.join(meta, name)) as f:
        for line in f:
            s = line.strip()
            if s == "{":
                buf = [s]
            elif s in ("}", "},"):
                buf.append("}")
                txt = "".join(buf)
                if keep(txt):
                    out.append(json.loads(txt))
                buf = []
            elif buf:
                buf.append(s)
    return out


def load(name):
    with open(os.path.join(meta, name)) as f:
        return {r["token"]: r for r in json.load(f)}


class Light:
    """the few devkit calls this check needs: get(table, token), scene list"""
    def __init__(self):
        self.t = {n: load(n + ".json") for n in ("scene", "log", "sample", "calibrated_sensor")}
        sds = scan("sample_data.json", lambda t: '"samples/CAM_FRONT/' in t)
        self.t["sample_data"] = {r["token"]: r for r in sds}
        self.cam = {r["sample_token"]: r for r in sds}
        need = {r["ego_pose_token"] for r in sds}
        self.t["ego_pose"] = {r["token"]: r for r in scan("ego_pose.json", lambda t: t[11:43] in need)}
        for s in self.t["sample"].values():
            s["data"] = {"CAM_FRONT": self.cam[s["token"]]["token"]}
        self.scene = list(self.t["scene"].values())

    def get(self, table, token):
        return self.t[table][token]


nusc = Light()
print(f"loaded {len(nusc.cam)} CAM_FRONT keyframes, {len(nusc.t['ego_pose'])} poses", flush=True)


class Src:
    def __init__(self):
        self._maps = {}

    def map_for(self, scene_token):
        from nuscenes.map_expansion.map_api import NuScenesMap
        loc = nusc.get("log", nusc.get("scene", scene_token)["log_token"])["location"]
        if loc not in self._maps:
            self._maps = {loc: NuScenesMap(dataroot=root, map_name=loc)}   # one map in memory at a time
        return self._maps[loc]

    def ego_pose(self, sd):
        return pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))


src = Src()


def contrast_profile(img, K, T_e2c, xy):
    def b(d):
        P = np.c_[xy[:, 0], xy[:, 1] + d, np.zeros(len(xy)), np.ones(len(xy))] @ T_e2c.T
        z = P[:, 2]
        uv = np.stack([K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1)
        v = bilinear(img, uv)
        v[z <= 1] = np.nan
        return v
    return np.array([np.nanmean(b(d) - 0.5 * (b(d - 0.4) + b(d + 0.4))) for d in DY])


by_loc = collections.defaultdict(list)
for sc in nusc.scene:
    log = nusc.get("log", sc["log_token"])
    first = nusc.get("sample", sc["first_sample_token"])
    if not os.path.exists(os.path.join(root, nusc.get("sample_data", first["data"]["CAM_FRONT"])["filename"])):
        continue
    by_loc[log["location"]].append((sc, log))

rows = []                                  # (location, log, scene, element, layer, dy, frame id)
rng = np.random.default_rng(0)
for loc, scenes in sorted(by_loc.items()):
    pick = [scenes[i] for i in rng.choice(len(scenes), min(15, len(scenes)), replace=False)]
    nmap = src.map_for(pick[0][0]["token"])
    for sc, log in pick:
        tok = sc["first_sample_token"]
        for k in range(40):
            s = nusc.get("sample", tok)
            if k % 9 == 3:
                sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
                T_e2g = src.ego_pose(sd)
                T_g2e = np.linalg.inv(T_e2g)
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                K = np.array(cs["camera_intrinsic"])
                T_e2c = np.linalg.inv(pose_matrix(cs))
                c = (T_e2g @ np.array([15, 0, 0, 1]))[:2]
                recs = nmap.get_records_in_patch((c[0] - 25, c[1] - 25, c[0] + 25, c[1] + 25),
                                                 ["lane_divider", "road_divider"], mode="intersect")
                img = None
                for lay in ("lane_divider", "road_divider"):
                    for etok in recs[lay]:
                        line = nmap.extract_line(nmap.get(lay, etok)["line_token"])
                        if line.is_empty:
                            continue
                        g = np.asarray(line.coords)[:, :2]
                        e = (np.c_[g, np.zeros(len(g)), np.ones(len(g))] @ T_g2e.T)[:, :2]
                        d = dense([e])
                        d = d[(d[:, 0] > 6) & (d[:, 0] < 25) & (np.abs(d[:, 1]) < 10)] if len(d) else d
                        if len(d) < 40:
                            continue
                        if img is None:
                            img = np.asarray(Image.open(os.path.join(root, sd["filename"])).convert("L"), np.float32)
                        p = contrast_profile(img, K, T_e2c, d)
                        if not np.isfinite(p).any():
                            continue
                        j = int(np.nanargmax(p))
                        if p[j] - np.nanmedian(p) > 4.0:
                            rows.append((loc, log["token"], sc["name"], etok, lay, DY[j], sd["token"]))
            if not s["next"]:
                break
            tok = s["next"]
    print(f"{loc}: {len([r for r in rows if r[0] == loc])} element measurements", flush=True)

print("\nper location and layer: clear measurements | median dy (m) | IQR (m) | |dy| <= 0.25 m")
for loc in sorted(by_loc):
    for lay in ("lane_divider", "road_divider"):
        v = np.array([r[5] for r in rows if r[0] == loc and r[4] == lay])
        if len(v):
            print(f"{loc:26s} {lay:12s} | {len(v):4d} | {np.median(v):+.2f} | {np.percentile(v, 25):+.2f}..{np.percentile(v, 75):+.2f} | "
                  f"{np.mean(np.abs(v) <= 0.25):.0%}")

# same element seen from different logs: consistent offset (map) or not (localisation)?
print("\nelements measured from >= 2 logs (>= 2 clear measurements each): per-log median dy")
el = collections.defaultdict(lambda: collections.defaultdict(list))
for loc, log, scn, etok, lay, dy, fid in rows:
    el[(loc, etok, lay)][log].append(dy)
diffs = []
for (loc, etok, lay), logs in el.items():
    good = {lg: np.median(v) for lg, v in logs.items() if len(v) >= 2}
    if len(good) >= 2:
        vals = list(good.values())
        diffs.append(max(vals) - min(vals))
        print(f"{loc:26s} {lay:12s} {etok[:8]}: " + ", ".join(f"{v:+.2f}" for v in vals))
if diffs:
    print(f"spread between logs for the same element: median {np.median(diffs):.2f} m (n = {len(diffs)})")

# all elements of one frame move together? (localisation would shift the whole frame)
fr = collections.defaultdict(list)
for r in rows:
    fr[r[6]].append(r[5])
spreads = [np.std(v) for v in fr.values() if len(v) >= 3]
means = [abs(np.mean(v)) for v in fr.values() if len(v) >= 3]
if spreads:
    print(f"frames with >= 3 elements: {len(spreads)}; within-frame std of dy median {np.median(spreads):.2f} m, "
          f"|frame mean dy| median {np.median(means):.2f} m")
