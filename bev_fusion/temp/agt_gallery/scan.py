"""scratch (gallery of the GT quality overview): per-keyframe statistics of the STORED GT (no nuScenes devkit needed).

  scan.py SIGNATURE [STRIDE]  ->  temp/agt_gallery/scan_<signature>.json  (one record per keyframe)

Scene / location / description come from the nuScenes json tables (scene, log, sample), only the complete sequences are used
(a `_scene_<name>.pkl` exists in the cache folder).
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
ROOT = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
sig = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
D = os.path.join("data", "autogt", sig)

scenes = json.load(open(os.path.join(ROOT, "scene.json")))
logs = {r["token"]: r for r in json.load(open(os.path.join(ROOT, "log.json")))}
samples = {r["token"]: r for r in json.load(open(os.path.join(ROOT, "sample.json")))}
done = {os.path.basename(p)[len("_scene_"):-4] for p in glob.glob(os.path.join(D, "_scene_*.pkl"))}
out = []
for sc in scenes:
    if sc["name"] not in done:
        continue
    desc = sc["description"]
    loc = logs[sc["log_token"]]["location"]
    tok, k, chain = sc["first_sample_token"], 0, []
    while tok:
        chain.append(tok)
        tok = samples[tok]["next"]
    for k, tok in enumerate(chain):
        if k % stride or not os.path.exists(os.path.join(D, tok + ".npz")):
            continue
        z = np.load(os.path.join(D, tok + ".npz"), allow_pickle=True)
        meta = z["meta"].item()
        drv, walk, cross = z["drivable_area"].astype(np.float32), z["walkway"].astype(np.float32), z["ped_crossing"].astype(np.float32)
        occ = z["occluded"].astype(np.float32)
        r = dict(tok=tok, scene=sc["name"], k=k, n_kf=len(chain), loc=loc, desc=desc,
                 night="night" in desc.lower(), rain="rain" in desc.lower(),
                 ego_road=float(drv[2:8, 50:54].mean()), camera=float(z["camera"].astype(np.float32).mean()),
                 occluded=float(occ.mean()), undefined=float(z["undefined"].astype(np.float32).mean()),
                 drivable=float(drv.mean()), n_ign=0, flags=collections.Counter(), vmax=0.0,
                 veh=0, ped=0, two=0, mov_veh=0, mov_veh_road=0, ped_ok=0, off_plane=0, in40=0, vis4=0, vis4_occ=0, mov_ped=0, mov_two=0,
                 centreline=int(z["centreline"].astype(np.float32).astype(bool).sum()),
                 centreline_road=int(((z["centreline"].astype(np.float32) > 0) & (drv > 0.5)).sum()),
                 shift_max=0.0, n_shift=0)
        for b in meta["boxes"]:
            cx, cy, cz = b["center"]
            ix, iy = int(cx / 0.5), int((cy + 26.0) / 0.5)
            inside = 0 <= ix < 120 and 0 <= iy < 104
            if not b["observed"]:
                r["n_ign"] += int(inside)
                continue
            if not inside:
                continue
            grp, sp = b["group"], float(np.hypot(*b["vel"]))
            r["veh" if grp == "vehicle" else "ped" if grp == "pedestrian" else "two"] += 1
            r["flags"][b["vel_flag"]] += 1
            r["vmax"] = max(r["vmax"], sp * 3.6)
            moving = b["vel_known"] and sp > 1.0
            if grp == "vehicle" and moving:
                r["mov_veh"] += 1
                r["mov_veh_road"] += int(drv[ix, iy] >= 0.5)
            if grp == "pedestrian":
                r["ped_ok"] += int(max(drv[ix, iy], walk[ix, iy], cross[ix, iy]) >= 0.3)
                r["mov_ped"] += int(moving)
            if grp == "two_wheeler":
                r["mov_two"] += int(moving)
            if cx < 40:
                r["in40"] += 1
                r["off_plane"] += int(abs(cz - b["size"][2] / 2) > 0.5)
            if b["visibility"] == "4":
                r["vis4"] += 1
                r["vis4_occ"] += int(occ[ix, iy] > 0.5)
            if b["shift_s"] != 0.0:
                r["n_shift"] += 1
                r["shift_max"] = max(r["shift_max"], float(np.hypot(*b["shift"])))
        r["flags"] = dict(r["flags"])
        out.append(r)
    if len(out) % 500 < 20:
        print(sc["name"], len(out), flush=True)
json.dump(out, open(os.path.join("temp", "agt_gallery", f"scan_{sig}.json"), "w"))
print("keyframes scanned:", len(out), "sequences:", len({r["scene"] for r in out}))
