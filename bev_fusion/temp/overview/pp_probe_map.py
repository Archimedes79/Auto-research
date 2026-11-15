"""scratch: can the nuScenes map API give the intersection polygons cheaply? (probe for the precision follow-up)"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
froot = os.path.expanduser("~/data/nuscenes_full")
fr = json.load(open(os.path.join(HERE, "rp_frames.json")))
f = fr[1]
g = np.load(os.path.join(ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
m = g["meta"]
print({k: (m[k] if k in ("sample", "scene", "timestamp", "res") else type(m[k]).__name__) for k in m})
print(np.asarray(m["T_e2g"]))
scenes = {s["token"]: s for s in json.load(open(os.path.join(froot, "v1.0-trainval", "scene.json")))}
logs = {s["token"]: s for s in json.load(open(os.path.join(froot, "v1.0-trainval", "log.json")))}
sc = scenes.get(m["scene"])
print("scene record found:", sc is not None, sc["name"] if sc else None, f["scene"])
loc = logs[sc["log_token"]]["location"]
print("location", loc)
t0 = time.time()
from nuscenes.map_expansion.map_api import NuScenesMap
nmap = NuScenesMap(dataroot=froot, map_name=loc)
print("map loaded", round(time.time() - t0, 1), "s;", len(nmap.road_segment), "road segments")
inter = [r for r in nmap.road_segment if r.get("is_intersection")]
print("intersection road segments:", len(inter))
poly = nmap.extract_polygon(inter[0]["polygon_token"])
print(type(poly).__name__, poly.bounds)
