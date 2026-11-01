"""scratch: is the stored GT of the current generator complete and readable? (run from the bev_fusion folder)"""
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, ".")
import autogt
from autogt.suite import json_tables

sig = autogt.current_signature()
D = os.path.join("data", "autogt", sig)
idx = json.load(open(os.path.join(D, "index.json")))
print("current signature:", sig, "| store index signature:", idx["signature"], "| version", idx["version"], "| res", idx["res"], "ss", idx["ss"],
      "| options", idx["options"], "| stride", idx["stride"], "| keyframes in index", idx["keyframes"], "| scenes", len(idx["scenes"]), "| complete flag:", idx.get("complete", "(none: old index)"))
assert idx["signature"] == sig, "STORE IS NOT THE CURRENT GENERATOR"
root = autogt.suite.default_root("v1.0-trainval")
scenes, _ = json_tables(root, "v1.0-trainval")
names = {s["name"] for s in scenes}
print("scenes of nuScenes trainval:", len(names), "| in the index:", len(set(idx["scenes"])), "| missing in the index:", sorted(names - set(idx["scenes"]))[:5])
t0, missing, small, bad, n = time.time(), [], [], [], 0
need = {"drivable_area", "walkway", "ped_crossing", "stop_line", "carpark_area", "lane_divider", "road_divider", "centreline", "vehicle", "pedestrian",
        "two_wheeler", "instance", "velocity", "flow_ego", "flow", "flow_weight", "flow_sigma", "camera", "occluded", "radar", "undefined", "ignore", "meta"}
for name in idx["scenes"]:
    toks = autogt.scene_tokens(name, "v1.0-trainval", root)
    for tok in toks:
        n += 1
        p = os.path.join(D, tok + ".npz")
        if not os.path.exists(p):
            missing.append(tok)
            continue
        if os.path.getsize(p) < 5000:
            small.append(tok)
            continue
        try:
            with zipfile.ZipFile(p) as z:
                have = {x[:-4] for x in z.namelist()}
            if not need <= have:
                bad.append((tok, sorted(need - have)))
        except Exception as e:                                      # noqa: BLE001
            bad.append((tok, repr(e)[:60]))
print(f"expected keyframes {n} | missing {len(missing)} | tiny {len(small)} | unreadable / without a layer {len(bad)} | {time.time() - t0:.0f} s")
files = [f for f in os.listdir(D) if f.endswith(".npz")]
print("npz files in the folder:", len(files), "| tmp leftovers:", len([f for f in os.listdir(D) if ".tmp" in f]), "| scene pickles:", len([f for f in os.listdir(D) if f.startswith("_scene_")]))
print("extra npz not in the scene lists:", len(set(f[:-4] for f in files) - {t for nm in idx["scenes"] for t in autogt.scene_tokens(nm, "v1.0-trainval", root)}))
rng = np.random.RandomState(0)
for f in rng.choice(files, 40, replace=False):                      # full read of a sample: every array finite, shapes right
    g = autogt.load(f[:-4], "data/autogt")
    for k in ("drivable_area", "vehicle", "flow_weight"):
        assert g[k].shape == (120, 104) and np.isfinite(g[k]).all(), (f, k)
    assert g["velocity"].shape == (2, 120, 104) and np.isfinite(g["velocity"]).all(), f
    assert g["meta"]["boxes_at"] == "camera" if "boxes_at" in g["meta"] else True
print("40 random keyframes read completely: shapes ok, all finite")
print("RESULT: complete and current" if not (missing or small or bad) else "RESULT: PROBLEMS", (missing[:3], small[:3], bad[:3]))
