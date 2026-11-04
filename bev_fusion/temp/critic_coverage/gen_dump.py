"""Run keyframe_gt of the autogt on sys.path (old = HEAD copy, new = working tree) on some mini keyframes; dump to a pickle.
usage: gen_dump.py OUT.pkl N"""
import os, sys, pickle
import numpy as np
import autogt
from autogt.nuscenes_gt import NuScenesGT
from autogt.grid import GridSpec
print("autogt from", os.path.dirname(autogt.__file__))
out, N = sys.argv[1], int(sys.argv[2])
gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
g = GridSpec()
toks = [s["token"] for s in gt.nusc.sample]
rng = np.random.default_rng(11)
pick = sorted(rng.choice(len(toks), N, replace=False))
res = {}
for i in pick:
    k = gt.keyframe_gt(toks[i], g)
    # radar poses for later independent checks
    from autogt.nuscenes_adapter import pose_matrix
    sample = gt.nusc.get("sample", toks[i])
    rp = [pose_matrix(gt.nusc.get("calibrated_sensor", gt.nusc.get("sample_data", t)["calibrated_sensor_token"])) for ch, t in sample["data"].items() if ch.startswith("RADAR")]
    k["_radar_poses"] = rp
    res[toks[i]] = k
pickle.dump(res, open(out, "wb"))
print("dumped", len(res))
