"""scratch: the broad comparison set: every other keyframe of all ten mini sequences + a few 12 Hz sweep frames (the video path), with the
generator of the CURRENT WORKING DIRECTORY: python dump_kf2.py out.pkl"""
import os
import pickle
import sys
import time

sys.path.insert(0, ".")
import autogt
from autogt import AutoGT, GridSpec, scene_tokens

out = sys.argv[1]
root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True)                          # all ten sequences
names = sorted({s["name"] for s in gen.nusc.scene})
res, t0 = {}, time.time()
for s in names:
    for tok in scene_tokens(s, "v1.0-mini", root)[::2]:
        res[tok] = gen.keyframe(tok)
n_key = len(res)
g = GridSpec()
for s in ("scene-0061", "scene-1077", "scene-0796"):                  # sweep frames: the ego frame, camera and map of another frame, boxes moved
    toks = scene_tokens(s, "v1.0-mini", root)
    tok = toks[8]
    track = gen.camera_track(tok)
    i = [x["token"] for x in track].index(gen.cam(tok)["token"])
    for k in (1, 3):
        res[tok + f"@{k}"] = gen.keyframe_gt(tok, g, sd=track[i + k]) if hasattr(gen, "keyframe_gt") else None
pickle.dump({"sig": gen.signature(), "kf": res, "file": autogt.__file__}, open(out, "wb"))
print(autogt.__file__, gen.signature(), n_key, "keyframes +", len(res) - n_key, "sweep frames", f"{(time.time() - t0) / len(res) * 1000:.0f} ms each")
