"""scratch: the broad comparison set (every other keyframe of all ten mini sequences + sweep frames) as one digest per keyframe (every array byte and
every meta entry), so that nothing big is held in memory: python digest_kf.py out.json   (the generator of the CURRENT WORKING DIRECTORY)"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import autogt
from autogt import AutoGT, GridSpec, scene_tokens


def feed(h, x):
    if isinstance(x, np.ndarray):
        h.update(f"A{x.dtype}{x.shape}".encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            h.update(f"K{k}".encode())
            feed(h, x[k])
    elif isinstance(x, (list, tuple)):
        h.update(f"L{len(x)}".encode())
        for v in x:
            feed(h, v)
    else:
        h.update(f"{type(x).__name__}{x!r}".encode())


def digest(g):
    h = hashlib.sha1()
    feed(h, g)
    return h.hexdigest()


out = sys.argv[1]
root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True)
names = sorted({s["name"] for s in gen.nusc.scene})
res, t0 = {}, time.time()
for s in names:
    for tok in scene_tokens(s, "v1.0-mini", root)[::2]:
        res[tok] = digest(gen.keyframe(tok))
n_key = len(res)
grid = GridSpec()
for s in ("scene-0061", "scene-1077", "scene-0796"):
    tok = scene_tokens(s, "v1.0-mini", root)[8]
    track = gen.camera_track(tok)
    i = [x["token"] for x in track].index(gen.cam(tok)["token"])
    for k in (1, 3):
        res[f"{tok}@{k}"] = digest(gen.keyframe_gt(tok, grid, sd=track[i + k]))
json.dump({"sig": gen.signature(), "file": autogt.__file__, "digests": res}, open(out, "w"))
print(autogt.__file__, gen.signature(), n_key, "keyframes +", len(res) - n_key, "sweep frames", f"{(time.time() - t0) / len(res) * 1000:.0f} ms each")
