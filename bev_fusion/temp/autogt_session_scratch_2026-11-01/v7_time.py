"""scratch: what the v7 coverage work costs per keyframe: the whole keyframe with and without the shares / soft hiding"""
import os
import sys
import time

sys.path.insert(0, ".")
from autogt import AutoGT, scene_tokens
from autogt import coverage as C

root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-1077"])
toks = scene_tokens("scene-0061", "v1.0-mini", root)[::4] + scene_tokens("scene-1077", "v1.0-mini", root)[::4]


def run(label):
    t0 = time.time()
    for t in toks:
        gen.keyframe(t)
    print(f"{label}: {(time.time() - t0) / len(toks) * 1000:.0f} ms per keyframe over {len(toks)}")


run("warm-up")
run("full v7")
real_shares, real_hidden = C.object_shares, C.hidden_cells
C.object_shares = lambda *a, **k: {"camera": 0.0, "radar": 0.0, "fusion": 0.0}
run("without the object shares")
C.hidden_cells = lambda gx, gy, cam, prisms, soft=True: real_hidden(gx, gy, cam, [], soft)
run("without shares and without any hiding")
