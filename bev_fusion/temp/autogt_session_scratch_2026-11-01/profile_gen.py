"""scratch: where the time of one keyframe goes (cProfile over a few mini keyframes, the live generator)"""
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, ".")
from autogt import AutoGT, scene_tokens

root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-1077"])
toks = scene_tokens("scene-0061", "v1.0-mini", root)[8:20:2] + scene_tokens("scene-1077", "v1.0-mini", root)[8:20:2]
for t in toks[:2]:
    gen.keyframe(t)                                   # warm-up (map caches, scene objects)
t0 = time.time()
pr = cProfile.Profile()
pr.enable()
for t in toks:
    gen.keyframe(t)
pr.disable()
print(f"{(time.time() - t0) / len(toks) * 1000:.0f} ms per keyframe under the profiler over {len(toks)}")
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(28)
st.sort_stats("tottime").print_stats(14)
