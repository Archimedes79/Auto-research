"""scratch probe: keys of a cached GT frame, the frames the paint check picks, whether they are in the val dataset's frames"""
import os
import sys
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "temp", "lane_tests")]
import paint_vs_label as P  # noqa: E402

frames = P.choose_frames("val", 3, 0, 0)
print(len(frames), "frames,", len({s for s, _ in frames}), "scenes")
f = frames[0][1]
g = np.load(f, allow_pickle=True)["gt"].item()
print(list(g))
cur = g["cur"]
for k, v in cur.items():
    print(k, getattr(v, "shape", type(v)), getattr(v, "dtype", ""))
print(list(cur["meta"]))
print({k: len(v) for k, v in cur["meta"]["vectors"].items()})
