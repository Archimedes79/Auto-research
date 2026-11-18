"""Smoke test of the image and video tools on 24 validation ground-truth graphs (noisy copies play the prediction).  Run: scripts/_py14.sh ../tools/viz/selftest.py"""
import os
import pickle
import tempfile

import numpy as np

import make_video as MV
import render as RD

gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/graph_gt_val.pkl"), "rb"))[:24]
rng = np.random.RandomState(1)
pred = [dict(pts=[p + rng.normal(0, 0.5, p.shape) for p in g["pts"]], edges=list(g["edges"])) for g in gts]
d = tempfile.mkdtemp()
png = RD.render_frame(pred[0], gts[0], f"{d}/frame.png", title="frame 0", metrics={"F1@1": 0.93})
cmp_ = RD.render_compare({"noisy": pred[0], "truth": gts[0]}, gts[0], f"{d}/compare.png")
sheet = RD.render_grid(pred[:4], gts[:4], range(4), f"{d}/sheet.png")
mp4 = MV.graphs_to_video(pred, gts, f"{d}/seq.mp4", fps=6, limit=12)
gif = MV.graphs_to_video(pred, gts, f"{d}/seq.gif", fps=4, limit=6)
import cv2  # noqa: E402

cap = cv2.VideoCapture(mp4)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for p in (png, cmp_, sheet, mp4, gif):
    print(f"{os.path.basename(p):12s} {os.path.getsize(p)/1e3:8.1f} kB")
print("mp4 frames readable:", n, "(expected 12)")
assert n == 12 and all(os.path.getsize(p) > 1000 for p in (png, cmp_, sheet, mp4, gif))
sbs = MV.side_by_side([np.zeros((4, 6, 3), np.uint8)], [np.full((3, 2, 3), 7, np.uint8)])[0]       # a smaller right picture is padded with white
assert sbs.shape == (4, 8, 3) and sbs[0, 6, 0] == 7 and sbs[0, 7, 0] == 7 and sbs[3, 6, 0] == 255
try:
    MV.frames_to_video([], f"{d}/none.mp4")
    raise AssertionError("an empty frame list must be refused")
except ValueError:
    pass
print("viz / video selftest passed in", d)
