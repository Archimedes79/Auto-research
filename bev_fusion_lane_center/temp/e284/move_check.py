"""scratch: after the GT chat moved data/autogt/lane to D: (2026-11-09): do the lane GT reads still work through the link? 20 VAL frames (every 40th), lane() and ref(), time per frame, and the pooled lane length of the 20 frames against the value of a read before the move is not available, so
only: no error, every frame has pieces, the signature of the default store.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/move_check.py"""
import os
import sys
import time

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
from autogt import lane_gt  # noqa: E402

frames = QE.val_frames()[::40][:20]
t0 = time.time()
n_pieces = n_pts = 0
for f in frames:
    g = f.lane()
    r = f.ref()
    assert len(g["pieces"]) > 0 and len(r.pts) > 0
    n_pieces += len(g["pieces"])
    n_pts += len(r.pts)
print(f"{len(frames)} frames read through the link, {n_pieces} pieces, {n_pts} points, {time.time() - t0:.1f} s ({(time.time() - t0) / len(frames):.2f} s per frame)")
print("default store signature:", lane_gt.signature())
held = lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature()))
print("frames held in the default store:", len(held))
print("stored share of the 873 VAL frames:", round(QE.stored_share(QE.val_frames()), 3))
