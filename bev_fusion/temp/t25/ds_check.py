"""scratch: run the radar_per_frame part of nuscenes_dataset's selftest alone (the full selftest stops earlier in the GT code)"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))
import nuscenes_dataset as nd  # noqa: E402
from lift_pd import Grid  # noqa: E402

t0 = time.time()
root = os.path.expanduser("~/data/nuscenes")
ds = nd.FrontRadarDataset(root, "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=False)
print("mean points in the grid per keyframe with one sweep:", nd._selftest_per_frame(ds, root), f"({time.time() - t0:.0f} s)")
