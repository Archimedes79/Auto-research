import os, sys, time
sys.path.insert(0, "training")
from gt_bev import NuScenesGT
from lift_pd import Grid
src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
tok = src.nusc.scene[3]["first_sample_token"]
for i in range(4):
    t = time.time(); src.keyframe_gt(tok, Grid()); print(f"keyframe_gt {time.time()-t:.1f} s", flush=True)
    tok = src.nusc.get("sample", tok)["next"]
