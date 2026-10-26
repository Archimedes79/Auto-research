import cProfile, os, pstats, sys
sys.path.insert(0, "training")
from gt_bev import NuScenesGT
from lift_pd import Grid
src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
tok = src.nusc.get("sample", src.nusc.scene[0]["first_sample_token"])["next"]
src.keyframe_gt(tok, Grid())
pr = cProfile.Profile(); pr.enable()
for _ in range(3):
    src.keyframe_gt(tok, Grid())
pr.disable()
pstats.Stats(pr).sort_stats("cumtime").print_stats(14)
