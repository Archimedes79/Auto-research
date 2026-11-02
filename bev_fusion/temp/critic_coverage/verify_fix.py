import numpy as np, sys
sys.path.insert(0, "bev_fusion/temp/critic_coverage")
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres, occlusion
import importlib.util
spec = importlib.util.spec_from_file_location("wf", "bev_fusion/temp/critic_coverage/wedge_fix.py")
src = open("bev_fusion/temp/critic_coverage/wedge_fix.py").read().split("rng = np.random")[0]
ns = {}; exec(src, ns)
grid = GridSpec(); gx, gy = cell_centres(grid)
for tok in ("dc4af2033d834737bd567ddb0ab02460", "26ff943010a54658b8a9471ff433c3b0"):
    g = load_keyframe("data/autogt/15efbde2d1", tok); m = g["meta"]; T = m["T_c2e"]
    hid = [b for b in m["boxes"] if b["observed"]]
    old = occlusion(gx, gy, T[:2, 3], [b["corners"] for b in hid])
    big = [(b["corners"], -500.0, 500.0) for b in hid]
    cur = C.hidden_cells(gx, gy, T[:3, 3], big)
    C._wedge = ns["wedge_fixed"]; fix = C.hidden_cells(gx, gy, T[:3, 3], big)
    print(tok, "old vs current soft:", int((old != cur).sum()), " old vs soft with fixed wedge:", int((old != fix).sum()))
    # in-view cells
    camm = g["camera"] > 0
    C._wedge = ns["wedge_fixed"]
