"""the new spec tests must fail on the old code (L1 graph_from_vertices, L3 skeletons)"""
import os, subprocess, types, sys
R = os.environ["BFLC_ROOT"]
def old_module(path, name):
    src = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "show", "HEAD:bev_fusion_lane_center/" + path], capture_output=True, text=True).stdout
    m = types.ModuleType(name); exec(compile(src, name, "exec"), m.__dict__); return m
SMo = old_module("evaluation/sparse_lanes/sparse_metrics.py", "sm_old")
import numpy as np
xy = np.stack([np.arange(8.0), np.zeros(8)], 1)
print("old fork:", SMo.graph_from_vertices(xy, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)])["nodes"])
print("old cycle:", SMo.graph_from_vertices(xy, [(1, 2), (2, 0), (0, 1)])["nodes"])
import sparse_metrics as SM
print("new fork:", SM.graph_from_vertices(xy, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)])["nodes"])
GRo = old_module("evaluation/lane_grid/grid_eval.py", "gr_old")
GRo.dev = "cpu"
D = np.full((1, 200, 200), 10.0, np.float32); D[0, 195:198, 180:199] = 1.0
print("old skeletons keep far seed:", bool(GRo.skeletons(D, thr=0.5, low=0.8, rg=1.0)[0][195:198, 180:199].any()))
