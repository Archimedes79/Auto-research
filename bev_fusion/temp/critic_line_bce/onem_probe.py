import sys
from dataclasses import replace
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/training")
import torch
import losses as L
torch.manual_seed(0)
spec1 = L.layer_spec_1m(True)
print(spec1)
B, X, Y = 2, 8, 10
m = (torch.rand(B, 1, X, Y) > 0.85).float()
d = torch.where(m > 0.5, torch.zeros_like(m), torch.rand(B, 1, X, Y) * 2)
dirt = torch.randn(B, 2, X, Y)
T = {"centreline@1m": m, "centreline_dist@1m": d, "centreline_dir@1m": dirt}
lg = torch.randn(B, len(spec1), X, Y)
w = {"line": torch.ones(B, 1, X, Y)}
c0 = L.LossConfig(centreline_1m=True)
t0, g0 = L.map_terms(lg, T, spec1, w, c0)
t1, g1 = L.map_terms(lg, T, spec1, w, replace(c0, line_bce=10.0))
print("new terms in the 1 m branch:", sorted(set(t1) - set(t0)), " lines group", float(g0["lines"]), "->", float(g1["lines"]))
# the cam_aux spec (layer_spec(False, True)) also has the two dividers
cs = L.layer_spec(False, True)
print("cam_aux spec has line + dist channels:", [n for n, k, _ in cs if k in ("line", "dist")])
