import sys
from dataclasses import replace
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/training")
import torch, losses as L
torch.manual_seed(0)
spec = L.layer_spec(False, True)
B, X, Y = 2, 10, 12
T = {}
for n, k, key in spec:
    if k in ("area", "line", "object"): T[key] = (torch.rand(B, 1, X, Y) > 0.9).float()
    if k == "dist": T[key] = torch.rand(B, 1, X, Y) * 2
wc = torch.ones(B, 1, X, Y)
lg = torch.randn(B, len(spec), X, Y)
c = L.LossConfig(line_bce=10.0)
t, g = L.map_terms(lg, T, spec, {"area": wc, "line": wc, "object": wc}, c)
print("cam_aux map_terms gets:", [k for k in t if k.startswith("line_bce")], "lines group", float(g["lines"]))
print("w_cam_aux", c.w_cam_aux, "-> fixed term = 0.3 * sum(groups) without Kendall; line_bce share of it:", 0.3 * 10 * float(torch.stack([t[k] for k in t if k.startswith('line_bce')]).mean()))
