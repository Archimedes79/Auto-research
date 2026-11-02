import sys, numpy as np
sys.path[:0] = [".", ".."]
import evaluation.bev_gtcheck as G
from evaluation import bev_fusion_area as FA
from evaluation.metrics import gt_boxes_from_meta
from autogt import coverage as AC
from autogt.grid import GridSpec
grid = GridSpec(res=0.5)
d = {}
mism = {}
for name, s in G.pick(60, G.resolve_sig(None), 0):
    g = s[0.0]; area = FA.fusion_area(g)
    raw = g["meta"]["boxes"]
    for rb, b in zip(raw, gt_boxes_from_meta(raw, observed_only=False)):
        a = FA.footprint_share(b, area)
        c = AC.footprint_share(rb["corners"], AC.fusion_cells(g), grid)
        k = "circle" if "radius" in b else "box"
        d.setdefault(k, []).append((a, c))
for k, v in d.items():
    v = np.array(v); cls = lambda x: np.where(x >= .5, 2, np.where(x >= .1, 1, 0))
    print(k, len(v), "max |diff|", np.abs(v[:,0]-v[:,1]).max().round(3), "class mismatches", int((cls(v[:,0]) != cls(v[:,1])).sum()))
