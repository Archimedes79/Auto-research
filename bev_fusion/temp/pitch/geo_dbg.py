import sys, numpy as np
sys.path[:0] = ["."]
import evaluation.bev_gtcheck as G
from evaluation.bev_fusion_area import footprint_share, gt_with_shares
from evaluation.metrics import in_front
vals = {t: [] for t in (0.5, 0.1, 0.0)}
for _, s in G.pick(50, G.resolve_sig(None), 0):
    g = s[0.0]; v = np.asarray(g["vehicle"], np.float32)
    boxes, _ = gt_with_shares(g, ["vehicle"])
    raws = [r for r in g["meta"]["boxes"] if r["group"] == "vehicle"]
    for b, r in zip(boxes, raws):
        if in_front(b) and r.get("observed", True):
            for t in vals: vals[t].append(footprint_share(b, v > t))
print({t: round(float(np.mean(x)), 3) for t, x in vals.items()}, len(vals[0.5]), v.max(), v.dtype)
