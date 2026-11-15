import sys, numpy as np
sys.path.insert(0, "."); 
from evaluation import bev_junction as J
good = J.junction_free
def mk(kind):
    def bad(pred, gt, fov, inter):
        keep = ~np.asarray(inter, bool)
        cut = lambda d: {n: np.where(keep, np.asarray(d[n]), 0) for n in J.JUNCTION_LAYERS if n in d}
        p, g, f = cut(pred), cut(gt), np.asarray(fov, bool) & keep
        if kind == "gt_not_cut": g = {n: np.asarray(gt[n]) for n in g}
        if kind == "pred_not_cut": p = {n: np.asarray(pred[n]) for n in p}
        if kind == "fov_not_cut": f = np.asarray(fov, bool)
        if kind == "all_layers": p = {n: np.where(keep, v, 0) for n, v in pred.items()}; g = {n: np.where(keep, v, 0) for n, v in gt.items()}
        if kind == "inverted": 
            k2 = np.asarray(inter, bool); p = {n: np.where(k2, np.asarray(pred[n]), 0) for n in p}; g = {n: np.where(k2, np.asarray(gt[n]), 0) for n in g}; f = np.asarray(fov, bool) & k2
        return p, g, f
    return bad
for kind in ("gt_not_cut", "pred_not_cut", "fov_not_cut", "all_layers", "inverted"):
    J.junction_free = mk(kind)
    try:
        J.selftest(); print("ESCAPED", kind)
    except AssertionError as e:
        print("caught ", kind)
J.junction_free = good
print(J.selftest())
