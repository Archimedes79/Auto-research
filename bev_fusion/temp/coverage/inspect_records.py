"""What the stored eval records of a run hold per frame: keys, the GT boxes, any camera-cell mask (for the footprint-share study of D38 b). Run from bev_fusion (WSL, gcar python, CPU)."""
import pickle
import sys

import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else "runs/t10_dual_s0/eval/scene_records_n1000.pkl"
recs = pickle.load(open(p, "rb"))
print(type(recs).__name__, len(recs))
r0 = recs[0] if isinstance(recs, (list, tuple)) else next(iter(recs.values()))
print("record type", type(r0).__name__)


def show(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                print(f"{pre}{k}: ndarray {v.shape} {v.dtype}")
            elif isinstance(v, dict):
                print(f"{pre}{k}: dict {list(v)[:8]}")
            elif isinstance(v, (list, tuple)):
                print(f"{pre}{k}: {type(v).__name__} len {len(v)}" + (f" first {str(v[0])[:200]}" if v else ""))
            else:
                print(f"{pre}{k}: {type(v).__name__} {str(v)[:100]}")


show(r0)
