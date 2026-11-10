"""scratch: structure of runs/t20_base_s0/eval/scene_records_n1000.pkl (what a record holds)"""
import os, pickle, sys
import numpy as np
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path[:0] = [ROOT, f"{ROOT}/training", "/mnt/c/Auto-research", "/mnt/c/Auto-research/autoresearch"]
p = f"{ROOT}/runs/t20_base_s0/eval/scene_records_n1000.pkl"
print("size MB", round(os.path.getsize(p) / 1e6, 1))
d = pickle.load(open(p, "rb"))
print("type", type(d).__name__, "len", len(d) if hasattr(d, "__len__") else "-")
def show(o, depth=0, key=""):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key} dict {len(o)}: {list(o)[:12]}")
        if depth < 3:
            for k in list(o)[:6]:
                show(o[k], depth + 1, str(k))
    elif isinstance(o, (list, tuple)):
        print(f"{pad}{key} {type(o).__name__} {len(o)}")
        if o and depth < 3:
            show(o[0], depth + 1, "[0]")
    elif isinstance(o, np.ndarray):
        print(f"{pad}{key} ndarray {o.shape} {o.dtype}")
    else:
        print(f"{pad}{key} {type(o).__name__} {str(o)[:80]}")
show(d)
