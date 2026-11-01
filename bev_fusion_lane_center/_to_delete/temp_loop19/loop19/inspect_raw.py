"""scratch: structure of a cached raw network output (raw_val873_<run>_<M>.pkl)"""
import os
import pickle

import numpy as np

p = os.path.expanduser("~/exp_gcar_b123/p18/raw_val873_e123_D7_r3_j1_192.pkl")
z = pickle.load(open(p, "rb"))
print(type(z), len(z) if hasattr(z, "__len__") else "")


def show(o, ind=0, maxn=14):
    pad = " " * ind
    if isinstance(o, dict):
        for k, v in list(o.items())[:maxn]:
            print(f"{pad}{k}: {type(v).__name__} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
            if isinstance(v, (dict, list, tuple)) and ind < 4:
                show(v, ind + 2)
    elif isinstance(o, (list, tuple)):
        print(f"{pad}[{len(o)}] first: {type(o[0]).__name__} {getattr(o[0], 'shape', '')}")
        if len(o) and isinstance(o[0], (dict, list, tuple)) and ind < 4:
            show(o[0], ind + 2)


show(z)
