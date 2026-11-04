import glob, numpy as np
f = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[0]
g = np.load(f, allow_pickle=True)["gt"].item()
def show(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pre}{k}/"); show(v, pre + "  ")
        elif hasattr(v, "shape"):
            print(f"{pre}{k:24s} {v.dtype} {v.shape}")
        elif isinstance(v, list):
            print(f"{pre}{k:24s} list[{len(v)}] {repr(v[:2])[:150]}")
        else:
            print(f"{pre}{k:24s} {repr(v)[:150]}")
show(g)
