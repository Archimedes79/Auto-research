import numpy as np, glob
fs = sorted(glob.glob('data/cache/gt/*_15efbde2d1.npz'))
g = np.load(fs[0], allow_pickle=True)['gt'].item()
print(type(g), list(g)[:6])
def show(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict) and pre.count("/") < 1:
            show(v, pre + k + "/")
        elif hasattr(v, "shape"):
            print(pre + k, v.shape, v.dtype, round(float(np.nanmin(v.astype(np.float32))), 2), round(float(np.nanmax(v.astype(np.float32))), 2))
        else:
            print(pre + k, type(v).__name__, str(v)[:80])
show(g)
