"""E280 scratch: what does temp/e276/predfull_e273_p_s0.pkl hold (one look, then the pre-check is designed)"""
import pickle, sys
import numpy as np

p = sys.argv[1]
with open(p, "rb") as fh:
    d = pickle.load(fh)
print("type", type(d), "len", len(d) if hasattr(d, "__len__") else "-")


def show(x, depth=0, name=""):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{name} dict[{len(x)}] keys={list(x.keys())[:30]}")
        if depth < 3:
            for k in list(x.keys())[:30]:
                show(x[k], depth + 1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name} {type(x).__name__}[{len(x)}]")
        if len(x) and depth < 3:
            show(x[0], depth + 1, "[0]")
    elif isinstance(x, np.ndarray):
        print(f"{pad}{name} ndarray {x.shape} {x.dtype}")
    else:
        print(f"{pad}{name} {type(x).__name__} {str(x)[:80]}")


if isinstance(d, dict):
    k0 = list(d.keys())[0]
    print("first key", k0)
    show(d[k0], 0, "first value")
else:
    show(d[0], 0, "first")
