"""scratch: two keyframe pickles (old and new code) must be identical in every array and every meta entry"""
import pickle
import sys

import numpy as np

a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
print("old", a["file"], a["sig"], "| new", b["file"], b["sig"])
assert a["kf"].keys() == b["kf"].keys()


def same(x, y, path):
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray) and x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), path
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), (path, sorted(set(x) ^ set(y)))
        for k in x:
            same(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), path
        for i, (u, v) in enumerate(zip(x, y)):
            same(u, v, f"{path}[{i}]")
    elif isinstance(x, float) and np.isnan(x):
        assert isinstance(y, float) and np.isnan(y), path
    else:
        assert x == y, (path, x, y)


for tok in a["kf"]:
    same(a["kf"][tok], b["kf"][tok], tok[:8])
print(f"identical: {len(a['kf'])} keyframes, every layer and every meta entry")
