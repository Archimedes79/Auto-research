import os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
o, n = (pickle.load(open(os.path.join(HERE, f"out_{w}.pkl"), "rb")) for w in ("old", "new"))


def same_counts(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_counts(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_counts(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b))
    return a == b or (a != a and b != b)


print("counts equal:", same_counts(o["counts"], n["counts"]), len(o["counts"]), "metric/variant combinations")
red_ok = o["reduced"].keys() == n["reduced"].keys() and all(
    o["reduced"][k].keys() == n["reduced"][k].keys() and all(
        (o["reduced"][k][q] == n["reduced"][k][q]) or (o["reduced"][k][q] != o["reduced"][k][q] and n["reduced"][k][q] != n["reduced"][k][q]) for q in o["reduced"][k])
    for k in o["reduced"])
print("reduced numbers equal (same keys, same values, bit for bit):", red_ok, sum(len(v) for v in o["reduced"].values()))
print("any centreline key in the new numbers:", any("centreline" in q for v in n["reduced"].values() for q in v))
bad = [k for k in o["pixels"] if o["pixels"][k][0] != n["pixels"][k][0] or not np.array_equal(o["pixels"][k][1], n["pixels"][k][1])]
print("pictures identical:", not bad, len(o["pixels"]), "pages; differing:", bad)
