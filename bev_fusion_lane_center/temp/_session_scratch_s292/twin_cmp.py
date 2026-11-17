import json, pickle, sys
import os
R = os.environ["BFLC_ROOT"]
S = sys.argv[1]
a = json.load(open(R + f"/runs/e324_s_s{S}/ends_card_snap_nogeom_attach_thr0.1.json"))
b = json.load(open(R + f"/runs/e324_s_s{S}/ends_card_twinatt_snap_nogeom_attach_thr0.1.json"))
def walk(x, y, path=""):
    bad = []
    if isinstance(x, dict):
        for k in x:
            if k in ("date", "seconds"):
                continue
            if k not in y:
                bad.append(path + "/" + k + " missing"); continue
            bad += walk(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if len(x) != len(y): bad.append(path + " len")
        else:
            for i, (u, v) in enumerate(zip(x, y)): bad += walk(u, v, f"{path}[{i}]")
    elif isinstance(x, float):
        if x != y: bad.append(f"{path} {x} vs {y}")
    elif x != y:
        bad.append(f"{path} {x!r} vs {y!r}")
    return bad
d = walk(a, b)
print("card fields differing:", len(d), d[:8])
ca = pickle.load(open(R + f"/runs/e324_s_s{S}/ends_card_snap_nogeom_attach_thr0.1.counts.pkl", "rb"))
cb = pickle.load(open(R + f"/runs/e324_s_s{S}/ends_card_twinatt_snap_nogeom_attach_thr0.1.counts.pkl", "rb"))
import numpy as np
def eq(u, v):
    if isinstance(u, dict): return u.keys() == v.keys() and all(eq(u[k], v[k]) for k in u)
    if isinstance(u, (list, tuple)): return len(u) == len(v) and all(eq(p, q) for p, q in zip(u, v))
    try: return bool(np.array_equal(np.asarray(u), np.asarray(v)))
    except Exception: return u == v
print("counts pkl identical:", eq(ca, cb))
