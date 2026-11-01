import sys, os, json, importlib.util
import numpy as np
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import bootstrap as NB, topo_metrics as NT
OB = load("old_bootstrap", f"{S}/old_bootstrap.py"); OT = load("old_topo", f"{S}/old_topo.py")
from evalsuite import stats as ST
# RandomState.choice vs randint stream
for n in (1, 7, 82, 150):
    a = np.random.RandomState(0); b = np.random.RandomState(0)
    us = np.arange(n) * 3 + 5
    same = all(np.array_equal(a.choice(us, n, replace=True), us[b.randint(0, n, n)]) for _ in range(200))
    print("choice==randint", n, same)
rng = np.random.RandomState(1)
nf = 400
scenes = np.array([f"s{rng.randint(0,30):02d}" for _ in range(nf)])
X = rng.rand(nf, 3); Y = rng.rand(nf, 3)
def mkpooled(nanmode):
    def pooled(rec, idx=None):
        r = np.asarray(rec) if not isinstance(rec, list) else np.asarray(rec)
        sel = r if idx is None else r[np.asarray(idx)]
        m = {"S": float(sel[:,0].mean()), "Q": float(sel[:,1].mean()), "vert100": float(sel[:,2].sum()/len(sel))}
        if nanmode:
            m["Q"] = float("nan") if sel[:,1].max() > 0.9995 else m["Q"]
        return m
    return pooled
def topopooled_factory(nanmode):
    f = mkpooled(nanmode)
    def pooled(recs):
        a = np.asarray(recs)
        m = f(a)
        return {"topo_F1": m["S"], "topo_R": m["Q"], "topo_P": m["vert100"], "dup_frac": m["S"]*0.5, "R@1_far": m["Q"], "R@1_mid": m["S"], "R@1_near": m["S"]}
    return pooled
for nanmode in (False, True):
    for mod in (NB, OB):
        mod.pooled = mkpooled(nanmode)
    keys = ("S", "Q", "vert100")
    o = OB.paired(list(X), list(Y), scenes, B=300, keys=keys); n = NB.paired(list(X), list(Y), scenes, B=300, keys=keys)
    print("paired nan" if nanmode else "paired", json.dumps(o["S"]) == json.dumps(n["S"]), {k: (o[k], n[k]) for k in keys if json.dumps(o[k], sort_keys=True) != json.dumps(n[k], sort_keys=True)})
    o = OB.interval(list(X), scenes, B=300, keys=keys); n = NB.interval(list(X), scenes, B=300, keys=keys)
    print("interval", {k: (o[k], n[k]) for k in keys if json.dumps(o[k], sort_keys=True) != json.dumps(n[k], sort_keys=True)})
    for mod in (NT, OT):
        mod.pooled = topopooled_factory(nanmode)
    o = OT.paired(list(X), list(Y), scenes, B=300); n = NT.paired(list(X), list(Y), scenes, B=300)
    print("topo", {k: (o[k], n[k]) for k in o if json.dumps(o[k], sort_keys=True) != json.dumps(n[k], sort_keys=True)})
# evalsuite bootstrap_scenes: a = control, b = variant; direction
OS = load("old_stats", f"{S}/old_stats.py")
ctrl = {f"s{i}": [np.array([1.0, 2.0])] for i in range(20)}
var = {f"s{i}": [np.array([1.5 + 0.01*i, 2.0])] for i in range(20)}
def reduce(counts, keys=None):
    a = np.sum(counts, 0) if counts else np.zeros(2)
    return {"err": float(a[0] / a[1]) if a[1] else float("nan")}
r = ST.bootstrap_scenes(ctrl, var, reduce, ["err"], n=200, better={"err": "low"})
print("evalsuite low-better variant worse:", r["err"])
r2 = OS.bootstrap_scenes(ctrl, var, reduce, ["err"], n=200)
print("old:", r2["err"])
