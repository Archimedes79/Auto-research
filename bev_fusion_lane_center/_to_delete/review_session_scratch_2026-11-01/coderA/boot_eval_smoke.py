"""throw-away: boot_eval imports; its W-matrix use of the shared draws equals the frame-concatenation draws of evalsuite.stats"""
import numpy as np

import boot_eval as BE
from evalsuite import stats as ST

rng = np.random.RandomState(0)
scenes = np.array(["b", "a", "c", "a", "b", "d", "d", "d", "c"])
comp = {n: rng.randint(0, 20, size=(len(scenes), 16)).astype(float) + 1 for n in ("x", "y")}
us = np.unique(scenes)
inv = np.searchsorted(us, scenes)
B = 200
W = np.stack([np.bincount(p, minlength=len(us)) for p in ST.scene_picks(len(us), B, seed=0)]).astype(np.float64)
BS, FULL = {}, {}
for n, c in comp.items():
    S = np.zeros((len(us), 16))
    np.add.at(S, inv, c)
    BS[n], FULL[n] = BE.metrics(W @ S), BE.metrics(c.sum(0))


def stat_w(n):
    return lambda b: FULL[n] if b is None else {k: v[b] for k, v in BS[n].items()}


def stat_f(n):
    return lambda idx: BE.metrics(comp[n].sum(0) if idx is None else comp[n][idx].sum(0))


pw = ST.paired_draws(stat_w("x"), stat_w("y"), range(B), BE.METRICS)
pf = ST.paired_draws(stat_f("x"), stat_f("y"), ST.scene_draws(scenes, B, 0), BE.METRICS)
for k in BE.METRICS:
    for f in ("diff", "lo", "hi", "p_gt0"):
        assert abs(pw[k][f] - pf[k][f]) < 1e-9, (k, f, pw[k][f], pf[k][f])
print("boot_eval smoke ok:", {k: round(pw[k]["lo"], 4) for k in BE.METRICS[:3]})
