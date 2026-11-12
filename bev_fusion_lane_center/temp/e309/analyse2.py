import os, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", "perframe_s0.pkl"), "rb"))
g = lambda k, f: np.array([r[k]["num"][f] for r in PF])
l = lambda k, f: np.array([r[k]["len"][f] for r in PF])
for k in ("B", "e284", "e287p2", "e312"):
    print(f"{k:7s} per-frame mean topo_F1 {g(k,'topo_F1').mean():.3f} F1@1 {g(k,'F1@1.0').mean():.3f} lanes_found {g(k,'lanes_found').mean():.3f} junc_F1 {g(k,'junction_F1').mean():.3f} | lenient 2 m cov {np.nanmean(l(k,'cov')):.3f} prec {np.nanmean(l(k,'prec')):.3f}")
d = g("e312", "topo_F1") - g("B", "topo_F1")
print("E312 - B topo F1 per frame: improved (> +0.02)", (d > .02).sum(), " equal", (abs(d) <= .02).sum(), " worse (< -0.02)", (d < -.02).sum())
o = np.argsort(d)
sel = {"gain": [int(i) for i in o[::-1][:3]], "loss": [int(i) for i in o[:2]], "median": [int(o[len(o)//2])]}
for k, v in sel.items():
    for i in v: print(k, i, PF[i]["token"][:8], "scene", PF[i]["scene"], f"B {PF[i]['B']['num']['topo_F1']:.2f} -> E312 {PF[i]['e312']['num']['topo_F1']:.2f}", "GT pieces", PF[i]["n_gt_lines"])
print("tokens:", " ".join(PF[i]["token"] for v in sel.values() for i in v))
