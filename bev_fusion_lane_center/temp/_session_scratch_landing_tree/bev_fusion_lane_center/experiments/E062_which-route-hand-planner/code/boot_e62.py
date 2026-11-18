"""Szenen-Bootstrap (gepaart, 2000 Ziehungen) der Hauptunterschiede aus E62 (verankerter Querfehler, Pose-Rauschen 1 m, Registrierung ein): TEST, alle Samples / Kurven-Verzweigungen.
Aufruf (gcar): boot_e62.py -> out/loop2/e62_boot.json"""
import json
import pickle

import numpy as np

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
D = "/home/ml/exp_gcar_b123"
P = np.load(f"{D}/poses_val.npz")
rs = pickle.load(open(f"{OUT}/e62_s1_reg/persample_test.pkl", "rb"))
sc = np.array([int(P["scene"][r["row"]]) for r in rs])
M = {m: np.array([float(np.mean(r["lata"][m])) for r in rs]) for m in ("const", "gt_route", "system", "nomap", "oracle", "maponly", "regmap")}
curve = np.array([r["curve"] for r in rs])
dev = {m: np.array([float(r["lata"][m][2] > 2.0) for r in rs]) for m in M}
scenes = np.unique(sc)
rng = np.random.default_rng(0)
pairs = [("regmap", "system"), ("regmap", "maponly"), ("regmap", "const"), ("system", "maponly"), ("system", "nomap"), ("system", "const"), ("oracle", "system")]
out = {}
for grp, mask in (("all", np.ones(len(rs), bool)), ("curve", curve), ("straight", ~curve)):
    out[grp] = {"n": int(mask.sum()), "mean": {m: float(M[m][mask].mean()) for m in M}, "dev3": {m: float(dev[m][mask].mean()) for m in M}, "diff": {}}
    for a, b in pairs:
        d = M[a] - M[b]
        stat = []
        for _ in range(2000):
            pick = rng.choice(scenes, len(scenes))
            w = np.concatenate([np.nonzero((sc == s) & mask)[0] for s in pick])
            stat.append(float(d[w].mean()))
        out[grp]["diff"][f"{a}-{b}"] = [float(d[mask].mean()), float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))]
json.dump(out, open(f"{OUT}/e62_boot.json", "w"), indent=1)
for grp, v in out.items():
    print(grp, v["n"], {k: [round(x, 2) for x in b] for k, b in v["diff"].items()})
