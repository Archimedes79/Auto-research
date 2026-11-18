"""Loop 11 / E105 evaluation (CPU): graphs of p11_temporal.py against the ground truth in the protocol of boot_eval.py (micro F1 at 1 m, scene-level bootstrap, paired differences against 'single').
Phases: cl = first half of every scene (clean features), fail = second half (perturbed features). Rule baseline: the loose rule graph (clean cache / E57 graphs of the perturbation).
Usage (gcar): p11_eval.py --out e105_a"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_eval as BE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import p8_lib as L  # noqa: E402

OUT = L.OUT
M = ["F1@1m", "P@1m", "R@1m", "split6_F1", "merge6_F1", "dir_acc"]


def per_sample(graphs, gts, workers=6):
    jobs = [(p, g, i) for i, (p, g) in enumerate(zip(graphs, gts))]
    with mp.get_context("fork").Pool(workers) as pool:
        rs = pool.map(BE._one, jobs, chunksize=8)
    return np.stack([BE.comps(r) for r in rs])


def agg(comp, mask):
    m = BE.metrics(comp[mask].sum(0))
    return {k: float(m[k]) for k in M}


def boot(comp_a, comp_b, scene, mask, draws, rng, key="F1@1m"):
    """paired scene bootstrap of the difference a - b of a metric over the rows in mask"""
    us = np.unique(scene[mask])
    sc = scene[mask]
    inv = np.searchsorted(us, sc)
    Sa, Sb = np.zeros((len(us), comp_a.shape[1])), np.zeros((len(us), comp_a.shape[1]))
    np.add.at(Sa, inv, comp_a[mask])
    np.add.at(Sb, inv, comp_b[mask])
    idx = rng.integers(0, len(us), size=(draws, len(us)))
    W = np.stack([np.bincount(i, minlength=len(us)) for i in idx])
    d = BE.metrics(W @ Sa)[key] - BE.metrics(W @ Sb)[key]
    full = BE.metrics(Sa.sum(0))[key] - BE.metrics(Sb.sum(0))[key]
    return float(full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e105_a")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--perts", default="")
    a = ap.parse_args()
    src = L.load_src("test_clean")
    ids = [int(i) for i in src["ids"]]
    kof = {r: k for k, r in enumerate(ids)}
    gt_all = GD.load_gt_graphs("val", ids)
    scene_all = np.load(f"{L.D}/poses_val.npz")["scene"]
    rng = np.random.default_rng(0)
    perts = a.perts.split(",") if a.perts else sorted({os.path.basename(f).split("_")[1] if False else None for f in []})
    files = sorted(glob.glob(f"{OUT}/{a.out}/graphs_*.pkl"))
    tags = [os.path.basename(f)[7:-4] for f in files]
    known = ["cam_front3_off", "cam_front_off", "radar_off", "dark", "blur", "fog"]
    by_p = {}
    for t in tags:
        pn = next(p for p in known if t.startswith(p + "_"))
        by_p.setdefault(pn, []).append(t[len(pn) + 1 :])
    res = {}
    for pn, vars_ in by_p.items():
        if a.perts and pn not in a.perts.split(","):
            continue
        rule_p = pickle.load(open(f"{OUT}/e57_{pn}/graphs_test.pkl", "rb"))
        rule_map = {int(i): g for i, g in zip(rule_p["ids"], rule_p["graphs"])}
        res[pn] = {}
        comps, meta = {}, None
        for v in vars_:
            z = pickle.load(open(f"{OUT}/{a.out}/graphs_{pn}_{v}.pkl", "rb"))
            rows = z["rows"]
            gts = [gt_all[kof[r]] for r in rows]
            cf = f"{OUT}/{a.out}/comps_{pn}_{v}.npy"
            if os.path.exists(cf) and len(np.load(cf)) == len(rows):
                comps[v] = np.load(cf)
            else:
                comps[v] = per_sample(z["graphs"], gts, a.workers)
                np.save(cf, comps[v])
            if meta is None:
                meta = dict(rows=rows, phase=np.array(z["phase"]), scene=np.array(z["scene"]))
            assert meta["rows"] == rows
            res[pn][v] = {ph: agg(comps[v], meta["phase"] == ph) for ph in ("cl", "fail")}
            res[pn][v]["flag"] = {ph: float(np.mean(np.array(z["flag"])[meta["phase"] == ph])) for ph in ("cl", "fail")}
            print(f"[{pn}] {v:16s} cl F1 {res[pn][v]['cl']['F1@1m']:.3f} fail F1 {res[pn][v]['fail']['F1@1m']:.3f} (P {res[pn][v]['fail']['P@1m']:.3f} R {res[pn][v]['fail']['R@1m']:.3f} dir {res[pn][v]['fail']['dir_acc']:.3f}) flagged cl/fail {res[pn][v]['flag']['cl']:.2f}/{res[pn][v]['flag']['fail']:.2f}", flush=True)
        rows = meta["rows"]
        rg = [src["graphs"][kof[r]] if ph == "cl" else rule_map[r] for r, ph in zip(rows, meta["phase"])]
        rcf = f"{OUT}/{a.out}/comps_{pn}_rule.npy"
        comps["rule"] = np.load(rcf) if os.path.exists(rcf) and len(np.load(rcf)) == len(rows) else per_sample(rg, [gt_all[kof[r]] for r in rows], a.workers)
        np.save(rcf, comps["rule"])
        res[pn]["rule"] = {ph: agg(comps["rule"], meta["phase"] == ph) for ph in ("cl", "fail")}
        print(f"[{pn}] {'rule loose':16s} cl F1 {res[pn]['rule']['cl']['F1@1m']:.3f} fail F1 {res[pn]['rule']['fail']['F1@1m']:.3f}", flush=True)
        sc = scene_all[np.array(rows)]
        for v in list(vars_) + ["rule"]:
            if v == "single" or "single" not in comps:  # differences against the single frame only where it was run (no mislabelled keys)
                continue
            for ph in ("cl", "fail"):
                d = boot(comps[v], comps["single"], sc, meta["phase"] == ph, a.draws, rng)
                res[pn][v][f"diff_vs_single_{ph}"] = dict(diff=d[0], lo=d[1], hi=d[2])
        for v in vars_:  # every variant against the rule pipeline (paired scene-level interval)
            for ph in ("cl", "fail"):
                d = boot(comps[v], comps["rule"], sc, meta["phase"] == ph, a.draws, rng)
                res[pn][v][f"diff_vs_rule_{ph}"] = dict(diff=d[0], lo=d[1], hi=d[2])
        res[pn]["n"] = {ph: int((meta["phase"] == ph).sum()) for ph in ("cl", "fail")}
        res[pn]["n_scenes"] = int(len(np.unique(sc)))
    vk = lambda v: ":".join(v.rsplit("_", 1)) if v.rsplit("_", 1)[-1].replace(".", "").isdigit() else v
    res = {pn: {(vk(k) if k not in ("n", "n_scenes") else k): val for k, val in d_.items()} for pn, d_ in res.items()}
    json.dump(res, open(f"{OUT}/{a.out}/eval.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
