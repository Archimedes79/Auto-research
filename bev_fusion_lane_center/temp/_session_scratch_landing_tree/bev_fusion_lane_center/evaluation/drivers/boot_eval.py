"""Szenen-Bootstrap (Schleife 4, E54): gepaarte 95-%-Intervalle der Kopfzahlen und ihrer Differenzen.
Je System und Menge werden die Einzelergebnisse (GE.eval_sample, wie evaluate_parallel) berechnet und gepuffert (out/loop2/e54_boot/cache_<name>_<menge>.pkl); danach 2000 Ziehungen der Szenen mit Zuruecklegen (dieselbe Ziehung fuer alle Systeme)
-> Intervalle der Metriken F1@1 m, Route F, Split-/Merge-F1 (6 m), Richtung sowie der Differenzen zu einem Bezugssystem.  Aufruf (gcar): boot_eval.py [--sets val,test] [--draws 2000] [--workers 6]   Ausgabe: out/loop2/e54_boot/boot.json
Since D24 (2026-10-31) the scene draws and the intervals come from the one shared bootstrap, autoresearch/Evaluation/stats.py (scene_picks,
interval_draws, paired_draws; the RandomState stream of sparse_lanes/bootstrap.py): a re-run gives the same method on other random draws
than the stored E54 boot.json (loop 4 used numpy default_rng(0)). For new comparisons use sparse_lanes/bootstrap.py (evaluate.py --compare)."""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402

if os.path.join(BP.REPO, "autoresearch") not in sys.path:
    sys.path.append(os.path.join(BP.REPO, "autoresearch"))
from Evaluation import stats as ST  # noqa: E402  (the shared scene bootstrap)

OUT = BP.RUNS
HOME = os.path.expanduser("~")
B = f"{OUT}/e54_boot"

# (Name, Beschriftung, Graph-Ordner/Datei-Praefix unter out/loop2, Bezug fuer die Differenz)
SYSTEMS = [
    ("base", "previous final system, no map (loop 3)", "e43/graphs_rule_e41_A"),
    ("default", "+ key-point nodes, DEFAULT (E44)", "e44_cons_p/graphs"),
    ("loose", "+ key-point nodes, node priority (E44)", "e44_cut/graphs"),
    ("conn50", "+ Bezier connectors, heat >= 0.5 (E53)", "e53_f0.50/graphs"),
    ("conn60", "+ Bezier connectors, heat >= 0.6 (E53)", "e53_f0.60/graphs"),
    ("conn50dir", "+ Bezier connectors, heat >= 0.5 + direction cosine >= 0.6 (E56)", "e56_mid/graphs"),
    ("map_base", "map prior, exact pose (loop 3)", "e43/graphs_rule_e43_joint_noprop"),
    ("map_default", "map prior + key-point nodes, DEFAULT (E44)", "e44_cons_map/graphs"),
    ("map_conn50", "map prior + connectors, heat >= 0.5 (E53)", "e53_map_f0.50/graphs"),
    ("map_conn60", "map prior + connectors, heat >= 0.6 (E53)", "e53_map_f0.60/graphs"),
]
# Differenzen (Minuend, Subtrahend)
PAIRS = [("default", "base"), ("loose", "base"), ("conn50", "default"), ("conn50", "base"), ("conn60", "default"), ("conn50dir", "conn50"), ("conn50dir", "default"), ("conn60", "base"), ("map_base", "base"), ("map_default", "map_base"), ("map_default", "default"), ("map_conn50", "map_default"), ("map_conn50", "map_base"), ("map_conn60", "map_default"), ("map_conn50", "conn50")]
METRICS = ["F1@1m", "P@1m", "R@1m", "route_f", "split6_F1", "merge6_F1", "dir_acc"]


def _one(job):
    p, g, i = job
    return GE.eval_sample(p, g, i, True)


def comps(r):
    c = r["cnt"][1]  # tau = 1.0 m: (tp_pred, n_pred, tp_gt, n_gt)
    rr, rp = r.get("route_rec", np.nan), r.get("route_prec", np.nan)
    return np.array([c[0], c[1], c[2], c[3], r["spl6"][0], r["spl6"][1], r["spl6"][2], r["mrg6"][0], r["mrg6"][1], r["mrg6"][2],
                     0.0 if np.isnan(rr) else rr, 0.0 if np.isnan(rr) else 1.0, 0.0 if np.isnan(rp) else rp, 0.0 if np.isnan(rp) else 1.0,
                     r.get("dir_ok", 0), r.get("dir_n", 0)], np.float64)


def metrics(T):
    """T: (..., 16) summierte Komponenten -> dict Name -> (...) Werte"""
    d = lambda a, b: a / np.maximum(b, 1e-9)
    f1 = lambda p, r: 2 * p * r / np.maximum(p + r, 1e-9)
    P, R = d(T[..., 0], T[..., 1]), d(T[..., 2], T[..., 3])
    sp, sr = d(T[..., 4], T[..., 5]), d(T[..., 4], T[..., 6])
    mp_, mr = d(T[..., 7], T[..., 8]), d(T[..., 7], T[..., 9])
    rr, rp = d(T[..., 10], T[..., 11]), d(T[..., 12], T[..., 13])
    return {"F1@1m": f1(P, R), "P@1m": P, "R@1m": R, "route_f": f1(rr, rp), "split6_F1": f1(sp, sr), "merge6_F1": f1(mp_, mr), "dir_acc": d(T[..., 14], T[..., 15])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val,test")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--systems", default="")
    a = ap.parse_args()
    os.makedirs(B, exist_ok=True)
    scene = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"]
    want = set(a.systems.split(",")) if a.systems else None
    res = {}
    for st in a.sets.split(","):
        per = {}
        ids_ref = None
        for name, lab, pref in SYSTEMS:
            if want and name not in want:
                continue
            fn = f"{OUT}/{pref}_{st}.pkl"
            if not os.path.exists(fn):
                print(f"[{st}] {name}: {fn} fehlt - uebersprungen", flush=True)
                continue
            d = pickle.load(open(fn, "rb"))
            ids = [int(i) for i in d["ids"]]
            if ids_ref is None:
                ids_ref = ids
            assert ids == ids_ref, f"{name}: andere Sample-Reihenfolge"
            cache = f"{B}/cache_{name}_{st}.pkl"
            if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(fn):
                comp = pickle.load(open(cache, "rb"))
            else:
                gts = GD.load_gt_graphs("val", ids)
                jobs = [(p, g, i) for i, (p, g) in enumerate(zip(d["graphs"], gts))]
                with mp.get_context("fork").Pool(a.workers) as pool:
                    rs = pool.map(_one, jobs, chunksize=8)
                comp = np.stack([comps(r) for r in rs])
                pickle.dump(comp, open(cache, "wb"))
            per[name] = comp
            m = metrics(comp.sum(0))
            print(f"[{st}] {name:12s} " + " ".join(f"{k} {float(v):.3f}" for k, v in m.items()), flush=True)
        sc = scene[np.array(ids_ref)]
        us = np.unique(sc)
        inv = np.searchsorted(us, sc)
        W = np.stack([np.bincount(p, minlength=len(us)) for p in ST.scene_picks(len(us), a.draws, seed=0)]).astype(np.float64)  # draw b: how often each scene is in it
        Tb, FULL, BS = {}, {}, {}
        out = dict(n_scenes=int(len(us)), n_samples=int(len(ids_ref)), systems={}, pairs={})

        def stat(b, n):
            """numbers of system n on all frames (b None) or on scene draw b (the draws are the rows of W)"""
            return FULL[n] if b is None else {k: v[b] for k, v in BS[n].items()}

        for name, comp in per.items():
            S = np.zeros((len(us), comp.shape[1]))
            np.add.at(S, inv, comp)
            Tb[name] = W @ S
            FULL[name], BS[name] = metrics(comp.sum(0)), metrics(Tb[name])
            iv = ST.interval_draws(lambda b, n=name: stat(b, n), range(a.draws), METRICS)
            out["systems"][name] = {k: dict(value=float(v["value"]), lo=v["lo"], hi=v["hi"]) for k, v in iv.items()}
        for x, y in PAIRS:
            if x in per and y in per:
                pr = ST.paired_draws(lambda b: stat(b, x), lambda b: stat(b, y), range(a.draws), METRICS)
                out["pairs"][f"{x}-{y}"] = {k: dict(diff=v["diff"], lo=v["lo"], hi=v["hi"], p_pos=v["p_gt0"]) for k, v in pr.items()}
        res[st] = out
    prev = json.load(open(f"{B}/boot.json")) if os.path.exists(f"{B}/boot.json") else {}
    prev.update(res)
    json.dump(prev, open(f"{B}/boot.json", "w"), indent=1)
    for st, out in res.items():
        print(f"== {st}: {out['n_scenes']} scenes, {out['n_samples']} samples")
        for k, v in out["pairs"].items():
            print("  ", k, " ".join(f"{m} {v[m]['diff']:+.3f} [{v[m]['lo']:+.3f},{v[m]['hi']:+.3f}]" for m in ("F1@1m", "route_f", "split6_F1", "merge6_F1")))


if __name__ == "__main__":
    main()
