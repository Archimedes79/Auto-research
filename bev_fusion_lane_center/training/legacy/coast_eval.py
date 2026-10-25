"""Schleife 5 / E68: Ausfall der Wahrnehmung ueberbruecken - den zuletzt guten Spurgraphen per Eigenbewegung weitergeben ('coasting').
Szenario wie E67: je TEST-Szene bis zur Mitte die sauberen Graphen, danach die gestoerten (E57). Fuer ein Bild im Ausfall: das juengste Bild davor (oder dasselbe), das der Monitor als gut einstuft (p <= 0.5; Orakel: Graph-F1 >= 0.6), wird mit der Eigenbewegung (Posen) in das aktuelle Bild gefaltet.
Verfahren: perceived (gestoerter Graph) / coast_mon (gefaltet, Tor = Monitor) / coast_oracle / switch_mon (wahrgenommen, solange p <= 0.5, sonst gefaltet) / switch_oracle / merge_mon (Vereinigung bei p > 0.5).
Metrik: Punkt-F1 @1 m (Praezision, Recall; Summe ueber die Bilder), getrennt nach Phase und nach Alter des Quellbildes (s); Kilometerzaehler-Rauschen optional (--odo).
Aufruf (gcar): coast_eval.py --conds dark,... --out e68_main [--odo 0]"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_eval as AE  # noqa: E402
import graph_eval as GE  # noqa: E402
import hold_eval as HE  # noqa: E402

D, OUT = AE.D, AE.OUT
_G = {}
POLS = ["perceived", "coast_mon", "coast_oracle", "switch_mon", "switch_oracle", "merge_mon"]
AGE_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 1e9)]


def warp_pieces(pts, Rp, tp, Rc, tc, lim=48.0, min_pts=2):
    """Polylinien aus dem Vorgaenger-System ins aktuelle: p_cur = Rc^T (tp + Rp p - tc), auf |x|,|y| < lim beschnitten"""
    out = []
    for p in pts:
        p = np.asarray(p, np.float64)
        if len(p) < 2:
            continue
        w = ((p @ Rp.T) + tp - tc) @ Rc
        ok = (np.abs(w) < lim).all(1)
        a = 0
        while a < len(ok):
            if not ok[a]:
                a += 1
                continue
            b = a
            while b < len(ok) and ok[b]:
                b += 1
            if b - a >= min_pts:
                out.append(w[a:b].astype(np.float32))
            a = b
    return out


def counts(pred_pts, gt, tau=1.0):
    """(Treffer Vorhersage, Zahl Vorhersage, Treffer Soll, Zahl Soll) fuer das ganze Bild und nur vor dem Fahrzeug (y > 0): 8 Zahlen"""
    P, _, _ = GE.dense_points(dict(pts=pred_pts, edges=[]))
    G, _, _ = GE.dense_points(gt)
    if len(P) == 0 or len(G) == 0:
        fp, fg = int((P[:, 1] > 0).sum()) if len(P) else 0, int((G[:, 1] > 0).sum()) if len(G) else 0
        return (0, len(P), 0, len(G), 0, fp, 0, fg)
    dp = cKDTree(G).query(P)[0]
    dg = cKDTree(P).query(G)[0]
    mp_, mg_ = P[:, 1] > 0, G[:, 1] > 0
    return (int((dp <= tau).sum()), len(P), int((dg <= tau).sum()), len(G), int(((dp <= tau) & mp_).sum()), int(mp_.sum()), int(((dg <= tau) & mg_).sum()), int(mg_.sum()))


def run_scene(args):
    cond, sc, seq = args
    a = _G["a"]
    P, GT = _G["P"], _G["GT"]
    T, R_, ts = P["t"], P["R"], P["ts"]
    rng = np.random.default_rng(AE.seed_of(f"coast{sc}", 5))
    n = len(seq)
    f0 = n // 2
    graphs, pfs, fails = [], [], []
    for f, (row, kc, kp) in enumerate(seq):
        fp = f >= f0
        graphs.append(_G["pert"][cond][row] if fp else _G["clean"][row])
        pfs.append(float(_G["pf"][cond][kp]) if fp else float(_G["pf"]["stored"][kc]))
        fails.append(bool(_G["fail"][cond][kp]) if fp else bool(_G["fail"]["stored"][kc]))
    recs = []
    last_mon, last_orc = None, None
    for f, (row, kc, kp) in enumerate(seq):
        gt = GT[row]
        ex = graphs[f]
        if pfs[f] <= 0.5:
            last_mon = f
        if not fails[f]:
            last_orc = f
        res = dict(f=f, row=int(row), fail_phase=bool(f >= f0), pf=pfs[f], fail_true=fails[f], counts={}, age={}, src={})
        coast = {}
        for key, src in (("mon", last_mon), ("oracle", last_orc)):
            if src is None:
                coast[key] = None
                continue
            r0 = seq[src][0]
            age = float(ts[row] - ts[r0])  # Sekunden
            Rp, tp = R_[r0], T[r0]
            if a.odo > 0 and src != f:  # Kilometerzaehler-Fehler: Gier N(0, odo grad * sqrt(Alter s)), Versatz N(0, 1 % der Strecke)
                dyaw = np.deg2rad(rng.normal(0.0, a.odo * np.sqrt(max(age, 0.0))))
                c, s = np.cos(dyaw), np.sin(dyaw)
                Rp = Rp @ np.array([[c, -s], [s, c]])
                tp = tp + rng.normal(0.0, 0.01 * np.linalg.norm(T[row] - T[r0]) + 1e-6, 2)
            coast[key] = (warp_pieces(graphs[src]["pts"], Rp, tp, R_[row], T[row]), age)
            res["age"][key] = age
            res["src"][key] = int(src)
        per = list(ex["pts"])
        res["counts"]["perceived"] = counts(per, gt)
        for key, name in (("mon", "coast_mon"), ("oracle", "coast_oracle")):
            res["counts"][name] = counts(coast[key][0], gt) if coast[key] is not None else res["counts"]["perceived"]
        res["counts"]["switch_mon"] = res["counts"]["coast_mon"] if (pfs[f] > 0.5 and coast["mon"] is not None) else res["counts"]["perceived"]
        res["counts"]["switch_oracle"] = res["counts"]["coast_oracle"] if (fails[f] and coast["oracle"] is not None) else res["counts"]["perceived"]
        res["counts"]["merge_mon"] = counts(per + coast["mon"][0], gt) if (pfs[f] > 0.5 and coast["mon"] is not None) else res["counts"]["perceived"]
        recs.append(res)
    return dict(cond=cond, scene=int(sc), recs=recs, f0=f0)


def f1(c):
    out = {}
    for suf, (tp_p, n_p, tp_g, n_g) in (("", c[:4]), ("_fwd", c[4:])):
        P_, R_ = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
        out.update({"F1" + suf: float(2 * P_ * R_ / max(P_ + R_, 1e-9)), "P" + suf: float(P_), "R" + suf: float(R_)})
    return out


def pool(recs, pol, flt):
    c = np.zeros(8)
    for r in recs:
        if flt(r):
            c += np.array(r["counts"][pol], float)
    return f1(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="dark,cam_front3_off,blur,cam_front_off,fog,radar_off,extr3")
    ap.add_argument("--odo", type=float, default=0.0, help="Gierfehler des Kilometerzaehlers in Grad je sqrt(s) (Versatz 1 % der Strecke)")
    ap.add_argument("--scenecv", type=int, default=0, help="1: Monitor nach Szenen kreuzangepasst")
    ap.add_argument("--min_frames", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    S = pickle.load(open(f"{OUT}/e64_monitors/signals.pkl", "rb"))
    pf, _ = HE.fit_monitor(S, folds=HE.scene_folds(S, P)) if a.scenecv else HE.fit_monitor(S)
    fail = {c: S[c]["f1"] < 0.6 for c in S}
    clean = pickle.load(open(f"{OUT}/e57_stored/graphs_test.pkl", "rb"))
    ids_c = [int(i) for i in clean["ids"]]
    kc = {r: k for k, r in enumerate(ids_c)}
    clean_g = {r: g for r, g in zip(ids_c, clean["graphs"])}
    conds = a.conds.split(",")
    pert, kp = {}, {}
    for c in conds:
        g = pickle.load(open(f"{OUT}/e57_{c}/graphs_test.pkl", "rb"))
        ids = [int(i) for i in g["ids"]]
        pert[c] = {r: x for r, x in zip(ids, g["graphs"])}
        kp[c] = {r: k for k, r in enumerate(ids)}
    _G.update(a=a, P=P, GT=GT, clean=clean_g, pert=pert, pf=pf, fail=fail)
    res = dict(args=vars(a), pols=POLS, conds={})
    for c in conds:
        rows = sorted(kp[c].keys())
        by = {}
        for r in rows:
            by.setdefault(int(P["scene"][r]), []).append(r)
        jobs = []
        for sc, rr in sorted(by.items()):
            if len(rr) >= a.min_frames:
                rr = sorted(rr, key=lambda i: P["ts"][i])
                jobs.append((c, sc, [(r, kc[r], kp[c][r]) for r in rr]))
        with mp.get_context("fork").Pool(a.workers) as pool_:
            sc_res = pool_.map(run_scene, jobs, chunksize=1)
        recs = [r for s in sc_res for r in s["recs"]]
        out = dict(n_scenes=len(sc_res), n_fail=int(sum(r["fail_phase"] for r in recs)), n_clean=int(sum(not r["fail_phase"] for r in recs)), true_fail_share=float(np.mean([r["fail_true"] for r in recs if r["fail_phase"]])))
        out["fail"] = {p: pool(recs, p, lambda r: r["fail_phase"]) for p in POLS}
        out["clean"] = {p: pool(recs, p, lambda r: not r["fail_phase"]) for p in POLS}
        # nach Alter des Quellbildes (nur Ausfallphase)
        for key, name in (("mon", "coast_mon"), ("oracle", "coast_oracle")):
            rows_ = []
            for lo, hi in AGE_BINS:
                cc = np.zeros(8)
                nn = 0
                for r in recs:
                    flagged = r["pf"] > 0.5 if key == "mon" else r["fail_true"]  # nur Bilder, die wirklich weitergegeben werden (Quelle strikt aelter)
                    if r["fail_phase"] and flagged and key in r["age"] and lo <= r["age"][key] < hi:
                        cc += np.array(r["counts"][name], float)
                        nn += 1
                rows_.append(dict(n=nn, counts=cc.tolist(), **f1(cc)) if nn else dict(n=0, F1=float("nan"), P=float("nan"), R=float("nan"), F1_fwd=float("nan"), P_fwd=float("nan"), R_fwd=float("nan")))
            out[f"age_{name}"] = rows_
        # Wahrgenommen (Referenz) im selben Alters-Raster nach demselben Bild (gleiche Bilder)
        rows_ = []
        for lo, hi in AGE_BINS:
            cc = np.zeros(8)
            nn = 0
            for r in recs:
                if r["fail_phase"] and r["fail_true"] and "oracle" in r["age"] and lo <= r["age"]["oracle"] < hi:
                    cc += np.array(r["counts"]["perceived"], float)
                    nn += 1
            rows_.append(dict(n=nn, counts=cc.tolist(), **f1(cc)) if nn else dict(n=0, F1=float("nan"), P=float("nan"), R=float("nan"), F1_fwd=float("nan"), P_fwd=float("nan"), R_fwd=float("nan")))
        out["age_perceived"] = rows_
        res["conds"][c] = out
        print(f"[{c}] fail frames {out['n_fail']} (true failures {out['true_fail_share']:.2f}) F1: " + " ".join(f"{p} {out['fail'][p]['F1']:.3f}/{out['fail'][p]['F1_fwd']:.3f}" for p in POLS) + " (full/forward) | clean: " + " ".join(f"{p} {out['clean'][p]['F1']:.3f}" for p in ("perceived", "switch_mon", "switch_oracle")), flush=True)
        pickle.dump(sc_res, open(f"{OUT}/{a.out}/frames_{c}.pkl", "wb"))
        json.dump(res, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)
    mean = {ph: {p: float(np.mean([res["conds"][c][ph][p]["F1"] for c in conds])) for p in POLS} for ph in ("clean", "fail")}
    mean["fail_fwd"] = {p: float(np.mean([res["conds"][c]["fail"][p]["F1_fwd"] for c in conds])) for p in POLS}
    mean["clean_fwd"] = {p: float(np.mean([res["conds"][c]["clean"][p]["F1_fwd"] for c in conds])) for p in POLS}
    res["mean"] = mean
    json.dump(res, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)
    print("Mittel Ausfallphase:", {p: round(v, 3) for p, v in mean["fail"].items()}, flush=True)


if __name__ == "__main__":
    main()
