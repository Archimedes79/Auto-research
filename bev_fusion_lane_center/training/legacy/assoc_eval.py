"""Schleife 5 / E59: Zuordnung extrahierter Spurmitten zum Kartengraphen + Uebertragung des Navigationsziels (Route) auf die extrahierten Spuren.
Graphen des Systems (Standard: e44_cons_p/graphs_<menge>.pkl) gegen die Karte (Soll-Graph mit Token, ggf. gestoert) und die Route (route_gt_val.pkl).
Aufruf (gcar): assoc_eval.py --sets val,test --graphs e44_cons_p/graphs --sigma 0 --drop 0 --reg 1 --out e59_a  [--oracle 1] [--nomap 1] [--protect 1] [--workers 10]
Ausgabe: out/loop2/<out>/metrics.json (je Menge: Zuordnungs-Genauigkeit, Routen-F1 je Punkt, Routen-Abdeckung / -Praezision @1,5 m, Erfolgsrate, Registrierungsfehler, Erkennung 'nicht in der Karte')"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import zlib

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402

HOME = os.path.expanduser("~")
OUT = BP.RUNS
D = f"{HOME}/exp_gcar_b123"
_G = {}


def seed_of(x, salt=0):
    return (zlib.crc32(str(x).encode()) + 7919 * salt) % (2 ** 32)


def follow_straight(ex, max_len=50.0):
    """Keine Navigationsinformation: ab dem Stueck vor dem Fahrzeug an jeder Verzweigung den Nachfolger mit der kleinsten Richtungsaenderung nehmen -> Menge der Stuecke."""
    pts = [np.asarray(p, np.float64) for p in ex["pts"]]
    if not pts:
        return set()
    out_e = {}
    for i, j in ex["edges"]:
        out_e.setdefault(i, []).append(j)
    best, bd = None, 1e9
    for k, p in enumerate(pts):
        if len(p) < 2:
            continue
        q = AL.resample(p, 1.0)
        d = np.linalg.norm(q, axis=1)
        j = int(np.argmin(d))
        t = AL.tangents(q)[j]
        if d[j] < bd and t[1] > 0.5:
            best, bd = k, d[j]
    if best is None or bd > 6.0:
        return set()
    seq, cur, L = [best], best, 0.0
    while True:
        L += np.linalg.norm(np.diff(pts[cur], axis=0), axis=1).sum()
        if L > max_len:
            break
        nxt = out_e.get(cur, [])
        if not nxt:
            break
        t0 = AL.tangents(pts[cur])[-1]
        cand = [(float(AL.tangents(pts[j])[0] @ t0), j) for j in nxt if len(pts[j]) >= 2 and j not in seq]
        if not cand:
            break
        cur = max(cand)[1]
        seq.append(cur)
    return set(seq)


def one(args):
    k, row = args
    ex, gt, rt = _G["ex"][k], _G["gt"][k], _G["rt"][k]
    a = _G["a"]
    if rt is None or len(ex["pts"]) == 0:
        return None
    route_tok = {gt["tokens"][p] for p in rt["route"]}
    route_idx = set(rt["route"])
    rng = np.random.default_rng(seed_of(row, 1))
    mp_ = AL.perturb_map(gt, a.sigma, a.drop, protect=route_idx if a.protect else (), rng=rng, shift_frac=a.shift_frac)
    EV, ET, Epid, pa_gt, tok_gt = AL.gt_assign(ex, gt)
    n = len(EV)
    if n == 0:
        return None
    gt_matched = np.array([t is not None for t in tok_gt])
    gt_on = np.array([t in route_tok for t in tok_gt])
    gt_on_geo = AL.route_flags(AL.perturb_map(gt, 0.0, 0.0), route_tok, EV, ET, Epid)  # geometrische Referenz (Soll-Karte, genaue Pose)
    if a.nomap:
        sel = follow_straight(ex)
        pred_on = np.array([int(p) in sel for p in Epid])
        pred_on_geo = pred_on
        pred_tok = [None] * n
        reg_err = (np.nan, np.nan)
        assigned = np.zeros(n, bool)
    else:
        if a.oracle:
            pred_tok = list(tok_gt)
            pred_on_geo = gt_on_geo.copy()
            reg_err = (np.nan, np.nan)
        else:
            r = AL.associate(ex, mp_, register_graph=bool(a.reg), hmm=bool(a.hmm), route_tok=route_tok)
            pred_tok = r["point_tok"]
            pred_on_geo = r["point_route"]
            dn, Rn = mp_["shift"]
            # erwartet R ~ Rn, t ~ dn (x_map = R x_ex + t); Fehler der Schaetzung
            reg_err = (float(np.linalg.norm(r["t"] - dn)), float(np.degrees(np.arctan2(r["R"][1, 0], r["R"][0, 0]) - np.arctan2(Rn[1, 0], Rn[0, 0]))))
        assigned = np.array([t is not None for t in pred_tok])
        pred_on = np.array([t in route_tok for t in pred_tok]) if a.drop == 0 or not a.protect else np.array([t in route_tok for t in pred_tok])
    res = dict(n=n)
    # Zuordnung (nur Punkte mit Soll-Stueck)
    m = gt_matched
    res["n_matched"] = int(m.sum())
    res["acc_all"] = int(sum(1 for i in np.nonzero(m)[0] if pred_tok[i] == tok_gt[i]))
    res["acc_assigned_n"] = int(sum(1 for i in np.nonzero(m)[0] if pred_tok[i] is not None))
    res["acc_assigned"] = int(sum(1 for i in np.nonzero(m)[0] if pred_tok[i] is not None and pred_tok[i] == tok_gt[i]))
    # Route je Punkt
    res["tp"], res["fp"], res["fn"] = int((pred_on & gt_on).sum()), int((pred_on & ~gt_on).sum()), int((~pred_on & gt_on).sum())
    # 'nicht in der Karte'-Erkennung: Wahrheit = Punkt hat kein Soll-Stueck ODER sein Soll-Stueck fehlt in der (veralteten) Karte
    kept_tok = set(mp_["tokens"])
    truth_novel = np.array([(t is None) or (t not in kept_tok) for t in tok_gt])
    flag = ~assigned
    res["nov_tp"], res["nov_fp"], res["nov_fn"] = int((flag & truth_novel).sum()), int((flag & ~truth_novel).sum()), int((~flag & truth_novel).sum())
    # geometrische Routen-Markierung (Gleichstand zugunsten der Route): Punkt-Metriken gegen die geometrische Referenz
    res["tpG"], res["fpG"], res["fnG"] = int((pred_on_geo & gt_on_geo).sum()), int((pred_on_geo & ~gt_on_geo).sum()), int((~pred_on_geo & gt_on_geo).sum())
    # Routen-Mittellinie: Soll-Punkte der Routenstuecke gegen vorhergesagte Routenpunkte
    gtp = [AL.resample(gt["pts"][p], 1.0) for p in rt["route"] if len(gt["pts"][p]) >= 2]
    gtp = np.concatenate(gtp) if gtp else np.zeros((0, 2))
    for suf, on_ in (("", pred_on), ("G", pred_on_geo)):
        pp = EV[on_]
        if len(gtp):
            if len(pp):
                d_gp = cKDTree(pp).query(gtp)[0]
                res["cov" + suf], res["n_gt_route"] = int((d_gp <= 1.5).sum()), len(gtp)
                d_pg = cKDTree(gtp).query(pp)[0]
                res["prec" + suf], res["n_pred_route" + suf] = int((d_pg <= 1.5).sum()), len(pp)
            else:
                res["cov" + suf], res["n_gt_route"], res["prec" + suf], res["n_pred_route" + suf] = 0, len(gtp), 0, 0
            c, p_ = res["cov" + suf] / res["n_gt_route"], (res["prec" + suf] / res["n_pred_route" + suf]) if res["n_pred_route" + suf] else 0.0
            res["success" + suf] = int(c >= 0.8 and p_ >= 0.8)
    if not np.isnan(reg_err[0]):
        res["reg_pos"], res["reg_yaw"] = reg_err
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val,test")
    ap.add_argument("--graphs", default="e44_cons_p/graphs")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--shift_frac", type=float, default=0.0)
    ap.add_argument("--protect", type=int, default=1, help="1: Routenstuecke bleiben in der veralteten Karte")
    ap.add_argument("--reg", type=int, default=1)
    ap.add_argument("--oracle", type=int, default=0)
    ap.add_argument("--hmm", type=int, default=0, help="Zuordnung als Viterbi-Pfad laengs der Polylinien (Karten-Nachfolger) statt naechster Punkt")
    ap.add_argument("--nomap", type=int, default=0)
    ap.add_argument("--rt_file", default="route_gt_val.pkl", help="Routen-Datei (alternative Ziele: route_alt_val.pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/{a.rt_file}", "rb"))
    res_all = dict(args=vars(a))
    _G["a"] = a
    for st in a.sets.split(","):
        d = pickle.load(open(f"{OUT}/{a.graphs}_{st}.pkl", "rb"))
        ids = [int(i) for i in d["ids"]]
        _G["ex"], _G["gt"], _G["rt"] = d["graphs"], [GT[i] for i in ids], [RT[i] for i in ids]
        with mp.get_context("fork").Pool(a.workers) as pool:
            rs = pool.map(one, list(enumerate(ids)), chunksize=4)
        rs = [r for r in rs if r is not None]
        S = lambda key: float(sum(r.get(key, 0) for r in rs))
        m = dict(n_samples=len(rs))
        m["assoc_acc_all"] = S("acc_all") / max(S("n_matched"), 1)
        m["assoc_acc_assigned"] = S("acc_assigned") / max(S("acc_assigned_n"), 1)
        m["assigned_frac"] = S("acc_assigned_n") / max(S("n_matched"), 1)
        tp, fp, fn = S("tp"), S("fp"), S("fn")
        m["route_P"], m["route_R"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        m["route_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        m["route_cov"] = S("cov") / max(S("n_gt_route"), 1)
        m["route_prec"] = S("prec") / max(S("n_pred_route"), 1)
        m["route_success"] = float(np.mean([r["success"] for r in rs if "success" in r]))
        tpg, fpg, fng = S("tpG"), S("fpG"), S("fnG")
        m["routeG_P"], m["routeG_R"] = tpg / max(tpg + fpg, 1), tpg / max(tpg + fng, 1)
        m["routeG_F1"] = 2 * tpg / max(2 * tpg + fpg + fng, 1)
        m["routeG_cov"] = S("covG") / max(S("n_gt_route"), 1)
        m["routeG_prec"] = S("precG") / max(S("n_pred_routeG"), 1)
        m["routeG_success"] = float(np.mean([r["successG"] for r in rs if "successG" in r]))
        ntp, nfp, nfn = S("nov_tp"), S("nov_fp"), S("nov_fn")
        m["novel_P"], m["novel_R"] = ntp / max(ntp + nfp, 1), ntp / max(ntp + nfn, 1)
        if any("reg_pos" in r for r in rs):
            rp = np.array([r["reg_pos"] for r in rs if "reg_pos" in r])
            ry = np.array([r["reg_yaw"] for r in rs if "reg_yaw" in r])
            m["reg_pos_median"], m["reg_pos_p90"], m["reg_yaw_median"] = float(np.median(rp)), float(np.percentile(rp, 90)), float(np.median(np.abs(ry)))
        res_all[st] = m
        print(f"[{st}] {a.out}: n {m['n_samples']} | assoc acc (all matched pts) {m['assoc_acc_all']:.3f} (assigned {m['assigned_frac']:.3f}, acc among assigned {m['assoc_acc_assigned']:.3f}) | route F1 {m['route_F1']:.3f} (P {m['route_P']:.3f} R {m['route_R']:.3f}) | route centerline cov/prec @1.5 {m['route_cov']:.3f}/{m['route_prec']:.3f} success {m['route_success']:.3f} || GEO-tie flag: F1 {m['routeG_F1']:.3f} cov/prec {m['routeG_cov']:.3f}/{m['routeG_prec']:.3f} success {m['routeG_success']:.3f}" + (f" | reg err median {m['reg_pos_median']:.2f} m {m['reg_yaw_median']:.2f} deg" if "reg_pos_median" in m else ""), flush=True)
    json.dump(res_all, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
