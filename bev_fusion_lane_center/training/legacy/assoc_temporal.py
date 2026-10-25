"""Schleife 5 / E61: zeitliche Verfolgung der Karten-Zuordnung. Pose-Fehler der Karte als Verzerrung je Szene (Gier- und Versatz-Bias in globalen Koordinaten + kleine Rauschwerte je Bild), wie bei einer driftenden Lokalisierung.
Verfahren (je Bild, in Zeitreihenfolge): M0 Einzelbild-Registrierung (Graph-ICP) + Zuordnung; M1 zeitlich gefilterte Ausrichtung (Median der letzten K globalen Schaetzungen, dann feste Ausrichtung); M2 Zuordnung mit zeitlicher Stimmenglaettung (Stimmen des Vorgaengerbildes, per Eigenbewegung gewarpt); M12 beides.
Metriken: Zuordnungs-Genauigkeit (Punkte mit Soll-Token), Routen-Erfolgsrate, FLACKERN = Anteil der zwischen zwei Bildern zugeordneten Punktpaare mit verschiedenem Token (Soll-Token als Referenz), Registrierungsfehler.
Aufruf (gcar): assoc_temporal.py --sigma 1.0 --out e61_s1 [--jitter 0.2] [--K 5]"""
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
import assoc_lib as AL  # noqa: E402

D, OUT = AE.D, AE.OUT
_G = {}


def rot(deg):
    return AL.rot2(deg)


def ang(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def noisy_pieces(gt, ti, Ri, c0, psi_deg, b, jit):
    """Karte im Fahrzeugsystem des Bildes i mit globaler Verzerrung X' = R(psi)(X - c0) + c0 + b + jit."""
    Rp = rot(psi_deg)
    out = []
    for p in gt["pts"]:
        p = np.asarray(p, np.float64)
        X = ti + p @ Ri.T
        Xn = (X - c0) @ Rp.T + c0 + b + jit
        out.append((Xn - ti) @ Ri)  # R^T (X' - t)
    return out


def meas_bias(R, t, ti, Ri, c0):
    """aus der Einzelbild-Ausrichtung (x_map = R x_ex + t, Fahrzeugsystem i) die globale Verzerrung (theta, b) um c0."""
    b = (np.eye(2) - R) @ (ti - c0) + Ri @ t
    return ang(R), b


def fixed_from_bias(theta, b, ti, Ri, c0):
    R = rot(theta)
    t = Ri.T @ (b - (np.eye(2) - R) @ (ti - c0))
    return R, t


def warp_tokens(prev, cur, Pprev, Pcur, tokens_prev):
    """Token des Vorgaengerbildes auf die Punkte des aktuellen Bildes uebertragen: x_(i-1) = R_(i-1)^T (t_i + R_i x_i - t_(i-1)); naechster Punkt <= 1 m mit passender Richtung."""
    (EVp, ETp), (EVc, ETc) = prev, cur
    (tp, Rp), (tc, Rc) = Pprev, Pcur
    if len(EVp) == 0 or len(EVc) == 0:
        return [None] * len(EVc)
    q = (tc + EVc @ Rc.T - tp) @ Rp
    qt = ETc @ Rc.T @ Rp
    tree = cKDTree(EVp)
    d, j = tree.query(q, distance_upper_bound=1.0)
    out = []
    for k in range(len(EVc)):
        if np.isfinite(d[k]) and float(ETp[j[k]] @ qt[k]) >= AL.COS_DIR:
            out.append(tokens_prev[j[k]])
        else:
            out.append(None)
    return out


def run_scene(args):
    sc, rows = args
    a = _G["a"]
    P, GT, RT = _G["P"], _G["GT"], _G["RT"]
    T, R_ = P["t"], P["R"]
    rows = sorted(rows, key=lambda i: P["ts"][i])
    rng = np.random.default_rng(AE.seed_of(f"scene{sc}", 3))
    c0 = T[rows[0]].copy()
    psi = rng.normal(0.0, 2.0 * a.sigma)
    b0 = rng.normal(0.0, a.sigma, 2)
    methods = ["M0", "M1", "M2", "M12", "M3", "M32", "GT"]
    hist3 = []
    st = {m: dict(prev=None, tok=None) for m in methods}
    hist = []
    acc = {m: [0, 0] for m in methods}
    flick = {m: [0, 0] for m in methods}
    succ = {m: [] for m in methods}
    succG = {m: [] for m in ("M0", "M1", "M3")}  # geometrische Routen-Markierung (Gleichstand zugunsten der Route)
    flickG = {m: [0, 0] for m in ("M0", "M1", "M3")}
    stG = {m: None for m in ("M0", "M1", "M3")}
    regerr = {m: [] for m in ("M0", "M1", "M3")}
    cache = _G["ex"]
    for i in rows:
        ex, gt, rt = cache[i], GT[i], RT[i]
        if len(ex["pts"]) == 0:
            hist.clear()
            for m in methods:
                st[m] = dict(prev=None, tok=None)
            for m in stG:
                stG[m] = None
            continue
        ti, Ri = T[i], R_[i]
        jit = rng.normal(0.0, a.jitter, 2)
        psi_i = psi + rng.normal(0.0, a.jitter_deg)
        mp_ = dict(pieces=noisy_pieces(gt, ti, Ri, c0, psi_i, b0, jit), tokens=gt["tokens"], idx=list(range(len(gt["pts"]))))
        EV, ET, Epid, pa_gt, tok_gt = AL.gt_assign(ex, gt)
        n = len(EV)
        if n == 0:
            continue
        gm = np.array([t is not None for t in tok_gt])
        route_tok = {gt["tokens"][p] for p in rt["route"]} if rt is not None else set()
        # M0: Einzelbild
        r0 = AL.associate(ex, mp_, register_graph=True, route_tok=route_tok)
        th_i, b_i = meas_bias(r0["R"], r0["t"], ti, Ri, c0)
        hist.append((th_i, b_i))
        regerr["M0"].append(float(np.linalg.norm(r0["t"] - ((np.eye(2) - rot(psi_i)) @ (Ri.T @ (ti - c0)) * 0 + Ri.T @ (b0 + jit - (np.eye(2) - rot(psi_i)) @ (ti - c0))))))
        # M1: gefilterte Ausrichtung
        w = hist[-a.K :]
        thf = float(np.median([h[0] for h in w]))
        bf = np.median(np.stack([h[1] for h in w]), axis=0)
        Rf, tf = fixed_from_bias(thf, bf, ti, Ri, c0)
        r1 = AL.associate(ex, mp_, register_graph=False, fixed=(Rf, tf), route_tok=route_tok)
        regerr["M1"].append(float(np.linalg.norm(tf - (Ri.T @ (b0 + jit - (np.eye(2) - rot(psi_i)) @ (ti - c0))))))
        # M3: nachgefuehrte Registrierung - Suche nur in einem kleinen Fenster um die aus den letzten Bildern vorhergesagte Ausrichtung
        MV_, MT_, _, _ = AL.vertices(mp_["pieces"])
        if len(hist3) == 0:
            R3, t3 = r0["R"], r0["t"]
        else:
            w3 = hist3[-a.K :]
            th3 = float(np.median([h[0] for h in w3]))
            b3 = np.median(np.stack([h[1] for h in w3]), axis=0)
            R03, t03 = fixed_from_bias(th3, b3, ti, Ri, c0)
            R3, t3, _ = AL.register_around(MV_, MT_, EV, ET, R03, t03)
        hist3.append(meas_bias(R3, t3, ti, Ri, c0))
        regerr["M3"].append(float(np.linalg.norm(t3 - (Ri.T @ (b0 + jit - (np.eye(2) - rot(psi_i)) @ (ti - c0))))))
        r3 = AL.associate(ex, mp_, register_graph=False, fixed=(R3, t3), route_tok=route_tok)
        toks = {"M0": list(r0["point_tok"]), "M1": list(r1["point_tok"]), "M3": list(r3["point_tok"]), "GT": list(tok_gt)}
        geo = {"M0": np.asarray(r0["point_route"], bool), "M1": np.asarray(r1["point_route"], bool), "M3": np.asarray(r3["point_route"], bool)}
        for m, fl in geo.items():
            if stG[m] is not None:
                ptf = warp_tokens(stG[m][0], (EV, ET), stG[m][1], (ti, Ri), stG[m][2])
                pairs = [(ptf[k], "R" if fl[k] else "N") for k in range(n) if ptf[k] is not None]
                flickG[m][0] += int(sum(1 for x, y in pairs if x != y))
                flickG[m][1] += len(pairs)
            if rt is not None:
                gtp_ = [AL.resample(gt["pts"][p], 1.0) for p in rt["route"] if len(gt["pts"][p]) >= 2]
                if gtp_:
                    gtp_ = np.concatenate(gtp_)
                    pp_ = EV[fl]
                    if len(pp_):
                        succG[m].append(int((cKDTree(pp_).query(gtp_)[0] <= 1.5).mean() >= 0.8 and (cKDTree(gtp_).query(pp_)[0] <= 1.5).mean() >= 0.8))
                    else:
                        succG[m].append(0)
            stG[m] = ((EV, ET), (ti, Ri), ["R" if f else "N" for f in fl])
        # M2 / M12: zeitliche Stimmenglaettung auf M0 bzw. M1
        for m, base in (("M2", "M0"), ("M12", "M1"), ("M32", "M3")):
            cur = list(toks[base])
            if st[m]["prev"] is not None:
                pt = warp_tokens(st[m]["prev"][0], (EV, ET), st[m]["prev"][1], (ti, Ri), st[m]["tok"])
                cur = []
                for k in range(n):
                    votes = {}
                    if toks[base][k] is not None:
                        votes[toks[base][k]] = votes.get(toks[base][k], 0.0) + 1.0
                    if pt[k] is not None:
                        votes[pt[k]] = votes.get(pt[k], 0.0) + a.wprev
                    cur.append(max(votes, key=votes.get) if votes else None)
            toks[m] = cur
        for m in methods:
            tk = toks[m]
            acc[m][0] += int(sum(1 for k in np.nonzero(gm)[0] if tk[k] == tok_gt[k]))
            acc[m][1] += int(gm.sum())
            # Flackern: Paare (Vorgaenger -> aktuell)
            if st[m]["prev"] is not None:
                pt = warp_tokens(st[m]["prev"][0], (EV, ET), st[m]["prev"][1], (ti, Ri), st[m]["tok"])
                pairs = [(pt[k], tk[k]) for k in range(n) if pt[k] is not None and tk[k] is not None]
                flick[m][0] += int(sum(1 for x, y in pairs if x != y))
                flick[m][1] += len(pairs)
            if rt is not None:
                on = np.array([t in route_tok for t in tk])
                gtp = [AL.resample(gt["pts"][p], 1.0) for p in rt["route"] if len(gt["pts"][p]) >= 2]
                if gtp:
                    gtp = np.concatenate(gtp)
                    pp = EV[on]
                    if len(pp):
                        c1 = (cKDTree(pp).query(gtp)[0] <= 1.5).mean()
                        c2 = (cKDTree(gtp).query(pp)[0] <= 1.5).mean()
                        succ[m].append(int(c1 >= 0.8 and c2 >= 0.8))
                    else:
                        succ[m].append(0)
            st[m] = dict(prev=((EV, ET), (ti, Ri)), tok=tk)
            st[m]["prev"] = ((EV, ET), (ti, Ri))
    return dict(acc=acc, flick=flick, succ=succ, regerr=regerr, succG=succG, flickG=flickG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="e44_cons_p/graphs")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--jitter_deg", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--wprev", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    d = pickle.load(open(f"{OUT}/{a.graphs}_test.pkl", "rb"))
    ids = [int(i) for i in d["ids"]]
    ex = {i: g for i, g in zip(ids, d["graphs"])}
    scenes = {}
    for i in ids:
        scenes.setdefault(int(P["scene"][i]), []).append(i)
    _G.update(a=a, P=P, GT=GT, RT=RT, ex=ex)
    with mp.get_context("fork").Pool(a.workers) as pool:
        rs = pool.map(run_scene, list(scenes.items()), chunksize=1)
    res = dict(args=vars(a), n_scenes=len(scenes))
    for m in ["M0", "M1", "M2", "M12", "M3", "M32", "GT"]:
        A0 = sum(r["acc"][m][0] for r in rs)
        A1 = sum(r["acc"][m][1] for r in rs)
        F0 = sum(r["flick"][m][0] for r in rs)
        F1 = sum(r["flick"][m][1] for r in rs)
        S = [x for r in rs for x in r["succ"][m]]
        res[m] = dict(acc=A0 / max(A1, 1), flicker=F0 / max(F1, 1), success=float(np.mean(S)) if S else 0.0, n_pairs=F1)
    cand = [r for r in rs if len(r["regerr"]["M0"]) >= 15]
    longest = max(cand or rs, key=lambda r: float(np.mean(r["regerr"]["M0"]) - np.mean(r["regerr"]["M3"])))
    res["trace"] = {m: [float(x) for x in longest["regerr"][m]] for m in ("M0", "M1", "M3")}  # Registrierungsfehler je Bild in der Szene mit dem groessten Unterschied M0 - M3 (Abbildung)
    for m in ("M0", "M1", "M3"):
        e = [x for r in rs for x in r["regerr"][m]]
        res[m]["reg_err_median"], res[m]["reg_err_mean"] = float(np.median(e)), float(np.mean(e))
        SG = [x for r in rs for x in r["succG"][m]]
        res[m]["successG"] = float(np.mean(SG)) if SG else 0.0
        res[m]["flickerG"] = sum(r["flickG"][m][0] for r in rs) / max(sum(r["flickG"][m][1] for r in rs), 1)
    print(f"[{a.out}] sigma {a.sigma} jitter {a.jitter}: " + " | ".join(f"{m}: acc {res[m]['acc']:.3f} flicker {res[m]['flicker']:.3f} success {res[m]['success']:.3f}" + (f" reg err med {res[m]['reg_err_median']:.2f} GEOflag success {res[m]['successG']:.3f} flicker {res[m]['flickerG']:.3f}" if "reg_err_median" in res[m] else "") for m in ["M0", "M1", "M2", "M12", "M3", "M32", "GT"]), flush=True)
    json.dump(res, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
