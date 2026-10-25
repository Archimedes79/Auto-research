"""Schleife 5 / E62: Planer-nahe Auswertung der Routen-Uebertragung (offene Schleife, 'planning-lite'): folgt ein Planer der uebertragenen Routen-Mittellinie mit der aktuellen Geschwindigkeit, wie weit liegt er nach 1 / 2 / 3 s von der tatsaechlich gefahrenen Position?
Vergleich: (a) Konstantfahrt geradeaus, (b) Soll-Route der Karte (Obergrenze des Routenfolgens), (c) extrahierter Graph + Routen-Zuordnung aus der Karte (System), (d) extrahierter Graph ohne Karte (an Verzweigungen geradeaus), (e) Zuordnung mit Soll-Token (Orakel).
Gruppen: Kurven/Verzweigungen (die gefahrene Route enthaelt eine Verzweigung oder die Zukunftslage weicht > 3 m seitlich von der Geraden ab) und Geradeausfahrt. Metriken: mittlerer L2-Fehler 1-3 s, Abweichungsrate (L2 3 s > 2 m), Routen-Konformitaet = Anteil der Samples, deren Pfadende (3 s) <= 1,5 m von der Soll-Route liegt.
Aufruf (gcar): plan_eval.py --sets val,test --graphs e44_cons_p/graphs --sigma 0 --out e62_a"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_eval as AE  # noqa: E402
import assoc_lib as AL  # noqa: E402

HOME = os.path.expanduser("~")
OUT = AE.OUT
D = AE.D
_G = {}


def chain_polyline(pieces_in_order):
    """Stuecke (Polylinien in Fahrtreihenfolge) -> eine Polylinie ab dem Punkt, der dem Fahrzeug (0,0) am naechsten liegt, 1 m abgetastet."""
    if not pieces_in_order:
        return np.zeros((0, 2))
    q = np.concatenate([AL.resample(p, 1.0) for p in pieces_in_order if len(p) >= 2])
    j = int(np.argmin(np.linalg.norm(q, axis=1)))
    return q[j:]


def point_at(poly, s):
    """Position nach der Weglaenge s entlang der Polylinie (vor dem Ende: geradeaus weiter in Endrichtung)."""
    if len(poly) < 2:
        return np.array([0.0, s])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    if s <= cs[-1]:
        return np.array([np.interp(s, cs, poly[:, 0]), np.interp(s, cs, poly[:, 1])])
    t = poly[-1] - poly[-2]
    t = t / (np.linalg.norm(t) + 1e-9)
    return poly[-1] + t * (s - cs[-1])


def pred_positions(poly, v0, times=(1.0, 2.0, 3.0)):
    return np.stack([point_at(poly, v0 * t) for t in times])


def route_polyline_from_flags(ex, on_pts, EV, Epid, gap=3):
    """Punkt-Ebene: Kette der als 'Route' markierten Punkte. Start = markierter Punkt, der dem Fahrzeug am naechsten liegt; dann laengs des Stuecks (Luecken <= gap Punkte werden ueberbrueckt) und am Stueckende ueber
    die Nachfolger-Kante, deren Anfang markiert ist (bei mehreren: kleinste Richtungsaenderung). (Die Stuecke sind oft 100 m lang, die Route deckt nur einen Teil davon ab - daher Punkte statt Stueck-Anteile.)"""
    n = len(ex["pts"])
    on_pts = np.asarray(on_pts, bool)
    if len(EV) == 0 or not on_pts.any():
        return np.zeros((0, 2)), []
    idx = [np.nonzero(Epid == k)[0] for k in range(n)]
    out_e = {}
    for i, j in ex["edges"]:
        out_e.setdefault(i, []).append(j)
    fl = np.nonzero(on_pts)[0]
    s = int(fl[int(np.argmin(np.linalg.norm(EV[fl], axis=1)))])
    k = int(Epid[s])
    pos = int(np.nonzero(idx[k] == s)[0][0])
    poly, seq, seen = [EV[s]], [k], {k}
    while True:
        v = idx[k]
        last, miss = pos, 0
        for j in range(pos + 1, len(v)):
            if on_pts[v[j]]:
                last, miss = j, 0
            else:
                miss += 1
                if miss > gap:
                    break
        poly.extend(EV[v[pos + 1 : last + 1]])
        if last < len(v) - 1 - gap:  # Markierung endet vor dem Stueckende: Route zu Ende
            break
        cands = []
        for c in out_e.get(k, []):
            if c in seen or len(idx[c]) < 2:
                continue
            fo = [m for m in range(min(gap + 1, len(idx[c]))) if on_pts[idx[c][m]]]
            if fo:
                cands.append((c, fo[0]))
        if not cands:
            break
        t0 = poly[-1] - poly[max(len(poly) - 4, 0)]
        t0 = t0 / (np.linalg.norm(t0) + 1e-9)
        c, m = max(cands, key=lambda cm: float(AL.tangents(EV[idx[cm[0]]])[cm[1]] @ t0))
        poly.append(EV[idx[c][m]])
        seq.append(c)
        seen.add(c)
        k, pos = c, m
    return np.asarray(poly, np.float64), seq


def one(args):
    k, row = args
    ex, gt, rt, fut = _G["ex"][k], _G["gt"][k], _G["rt"][k], _G["fut"][row]
    a = _G["a"]
    if rt is None or fut is None or not np.isfinite(rt["v0"]):
        return None  # (Samples mit LEEREM extrahierten Graphen bleiben drin: das System faellt dort auf 'geradeaus' zurueck)
    v0 = float(rt["v0"])
    seq_pos = np.vstack([[0.0, 0.0], fut])  # [7,2]: Start + 6 Folgebilder (0,5 s Abstand)
    s_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seq_pos, axis=0), axis=1))])
    gtpos = fut[[1, 3, 5]]  # Zukunftspositionen nach 1, 2, 3 s im Fahrzeugsystem
    s_true = s_cum[[2, 4, 6]]  # tatsaechlich gefahrene Weglaenge nach 1, 2, 3 s
    out = {}
    # (a) geradeaus
    out["const"] = pred_positions(np.array([[0.0, 0.0], [0.0, 1.0]]), v0)
    # (b) Soll-Route der Karte
    out["gt_route"] = pred_positions(chain_polyline([np.asarray(gt["pts"][p], np.float64) for p in rt["route"] if len(gt["pts"][p]) >= 2]), v0)
    route_tok = {gt["tokens"][p] for p in rt["route"]}
    EV, ET, Epid, pa_gt, tok_gt = AL.gt_assign(ex, gt)
    # (c) System: Zuordnung aus der (ggf. gestoerten) Karte
    rng = np.random.default_rng(AE.seed_of(row, 1))
    mp_ = AL.perturb_map(gt, a.sigma, a.drop, protect=set(rt["route"]) if a.protect else (), rng=rng)
    r = AL.associate(ex, mp_, register_graph=bool(a.reg), route_tok=route_tok)
    on = r["point_route"]
    poly_c, seq_c = route_polyline_from_flags(ex, on, EV, Epid)
    out["system"] = pred_positions(poly_c, v0) if len(poly_c) >= 2 else out["const"]
    # (d) ohne Karte
    sel = AE.follow_straight(ex)
    poly_d = chain_polyline([np.asarray(ex["pts"][k_], np.float64) for k_ in sorted(sel, key=lambda x: 0)]) if sel else np.zeros((0, 2))
    # Reihenfolge der Kette neu bestimmen
    out_e = {}
    for i_, j_ in ex["edges"]:
        out_e.setdefault(i_, []).append(j_)
    out["nomap"] = None
    if sel:
        start = min(sel, key=lambda k_: float(np.min(np.linalg.norm(AL.resample(ex["pts"][k_], 1.0), axis=1))) if len(ex["pts"][k_]) >= 2 else 1e9)
        seq, cur = [start], start
        while True:
            nxt = [j for j in out_e.get(cur, []) if j in sel and j not in seq]
            if not nxt:
                break
            t0 = AL.tangents(np.asarray(ex["pts"][cur], np.float64))[-1]
            cur = max(nxt, key=lambda j: float(AL.tangents(np.asarray(ex["pts"][j], np.float64))[0] @ t0))
            seq.append(cur)
        poly_d = chain_polyline([np.asarray(ex["pts"][k_], np.float64) for k_ in seq])
    out["nomap"] = pred_positions(poly_d, v0) if len(poly_d) >= 2 else out["const"]
    # (f) nur Karte (ohne Wahrnehmung): Route der (ggf. gestoerten) Karte ohne Registrierung - Rueckfall, wenn die Wahrnehmung ausfaellt
    kept = {k_: n_ for n_, k_ in enumerate(mp_["idx"])}
    poly_m = chain_polyline([mp_["pieces"][kept[p_]] for p_ in rt["route"] if p_ in kept and len(mp_["pieces"][kept[p_]]) >= 2])
    out["maponly"] = pred_positions(poly_m, v0) if len(poly_m) >= 2 else out["const"]
    # (g) Karte + Registrierung aus der Wahrnehmung: Routen-Polylinie der Karte mit der geschaetzten Ausrichtung ins Fahrzeugsystem (x_ex = (x_map - t) R)
    poly_rm = chain_polyline([(mp_["pieces"][kept[p_]] - r["t"]) @ r["R"] for p_ in rt["route"] if p_ in kept and len(mp_["pieces"][kept[p_]]) >= 2])
    out["regmap"] = pred_positions(poly_rm, v0) if len(poly_rm) >= 2 else out["const"]
    # (e) Orakel-Zuordnung
    on_o = AL.route_flags(AL.perturb_map(gt, 0.0, 0.0), route_tok, EV, ET, Epid)
    poly_o, _ = route_polyline_from_flags(ex, on_o, EV, Epid)
    out["oracle"] = pred_positions(poly_o, v0) if len(poly_o) >= 2 else out["const"]
    res = {k_: np.linalg.norm(v - gtpos, axis=1) for k_, v in out.items()}  # L2 je Zeithorizont
    polys = dict(const=np.array([[0.0, 0.0], [0.0, 1.0]]), gt_route=chain_polyline([np.asarray(gt["pts"][p], np.float64) for p in rt["route"] if len(gt["pts"][p]) >= 2]), system=poly_c if len(poly_c) >= 2 else np.array([[0.0, 0.0], [0.0, 1.0]]),
                 nomap=poly_d if len(poly_d) >= 2 else np.array([[0.0, 0.0], [0.0, 1.0]]), oracle=poly_o if len(poly_o) >= 2 else np.array([[0.0, 0.0], [0.0, 1.0]]), maponly=poly_m if len(poly_m) >= 2 else np.array([[0.0, 0.0], [0.0, 1.0]]), regmap=poly_rm if len(poly_rm) >= 2 else np.array([[0.0, 0.0], [0.0, 1.0]]))
    lat = {k_: np.linalg.norm(np.stack([point_at(pl, s_) for s_ in s_true]) - gtpos, axis=1) for k_, pl in polys.items()}
    # verankert: das Planer-Verhalten 'von der Ist-Lage auf die Route einlenken' - der anfaengliche Versatz zur Routen-Mittellinie klingt mit s ab (Skala 15 m)
    lata = {k_: np.linalg.norm(np.stack([point_at(pl, s_) + (-pl[0]) * np.exp(-s_ / 15.0) for s_ in s_true]) - gtpos, axis=1) for k_, pl in polys.items()}
    # Gruppen
    gtroute_pts = np.concatenate([AL.resample(gt["pts"][p], 0.5) for p in rt["route"] if len(gt["pts"][p]) >= 2])
    from scipy.spatial import cKDTree

    tr = cKDTree(gtroute_pts)
    conf = {k_: float(tr.query(v[-1])[0]) for k_, v in out.items()}
    curve = abs(gtpos[-1, 0]) > 3.0  # seitliche Abweichung der Zukunftslage von der Geraden (x = rechts)
    n_split = 0
    outd = {}
    for i_, j_ in gt["edges"]:
        outd[i_] = outd.get(i_, 0) + 1
    n_split = int(sum(1 for p in rt["route"] if outd.get(p, 0) >= 2))
    nopoly = dict(system=len(poly_c) < 2, oracle=len(poly_o) < 2, nomap=len(poly_d) < 2, maponly=len(poly_m) < 2, regmap=len(poly_rm) < 2)
    return dict(l2=res, lat=lat, lata=lata, conf=conf, curve=bool(curve or n_split > 0), v0=v0, nopoly=nopoly, row=int(row), k=int(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val,test")
    ap.add_argument("--graphs", default="e44_cons_p/graphs")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--protect", type=int, default=1)
    ap.add_argument("--reg", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    P = np.load(f"{D}/poses_val.npz")
    T, R, prev = P["t"], P["R"], P["prev"]
    N = len(T)
    nxt = -np.ones(N, int)
    for i in range(N):
        if prev[i] >= 0:
            nxt[prev[i]] = i
    fut = [None] * N  # Zukunftspositionen nach 1, 2, 3 s = 2., 4., 6. Folgebild
    for i in range(N):
        j, seq = nxt[i], []
        while j >= 0 and len(seq) < 6:
            seq.append(j)
            j = nxt[j]
        if len(seq) >= 6:
            fut[i] = np.stack([(T[seq[m]] - T[i]) @ R[i] for m in range(6)])
    _G["a"], _G["fut"] = a, fut
    res_all = dict(args=vars(a))
    for st in a.sets.split(","):
        d = pickle.load(open(f"{OUT}/{a.graphs}_{st}.pkl", "rb"))
        ids = [int(i) for i in d["ids"]]
        _G["ex"], _G["gt"], _G["rt"] = d["graphs"], [GT[i] for i in ids], [RT[i] for i in ids]
        with mp.get_context("fork").Pool(a.workers) as pool:
            rs = pool.map(one, list(enumerate(ids)), chunksize=4)
        rs = [r for r in rs if r is not None]
        pickle.dump(rs, open(f"{OUT}/{a.out}/persample_{st}.pkl", "wb"))  # je Sample (Gating-Analyse E64)
        m = dict(n=len(rs), n_curve=int(sum(r["curve"] for r in rs)))
        for grp, flt in (("all", lambda r: True), ("curve", lambda r: r["curve"]), ("straight", lambda r: not r["curve"])):
            sub = [r for r in rs if flt(r)]
            if not sub:
                continue
            for k_ in ("const", "gt_route", "system", "nomap", "oracle", "maponly", "regmap"):
                L = np.stack([r["l2"][k_] for r in sub])
                Lt = np.stack([r["lat"][k_] for r in sub])
                La = np.stack([r["lata"][k_] for r in sub])
                m[f"{grp}_{k_}_lata"] = float(La.mean())
                m[f"{grp}_{k_}_lata_med"] = float(np.median(La))
                m[f"{grp}_{k_}_deva"] = float((La[:, 2] > 2.0).mean())
                m[f"{grp}_{k_}_lat"] = float(Lt.mean())
                m[f"{grp}_{k_}_lat_med"] = float(np.median(Lt))
                m[f"{grp}_{k_}_l2_med"] = float(np.median(L))
                m[f"{grp}_{k_}_l2"] = float(L.mean())
                m[f"{grp}_{k_}_l2_3s"] = float(L[:, 2].mean())
                m[f"{grp}_{k_}_dev"] = float((L[:, 2] > 2.0).mean())
                m[f"{grp}_{k_}_conf"] = float(np.mean([r["conf"][k_] <= 1.5 for r in sub]))
                m[f"{grp}_{k_}_nopoly"] = float(np.mean([r["nopoly"].get(k_, False) for r in sub]))
                m[f"{grp}_{k_}_dev_lat"] = float((Lt[:, 2] > 2.0).mean())
            m[f"{grp}_n"] = len(sub)
        res_all[st] = m
        print(f"[{st}] {a.out}: n {m['n']} (Kurve/Verzweigung {m['n_curve']}) | L2 mit wahrer Laengsbewegung (nur Route), alle: const {m['all_const_lat']:.2f} nomap {m['all_nomap_lat']:.2f} SYSTEM {m['all_system_lat']:.2f} oracle {m['all_oracle_lat']:.2f} gt_route {m['all_gt_route_lat']:.2f} | Kurve: const {m['curve_const_lat']:.2f} nomap {m['curve_nomap_lat']:.2f} SYSTEM {m['curve_system_lat']:.2f} gt {m['curve_gt_route_lat']:.2f} | mit v0: alle (Median) const {m['all_const_l2_med']:.2f} SYSTEM {m['all_system_l2_med']:.2f} gt {m['all_gt_route_l2_med']:.2f} | Konformitaet 3 s <= 1,5 m: nomap {m['all_nomap_conf']:.3f} SYSTEM {m['all_system_conf']:.3f} gt {m['all_gt_route_conf']:.3f}", flush=True)
    json.dump(res_all, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
