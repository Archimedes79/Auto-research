"""Schleife 5 / E67: Ausfall der Wahrnehmung MITTEN in der Fahrt - die Registrierung der Karte halten statt neu schaetzen (zeitliche Verfolgung + Monitor).
Szenario: je TEST-Szene (nach Zeit sortiert) bis zur Mitte die sauberen Graphen (E57 'stored'), danach die Graphen einer gestoerten Bedingung (E57: dark, cam_front3_off, blur, ...). Pose-Fehler der Karte wie in E61 (Bias je Szene + Rauschen je Bild, optional Drift).
Verfahren, die Karten-Route ins Fahrzeugsystem zu bringen:
  asgiven     Karte wie gegeben (keine Registrierung)
  perframe    Einzelbild-Graph-ICP (E59/E62 'regmap')
  track       nachgefuehrte Registrierung (E61 M3: Fenstersuche um die aus den letzten K Bildern vorhergesagte Ausrichtung), Verlauf immer fortgeschrieben
  gmon@t      wie track, aber Bilder mit Ausfallwahrscheinlichkeit p > t (kombinierter Monitor E64, ueber die Paritaet kreuzangepasst) gehen NICHT in den Verlauf ein; die Ausrichtung wird gehalten (Vorhersage)
  gscore      wie gmon, Tor = Registrierungsguete des Einzelbildes unter dem 5-%-Quantil der sauberen Bilder (kein Zusatznetz, keine Etiketten)
  goracle     Tor = wahres Versagen (Graph-F1 < 0.6)
  gt          wahre Ausrichtung (Untergrenze)
Metriken: verankerter Querfehler nach 1/2/3 s (wie E62 'regmap': Weglaenge der wahren Fahrt, Anfangsversatz klingt mit Skala 15 m ab) und Registrierungsfehler (Versatz m, Gier Grad); getrennt nach sauberer Phase und Ausfallphase, nach Bildern seit dem Ausfall binniert; Szenen-Bootstrap.
Aufruf (gcar): hold_eval.py --conds dark,cam_front3_off,blur --sigma 1.0 --out e67_s1"""
import bflc_paths as BP  # project paths from the location of this checkout
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
import assoc_temporal as AT  # noqa: E402
import plan_eval as PE  # noqa: E402

D, OUT = AE.D, AE.OUT
_G = {}
SIGN = dict(dis=1, mass=-1, unassigned=1, regres=1, routegap=1, npieces=-1)
NAMES = ["dis", "mass", "unassigned", "regres", "routegap", "npieces"]
GATES = (0.35, 0.5, 0.9)
TRACKED = ["track"] + [f"gmon@{g}" for g in GATES] + ["gscore", "goracle"]
POLS = ["asgiven", "perframe", "pfdrop@0.5"] + TRACKED + ["gt"]  # pfdrop@0.5 = Strategie aus E64: Einzelbild-Registrierung, bei p > 0.5 verworfen (Karte wie gegeben)
BINS = [(0, 3), (3, 6), (6, 10), (10, 10 ** 6)]


def scene_folds(S, P):
    """Faltung der Bilder nach SZENEN (zwei Haelften der Szenen) statt nach Paritaet: {Bedingung: 0/1-Feld je Bild}. Zeilen: 'stored' = splits.json TEST, sonst test_rows.npy der Stoerung."""
    rows_of = {}
    for c in S:
        if c == "stored":
            rows_of[c] = np.array(json.load(open(BP.ROOT + "/data/splits.json"))["test"])
        else:
            rows_of[c] = np.load(f"{D}/pert/{c}/test_rows.npy")
    scenes = sorted({int(P["scene"][r]) for c in rows_of for r in rows_of[c]})
    fold_of = {sc: i % 2 for i, sc in enumerate(scenes)}
    return {c: np.array([fold_of[int(P["scene"][r])] for r in rows_of[c]]) for c in S}


def fit_monitor(S, thr=0.6, folds=None):
    """kombinierter Monitor wie in monitor_eval.py, aber KREUZANGEPASST: Samples gerader Nummer erhalten das auf den ungeraden Samples angepasste Modell und umgekehrt (kein Sample wird vom eigenen Modell bewertet)."""
    X, y, half, off = [], [], [], {}
    o = 0
    for c in S:
        n = len(S[c]["f1"])
        X.append(np.stack([SIGN[k] * S[c][k] for k in NAMES], 1))
        y.append(S[c]["f1"] < thr)
        half.append(folds[c] if folds is not None else np.arange(n) % 2)
        off[c] = (o, n)
        o += n
    X, y, half = np.concatenate(X), np.concatenate(y), np.concatenate(half)
    models = {}
    for h in (0, 1):
        tr = half == h
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xn = (X - mu) / sd
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(500):
            p = 1 / (1 + np.exp(-(Xn[tr] @ w + b)))
            g = Xn[tr].T @ (p - y[tr]) / tr.sum() + 1e-3 * w
            w -= 0.5 * g
            b -= 0.5 * float(np.mean(p - y[tr]))
        models[h] = (mu, sd, w, b)
    pf = {}
    for c in S:
        o, n = off[c]
        Z = X[o : o + n]
        k = np.arange(n)
        fc = folds[c] if folds is not None else k % 2
        out = np.zeros(n)
        for h in (0, 1):
            mu, sd, w, b = models[h]
            p = 1 / (1 + np.exp(-(((Z - mu) / sd) @ w + b)))
            out[fc != h] = p[fc != h]
        pf[c] = out
    return pf, models


def fit_loco(S, hold, thr=0.6):
    """Monitor ohne die Stoerung 'hold': Anpassung auf allen Bildern der uebrigen Bedingungen (einschliesslich der sauberen); -> Ausfallwahrscheinlichkeit der sauberen Bilder und der Bilder von 'hold'"""
    tr = [c for c in S if c != hold]
    X = np.concatenate([np.stack([SIGN[k] * S[c][k] for k in NAMES], 1) for c in tr])
    y = np.concatenate([S[c]["f1"] < thr for c in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Xn @ w + b)))
        g = Xn.T @ (p - y) / len(y) + 1e-3 * w
        w -= 0.5 * g
        b -= 0.5 * float(np.mean(p - y))

    def prob(c):
        Z = (np.stack([SIGN[k] * S[c][k] for k in NAMES], 1) - mu) / sd
        return 1 / (1 + np.exp(-(Z @ w + b)))

    return prob("stored"), prob(hold)


def build_future(P):
    T, R, prev = P["t"], P["R"], P["prev"]
    N = len(T)
    nxt = -np.ones(N, int)
    for i in range(N):
        if prev[i] >= 0:
            nxt[prev[i]] = i
    fut = [None] * N
    for i in range(N):
        j, seq = nxt[i], []
        while j >= 0 and len(seq) < 6:
            seq.append(j)
            j = nxt[j]
        if len(seq) >= 6:
            fut[i] = np.stack([(T[seq[m]] - T[i]) @ R[i] for m in range(6)])
    return fut


def lata_poly(poly, s_true, gtpos):
    if len(poly) < 2:
        poly = np.array([[0.0, 0.0], [0.0, 1.0]])
    pts = np.stack([PE.point_at(poly, s) + (-poly[0]) * np.exp(-s / 15.0) for s in s_true])
    return np.linalg.norm(pts - gtpos, axis=1)


def run_scene(args):
    cond, sc, seq, thr_s, prepass = args  # seq: Liste (Zeile, k_sauber, k_gestoert), nach Zeit sortiert
    a = _G["a"]
    P, GT, RT, fut = _G["P"], _G["GT"], _G["RT"], _G["fut"]
    T, R_ = P["t"], P["R"]
    rng = np.random.default_rng(AE.seed_of(f"scene{sc}", 3 + 100 * a.seed))  # seed 0 = Hauptlauf; andere Seeds: andere Pose-Fehler-Realisierung
    n = len(seq)
    f0 = int(n * a.onset) if cond is not None else n + 1
    f1 = int(n * a.end) if cond is not None else n + 1  # Ausfall in [f0, f1); danach (Erholung) wieder saubere Graphen
    c0 = T[seq[0][0]].copy()
    psi0 = rng.normal(0.0, 2.0 * a.sigma)
    b0 = rng.normal(0.0, a.sigma, 2)
    dpsi_cum, db_cum = 0.0, np.zeros(2)
    hist = {m: [] for m in TRACKED}
    recs = []
    for f, (row, kc, kp) in enumerate(seq):
        fail_phase = f0 <= f < f1
        recovery = (f >= f1) and cond is not None
        # Zufallszahlen immer in derselben Reihenfolge ziehen (gleiche Pose-Realisierung fuer alle Verfahren und Bedingungen)
        jit = rng.normal(0.0, a.jitter, 2)
        jpsi = rng.normal(0.0, a.jitter_deg)
        db_cum = db_cum + rng.normal(0.0, a.drift, 2)
        dpsi_cum = dpsi_cum + rng.normal(0.0, a.drift_deg)
        ex = _G["pert"][cond][row] if fail_phase else _G["clean"][row]
        if a.loco and cond is not None:  # Monitor, der die Stoerungsart nie gesehen hat
            pf = float(_G["pf_loco"][cond][1][kp]) if fail_phase else float(_G["pf_loco"][cond][0][kc])
        else:
            pf = float(_G["pf"][cond][kp]) if fail_phase else float(_G["pf"]["stored"][kc])
        fail_true = bool(_G["fail"][cond][kp]) if fail_phase else bool(_G["fail"]["stored"][kc])
        ti, Ri = T[row], R_[row]
        psi_i = psi0 + dpsi_cum + jpsi
        b_i = b0 + db_cum
        gt, rt = GT[row], RT[row]
        pieces = AT.noisy_pieces(gt, ti, Ri, c0, psi_i, b_i, jit)
        MV, MT, _, _ = AL.vertices(pieces)
        EV, ET, _, _ = AL.vertices(ex["pts"])
        ok_ex = len(EV) >= 5 and len(MV) >= 5
        if ok_ex:
            R_pf, t_pf, s_pf = AL.register(MV, MT, EV, ET)
        else:
            R_pf, t_pf, s_pf = np.eye(2), np.zeros(2), 0.0
        if prepass:
            recs.append(dict(f=f, s_pf=float(s_pf), ok_ex=bool(ok_ex)))
            continue
        t_gt = Ri.T @ (b_i + jit - (np.eye(2) - AT.rot(psi_i)) @ (ti - c0))
        R_gt = AT.rot(psi_i)
        reg = {"asgiven": (np.eye(2), np.zeros(2)), "perframe": (R_pf, t_pf), "gt": (R_gt, t_gt), "pfdrop@0.5": (np.eye(2), np.zeros(2)) if pf > 0.5 else (R_pf, t_pf)}
        acc = {}
        for m in TRACKED:
            pred = None
            if hist[m]:
                w = hist[m][-a.K :]
                th = float(np.median([h[0] for h in w]))
                bb = np.median(np.stack([h[1] for h in w]), axis=0)
                pred = AT.fixed_from_bias(th, bb, ti, Ri, c0)
            if m == "track":
                accept = ok_ex
            elif m.startswith("gmon@"):
                accept = ok_ex and pf <= float(m.split("@")[1])
            elif m == "gscore":
                accept = ok_ex and s_pf >= thr_s
            else:
                accept = ok_ex and not fail_true
            if accept:
                if pred is None:
                    R3, t3 = R_pf, t_pf  # Start: Gittersuche
                else:
                    R3, t3, _ = AL.register_around(MV, MT, EV, ET, pred[0], pred[1])
                hist[m].append(AT.meas_bias(R3, t3, ti, Ri, c0))
                reg[m] = (R3, t3)
            else:
                reg[m] = pred if pred is not None else (np.eye(2), np.zeros(2))  # halten (Vorhersage); noch keine Vorhersage: Karte wie gegeben
            acc[m] = bool(accept)
        rec = dict(f=f, fail_phase=bool(fail_phase), recovery=bool(recovery), pf=pf, fail_true=fail_true, ok_ex=bool(ok_ex), s_pf=float(s_pf), n_pts=int(len(EV)), acc=acc, err={}, lata=None, row=int(row),
                   reg={pol: (np.asarray(R, float).tolist(), np.asarray(t, float).tolist()) for pol, (R, t) in reg.items()}, noise=dict(psi=float(psi_i), b=np.asarray(b_i, float).tolist(), jit=np.asarray(jit, float).tolist(), c0=np.asarray(c0, float).tolist()))  # (Abbildung der Szene)
        for pol, (R, t) in reg.items():
            rec["err"][pol] = (float(np.linalg.norm(t - t_gt)), float(abs(AT.ang(R) - psi_i)))
        if rt is not None and fut[row] is not None and np.isfinite(rt["v0"]):
            seq_pos = np.vstack([[0.0, 0.0], fut[row]])
            s_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seq_pos, axis=0), axis=1))])
            s_true, gtpos = s_cum[[2, 4, 6]], fut[row][[1, 3, 5]]
            rec["lata"] = {}
            for pol, (R, t) in reg.items():
                poly = PE.chain_polyline([(pieces[p] - t) @ R for p in rt["route"] if len(pieces[p]) >= 2])
                rec["lata"][pol] = lata_poly(poly, s_true, gtpos)
        recs.append(rec)
    return dict(cond=cond, scene=int(sc), recs=recs)


def summarize(scenes, pols):
    """scenes: Liste Szenen-Ergebnisse (eine Bedingung) -> Kennzahlen"""
    out = {}
    for phase in ("clean", "fail", "recovery"):
        if phase == "recovery" and not any(r.get("recovery") for s in scenes for r in s["recs"]):
            continue
        R = [r for s in scenes for r in s["recs"] if ((r["fail_phase"] and not r.get("recovery")) if phase == "fail" else (bool(r.get("recovery")) if phase == "recovery" else (not r["fail_phase"] and not r.get("recovery"))))]
        d = {}
        L = [r for r in R if r["lata"] is not None]
        for pol in pols:
            la = np.stack([r["lata"][pol] for r in L]) if L else np.zeros((0, 3))
            e = np.array([r["err"][pol] for r in R]) if R else np.zeros((0, 2))
            d[pol] = dict(lata=float(la.mean()) if len(la) else float("nan"), lata_med=float(np.median(la.mean(1))) if len(la) else float("nan"), dev3=float((la[:, 2] > 2.0).mean()) if len(la) else float("nan"),
                          lata_1s=float(la[:, 0].mean()) if len(la) else float("nan"), lata_3s=float(la[:, 2].mean()) if len(la) else float("nan"),
                          trans=float(e[:, 0].mean()) if len(e) else float("nan"), trans_med=float(np.median(e[:, 0])) if len(e) else float("nan"), yaw=float(e[:, 1].mean()) if len(e) else float("nan"))
        out[phase] = dict(n=len(R), n_plan=len(L), pols=d, true_fail_share=float(np.mean([r["fail_true"] for r in R])) if R else float("nan"))
        if R:
            out[phase]["reject_share"] = {m: float(np.mean([not r["acc"][m] for r in R])) for m in TRACKED}
            out[phase]["flag_share_p05"] = float(np.mean([r["pf"] > 0.5 for r in R]))
    # nach Bildern seit dem Ausfall
    bins = {pol: [] for pol in pols}
    for lo, hi in BINS:
        L = [r for s in scenes for r in s["recs"] if r["fail_phase"] and r["lata"] is not None and lo <= r["f"] - s["f0"] < hi]
        for pol in pols:
            bins[pol].append(float(np.mean([r["lata"][pol].mean() for r in L])) if L else float("nan"))
    out["bins"] = bins
    out["bins_n"] = [int(sum(1 for s in scenes for r in s["recs"] if r["fail_phase"] and r["lata"] is not None and lo <= r["f"] - s["f0"] < hi)) for lo, hi in BINS]
    return out


def bootstrap(scenes, pols, pairs, B=2000, seed=0):
    """Szenen-Bootstrap der mittleren Querfehler-Differenz in der Ausfallphase (a - b), gewichtet mit der Zahl der Bilder je Szene"""
    per = []
    for s in scenes:
        L = [r for r in s["recs"] if r["fail_phase"] and r["lata"] is not None]
        if L:
            per.append((len(L), {pol: float(np.mean([r["lata"][pol].mean() for r in L])) for pol in pols}))
    rng = np.random.default_rng(seed)
    out = {}
    for a_, b_ in pairs:
        diffs = []
        for _ in range(B):
            idx = rng.integers(0, len(per), len(per))
            w = np.array([per[i][0] for i in idx], float)
            da = np.array([per[i][1][a_] - per[i][1][b_] for i in idx])
            diffs.append(float((w * da).sum() / w.sum()))
        w = np.array([p[0] for p in per], float)
        mean = float((w * np.array([p[1][a_] - p[1][b_] for p in per])).sum() / w.sum())
        out[f"{a_}-{b_}"] = [mean, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="dark,cam_front3_off,blur,cam_front_off,fog,radar_off,extr3")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--jitter_deg", type=float, default=0.3)
    ap.add_argument("--drift", type=float, default=0.0, help="Zufallslauf des Karten-Offsets je Bild (m)")
    ap.add_argument("--drift_deg", type=float, default=0.0, help="Zufallslauf der Karten-Gierung je Bild (Grad)")
    ap.add_argument("--onset", type=float, default=0.5, help="Beginn des Ausfalls als Anteil der Szene")
    ap.add_argument("--end", type=float, default=1.0, help="Ende des Ausfalls als Anteil der Szene (< 1: danach Erholung mit sauberen Graphen)")
    ap.add_argument("--loco", type=int, default=0, help="1: Monitor ohne die jeweils ausgewertete Stoerungsart angepasst (leave-one-condition-out)")
    ap.add_argument("--seed", type=int, default=0, help="Seed der Pose-Fehler-Realisierung (0 = Hauptlauf)")
    ap.add_argument("--scenecv", type=int, default=0, help="1: Monitor nach Szenen statt nach Bildparitaet kreuzangepasst")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--min_frames", type=int, default=12)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    S = pickle.load(open(f"{OUT}/e64_monitors/signals.pkl", "rb"))
    if a.scenecv:  # Monitor nach Szenen kreuzangepasst (Bilder derselben Szene nie in Anpassung und Bewertung)
        pf, models = fit_monitor(S, folds=scene_folds(S, P))
    else:
        pf, models = fit_monitor(S)
    chk = json.load(open(f"{OUT}/e64_monitors/metrics.json"))["combined_weights"]
    w_ref = np.array([chk[k] for k in NAMES])
    print("Monitor-Gewichte (Modell 0) gegen metrics.json:", np.round(models[0][2], 3), np.round(w_ref, 3), flush=True)
    if not a.scenecv:
        assert np.allclose(models[0][2], w_ref, atol=1e-3)
    fail = {c: S[c]["f1"] < 0.6 for c in S}
    clean = pickle.load(open(f"{OUT}/e57_stored/graphs_test.pkl", "rb"))
    ids_c = [int(i) for i in clean["ids"]]
    kc = {r: k for k, r in enumerate(ids_c)}
    clean_g = {r: g for r, g in zip(ids_c, clean["graphs"])}
    conds = a.conds.split(",")
    pert, kp = {None: {}}, {}
    for c in conds:
        g = pickle.load(open(f"{OUT}/e57_{c}/graphs_test.pkl", "rb"))
        ids = [int(i) for i in g["ids"]]
        assert len(ids) == len(pf[c]), (c, len(ids), len(pf[c]))
        pert[c] = {r: x for r, x in zip(ids, g["graphs"])}
        kp[c] = {r: k for k, r in enumerate(ids)}
    pert["__none__"] = clean_g
    _G.update(a=a, P=P, GT=GT, RT=RT, fut=build_future(P), clean=clean_g, pert=pert, pf=pf, fail=fail)
    if a.loco:
        _G["pf_loco"] = {c: fit_loco(S, c) for c in conds}

    def scene_jobs(cond, rows, thr_s, prepass):
        by = {}
        for r in rows:
            by.setdefault(int(P["scene"][r]), []).append(r)
        jobs = []
        for sc, rr in sorted(by.items()):
            if len(rr) < a.min_frames:
                continue
            rr = sorted(rr, key=lambda i: P["ts"][i])
            jobs.append((cond, sc, [(r, kc[r], kp[cond][r] if cond in kp else -1) for r in rr], thr_s, prepass))
        if a.max_scenes:
            jobs = jobs[: a.max_scenes]
        return jobs

    # Vorlauf: Tor-Schwelle fuer 'gscore' = 5-%-Quantil der Einzelbild-Registrierungsguete aller sauberen Bilder
    _G["pert"][None] = clean_g
    _G["pf"][None] = pf["stored"]
    _G["fail"][None] = fail["stored"]
    with mp.get_context("fork").Pool(a.workers) as pool:
        pre = pool.map(run_scene, scene_jobs(None, ids_c, 0.0, True), chunksize=1)
    s_all = np.array([r["s_pf"] for s in pre for r in s["recs"] if r["ok_ex"]])
    thr_s = float(np.quantile(s_all, 0.05))
    print(f"Tor-Schwelle gscore: {thr_s:.3f} (5-%-Quantil der Registrierungsguete auf {len(s_all)} sauberen Bildern)", flush=True)
    res = dict(args=vars(a), thr_s=thr_s, conds={}, pols=POLS)
    pairs = [("gmon@0.5", "track"), ("gmon@0.5", "perframe"), ("gmon@0.5", "asgiven"), ("gmon@0.5", "goracle"), ("gscore", "track"), ("track", "perframe")]
    for c in conds:
        rows = sorted(kp[c].keys())
        jobs = scene_jobs(c, rows, thr_s, False)
        with mp.get_context("fork").Pool(a.workers) as pool:
            sc_res = pool.map(run_scene, jobs, chunksize=1)
        for s, j in zip(sc_res, jobs):
            n = len(j[2])
            s["f0"] = int(n * a.onset)
        summ = summarize(sc_res, POLS)
        summ["n_scenes"] = len(sc_res)
        summ["boot"] = bootstrap(sc_res, POLS, pairs)
        # Spur: Szene mit dem groessten Unterschied track - gmon@0.5 im Versatz der Ausfallphase
        best, bd = None, -1.0
        for s in sc_res:
            fr = [r for r in s["recs"] if r["fail_phase"]]
            if len(fr) >= 8:
                d_ = float(np.mean([r["err"]["track"][0] - r["err"]["gmon@0.5"][0] for r in fr]))
                if d_ > bd:
                    best, bd = s, d_
        if best is not None:
            summ["trace"] = dict(scene=best["scene"], f0=best["f0"], f=[r["f"] for r in best["recs"]], pf=[r["pf"] for r in best["recs"]], fail_true=[r["fail_true"] for r in best["recs"]],
                                 err={p: [r["err"][p][0] for r in best["recs"]] for p in ("asgiven", "perframe", "track", "gmon@0.5", "goracle")}, acc={p: [r["acc"][p] for r in best["recs"]] for p in ("gmon@0.5",)})
        res["conds"][c] = summ
        f, cl = summ["fail"], summ["clean"]
        print(f"[{c}] scenes {summ['n_scenes']}, fail-phase frames {f['n']} (true failures {f['true_fail_share']:.2f}); lateral error m (fail phase): " + " ".join(f"{p} {f['pols'][p]['lata']:.2f}" for p in POLS) + " | clean phase: " + " ".join(f"{p} {cl['pols'][p]['lata']:.2f}" for p in ("perframe", "track", "gmon@0.5", "gscore")), flush=True)
        json.dump(res, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)
        pickle.dump(sc_res, open(f"{OUT}/{a.out}/frames_{c}.pkl", "wb"))
    # Mittel ueber die Bedingungen (Ausfallphase, jede Bedingung gleiches Gewicht)
    mean = {}
    for ph in ("clean", "fail", "recovery"):
        if not all(ph in res["conds"][c] for c in conds):
            continue
        mean[ph] = {p: {k: float(np.nanmean([res["conds"][c][ph]["pols"][p][k] for c in conds])) for k in ("lata", "trans", "yaw", "dev3")} for p in POLS}
    res["mean"] = mean
    json.dump(res, open(f"{OUT}/{a.out}/metrics.json", "w"), indent=1)
    print("Mittel (Ausfallphase):", {p: round(mean["fail"][p]["lata"], 3) for p in POLS}, flush=True)


if __name__ == "__main__":
    main()
