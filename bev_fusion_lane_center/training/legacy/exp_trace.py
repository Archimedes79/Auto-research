"""E26: Sparsifizierung auf dem Abstandsfeld - Punkt-fuer-Punkt-Linienverfolger (line_tracer.py) gegen Schwelle + Skelett.

Quellen (Validierung, Teile 1-3, 1.706 Samples):
  crn_orakel                : Soll-Abstandsfeld (perfektes Feld) -> misst den Verlust der Linienbildung selbst
  crn_unet64                : Kamera+Radar (CRN), U-Netz 64 (bestes CRN-Netz, E17/E19)
  gcar_unet64 (E27 ff.)     : Kamera+Radar (GaussianCaR), U-Netz 64 auf den Karten-Merkmalen, Felder aus out/crn/fields_gcar_unet64.npy
Arme: Skelett (Standard 0,5 m), Skelett (fein abgestimmt, E19), Verfolger (Standardparameter), Verfolger (abgestimmt), Verfolger auf 3-m-Knoten ausgeduennt.
Abstimmung der Verfolger-Parameter auf den geraden Samples (wie E19 beim Skelett); berichtet wird auf den ungeraden (ungesehen) und auf allen.
Aufruf: exp_trace.py <quelle>   -> out/crn/exp_trace_<quelle>.json, out/crn/trace_ex_<quelle>.pkl
"""
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from numba import njit
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import centerline_estimator as CE  # noqa: E402
import field_tools as FT  # noqa: E402
import line_tracer as LT  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn"
home = os.path.expanduser("~")
SRC = {
    "crn_orakel": dict(name=None, exp="exp_crn_b123", gtd="val_gt_dist_exact.npy", est="unet64"),
    "crn_unet64": dict(name="unet64", exp="exp_crn_b123", est="unet64"),
    # neues Fundament (GaussianCaR): gleiche Samples, gleiches Soll im Fahrzeug-System; Felder und Einstellung des Skeletts aus dem Lauf des Fundaments
    "gcar_unet64": dict(name="gcar_unet64", exp="exp_crn_b123", est="gcar_unet64"),
}
TAUS = (0.5, 1.0, 1.5)
MIN_HOLE = 16  # Pixel (= 4 m^2)
EX_IDS = [40, 500, 1100, 1500]
GRID = [("sig_nav", [0.7, 1.0, 1.5]), ("t_on", [0.3, 0.6, 1.0, 1.5]), ("t_seed", [0.2, 0.35, 0.6]), ("min_len", [2.0, 4.0, 8.0, 15.0]), ("gap_max", [0, 2, 4]),
        ("lam", [0.05, 0.15, 0.4]), ("r_seed", [2.0, 3.0, 4.0]), ("merge_deg", [25.0, 40.0, 60.0]), ("beta", [0.0, 0.03, 0.06, 0.1, 0.15]), ("dup_m", [0.0, 1.0, 1.5, 2.0])]
OBJ = os.environ.get("TRACE_OBJ", "o2o")  # Abstimmungsziel: 'o2o' = F1 bei 1 m mit 1:1-Zuordnung (bestraft doppelte Linien), 'f1' = Punkt-F1 bei 1 m


# ------------------------------------------------------------------ Daten


def load_source(key):
    s = SRC[key]
    if s["name"]:
        D = np.load(f"{OUT}/fields_{s['name']}.npy").astype(np.float32)
    else:
        D = np.load(f"{home}/{s['exp']}/feats/{s['gtd']}").astype(np.float32) * 0.5  # Soll-Feld liegt in Pixeln (8 px = 4 m)
    g = np.load(f"{home}/{s['exp']}/graph/val_graph.npz")
    ex = f"{home}/{s['exp']}/feats/val_gt_pts_exact.npy"
    gts = list(np.load(ex, allow_pickle=True)) if os.path.exists(ex) else [CE.gt_points(g["nodes"][i], g["poly"][i], g["mask"][i]) for i in range(len(D))]
    g = {k: g[k] for k in ("nodes", "poly", "mask")}
    if os.environ.get("TRACE_SUB"):  # Rauchtest mit wenigen Samples
        n = int(os.environ["TRACE_SUB"])
        D, g, gts = D[:n], {k: v[:n] for k, v in g.items()}, gts[:n]
    return D, g, gts


def lane_data(nodes, poly, mask, step=0.5):
    """dichte Punkte je Spur (Soll-Punkte) mit Spur-Nummer; zusaetzlich Polylinien in Pixeln fuer die Rasterung."""
    pts, ids, polys_px = [], [], []
    for k, pid in enumerate(np.unique(poly[mask])):
        q = nodes[(poly == pid) & mask]
        seg = []
        run = [q[0]] if len(q) else []
        for a, b in zip(q[:-1], q[1:]):
            L = np.linalg.norm(b - a)
            if L < 5.0:
                n = max(int(L / step), 1)
                seg.append(a + (b - a) * (np.arange(n)[:, None] / n))
            if L < 6.0:
                run.append(b)
            else:
                if len(run) > 1:
                    polys_px.append(np.stack([(np.array(run)[:, 0] + 50) / 0.5 - 0.5, (np.array(run)[:, 1] + 50) / 0.5 - 0.5], 1))
                run = [b]
        if len(run) > 1:
            polys_px.append(np.stack([(np.array(run)[:, 0] + 50) / 0.5 - 0.5, (np.array(run)[:, 1] + 50) / 0.5 - 0.5], 1))
        if len(q):
            seg.append(q[-1:])
        if seg:
            s = np.concatenate(seg)
            pts.append(s)
            ids.append(np.full(len(s), k))
    if not pts:
        return np.zeros((0, 2), np.float32), np.zeros(0, int), polys_px
    return np.concatenate(pts).astype(np.float32), np.concatenate(ids), polys_px


# ------------------------------------------------------------------ Kennzahlen je Sample


@njit(cache=True)
def _greedy(pi, gi, n_p, n_g):
    up = np.zeros(n_p, np.bool_)
    ug = np.zeros(n_g, np.bool_)
    m = 0
    for t in range(pi.shape[0]):
        p = pi[t]
        g = gi[t]
        if (not up[p]) and (not ug[g]):
            up[p] = True
            ug[g] = True
            m += 1
    return m


def match_1to1(P, G, tG, taus=TAUS, K=8):
    """Gierige 1:1-Zuordnung geschaetzter Punkte zu Soll-Punkten (naechste Paare zuerst) innerhalb der Toleranzen -> Zahl der Paare je Toleranz.
    Doppelte Linien (zwei geschaetzte Linien an einer Soll-Linie) koennen nur einmal zuordnen: bestraft Gewirr, das die Punkt-F1 nicht sieht."""
    d, idx = tG.query(P, k=K, distance_upper_bound=max(taus) + 1e-6)
    pi = np.repeat(np.arange(len(P)), K)
    d, gi = d.ravel(), idx.ravel()
    ok = np.isfinite(d)
    pi, gi, d = pi[ok], gi[ok], d[ok]
    o = np.argsort(d, kind="stable")
    pi, gi, d = pi[o], gi[o], d[o]
    return [_greedy(pi[d <= t].astype(np.int64), gi[d <= t].astype(np.int64), len(P), len(G)) for t in taus]


def geom_stats(P, G, Lp, Lid):
    """Punkt-Zaehler (0,5/1/1,5 m), 1:1-Paare, Spur-Ausbeute (Anteil der Spuren, von denen >= 80 % der Punkte innerhalb 1 m liegen), Pruefwert-Daten."""
    out = dict(cnt=np.zeros((3, 4)), m1=np.zeros(3), lanes_found=0, lanes_total=0, lane_cov=0.0)
    nl = int(Lid.max()) + 1 if len(Lid) else 0
    out["lanes_total"] = nl
    if len(P) == 0 or len(G) == 0:
        out["cnt"][:, 1] = len(P)
        out["cnt"][:, 3] = len(G)
        return out, None
    tG = cKDTree(G)
    dp = tG.query(P)[0]
    dg = cKDTree(P).query(G)[0]
    out["m1"] = np.array(match_1to1(P, G, tG), float)
    for k, t in enumerate(TAUS):
        out["cnt"][k] = (np.sum(dp <= t), len(P), np.sum(dg <= t), len(G))
    if nl:
        dl = cKDTree(P).query(Lp)[0] <= 1.0
        cov = np.bincount(Lid, weights=dl.astype(float), minlength=nl) / np.maximum(np.bincount(Lid, minlength=nl), 1)
        out["lanes_found"] = int((cov >= 0.8).sum())
        out["lane_cov"] = float(cov.sum())
    return out, dp


def topo_desc(sk):
    """Zusammenhangskomponenten, Loecher (eingeschlossene Hintergrundflaechen), Endpunkte, Kreuzungs-Cluster eines duennen Rasters."""
    _, ncomp = ndi.label(sk, structure=np.ones((3, 3)))
    lb, nb = ndi.label(~sk)  # 4er-Nachbarschaft fuer den Hintergrund
    border = set(np.unique(np.concatenate([lb[0, :], lb[-1, :], lb[:, 0], lb[:, -1]]))) - {0}
    area = np.bincount(lb.ravel(), minlength=nb + 1)
    holes = sum(1 for b in range(1, nb + 1) if b not in border and area[b] >= MIN_HOLE)  # Schleifen: eingeschlossene Flaechen ab 4 m^2 (kleinere sind Rasterartefakte)
    p = np.pad(sk, 1)
    nb8 = [p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[0:-2, :-2]]
    runs = sum(((~nb8[i]) & nb8[(i + 1) % 8]).astype(int) for i in range(8))
    cnt = sum(n.astype(int) for n in nb8)
    ends = int((sk & (cnt >= 1) & (cnt <= 2) & (runs == 1)).sum())
    jm = ndi.binary_dilation(sk & (runs >= 3), structure=np.ones((3, 3)), iterations=2)  # Kreuzungspixel im Umkreis von 2 Pixeln zu einer Kreuzung
    junc = ndi.label(jm, structure=np.ones((3, 3)))[1]
    return np.array([ncomp, holes, ends, junc], float)


def thin_rasters(R, bs=64):
    out = np.zeros_like(R)
    for i in range(0, len(R), bs):
        out[i : i + bs] = FT.thin_batch(torch.from_numpy(R[i : i + bs]).to("cuda")).cpu().numpy()
    return out


def aggregate(st, idx):
    """st: Dict mit Arrays je Sample -> Kennzahlen ueber die Teilmenge idx."""
    c = st["cnt"][idx].sum(0)
    r = {}
    for k, t in enumerate(TAUS):
        pr, rc = c[k, 0] / max(c[k, 1], 1), c[k, 2] / max(c[k, 3], 1)
        r[str(t)] = [float(pr), float(rc), float(2 * pr * rc / max(pr + rc, 1e-9))]
    m1 = st["m1"][idx].sum(0)
    r["o2o"] = {str(t): [float(m1[k] / max(c[k, 1], 1)), float(m1[k] / max(c[k, 3], 1)), float(2 * m1[k] / max(c[k, 1] + c[k, 3], 1))] for k, t in enumerate(TAUS)}  # 1:1-Zuordnung: Praez., Ausb., F1
    r["lane_recall"] = float(st["lanes_found"][idx].sum() / max(st["lanes_total"][idx].sum(), 1))
    r["lane_cov"] = float(st["lane_cov"][idx].sum() / max(st["lanes_total"][idx].sum(), 1))
    if st.get("topo") is not None:
        r["topo"] = [float(x) for x in st["topo"][idx].mean(0)]  # Komponenten, Loecher, Endpunkte, Kreuzungen je Sample
    if st.get("nodes") is not None:
        r["nodes"] = float(np.mean(st["nodes"][idx]))
    if st.get("points") is not None:
        r["points_per_sample"] = float(np.mean(st["points"][idx]))
    return r


def empty_stats(S):
    return dict(cnt=np.zeros((S, 3, 4)), m1=np.zeros((S, 3)), lanes_found=np.zeros(S, int), lanes_total=np.zeros(S, int), lane_cov=np.zeros(S), topo=None, nodes=None, points=np.zeros(S))


# ------------------------------------------------------------------ Arme


def arm_skeleton(D, gts, lanes, sk):
    S = len(D)
    st = empty_stats(S)
    for i in range(S):
        P = np.argwhere(sk[i]).astype(np.float32)
        P = np.stack([-50 + (P[:, 0] + 0.5) * 0.5, -50 + (P[:, 1] + 0.5) * 0.5], 1)
        P = P[(np.abs(P) < 48).all(1)]
        o, _ = geom_stats(P, gts[i], lanes[i][0], lanes[i][1])
        st["cnt"][i], st["m1"][i], st["lanes_found"][i], st["lanes_total"][i], st["lane_cov"][i] = o["cnt"], o["m1"], o["lanes_found"], o["lanes_total"], o["lane_cov"]
        st["points"][i] = len(P)
    st["topo"] = np.stack([topo_desc(sk[i]) for i in range(S)])
    st["nodes"] = sk.reshape(S, -1).sum(1).astype(float)
    return st


def run_tracer(D, params, idx=None):
    idx = range(len(D)) if idx is None else idx
    res = {}
    for i in idx:
        res[i] = LT.trace_field(D[i], params)
    return res


def arm_tracer(D, gts, lanes, res, node_m=None, calib=None):
    """node_m: None = dichte Polylinie (Knoten alle 1 m, Auswertung alle 0,5 m); sonst Knoten alle node_m m (lineare Zwischenpunkte)."""
    S = len(D)
    st = empty_stats(S)
    rast = np.zeros((S, 200, 200), bool)
    nodes = np.zeros(S)
    for i in range(S):
        polys = res[i]["polys"] if i in res else []
        if node_m:
            polys = LT.sparse_nodes(polys, node_m)
        nodes[i] = sum(len(a) for a in polys)
        P = LT.polys_points_m(polys, 0.5)
        o, dp = geom_stats(P, gts[i], lanes[i][0], lanes[i][1])
        st["cnt"][i], st["m1"][i], st["lanes_found"][i], st["lanes_total"][i], st["lane_cov"][i] = o["cnt"], o["m1"], o["lanes_found"], o["lanes_total"], o["lane_cov"]
        st["points"][i] = len(P)
        rast[i] = LT.raster(polys)
        if calib is not None and dp is not None and len(P):
            r = (P[:, 0] + 50) / 0.5 - 0.5
            c = (P[:, 1] + 50) / 0.5 - 0.5
            v = ndi.map_coordinates(D[i], [r, c], order=1, mode="nearest")
            b = np.digitize(v, calib["bins"])
            calib["n"] += np.bincount(b, minlength=len(calib["bins"]) + 1)
            calib["ok1"] += np.bincount(b, weights=(dp <= 1.0).astype(float), minlength=len(calib["bins"]) + 1)
            calib["ok05"] += np.bincount(b, weights=(dp <= 0.5).astype(float), minlength=len(calib["bins"]) + 1)
    rt = thin_rasters(rast)
    st["topo"] = np.stack([topo_desc(rt[i]) for i in range(S)])
    st["nodes"] = nodes
    return st


def tune(D, gts, lanes, idx_even, base):
    """Koordinatenabstieg auf den geraden Samples; Ziel: F1 bei 1 m mit 1:1-Zuordnung (OBJ='o2o') oder Punkt-F1 (OBJ='f1')."""
    cur = dict(base)

    def score(p):
        cnt = np.zeros(4)
        m1 = 0.0
        for i in idx_even:
            polys = LT.trace_field(D[i], p)["polys"]
            P = LT.polys_points_m(polys, 0.5)
            G = gts[i]
            if len(P) == 0 or len(G) == 0:
                cnt += (0, len(P), 0, len(G))
                continue
            tG = cKDTree(G)
            dp = tG.query(P)[0]
            dg = cKDTree(P).query(G)[0]
            cnt += (np.sum(dp <= 1.0), len(P), np.sum(dg <= 1.0), len(G))
            if OBJ == "o2o":
                m1 += match_1to1(P, G, tG, taus=(1.0,))[0]
        if OBJ == "o2o":
            return 2 * m1 / max(cnt[1] + cnt[3], 1)
        pr, rc = cnt[0] / max(cnt[1], 1), cnt[2] / max(cnt[3], 1)
        return 2 * pr * rc / max(pr + rc, 1e-9)

    best = score(cur)
    hist = [dict(round=0, param="Start", value="-", f1=float(best))]
    fast = os.environ.get("TRACE_FAST") == "1"
    for rnd in ((1,) if fast else (1, 2)):
        for name, vals in (GRID[:2] if fast else GRID):
            for v in vals:
                if v == cur[name]:
                    continue
                p = dict(cur)
                p[name] = v
                f = score(p)
                hist.append(dict(round=rnd, param=name, value=v, f1=float(f)))
                if f > best + 1e-6:
                    best, cur = f, p
        print(f"  Runde {rnd}: F1@1m (gerade Samples) {best:.4f}  {({k: cur[k] for k, _ in GRID})}", flush=True)
    return cur, hist


# ------------------------------------------------------------------ Hauptlauf


def main(key):
    t00 = time.time()
    D, g, gts = load_source(key)
    S = len(D)
    est = SRC[key]["est"]
    tuned_sk = json.load(open(f"{OUT}/exp_est_{est}.json"))["tuned"]
    lanes = [lane_data(g["nodes"][i], g["poly"][i], g["mask"][i]) for i in range(S)]
    lane_arr = [(a, b) for a, b, _ in lanes]
    gt_ras = np.zeros((S, 200, 200), bool)
    for i in range(S):
        gt_ras[i] = LT.raster(lanes[i][2])
    gt_topo = np.stack([topo_desc(x) for x in thin_rasters(gt_ras)])
    idx_all = np.arange(S)
    idx_even, idx_odd = idx_all[::2], idx_all[1::2]
    J = dict(key=key, S=S, gt=dict(topo=[float(x) for x in gt_topo.mean(0)], nodes=float(g["mask"].sum(1).mean()), points=float(np.mean([len(x) for x in gts]))), arms={}, skeleton_tuned_setting=dict(thr=tuned_sk["thr"], min_comp=tuned_sk["min_comp"]))
    print(f"[{key}] Daten geladen ({time.time() - t00:.0f} s); Soll-Topologie je Sample (Komp., Loecher, Endpunkte, Kreuzungen): {gt_topo.mean(0).round(2)}", flush=True)

    def put(name, st, extra=None):
        J["arms"][name] = dict(alle=aggregate(st, idx_all), ungesehen=aggregate(st, idx_odd), gerade=aggregate(st, idx_even))
        if extra:
            J["arms"][name].update(extra)
        a = J["arms"][name]["ungesehen"]
        print(f"[{key}] {name:40s} F1 @0,5/1/1,5 (ungesehen): {a['0.5'][2]:.3f} / {a['1.0'][2]:.3f} / {a['1.5'][2]:.3f} | 1:1 @1 m {a['o2o']['1.0'][2]:.3f} | P/R@1 {a['1.0'][0]:.3f}/{a['1.0'][1]:.3f} | Spuren {a['lane_recall']:.3f} | Topologie {np.round(a['topo'], 2)} | Knoten {a.get('nodes', 0):.0f}", flush=True)

    # Skelett-Arme (GPU-Ausduennen)
    t0 = time.time()
    sk_def = FT.skeletons(D, 0.5, 0.0, None, 0)
    t_sk_def = time.time() - t0
    t0 = time.time()
    sk_tun = FT.skeletons(D, tuned_sk["thr"], 0.0, None, tuned_sk["min_comp"])
    t_sk_tun = time.time() - t0
    put("Skelett Standard (0,5 m)", arm_skeleton(D, gts, lane_arr, sk_def), dict(zeit_ms=1000 * t_sk_def / S))
    put("Skelett abgestimmt", arm_skeleton(D, gts, lane_arr, sk_tun), dict(zeit_ms=1000 * t_sk_tun / S))

    # Skelett -> Douglas-Peucker-Knoten (nur jedes 4. Sample, Zaehlung der Knoten)
    sub = np.arange(1, S, 4)
    nn = []
    for i in sub:
        ls = CE.CenterlineEstimator.vectorize(sk_tun[i], 0.4)
        nn.append(sum(len(a) for a in ls))
    J["skelett_dp_knoten"] = float(np.mean(nn))

    # Verfolger: Standardparameter
    t0 = time.time()
    res_def = run_tracer(D, None)
    t_tr_def = time.time() - t0
    put("Verfolger Standardparameter", arm_tracer(D, gts, lane_arr, res_def), dict(zeit_ms=1000 * t_tr_def / S, params=LT.DEFAULT))

    # fruehere Fassung v1 (nur auf Punkt-F1 abgestimmt, ohne Mittelachsenzug und Duplikatentfernung) und Einzelschritte (nur echte Felder)
    res_v1 = None
    pth = f"{OUT}/exp_trace_v1_{key}.json"
    if os.path.exists(pth):
        v1p = dict(json.load(open(pth))["arms"]["Verfolger abgestimmt"]["params"])
        v1p.update(beta=0.0, dup_m=0.0)
        res_v1 = run_tracer(D, v1p)
        put("Verfolger v1 (Punkt-F1 abgestimmt)", arm_tracer(D, gts, lane_arr, res_v1), dict(params=v1p))
        put("v1 + Duplikate entfernt", arm_tracer(D, gts, lane_arr, run_tracer(D, dict(v1p, dup_m=1.0))), dict(params=dict(v1p, dup_m=1.0)))
        put("v1 + Mittelachsenzug", arm_tracer(D, gts, lane_arr, run_tracer(D, dict(v1p, beta=0.06))), dict(params=dict(v1p, beta=0.06)))

    # Abstimmung auf den geraden Samples
    print(f"[{key}] Abstimmung der Verfolger-Parameter ...", flush=True)
    best, hist = tune(D, gts, lane_arr, idx_even, LT.DEFAULT)
    J["tuning"] = hist
    calib = dict(bins=np.array([0.05, 0.15, 0.3, 0.6, 1.0]), n=np.zeros(6), ok1=np.zeros(6), ok05=np.zeros(6))
    t0 = time.time()
    res = run_tracer(D, best)
    t_tr = time.time() - t0
    put("Verfolger abgestimmt", arm_tracer(D, gts, lane_arr, res, calib=calib), dict(zeit_ms=1000 * t_tr / S, params=best))
    put("Verfolger abgestimmt, 3-m-Knoten", arm_tracer(D, gts, lane_arr, res, node_m=3.0))
    put("Verfolger abgestimmt, 6-m-Knoten", arm_tracer(D, gts, lane_arr, res, node_m=6.0))
    J["pruefwert"] = dict(bins=[float(b) for b in calib["bins"]], n=[float(x) for x in calib["n"]], ok1=[float(x) for x in calib["ok1"]], ok05=[float(x) for x in calib["ok05"]])
    # Endgruende und Linien je Sample
    rs = np.zeros(5)
    nlines, lens, snaps = [], [], 0
    for i in range(S):
        meta = res[i]["info"]["meta"]
        nlines.append(len(meta))
        lens.append(sum(m["length"] for m in meta))
        for m in meta:
            rs[m["reason_start"]] += 1
            rs[m["reason_end"]] += 1
            snaps += int(m["snap_start"]) + int(m["snap_end"])
    J["tracer_stats"] = dict(linien_je_sample=float(np.mean(nlines)), laenge_je_sample_m=float(np.mean(lens)), endgruende=[float(x) for x in rs], anbindungen_je_sample=snaps / S)

    # Beispiele fuer die Bilder
    ex = {}
    for i in [i for i in EX_IDS if i < S]:
        ex[i] = dict(sk_tun=sk_tun[i], sk_def=sk_def[i], polys=res[i]["polys"], polys_def=res_def[i]["polys"], polys_v1=(res_v1[i]["polys"] if res_v1 else None))
    pickle.dump(ex, open(f"{OUT}/trace_ex_{key}.pkl", "wb"))
    json.dump(J, open(f"{OUT}/exp_trace_{key}.json", "w"), indent=1)
    print(f"[{key}] FERTIG nach {time.time() - t00:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
