"""E31: Raster-Ansatz -> Graph. Dichte Karte (Heatmap + Richtung) -> gerichteter Spurmitten-Graph (Polylinien + Nachfolger-Kanten, Abzweige/Zusammenfuehrungen).

Zwei Bauarten, gleiche Nachbearbeitung:
  'skel'  Schwelle -> Ausduennen (Zhang-Suen) -> Skelett-Graph (Kreuzungsklumpen, Segmente dazwischen)
  'trace' Linienverfolger (E26) auf einem Pseudo-Abstandsfeld d = sigma * sqrt(-2 ln p); Treffer auf fremde Linien (snap) werden zu Abzweigen
Gemeinsam (assemble): jede Polylinie nach der Richtungskarte orientieren (mittleres Skalarprodukt), an Treffpunkten teilen, Ende i -> Anfang j verbinden,
wenn nah genug und Kurswinkel passt. Alles in Pixeln (Zeile = x, Spalte = y, 0,5 m je Pixel), Ausgabe in Metern.
Aufruf als Programm: graph_from_dense.py --name <dichter Lauf> --set val --variant trace [--oracle]  (Umgebung gcar oder bev)
"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import line_tracer as LT  # noqa: E402

PX = 0.5
SIGMA_M = 0.75
HOME = os.path.expanduser("~")
OUT = BP.RUNS
NB = np.ones((3, 3), np.uint8)
NB[1, 1] = 0
S8 = np.ones((3, 3), bool)

SKEL_DEFAULT = dict(tau=0.5, min_comp=30, spur_px=8, join_px=2.5)
TRACE_DEFAULT = dict(t_seed=0.55, t_on=0.9, beta=0.06, min_len=4.0, dup_m=1.0)
ASM_DEFAULT = dict(link_px=3.0, gap_px=8.0, gap_deg=30.0, max_turn=100.0, orient=True, min_len_m=2.0, lead_m=0.0)


def px_to_m(a):
    return -50.0 + (np.asarray(a, np.float64) + 0.5) * PX


def prob_to_dist(p, sigma=SIGMA_M, floor=1e-3):
    """Heatmap p = exp(-d^2 / 2 sigma^2) -> Pseudo-Abstand d in Metern (so wurde das Soll gebaut)."""
    return (sigma * np.sqrt(-2.0 * np.log(np.clip(np.asarray(p, np.float32), floor, 1.0)))).astype(np.float32)


def path_len(a):
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) if len(a) > 1 else 0.0


def tangent_end(a, m=4, end=True):
    """Fahrtrichtung am Ende (end=True) bzw. Anfang einer Polylinie in Pixeln, normiert."""
    if len(a) < 2:
        return np.array([1.0, 0.0])
    if end:
        i = max(len(a) - 1 - m, 0)
        v = a[-1] - a[i]
    else:
        j = min(m, len(a) - 1)
        v = a[j] - a[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


# ------------------------------------------------------------------ Skelett -> Segmente


def _bfs(nbrs, s):
    n = len(nbrs)
    dist, par = [-1] * n, [-1] * n
    dist[s] = 0
    q = [s]
    for u in q:
        for v in nbrs[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                par[v] = u
                q.append(v)
    return dist, par


def order_path(pix):
    """8-zusammenhaengende duenne Pixelmenge (n,2) -> laengster geodaetischer Weg (geordnete Pixel)."""
    from scipy.spatial import cKDTree

    n = len(pix)
    if n <= 2:
        return pix
    t = cKDTree(pix)
    nbrs = [[] for _ in range(n)]
    for a, b in t.query_pairs(1.5):
        nbrs[a].append(b)
        nbrs[b].append(a)
    d0, _ = _bfs(nbrs, 0)
    u = int(np.argmax(d0))
    d1, par = _bfs(nbrs, u)
    v = int(np.argmax(d1))
    path = []
    while v >= 0:
        path.append(v)
        v = par[v]
    return pix[path]


def skeleton_polylines(sk, spur_px=8.0, join_px=2.5):
    """Skelett (bool) -> Segmente dict(path (n,2) Pixel, att [Klumpen am Anfang, am Ende] (0 = frei))."""
    sk = np.asarray(sk, bool)
    nb = ndi.convolve(sk.astype(np.uint8), NB, mode="constant")
    junc = sk & (nb >= 3)
    jreg = ndi.binary_dilation(junc, S8) & sk
    jl, nj = ndi.label(jreg, S8)
    seg = sk & ~jreg
    sl, ns = ndi.label(seg, S8)
    if nj:
        dj, (ir, ic) = ndi.distance_transform_edt(~jreg, return_indices=True)
        cen = np.array(ndi.center_of_mass(jreg, jl, range(1, nj + 1)))
    segs = []
    for s, sl_ in enumerate(ndi.find_objects(sl), start=1):
        pix = np.argwhere(sl[sl_] == s) + np.array([sl_[0].start, sl_[1].start])
        path = order_path(pix).astype(np.float64)
        att = [0, 0]
        for k, e in enumerate((path[0], path[-1])):
            r, c = int(e[0]), int(e[1])
            if nj and dj[r, c] <= join_px:
                att[k] = int(jl[ir[r, c], ic[r, c]])
        if att[0]:
            path = np.vstack([cen[att[0] - 1][None], path])
        if att[1]:
            path = np.vstack([path, cen[att[1] - 1][None]])
        if (att[0] == 0 or att[1] == 0) and path_len(path) < spur_px:
            continue  # kurzer Stummel mit freiem Ende
        if path_len(path) < 2.0:
            continue
        segs.append(dict(path=path, att=att))
    return segs


# ------------------------------------------------------------------ gemeinsame Nachbearbeitung


def orient(a, dirf):
    """Polylinie (Pixel) nach Richtungskarte dirf [2,H,W] (dx = Zeilenrichtung, dy = Spaltenrichtung) orientieren. -> (a', reversed, confidence)"""
    if dirf is None or len(a) < 2:
        return a, False, 0.0
    t = np.gradient(a, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    r = np.clip(np.round(a[:, 0]).astype(int), 0, dirf.shape[1] - 1)
    c = np.clip(np.round(a[:, 1]).astype(int), 0, dirf.shape[2] - 1)
    d = np.stack([dirf[0][r, c], dirf[1][r, c]], 1).astype(np.float64)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    s = float((t * d).sum(1).mean())
    return (a[::-1].copy(), True, -s) if s < 0 else (a, False, s)


def link_edges(polys, link_px, gap_px, gap_deg, max_turn):
    """Kanten (i, j): Ende von i -> Anfang von j, wenn nah (link_px) mit Kurswinkel <= max_turn oder (Luecke bis gap_px) fast geradeaus (gap_deg)."""
    n = len(polys)
    if n < 2:
        return []
    ends = np.array([p[-1] for p in polys])
    starts = np.array([p[0] for p in polys])
    te = [tangent_end(p, 4, True) for p in polys]
    ts = [tangent_end(p, 4, False) for p in polys]
    D = np.linalg.norm(ends[:, None] - starts[None], axis=2)
    out = []
    for i in range(n):
        for j in range(n):
            if i == j or D[i, j] > gap_px:
                continue
            cosang = float(np.clip(te[i] @ ts[j], -1, 1))
            ang = np.degrees(np.arccos(cosang))
            if D[i, j] <= link_px:
                if ang <= max_turn:
                    out.append((i, j))
            elif ang <= gap_deg:
                v = starts[j] - ends[i]
                nv = np.linalg.norm(v)
                if nv > 1e-9 and np.degrees(np.arccos(np.clip(te[i] @ (v / nv), -1, 1))) <= gap_deg:
                    out.append((i, j))
    return out


def finish(polys, edges, min_len_m):
    """Pixel-Polylinien -> Graph in Metern; zu kurze Stuecke fallen mit ihren Kanten weg."""
    keep = [k for k, p in enumerate(polys) if len(p) >= 2 and path_len(p) * PX >= min_len_m]
    ren = {k: n for n, k in enumerate(keep)}
    pts = [px_to_m(polys[k]) for k in keep]
    ed = [(ren[i], ren[j]) for i, j in edges if i in ren and j in ren]
    return dict(pts=pts, edges=ed)


def assemble(polys, dirf, snaps=None, ap=None):
    """polys: Liste (n,2) Pixel; snaps: je Polylinie (Anfang-Ziel, Ende-Ziel) mit Ziel = (Index der Zielpolylinie, (r, c)) oder None.
    lead_m: Orts-Konvention der Abzweige/Zusammenfuehrungen: die Elternlinie wird lead_m VOR (Abzweig) bzw. NACH (Merge) dem Trefferpunkt geteilt, weil das Soll Abzweige am Ende des Spurstuecks
    (Beginn der Aufspaltung) setzt, die Heatmap-Linien sich aber erst einige Meter weiter trennen; die Kanten werden dann explizit gesetzt."""
    a = dict(ASM_DEFAULT)
    if ap:
        a.update(ap)
    P = [np.asarray(p, np.float64) for p in polys]
    sn = [list(s_) for s_ in snaps] if snaps is not None else [[None, None] for _ in P]
    # 1. orientieren (vor dem Teilen, damit Teile die Richtung erben)
    if a["orient"] and dirf is not None:
        for k in range(len(P)):
            P[k], rev, _ = orient(P[k], dirf)
            if rev:
                sn[k] = [sn[k][1], sn[k][0]]
    # 2. Zielpolylinien an Treffpunkten teilen
    cuts = {}
    lead = a["lead_m"] / PX
    for k in range(len(P)):
        for side, s_ in enumerate(sn[k]):  # side 0: Anfang der Linie k trifft, 1: Ende trifft
            if s_ is None:
                continue
            t, rc = s_
            if t is None or t >= len(P) or t == k or len(P[t]) < 4:
                continue
            kk = int(np.argmin(np.linalg.norm(P[t] - np.asarray(rc)[None], axis=1)))
            if kk <= 0 or kk >= len(P[t]) - 1:
                continue
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[t], axis=0), axis=1))])
            ts = cum[kk] - lead if side == 0 else cum[kk] + lead
            idx = int(np.argmin(np.abs(cum - ts)))
            idx = min(max(idx, 1), len(P[t]) - 2)
            cuts.setdefault(t, []).append((idx, side, k))
    pieces, piece_of, forced = [], {}, []
    for k in range(len(P)):
        if k in cuts:
            recs = sorted(cuts[k])
            idxs, rmap = [], []
            for ci, side, br in recs:
                if idxs and ci - idxs[-1] < 3:
                    rmap.append(len(idxs) - 1)
                else:
                    idxs.append(ci)
                    rmap.append(len(idxs) - 1)
            bounds = [0] + idxs + [len(P[k]) - 1]
            ids = []
            for u, v in zip(bounds[:-1], bounds[1:]):
                ids.append(len(pieces))
                pieces.append(P[k][u : v + 1])
            piece_of[k] = ids
            for (ci, side, br), mi in zip(recs, rmap):
                forced.append((("p", ids[mi]), ("p", ids[mi + 1])))
                forced.append((("p", ids[mi]), ("s", br)) if side == 0 else (("e", br), ("p", ids[mi + 1])))
        else:
            piece_of[k] = [len(pieces)]
            pieces.append(P[k])
    # 3. auf 1 Pixel Abstand bringen (glatter, gleichmaessig); Zaehlung der Stuecke bleibt erhalten
    pieces = [LT.resample(q, 1.0) if len(q) >= 2 else q for q in pieces]

    def res(r):
        kind, v = r
        return v if kind == "p" else (piece_of[v][0] if kind == "s" else piece_of[v][-1])

    edges = set(link_edges(pieces, a["link_px"], a["gap_px"], a["gap_deg"], a["max_turn"]))
    for x, y in forced:
        i, j = res(x), res(y)
        if i != j:
            edges.add((i, j))
    return finish(pieces, sorted(edges), a["min_len_m"])


# ------------------------------------------------------------------ Bauarten


def graph_from_skeleton(sk, dirf, sp=None, ap=None):
    p = dict(SKEL_DEFAULT)
    if sp:
        p.update(sp)
    segs = skeleton_polylines(sk, p["spur_px"], p["join_px"])
    polys = [s["path"] for s in segs]
    polys = [LT.smooth_poly(LT.resample(q, 1.0), 2) if len(q) >= 3 else q for q in polys]
    return assemble(polys, dirf, None, ap)


def graph_from_trace(prob, dirf, tp=None, ap=None):
    t = dict(TRACE_DEFAULT)
    if tp:
        t.update(tp)
    D = prob_to_dist(prob)
    r = LT.trace_field(D, t)
    polys, meta = r["polys"], r["info"]["meta"]
    pos = {m["id"]: k for k, m in enumerate(meta)}
    snaps = []
    for m in meta:
        s0, s1 = m["snap_start_to"], m["snap_end_to"]
        snaps.append([None if s0 is None or s0[0] not in pos else (pos[s0[0]], s0[1]), None if s1 is None or s1[0] not in pos else (pos[s1[0]], s1[1])])
    return assemble(polys, dirf, snaps, ap)


def build(prob, dirf, variant, sk=None, params=None, ap=None):
    if variant == "skel":
        return graph_from_skeleton(sk, dirf, params, ap)
    return graph_from_trace(prob, dirf, params, ap)


# ------------------------------------------------------------------ Auswertung eines gespeicherten dichten Laufs


def load_gt_graphs(part, ids):
    G = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_{part}.pkl", "rb"))
    return [G[int(i)] for i in ids]


def thin_all(prob, tau, min_comp=30, dev="cpu", bs=32):
    import torch

    import dense_head as DH

    out = []
    for i in range(0, len(prob), bs):
        x = torch.from_numpy(np.asarray(prob[i : i + bs], np.float32) >= tau).to(dev)
        sk = DH.thin_batch(x).cpu().numpy()
        for s in sk:
            if min_comp > 0:
                lab, n = ndi.label(s, S8)
                if n:
                    sizes = ndi.sum(s, lab, range(1, n + 1))
                    s = np.isin(lab, 1 + np.nonzero(sizes >= min_comp)[0])
            out.append(s)
    return out


def _work(args):
    prob, dirf, variant, sk, params, ap = args
    return build(prob, dirf, variant, sk, params, ap)


def eval_variant(prob, dirf, gts, variant, params=None, ap=None, workers=6, with_route=True):
    """prob [N,200,200], dirf [N,2,200,200] oder None, gts: Liste Soll-Graphen -> (Kennzahlen, Vorhersagegraphen)."""
    import multiprocessing as mp

    sks = [None] * len(prob)
    if variant == "skel":
        p = dict(SKEL_DEFAULT)
        if params:
            p.update(params)
        sks = thin_all(prob, p["tau"], p["min_comp"], dev="cuda" if _has_cuda() else "cpu")
    if workers > 1:  # numba einmal im Hauptprozess uebersetzen (keine gleichzeitigen Cache-Schreibzugriffe der Arbeiter)
        LT.trace_field(np.full((200, 200), 3.0, np.float32), None)
    jobs = [(np.asarray(prob[i], np.float32), None if dirf is None else np.asarray(dirf[i], np.float32), variant, sks[i], params, ap) for i in range(len(prob))]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            preds = pool.map(_work, jobs, chunksize=8)
    else:
        preds = [_work(j) for j in jobs]
    return evaluate_parallel(preds, gts, workers, with_route), preds


def _eval_one(a):
    p, g, i, wr = a
    return GE.eval_sample(p, g, i, wr)


def evaluate_parallel(preds, gts, workers=6, with_route=True):
    import multiprocessing as mp

    jobs = [(p, g, i, with_route) for i, (p, g) in enumerate(zip(preds, gts))]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rs = pool.map(_eval_one, jobs, chunksize=8)
    else:
        rs = [_eval_one(j) for j in jobs]
    return GE.aggregate(rs)


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def oracle_inputs(split, part, ids):
    """Soll-Heatmap (Gauss, wie im Training) und Soll-Richtung als 'perfekte Karte' (Obergrenze der Graph-Bauart)."""
    d = np.load(f"{HOME}/exp_crn_{split}/feats/{part}_gt_dist_exact.npy", mmap_mode="r")
    dr = np.load(f"{HOME}/exp_gcar_{split}/dir_{part}.npy", mmap_mode="r")
    prob = np.stack([np.exp(-((np.asarray(d[int(i)], np.float32) * 0.5) ** 2) / (2 * SIGMA_M ** 2)) for i in ids])
    dirf = np.stack([np.asarray(dr[int(i)], np.float32) for i in ids])
    return prob, dirf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--set", default="val")
    ap.add_argument("--variant", default="trace", choices=["skel", "trace"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--split", default="b123")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--asm", default="{}")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gt_dir", action="store_true", help="Richtung aus dem Soll statt aus der Vorhersage (Zerlegung der Fehler)")
    ap.add_argument("--gt_heat", action="store_true", help="Heatmap aus dem Soll statt aus der Vorhersage")
    a = ap.parse_args()
    part = "train" if a.set in ("train", "train_eval", "oof") else "val"
    params, asm = json.loads(a.params), json.loads(a.asm)
    if a.oracle:
        sp = json.load(open(BP.ROOT + "/data/splits.json")) if a.split == "b123" else None
        ids = np.array(sp[a.set] if sp else list(range(81)))
        if a.n:
            ids = ids[: a.n]
        prob, dirf = oracle_inputs(a.split, part, ids)
    else:
        d = f"{OUT}/{a.name}"
        prob = np.load(f"{d}/prob_{a.set}.npy" if os.path.exists(f"{d}/prob_{a.set}.npy") else f"{d}/heat_{a.set}.npy")
        ids = np.load(f"{d}/ids_{a.set}.npy")
        dirf = np.load(f"{d}/dir_{a.set}.npy") if os.path.exists(f"{d}/dir_{a.set}.npy") else None
        if a.n:
            prob, ids, dirf = prob[: a.n], ids[: a.n], (None if dirf is None else dirf[: a.n])
    if (a.gt_dir or a.gt_heat) and not a.oracle:
        ph, dh = oracle_inputs(a.split, part, ids)
        if a.gt_heat:
            prob = ph
        if a.gt_dir:
            dirf = dh
    gts = load_gt_graphs(part, ids) if a.split == "b123" else pickle.load(open(f"{HOME}/exp_gcar_mini/graph_gt_{part}.pkl", "rb"))[: len(ids)]
    t0 = time.time()
    m, preds = eval_variant(prob, dirf, gts, a.variant, params, asm, a.workers)
    keys = ["F1@0.5", "F1@1.0", "P@1.0", "R@1.0", "F1o2o@1.0", "lanes_found", "false_len_frac", "split_F1", "merge_F1", "split6_F1", "merge6_F1", "route_rec", "route_prec", "route_f", "dir_acc"]
    print(f"[{a.name or 'oracle'} {a.set} {a.variant}] n={len(ids)} ({time.time() - t0:.0f} s)", " ".join(f"{k}={m.get(k, float('nan')):.3f}" for k in keys), flush=True)
    if a.name:
        os.makedirs(f"{OUT}/{a.name}", exist_ok=True)
        json.dump(dict(variant=a.variant, params=params, asm=asm, metrics=m), open(f"{OUT}/{a.name}/graph_{a.set}_{a.variant}{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
