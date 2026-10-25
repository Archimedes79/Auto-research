"""Schleife 5 / E59: Navigationsziel (Route) je Sample aus der Karte + der tatsaechlich gefahrenen Zukunft.
Die Karte (Soll-Graph graph_gt_<teil>.pkl: Stuecke mit Karten-Token) enthaelt das Navigationsziel; hier wird die Route so nachgebildet, wie ein Router sie liefern wuerde: die Folge der Karten-Spuren, die das Fahrzeug in den naechsten Sekunden tatsaechlich befaehrt
(Zukunftspositionen der folgenden Schluesselbilder derselben Szene, 2 Hz, bis 6 s), im Fenster +-50 m.
Je Sample: dict(route=[Stueck-Indizes in Fahrtreihenfolge], path=[M,2] Zukunftspfad im Fahrzeugsystem (1 m abgetastet, nur im Fenster), goal=(x, y) letzter Pfadpunkt im Fenster, v0 Geschwindigkeit (m/s, aus dem Vorgaenger), n_future Zahl der Folgebilder)  oder None (kein Folgebild / Pfad ausserhalb jeder Spur).
Ausgabe: ~/exp_gcar_b123/route_gt_<teil>.pkl.  Aufruf (gcar): make_route_gt.py [<tag>]   (tag: b123 by default; b456 or a check set: ~/exp_gcar_<tag>/)"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_{sys.argv[1] if len(sys.argv) > 1 else 'b123'}"
K_FUT = 12  # Folgebilder (6 s)
HALF = 50.0


def piece_vertices(g):
    V, pid, tan = [], [], []
    for k, p in enumerate(g["pts"]):
        p = np.asarray(p, np.float64)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        V.append(p)
        tan.append(t)
        pid.append(np.full(len(p), k))
    if not V:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 2))
    return np.concatenate(V), np.concatenate(pid), np.concatenate(tan)


def densify(path, step=1.0):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b - a)
        n = max(int(np.ceil(L / step)), 1)
        for u in range(1, n + 1):
            out.append(a + (b - a) * (u / n))
    return np.asarray(out)


def route_for(g, path_vehicle, max_d=2.5, cos_min=np.cos(np.deg2rad(45.0))):
    """Folge der Stuecke, auf denen der (dichte) Pfad entlang laeuft: Viterbi ueber die Pfadpunkte (Kosten = Abstand; Wechsel auf einen Nachfolger 0,3, auf ein nicht verbundenes Stueck 3,0), damit an Kreuzungseinfahrten
    nicht mehrere nebeneinander laufende Verbinder in die Route geraten."""
    V, pid, tan = piece_vertices(g)
    if len(V) == 0 or len(path_vehicle) < 2:
        return []
    tree = cKDTree(V)
    succ = set(map(tuple, g["edges"]))
    dp = densify(path_vehicle, 1.0)
    inside = (np.abs(dp[:, 0]) <= HALF) & (np.abs(dp[:, 1]) <= HALF)
    tp = np.gradient(dp, axis=0)
    tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-9
    cands = []
    for q, tq, ins in zip(dp, tp, inside):
        if not ins:
            continue
        cd = {}
        for c in tree.query_ball_point(q, max_d):
            if float(tan[c] @ tq) < cos_min:
                continue
            d = float(np.linalg.norm(V[c] - q))
            k = int(pid[c])
            if k not in cd or d < cd[k]:
                cd[k] = d
        if cd:
            cands.append(cd)
    if not cands:
        return []
    prev_cost, back = {}, []
    for cd in cands:
        cost, bp = {}, {}
        for k, d in cd.items():
            if not prev_cost:
                cost[k], bp[k] = d, None
                continue
            best = None
            for kk, c in prev_cost.items():
                tr = 0.0 if kk == k else (0.3 if (kk, k) in succ else 3.0)
                v = c + tr + d
                if best is None or v < best[0]:
                    best = (v, kk)
            cost[k], bp[k] = best
        back.append(bp)
        prev_cost = cost
    k = min(prev_cost, key=prev_cost.get)
    seq = [k]
    for bp in reversed(back[1:]):
        k = bp[k]
        seq.append(k)
    seq.reverse()
    out = []
    for k in seq:
        if k not in out:
            out.append(k)
    return out


def main():
    for part in ("train", "val"):
        P = np.load(f"{D}/poses_{part}.npz")
        T, R, scene, prev, dt = P["t"], P["R"], P["scene"], P["prev"], P["dt"]
        G = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        N = len(G)
        nxt = -np.ones(N, int)
        for i in range(N):
            if prev[i] >= 0:
                nxt[prev[i]] = i
        res = []
        n_ok = 0
        for i in range(N):
            fut = []
            j = nxt[i]
            while j >= 0 and len(fut) < K_FUT:
                fut.append(j)
                j = nxt[j]
            if not fut:
                res.append(None)
                continue
            pg = np.stack([T[j] for j in fut])
            path = np.vstack([[0.0, 0.0], (pg - T[i]) @ R[i]])  # R^T (p - t) = (p - t) @ R
            v0 = float(np.linalg.norm(T[i] - T[prev[i]]) / max(dt[i], 1e-3)) if prev[i] >= 0 else float("nan")
            route = route_for(G[i], path)
            dp = densify(path, 1.0)
            ins = (np.abs(dp[:, 0]) <= HALF) & (np.abs(dp[:, 1]) <= HALF)
            # nur der zusammenhaengende Anfangsteil im Fenster
            k_out = np.argmax(~ins) if (~ins).any() else len(ins)
            dp = dp[:k_out]
            if not route or len(dp) < 3:
                res.append(None)
                continue
            n_ok += 1
            res.append(dict(route=route, path=dp.astype(np.float32), goal=tuple(dp[-1]), v0=v0, n_future=len(fut)))
            if i % 1000 == 0:
                print(part, i, N, flush=True)
        pickle.dump(res, open(f"{D}/route_gt_{part}.pkl", "wb"))
        L = [len(r["route"]) for r in res if r]
        PL = [len(r["path"]) for r in res if r]
        print(f"{part}: {N} Samples, Route vorhanden {n_ok} ({100.0 * n_ok / N:.1f} %), Stuecke je Route {np.mean(L):.2f}, Pfadlaenge {np.mean(PL):.1f} m", flush=True)


if __name__ == "__main__":
    main()
