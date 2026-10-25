"""Loop 3, E41 Variante B: Weitergabe der Anfragen ('query propagation') am Verfeinerer.
Die verfeinerten Linien des Vorgaenger-Bildes (Graph nach Schnitt/Orientierung/Verbindung) werden per Eigenbewegung ins aktuelle Bild gefaltet und als ZUSAETZLICHE Vorschlaege zu den Gitter-Vorschlaegen gegeben;
der Verfeinerer (gleiche Gewichte, ohne Neutraining) zieht sie auf die Merkmale des aktuellen Bildes und bewertet ihre Existenz; Doppelte werden unterdrueckt. Bilder werden in Wellen (k-tes Bild jeder Szene) nacheinander abgearbeitet.
Aufruf (gcar): temporal_b.py --refiner e39_main --run e37_final --set val --part val [--tau_e 0.3] [--tag relax] [--name e41_B]
Ausgabe: out/loop2/<name>/tb_<run>_<menge>.json  (Kennzahlen: ohne / mit Weitergabe, Stabilitaet)"""
import argparse, json, os, pickle, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refiner as R
import graph_eval as GE
import graph_from_dense as GD
import lf_inputs as LI
import stability as ST
from scipy.spatial import cKDTree

OUT = R.OUT


def warp_graph(g, Q, s, lim=49.0, spacing=1.0, nmax=100, min_pts=3):
    """Graph (Meter, Vorgaenger-System) -> Stuecke + Kanten im aktuellen System (p_cur = Q p_prev + s): auf |x|,|y| < lim beschneiden, neu abtasten, lange Stuecke teilen."""
    pts, first, last = [], {}, {}
    edges = []
    for i, p in enumerate(g["pts"]):
        w = np.asarray(p, np.float64) @ Q.T + s
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
                q = GE.resample(w[a:b], spacing)
                if len(q) >= 2:
                    parts = [q]
                    if len(q) > nmax:
                        k = int(np.ceil((len(q) - 1) / (nmax - 1)))
                        idx = np.linspace(0, len(q) - 1, k + 1).round().astype(int)
                        parts = [q[idx[c] : idx[c + 1] + 1] for c in range(k)]
                    ids = []
                    for part in parts:
                        ids.append(len(pts))
                        pts.append(part.astype(np.float32))
                    edges += [(ids[c], ids[c + 1]) for c in range(len(ids) - 1)]
                    first.setdefault(i, ids[0])
                    last[i] = ids[-1]
            a = b
    for i, j in g.get("edges", []):
        if i in last and j in first:
            edges.append((last[i], first[j]))
    return pts, sorted(set(edges))


class Sub(torch.utils.data.Dataset):
    """Teilmenge eines RData (globale Nummern bleiben erhalten)."""

    def __init__(self, ds, ns):
        self.ds, self.ns = ds, list(ns)

    def __len__(self):
        return len(self.ns)

    def __getitem__(self, k):
        return self.ds[self.ns[k]]

    def edges_of(self, n):
        return self.ds.edges_of(n)


def load_model(name, dev="cuda"):
    ck = torch.load(f"{OUT}/{name}/model.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    m = R.Refiner(39, 32, a["D"], a["blocks"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, a


def covered(q, tree, tol=1.0):
    """Anteil der Punkte des Stuecks q (m), die innerhalb tol eines Gitter-Vorschlagspunktes liegen."""
    return float((tree.query(GE.resample(q, 0.5))[0] <= tol).mean()) if tree is not None else 0.0


def run(model, mean, std, run_, st, part, tag, tau_e, propagate, workers, pose, pmax=96, label="", gate=0.0, tau_prop=0.0, max_rows=0):
    """gate > 0: weitergegebene Stuecke, die zu >= gate schon von Gitter-Vorschlaegen abgedeckt sind (<= 1 m), entfallen (nur Neues kommt hinzu); tau_prop > 0: eigene Existenz-Schwelle fuer weitergegebene Stuecke."""
    R.PMAX = pmax
    ds = R.RData([dict(run=run_, st=st, part=part)], tag, train=False)
    if max_rows:
        ds.items = ds.items[:max_rows]
    ids = ds.S[0]["ids"][: len(ds.items)]
    pos = {int(i): n for n, i in enumerate(ids)}
    base = [dict(pts=list(p["pts"]), edges=list(p["edges"])) for p in ds.S[0]["props"]]
    prev = [-1] * len(ids)
    for n, i in enumerate(ids):
        j = pose.prev(part, int(i))
        prev[n] = pos.get(j, -1) if j >= 0 else -1
    depth = [0] * len(ids)
    for n in range(len(ids)):  # Zeilen aufsteigend: Vorgaenger hat kleinere Nummer
        depth[n] = depth[prev[n]] + 1 if (propagate and prev[n] >= 0) else 0
    graphs, results = [None] * len(ids), [None] * len(ids)
    n_prop = []
    t0 = time.time()
    waves = sorted(set(depth))
    for w in waves:
        ns = [n for n in range(len(ids)) if depth[n] == w]
        for n in ns:
            pts, edges = list(base[n]["pts"]), list(base[n]["edges"])
            if propagate and prev[n] >= 0 and graphs[prev[n]] is not None:
                Q, s, _ = pose.rel(part, int(ids[n]))
                gp = graphs[prev[n]]
                ep, ee = warp_graph(gp, Q.reshape(2, 2), s)
                if gate > 0 and len(ep):
                    gp_pts = [np.asarray(q, np.float64) for q in base[n]["pts"] if len(q) >= 2]
                    tree = cKDTree(np.concatenate([GE.resample(q, 0.5) for q in gp_pts])) if gp_pts else None
                    keep = [k for k, q in enumerate(ep) if covered(q, tree) < gate]
                    remap = {o: i for i, o in enumerate(keep)}
                    ep = [ep[k] for k in keep]
                    ee = [(remap[a], remap[b]) for a, b in ee if a in remap and b in remap]
                off = len(pts)
                pts += ep
                edges += [(a + off, b + off) for a, b in ee]
                n_prop.append(len(ep))
            ds.S[0]["props"][n] = dict(pts=pts, edges=sorted(set(edges)))
        r = R.infer(model, Sub(ds, ns), mean, std, bs=16, workers=workers)
        for n, rr in zip(ns, r):
            if tau_prop > 0 and propagate:
                order = ds.order_of(n)
                ng = len(base[n]["pts"])
                rr["ex"] = [e * (tau_e / tau_prop) if q >= ng else e for e, q in zip(rr["ex"], order)]
            results[n] = rr
            graphs[n] = R.graph_from_refined(rr, tau_e=tau_e)
    print(f"{label} Wellen {len(waves)} ({time.time() - t0:.0f} s), mittlere weitergegebene Stuecke je Bild {np.mean(n_prop) if n_prop else 0:.1f}", flush=True)
    return graphs, results, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="e39_main")
    ap.add_argument("--run", default="e37_final")
    ap.add_argument("--set", default="val")
    ap.add_argument("--part", default="val")
    ap.add_argument("--tag", default="relax")
    ap.add_argument("--tau_e", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name", default="e41_B")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--variants", default="plain:0:0,gate:0.6:0,gate_tau:0.6:0.5", help="Name:gate:tau_prop[,..] fuer die Weitergabe; 'single' (ohne Weitergabe) wird immer zuerst gerechnet")
    ap.add_argument("--max_rows", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.name}", exist_ok=True)
    model, ma = load_model(a.refiner)
    mean, std = R.channel_stats()
    mean, std = mean.cuda(), std.cuda()
    pose = LI.PoseTable("b123", a.gap)
    res = dict(args=vars(a), refiner_args=ma)
    gts = None
    todo = [("single", False, 0.0, 0.0)] + [(v.split(":")[0], True, float(v.split(":")[1]), float(v.split(":")[2])) for v in a.variants.split(",") if v]
    for key, propagate, gate, tau_prop in todo:
        graphs, rr, ids = run(model, mean, std, a.run, a.set, a.part, a.tag, a.tau_e, propagate, a.workers, pose, label=key, gate=gate, tau_prop=tau_prop, max_rows=a.max_rows)
        if gts is None:
            gts = GD.load_gt_graphs(a.part, ids)
        m = GD.evaluate_parallel(graphs, gts, 6, True)
        st = ST.stability(graphs, gts, ids, a.part, a.gap, pose)
        res[key] = dict(metrics=m, stability=st, gate=gate, tau_prop=tau_prop)
        pickle.dump(dict(graphs=graphs, ids=ids), open(f"{OUT}/{a.name}/graphs_{key}_{a.run}_{a.set}.pkl", "wb"))
        print(f"[{key}] F1@0.5/1/1.5 {m['F1@0.5']:.3f}/{m['F1@1.0']:.3f}/{m['F1@1.5']:.3f} P {m['P@1.0']:.3f} R {m['R@1.0']:.3f} route F {m['route_f']:.3f} split6/merge6 {m.get('split6_F1', float('nan')):.3f}/{m.get('merge6_F1', float('nan')):.3f} | Stabilitaet {st['pred']['F1']:.3f} (GT-Referenz {st['gt_reference']['F1']:.3f}, Paare {st['pairs']})", flush=True)
    json.dump(res, open(f"{OUT}/{a.name}/tb_{a.run}_{a.set}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
