"""Loop 4 / E44: Soll fuer die Abzweig-Schluesselpunkte (SAM-Road-artige Knoten-Heatmaps): je Sample 2 Kanaele (Split, Merge) mit Gauss (sigma 1,5 m) um die Soll-Knoten.
Split-Ort = Ende eines Stuecks mit >= 2 Nachfolgern, Merge-Ort = Anfang eines Stuecks mit >= 2 Vorgaengern (graph_eval.junctions, wie in der Auswertung).
Aufruf (gcar): make_node_gt.py  ->  ~/exp_gcar_b123/node_{train,val}.npy (uint8, [N,2,200,200], Zeile = x, Spalte = y wie die Heatmaps)"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402

HOME = os.path.expanduser("~")
SIG = 1.5 / 0.5  # Pixel
R = int(4 * SIG)


def render(nodes_list):
    out = np.zeros((2, 200, 200), np.float32)
    for c, P in enumerate(nodes_list):
        for x, y in P:
            r0, c0 = (x + 50.0) / 0.5 - 0.5, (y + 50.0) / 0.5 - 0.5
            if not (-R <= r0 < 200 + R and -R <= c0 < 200 + R):
                continue
            ri, ci = int(round(r0)), int(round(c0))
            rr = np.arange(max(ri - R, 0), min(ri + R + 1, 200))
            cc = np.arange(max(ci - R, 0), min(ci + R + 1, 200))
            if len(rr) == 0 or len(cc) == 0:
                continue
            g = np.exp(-(((rr[:, None] - r0) ** 2 + (cc[None, :] - c0) ** 2) / (2 * SIG ** 2)))
            out[c, rr[0] : rr[-1] + 1, cc[0] : cc[-1] + 1] = np.maximum(out[c, rr[0] : rr[-1] + 1, cc[0] : cc[-1] + 1], g)
    return out


for part in ("val", "train"):
    t0 = time.time()
    G = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_{part}.pkl", "rb"))
    arr = np.lib.format.open_memmap(f"{HOME}/exp_gcar_b123/node_{part}.npy", mode="w+", dtype=np.uint8, shape=(len(G), 2, 200, 200))
    cnt = np.zeros(2)
    for i, g in enumerate(G):
        sp, mg = GE.junctions(g)
        m = np.abs(np.concatenate([sp, mg]).reshape(-1, 2)).max(1) < 50 if len(sp) + len(mg) else None
        sp = sp[(np.abs(sp) < 50).all(1)] if len(sp) else sp
        mg = mg[(np.abs(mg) < 50).all(1)] if len(mg) else mg
        cnt += [len(sp), len(mg)]
        arr[i] = np.round(render([sp, mg]) * 255).astype(np.uint8)
    arr.flush()
    print(f"{part}: {len(G)} Samples, Knoten je Sample: Split {cnt[0] / len(G):.2f}, Merge {cnt[1] / len(G):.2f} ({time.time() - t0:.0f} s)", flush=True)
