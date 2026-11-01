"""Diagnosis of the post-check LOOK on b456 val (edges +26 %, route share -6 points): the same statistics per city, b123 next to b456.

    wsl bash tools/env/_py14.sh temp/loop19/diag_b456_by_city.py
"""
import os
import pickle

import numpy as np

H = os.path.expanduser("~")
LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]
for tag in ("b123", "b456"):
    for part in ("train", "val"):
        d = f"{H}/exp_gcar_{tag}"
        G = pickle.load(open(f"{d}/graph_gt_{part}.pkl", "rb"))
        RT = pickle.load(open(f"{d}/route_gt_{part}.pkl", "rb"))
        P = np.load(f"{d}/poses_{part}.npz")
        nxt = np.zeros(len(G), bool)
        nxt[P["prev"][P["prev"] >= 0]] = True  # frames that have a following frame (a route needs the future)
        for li, name in enumerate(LOCS):
            m = P["loc"] == li
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            route = np.mean([RT[i] is not None for i in idx])
            route_nxt = np.mean([RT[i] is not None for i in idx if nxt[i]])
            print(f"{tag} {part:5} {name:25} frames {m.sum():5}  pieces {np.mean([len(G[i]['pts']) for i in idx]):5.1f}  edges {np.mean([len(G[i]['edges']) for i in idx]):5.1f}"
                  f"  route {100 * route:5.1f} %  (frames with a next frame: {100 * route_nxt:5.1f} %)")
