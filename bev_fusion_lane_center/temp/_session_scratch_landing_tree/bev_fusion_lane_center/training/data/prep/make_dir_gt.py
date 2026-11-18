"""Richtungs-Ziele fuer das Mehrfach-Ausgabe-Netz (Test H).

Fuer jedes Sample ein 2-Kanal-Raster (cos 2*theta, sin 2*theta) der Spurrichtung theta (Doppelwinkel, weil eine Spur keine
Vorzugsrichtung hat) im Lidar-System: jedes Pixel erhaelt die Richtung des naechsten Karten-Knotens, wenn dieser naeher als
2.5 m liegt (sonst 0). Winkel in der (Zeile=x, Spalte=y)-Konvention.
Aufruf: python make_dir_gt.py <feats-Ordner> <graph-Ordner>
"""
import os
import sys

import numpy as np
import torch

fdir, gdir = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2])
dev = "cuda"
cen = -50.0 + (torch.arange(200, device=dev).float() + 0.5) * 0.5
gx, gy = torch.meshgrid(cen, cen, indexing="ij")
pix = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)  # [40000,2] (x,y)

for split in ("train", "val"):
    g = np.load(f"{gdir}/{split}_graph.npz")
    nodes, tang, mask = [torch.from_numpy(g[k]).to(dev) for k in ("nodes", "tangent", "mask")]
    S = nodes.shape[0]
    out = np.zeros((S, 2, 200, 200), np.float16)
    for i in range(S):
        n = nodes[i][mask[i]]
        t = tang[i][mask[i]]
        d = torch.cdist(pix, n)  # [40000, M]
        dmin, j = d.min(1)
        tt = t[j]  # Tangente des naechsten Knotens [40000,2] (tx, ty)
        th = torch.atan2(tt[:, 1], tt[:, 0])
        c, s = torch.cos(2 * th), torch.sin(2 * th)
        ok = (dmin < 2.5).float()
        out[i, 0] = (c * ok).view(200, 200).cpu().numpy().astype(np.float16)
        out[i, 1] = (s * ok).view(200, 200).cpu().numpy().astype(np.float16)
    np.save(f"{fdir}/{split}_dir.npy", out)
    print(f"{split}: {S} Samples -> {fdir}/{split}_dir.npy", flush=True)
