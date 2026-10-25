"""Orientierung und Rahmen der GaussianCaR-Karten-Vorhersage gegen unser Soll pruefen (Mini-Validierung) und ein Bild schreiben.
Aufruf: gcar_orient.py   (liest ~/work/gcar_probe/probe_mini_val.npz aus 'gcar_tools.py probe' und ~/exp_crn_mini/feats/val_gt_raster.npy; schreibt out/crn/viz_gcar_orientierung.png)"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcar_tools import D4, D4_NAME  # noqa: E402

HOME = os.path.expanduser("~")
z = np.load(f"{HOME}/work/gcar_probe/probe_mini_val.npz")
P = z["probs"].astype(np.float32)  # [N,7,200,200]
n = len(P)
G = np.asarray(np.load(f"{HOME}/exp_crn_mini/feats/val_gt_raster.npy", mmap_mode="r")[:n]).astype(bool)
names = ["lane", "road_segment", "road_divider", "lane_divider", "ped_crossing", "walkway", "carpark_area"]
Gd = np.stack([ndi.binary_dilation(g, iterations=2) for g in G])
print(f"{n} Samples; Soll-Pixel je Sample {G.sum() / n:.0f}")
print("Orientierung (Zeile: Schicht; Spalten: " + " | ".join(D4_NAME) + ")")
tab = {}
for li, nm in ((0, "lane"), (1, "road_segment"), (3, "lane_divider"), (2, "road_divider")):
    mean_gt, frac_in = [], []
    for f in D4:
        p = np.ascontiguousarray(f(P[:, li]))
        mean_gt.append(float((p * G).sum() / G.sum()))
        frac_in.append(float(((p > 0.5) & G).sum() / G.sum()))
    tab[nm] = (mean_gt, frac_in)
    print(f"{nm:13s} mittlere Wahrsch. auf Soll:", " ".join(f"{x:.3f}" for x in mean_gt), "| Anteil Soll in Schicht>0,5:", " ".join(f"{x:.3f}" for x in frac_in))
# beste Orientierung nach lane + road_segment
score = np.array(tab["lane"][0]) + np.array(tab["road_segment"][0])
best = int(np.argmax(score))
print("beste Orientierung:", best, D4_NAME[best])
# Verschiebung: Kreuzkorrelation der Schicht 'lane' (>0,5) mit dem Soll, in +-6 Pixeln
pb = np.ascontiguousarray(D4[best](P[:, 0])) > 0.5
res = {}
for dr in range(-6, 7):
    for dc in range(-6, 7):
        sh = np.roll(np.roll(pb, dr, axis=1), dc, axis=2)
        res[(dr, dc)] = float((sh & G).sum() / G.sum())
bs = max(res, key=res.get)
print(f"beste Verschiebung (Zeile, Spalte) in Pixeln: {bs} (Anteil {res[bs]:.3f}) gegen 0,0: {res[(0, 0)]:.3f}")
fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
for k, i in enumerate([0, 2, 4, 6][: min(4, n)]):
    for r, (li, ttl) in enumerate(((0, "lane"), (3, "lane_divider"))):
        ax = axs[r, k]
        p = np.ascontiguousarray(D4[best](P[i : i + 1, li]))[0]
        ax.imshow(p, cmap="gray_r", vmin=0, vmax=1)
        yy, xx = np.nonzero(G[i])
        ax.scatter(xx, yy, s=1.0, c="#d93025")
        ax.set_title(f"Mini-Val {i}: Schicht {ttl} (Orientierung {best}), Soll rot", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle(f"GaussianCaR (Karten-Gewichte), BEV-Rahmen = Hinterachse zum Zeitpunkt von CAM_FRONT: Vorhersage gegen unsere Soll-Mittellinien; Orientierung '{D4_NAME[best]}'", fontsize=10)
fig.tight_layout()
fig.savefig("/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn/viz_gcar_orientierung.png", dpi=85)
print("Bild geschrieben")
