"""scratch (Mirko's lower circle of 2026-11-08: 'all lines are distorted to the left'): the picture frame only. For horizontal strips of the circled region (x -6 .. 26 m) the centroid in x of p and of T and the mass ratio, row by row (1 m rows,
y = -8 .. 4), and the same for the whole circle. A centroid of p left of the centroid of T = p sits to the left of the target.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/circle_check.py"""
import os

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
i = toks.index("5dd954d0ac0242a990ba35eef950c937")
p, t = z["p"].astype(np.float32)[i].reshape(100, 100), z["t"].astype(np.float32)[i].reshape(100, 100)
c = -50.0 + (np.arange(100) + 0.5)
X = c[:, None] * np.ones((1, 100))
Y = np.ones((100, 1)) * c[None, :]
print("row y   centroid x of p   of T   difference (p - T)   mass p / mass T")
for y0 in range(-8, 4):
    m = (X >= -6) & (X <= 26) & (Y >= y0) & (Y < y0 + 1)
    cp, ct = (p[m] * X[m]).sum() / p[m].sum(), (t[m] * X[m]).sum() / t[m].sum()
    print(f"{y0:+3d} .. {y0 + 1:+3d}   {cp:7.2f}   {ct:7.2f}   {cp - ct:+6.2f}   {p[m].sum() / t[m].sum():5.2f}")
m = (X >= -5) & (X <= 25) & (Y >= -5) & (Y <= 2)
print(f"whole lower circle (x -5..25, y -5..2): centroid x p {(p[m] * X[m]).sum() / p[m].sum():.2f}  T {(t[m] * X[m]).sum() / t[m].sum():.2f}  mass p / T {p[m].sum() / t[m].sum():.2f}")
# also the heading channel is not used here; the offset channels neither
