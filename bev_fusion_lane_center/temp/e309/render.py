"""E309 scratch: render chosen frames: front camera + delivered graph of B, B + E284, B + E287 p2 against the l6 lane GT with direction arrows.
usage: wsl bash tools/env/_py14.sh render.py <seed> <out dir> <token|row> [<token|row> ...]  or  --pick N   (N frames spread over the per-frame topo F1 of B, plus the junction scene 11)"""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = os.environ["BFLC_ROOT"]
for sub in ("tools/viz", "evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_view as QV  # noqa: E402

seed = int(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", f"perframe_s{seed}.pkl"), "rb"))
BY = {r["token"]: r for r in PF}


def resample(p, ds=0.5):
    p = np.asarray(p, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if len(p) < 2 or seg[-1] < 1e-6:
        return p
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


def panel(ax, rec, key, title):
    ax.set_xlim(-48, 48)
    ax.set_ylim(-48, 48)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    gts = [resample(p) for p in rec["gt_pieces"] if len(p) > 1]
    G = np.vstack(gts) if gts else np.zeros((0, 2))
    lines = [resample(l) for l in rec[key]["lines"] if len(l) > 1]
    P = np.vstack(lines) if lines else np.zeros((0, 2))
    tg = cKDTree(G) if len(G) else None
    tp = cKDTree(P) if len(P) else None
    for q in gts:                                                          # GT: grey, with the travel direction
        ax.plot(q[:, 0], q[:, 1], color="#b0b0b0", lw=3.5, solid_capstyle="round", zorder=1)
        for m in range(10, len(q) - 1, 20):                               # an arrow every 10 m
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0), zorder=2)
    if tp is not None and len(G):                                         # GT not covered within 2 m: blue
        miss = tp.query(G)[0] > 2.0
        ax.scatter(G[miss, 0], G[miss, 1], s=4, color="#1f4fff", zorder=3)
    for q in lines:                                                       # delivered: green <= 1 m, orange <= 2 m, red beyond
        d = tg.query(q)[0] if tg is not None else np.full(len(q), 9.0)
        col = np.where(d <= 1.0, 0, np.where(d <= 2.0, 1, 2))
        for c, cc in enumerate(("#1a9a2a", "#f0a000", "#e02020")):
            m = col == c
            if m.any():
                ax.scatter(q[m, 0], q[m, 1], s=3, color=cc, zorder=4)
    for q in lines:
        for m in range(15, len(q) - 1, 30):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#0b4d12", lw=1.0), zorder=5)
    ax.plot([0], [0], marker="^", color="k", ms=7, zorder=6)


def render(rec):
    fig = plt.figure(figsize=(21, 7.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    try:
        g = QV.load_gt(rec["token"])
        t = QV.camera_tiles(rec["token"], g, sizes={"CAM_FRONT": (640, 400)}, log=lambda *a: None) if False else None
    except Exception:
        t = None
    try:
        import cv2
        g = QV.load_gt(rec["token"])
        tiles = QV.camera_tiles(rec["token"], g, {c: (640, 360) for c in QV.CAMS}, log=lambda *a: None)
        ax.imshow(tiles["CAM_FRONT"])
    except Exception as ex:
        ax.text(0.5, 0.5, f"no camera tile\n{type(ex).__name__}", ha="center")
    for k, (key, name) in enumerate((("B", "B (E273 + hard step) = the plain graph"), ("e312", "E312 final system (E307 maps + pair head + snapped p>=0.1)"), ("e287p2", "B + E287 p2 pair head (old)"))):
        a = fig.add_subplot(gs[k + 1])
        r = rec[key]
        panel(a, rec, key, f"{name}\ntopo F1 {r['num']['topo_F1']:.2f}  F1@1 {r['num']['F1@1.0']:.2f}  lanes found {r['num']['lanes_found']:.2f}  junc F1 {r['num']['junction_F1']:.2f}\n"
              f"lenient 2 m: GT covered {r['len']['cov']:.2f}, delivered right {r['len']['prec']:.2f}")
        if k:
            a.set_yticklabels([])
    fig.suptitle(f"{rec['token'][:8]}  scene {rec['scene']}  frame row {rec['row']}  | grey+arrows = l6 GT (travel direction), delivered points: green <= 1 m, orange <= 2 m, red > 2 m from GT; blue = GT not covered within 2 m", fontsize=10)
    p = os.path.join(out_dir, f"f{rec['row']:04d}_{rec['token'][:8]}.png")
    fig.savefig(p, dpi=70, bbox_inches="tight")
    plt.close(fig)
    return p


toks = []
if sys.argv[3] == "--pick":
    n = int(sys.argv[4])
    order = np.argsort([r["B"]["num"]["topo_F1"] for r in PF])
    idx = [int(order[int(i)]) for i in np.linspace(0, len(order) - 1, n)]
    toks = [PF[i]["token"] for i in idx]
else:
    for a in sys.argv[3:]:
        toks.append(PF[int(a)]["token"] if a.isdigit() and len(a) < 5 else a)
for t in toks:
    print(render(BY[t]), flush=True)
