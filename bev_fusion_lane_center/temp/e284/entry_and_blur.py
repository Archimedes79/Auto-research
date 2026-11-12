"""scratch (Mirko's marks of 2026-11-08 on the E279 output p: lower circle = the lines at the junction entry are all distorted to the left; upper circle = the structure inside the junction is blurred).
My earlier lateral_offset.py dropped the points that have a second piece within 3 m - exactly the lane ends where the connectors start. This script measures the entry again:
 1. entry: LANE points (not connector segments) by their distance to the nearest connector point inside an intersection (0-3, 3-6, 6-10, 10-15, 15-25 m, none): the shift of the map's cross-lane centroid against the target's
    (+ = left of the travel direction; the same profile window +-1.7 m on p and on T, so the pull of neighbouring lines cancels), the share with |shift| > 0.5 m, the world-frame shift (dx, dy);
 2. blur: cells inside the junction (< 8 m) against cells outside: correlation of p and T, the gradient magnitude of p relative to T, and the share of mid values (0.2 < p < 0.8) among the cells with p > 0.2;
 3. the picture of the frame (zoom, x -12 .. 52, y -12 .. 28): p with the GT lane lines and the T = 0.5 contour, and T.
Reads temp/e284/target_vs_pred.npz. CPU.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/entry_and_blur.py"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

TOK = "5dd954d0ac0242a990ba35eef950c937"
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
P = z["p"].astype(np.float32).reshape(-1, 100, 100)
T = z["t"].astype(np.float32).reshape(-1, 100, 100)
DJ = z["dj"].astype(np.float32).reshape(-1, 100, 100)
by_tok = {f.token: f for f in QE.val_frames()}
S = np.arange(-1.7, 1.71, 0.17)
BINS = (("0-3 m", 0, 3), ("3-6 m", 3, 6), ("6-10 m", 6, 10), ("10-15 m", 10, 15), ("15-25 m", 15, 25), ("25 m and more / none", 25, 1e12))
n_stat = len(toks) - (1 if len(toks) > 88 else 0)


def profile(M, xy):
    v = map_coordinates(M, [(xy[..., 0] + 49.5).ravel(), (xy[..., 1] + 49.5).ravel()], order=1, mode="nearest").reshape(xy.shape[:2])
    w = np.clip(v, 0, None)
    sw = w.sum(1) + 1e-9
    return (w * S[None]).sum(1) / sw, v.max(1)


rows = []
frame_pts = None
for i, tk in enumerate(toks):
    f = by_tok[tk]
    g = f.lane()
    nt = f.ref().nt
    sp = nt["pts"][nt["supp"]]
    jt = cKDTree(sp) if len(sp) else None
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        ok = (~conn) & (np.abs(q) < 46.0).all(1)
        if not ok.any():
            continue
        q, tan = q[ok], tan[ok]
        dj = jt.query(q)[0] if jt is not None else np.full(len(q), 1e9)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        xy = q[:, None, :] + S[None, :, None] * nrm[:, None, :]
        cp, mp = profile(P[i], xy)
        ct, mt = profile(T[i], xy)
        for k in range(len(q)):
            rows.append((dj[k], cp[k] - ct[k], mp[k], mt[k], (cp[k] - ct[k]) * nrm[k, 0], (cp[k] - ct[k]) * nrm[k, 1], q[k, 0], q[k, 1], float(i)))
    f._lane = f._ref = None
R = np.array(rows)
dj, sh, mp, mt, dx, dy, px, py, fi = R.T
m_all = (mt > 0.5)
entry = {}
for name, a, b in BINS:
    m = m_all & (dj >= a) & (dj < b) & (fi < n_stat)
    if m.sum() < 30:
        continue
    entry[name] = dict(points=int(m.sum()), shift_left_median=float(np.median(sh[m])), shift_left_mean=float(sh[m].mean()), abs_median=float(np.median(np.abs(sh[m]))), abs_p90=float(np.percentile(np.abs(sh[m]), 90)),
                       share_abs_over_0_5=float((np.abs(sh[m]) > 0.5).mean()), share_left_over_0_3=float((sh[m] > 0.3).mean()), share_right_over_0_3=float((sh[m] < -0.3).mean()), dx_mean=float(dx[m].mean()), dy_mean=float(dy[m].mean()),
                       peak_p=float(np.median(mp[m])))
# the circled region of the picture frame: lane points with x in [-5, 25], y in [-6, 3]
fi_pic = toks.index(TOK)
mc = m_all & (fi == fi_pic) & (px > -5) & (px < 25) & (py > -6) & (py < 3)
circle = dict(points=int(mc.sum()), shift_left_median=float(np.median(sh[mc])) if mc.any() else None, dx_mean=float(dx[mc].mean()) if mc.any() else None, dy_mean=float(dy[mc].mean()) if mc.any() else None,
              abs_median=float(np.median(np.abs(sh[mc]))) if mc.any() else None)

# ---- blur: cells in the junction against outside
inb = np.zeros((100, 100), bool)
inb[2:98, 2:98] = True
acc = {k: dict(n=0, sp=0.0, st=0.0, spp=0.0, stt=0.0, spt=0.0, gp=0.0, gt=0.0, mid=0, high=0, midT=0, highT=0) for k in ("junction", "outside")}
for i in range(n_stat):
    p, t, dj_c = P[i], T[i], DJ[i]
    gp = np.hypot(*np.gradient(p))
    gt = np.hypot(*np.gradient(t))
    rel = (p > 0.05) | (t > 0.05)
    for k, m in (("junction", (dj_c < 8.0) & rel & inb), ("outside", (dj_c >= 8.0) & rel & inb)):
        a = acc[k]
        n = float(m.sum())
        a["n"] += n
        a["sp"] += float(p[m].sum())
        a["st"] += float(t[m].sum())
        a["spp"] += float((p[m] ** 2).sum())
        a["stt"] += float((t[m] ** 2).sum())
        a["spt"] += float((p[m] * t[m]).sum())
        a["gp"] += float(gp[m].sum())
        a["gt"] += float(gt[m].sum())
        a["mid"] += int(((p > 0.2) & (p < 0.8) & m).sum())
        a["high"] += int(((p > 0.2) & m).sum())
        a["midT"] += int(((t > 0.2) & (t < 0.8) & m).sum())
        a["highT"] += int(((t > 0.2) & m).sum())
blur = {}
for k, a in acc.items():
    n = a["n"]
    cov = a["spt"] / n - a["sp"] / n * a["st"] / n
    vp, vt = a["spp"] / n - (a["sp"] / n) ** 2, a["stt"] / n - (a["st"] / n) ** 2
    blur[k] = dict(cells=n, corr_p_T=cov / np.sqrt(vp * vt), gradient_p=a["gp"] / n, gradient_T=a["gt"] / n, gradient_ratio=a["gp"] / a["gt"], mid_share_p=a["mid"] / max(a["high"], 1), mid_share_T=a["midT"] / max(a["highT"], 1))
# the picture frame, the upper circle (x 0 .. 45, y 2 .. 15)
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
zone = (CX > 0) & (CX < 45) & (CY > 2) & (CY < 15)
p, t = P[fi_pic], T[fi_pic]
gp, gt = np.hypot(*np.gradient(p)), np.hypot(*np.gradient(t))
blur["picture frame, upper circle"] = dict(cells=int(zone.sum()), corr_p_T=float(np.corrcoef(p[zone], t[zone])[0, 1]), gradient_ratio=float(gp[zone].sum() / gt[zone].sum()), mean_p=float(p[zone].mean()), mean_T=float(t[zone].mean()))
json.dump(dict(frames=n_stat, entry=entry, circle_picture_frame=circle, blur=blur), open(os.path.join(ROOT, "temp", "e284", "entry_and_blur.json"), "w"), indent=1)
print("LANE points at the junction entry (shift of p against T across the lane; + = left of the travel direction; world frame dx, dy in m)")
for k, v in entry.items():
    print(f"  {k:22s} n {v['points']:6d}  median {v['shift_left_median']:+.2f}  mean {v['shift_left_mean']:+.2f}  |median| {v['abs_median']:.2f}  p90 {v['abs_p90']:.2f}  >0.5 m: {100 * v['share_abs_over_0_5']:.0f} %  left > 0.3: {100 * v['share_left_over_0_3']:.0f} %  right > 0.3: {100 * v['share_right_over_0_3']:.0f} %  (dx {v['dx_mean']:+.2f}, dy {v['dy_mean']:+.2f})  peak {v['peak_p']:.2f}")
print("picture frame, circled lower region:", json.dumps(circle))
print("BLUR")
for k, v in blur.items():
    print(" ", k, json.dumps({a: round(b, 3) for a, b in v.items()}))

# ---- the zoom picture
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

f = by_tok[TOK]
g = f.lane()
fig, ax = plt.subplots(1, 2, figsize=(22, 9), dpi=60)
for a, M, ttl in ((ax[0], p, "E279 output p, GT lane lines (red = lane, orange = connector), blue: T = 0.5"), (ax[1], t, "target T, the same GT lines")):
    a.imshow(M.T, origin="lower", extent=(-50, 50, -50, 50), cmap="gray_r", vmin=0, vmax=1)
    a.set_xlim(-12, 52)
    a.set_ylim(-12, 28)
    a.set_aspect("equal")
    a.set_title(ttl)
    a.set_xlabel("x right (m)")
    a.set_ylabel("y forward (m)")
    a.contour(CX, CY, t, levels=[0.5], colors="tab:blue", linewidths=0.8)
    for pc in g["pieces"]:
        q = np.asarray(pc["pts"], float)
        for s in pc["segments"]:
            seg = q[s["i0"]: s["i1"] + 1]
            a.plot(seg[:, 0], seg[:, 1], "-", color="tab:orange" if s["kind"] == "connector" else "tab:red", lw=1.0)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "temp", "e284", "entry_zoom.png"), dpi=60)
print("written entry_zoom.png")
