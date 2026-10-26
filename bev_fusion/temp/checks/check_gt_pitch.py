"""Per-frame camera pitch correction: is the remaining offset a pitch effect or a single-line problem? (Mirko, 2026-10-26)

A pitch error of the camera against the road plane changes all projected ground lines together (distances stretch or
shrink, left and right lines fan in or out with range); it cannot move one line sideways while another one stays.
Per keyframe of nuScenes mini with dividers at 6-30 m:
1. search the pitch correction dp in -1.5..+1.5 deg (0.1 deg) that maximises the paint contrast of ALL divider lines
   (paint minus 0.4 m to either side, ego frame, bilinear image sampling)
2. at that pitch, per line element (>= 30 points at 6-30 m) its own lateral shift dy in -1..+1 m (0.1 m)
3. pictures before / after for chosen frames: the road-plane view (inverse perspective, 5 cm) with the vectors on top
Pitch here = a rotation of the camera about its own x axis (right), applied after the calibration.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_gt_pitch.py <out dir>
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402

G = Grid()
PITCH = np.round(np.arange(-1.5, 1.501, 0.1), 2)
DY = np.round(np.arange(-1.0, 1.001, 0.1), 2)
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")


def dense(polylines, step=0.05):
    out = []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 2))


def bilinear(img, uv):
    H, W = img.shape
    u, v = uv[:, 0], uv[:, 1]
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u0, v0 = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    fu, fv = u[ok] - u0, v[ok] - v0
    val = np.full(len(u), np.nan)
    val[ok] = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
               + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
    return val


def project_pitch(gt, xyz, dp_deg):
    T = np.linalg.inv(gt["meta"]["T_c2e"])
    P = (np.c_[xyz, np.ones(len(xyz))] @ T.T)[:, :3]
    a = math.radians(dp_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    P = P @ Rx.T
    K = gt["meta"]["K"]
    z = P[:, 2]
    uv = np.stack([K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1)
    return uv, z


def contrast(gt, img, xy, dp, dy=0.0):
    def b(d):
        uv, z = project_pitch(gt, np.c_[xy[:, 0], xy[:, 1] + d, np.zeros(len(xy))], dp)
        v = bilinear(img, uv)
        v[z <= 1] = np.nan
        return v
    return b(dy) - 0.5 * (b(dy - 0.4) + b(dy + 0.4))


def ipm(gt, img, dp, x_rng=(3.0, 30.0), y_rng=(-12.0, 12.0), res=0.05):
    xs = np.arange(*x_rng, res) + res / 2
    ys = np.arange(*y_rng, res) + res / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    uv, z = project_pitch(gt, np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1), dp)
    out = np.zeros((X.size, 3), np.float32)
    for c in range(3):
        v = bilinear(img[..., c].astype(np.float32) / 255, uv)
        v[z <= 0.5] = np.nan
        out[:, c] = np.nan_to_num(v)
    return out.reshape(X.shape + (3,))


def frames():
    for sc in src.nusc.scene:
        tok = sc["first_sample_token"]
        for k in range(40):
            s = src.nusc.get("sample", tok)
            if k % 4 == 0:
                yield sc["name"], k, tok
            if not s["next"]:
                break
            tok = s["next"]


if __name__ == "__main__":
    out_dir = sys.argv[1]
    show = {("scene-0103", 12), ("scene-0655", 20), ("scene-1077", 8)}
    rows = []
    for name, k, tok in frames():
        gt = src.keyframe_gt(tok, G)
        lines = [(lay, pl) for lay in ("lane_divider", "road_divider") for pl in gt["meta"]["vectors"][lay]]
        elems = []
        for lay, pl in lines:
            d = dense([pl])
            d = d[(d[:, 0] > 6) & (d[:, 0] < 30)] if len(d) else d
            if len(d) >= 30:
                elems.append((lay, d))
        if not elems:
            continue
        img_rgb = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB"))
        img = img_rgb.astype(np.float32).mean(2)
        allp = np.concatenate([e[1] for e in elems])
        sc_p = np.array([np.nanmean(contrast(gt, img, allp, dp)) for dp in PITCH])
        if not np.isfinite(sc_p).all():
            continue
        best = PITCH[int(np.argmax(sc_p))]
        res_dy = []
        for lay, d in elems:
            s = np.array([np.nanmean(contrast(gt, img, d, best, dy)) for dy in DY])
            res_dy.append((lay, DY[int(np.nanargmax(s))] if np.isfinite(s).any() else np.nan, len(d)))
        rows.append((name, k, best, sc_p.max() - sc_p[list(PITCH).index(0.0)], res_dy))
        print(f"{name} kf {k:2d}: pitch {best:+.1f} deg (gain {rows[-1][3]:5.1f}) | per line dy after pitch: "
              + ", ".join(f"{l[:4]} {dy:+.1f} m" for l, dy, n in res_dy))
        if (name, k) in show:
            fig, ax = plt.subplots(1, 2, figsize=(13, 7))
            for a, dp, t in ((ax[0], 0.0, "calibration as given"), (ax[1], best, f"pitch corrected {best:+.1f} deg")):
                a.imshow(np.flip(np.flip(ipm(gt, img_rgb, dp), 0), 1), extent=[12, -12, 3, 30])
                for lay, pl in lines:
                    a.plot(pl[:, 1], pl[:, 0], color=(1, 0.9, 0.1) if lay == "lane_divider" else (1, 0.45, 0), linewidth=1.4)
                a.set_xlim(12, -12)
                a.set_ylim(3, 30)
                a.set_title(f"{name} kf {k}: {t}", fontsize=10)
                a.set_xlabel("y (m, left +)")
                a.set_ylabel("x forward (m)")
            fig.tight_layout()
            p = os.path.join(out_dir, f"gt_pitch_{name}_k{k:02d}.png")
            fig.savefig(p, dpi=80)
            plt.close(fig)
            print("saved", p)
    P = np.array([r[2] for r in rows])
    print(f"\nframes {len(rows)}; best pitch correction: median {np.median(P):+.2f} deg, IQR "
          f"{np.percentile(P, 25):+.2f}..{np.percentile(P, 75):+.2f} deg, |pitch| <= 0.3 deg in {np.mean(np.abs(P) <= 0.3):.0%}")
    dys = [(lay, dy) for r in rows for lay, dy, n in r[4] if np.isfinite(dy)]
    for lay in ("lane_divider", "road_divider"):
        v = np.array([dy for l, dy in dys if l == lay])
        if len(v):
            print(f"{lay}: line elements {len(v)}, residual |dy| <= 0.2 m in {np.mean(np.abs(v) <= 0.2):.0%}, "
                  f"<= 0.4 m in {np.mean(np.abs(v) <= 0.4):.0%}, median dy {np.median(v):+.2f} m")
