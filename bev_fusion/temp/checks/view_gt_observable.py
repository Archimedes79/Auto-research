"""Observable GT (Mirko, 2026-10-26): pictures in which a wrong GT or a wrong projection has to show.

Per keyframe one figure, only camera matrices, the GT of training/gt_bev.py and the image:
1. the front image with the exact map vectors drawn as LINES (dividers yellow / orange, centre-lines cyan, drivable-area
   boundary green), clipped at the camera's near plane (no point clouds), and the boxes as 3D wireframes
2. the inverse view: the image projected onto the road plane z = 0 of the ego frame (inverse perspective mapping) at
   0.05 m, x 3-30 m, y -12..12 m, with the same vectors on top (no projection in this overlay: the vectors are already
   in the ego frame) and the 0.5 m GT cells of the dividers as squares - an offset between paint and lines reads
   directly in metres
3. a zoom of 2 on the near road (x 4-16 m)
The ego frame of the GT is the ego pose at the front camera's timestamp.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/view_gt_observable.py <out dir> [scene:k ...]
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
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from lift_pd import Grid  # noqa: E402

G = Grid()
COL = {"vehicle": (0.95, 0.2, 0.2), "pedestrian": (0.9, 0.3, 0.9), "two_wheeler": (1.0, 0.6, 0.1)}
LINE_COL = {"lane_divider": (1.0, 0.9, 0.1), "road_divider": (1.0, 0.45, 0.0), "centreline": (0.1, 0.9, 0.95)}


def to_cam(gt, xyz):
    T = np.linalg.inv(gt["meta"]["T_c2e"])
    return np.c_[xyz, np.ones(len(xyz))] @ T.T


def draw_polyline_3d(ax, gt, xyz, color, lw=1.2, near=0.5):
    """project a 3D polyline (ego frame), clipping each segment at the near plane z_cam = near."""
    K = gt["meta"]["K"]
    P = to_cam(gt, xyz)[:, :3]
    for a, b in zip(P[:-1], P[1:]):
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near or b[2] < near:                    # clip to the near plane
            t = (near - a[2]) / (b[2] - a[2])
            c = a + t * (b - a)
            a, b = (c, b) if a[2] < near else (a, c)
        uv = np.array([[K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]] for p in (a, b)])
        ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=lw)


def densify(pl, step=0.5):
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def drivable_boundaries(src, gt, sample_token):
    """exterior / interior rings of the drivable-area polygons near the car, in the ego frame."""
    nmap = src.map_for(src.nusc.get("sample", sample_token)["scene_token"])
    T_e2g = gt["meta"]["T_e2g"]
    T_g2e = np.linalg.inv(T_e2g)
    c = (T_e2g @ np.array([30, 0, 0, 1]))[:2]
    recs = nmap.get_records_in_patch((c[0] - 40, c[1] - 40, c[0] + 40, c[1] + 40), ["drivable_area"], mode="intersect")
    rings = []
    for tok in recs["drivable_area"]:
        for ptok in nmap.get("drivable_area", tok)["polygon_tokens"]:
            poly = nmap.extract_polygon(ptok)
            for ring in [poly.exterior] + list(poly.interiors):
                xy = np.asarray(ring.coords)[:, :2]
                rings.append((np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T_g2e.T)[:, :2])
    return rings


def ipm(gt, img, x_rng=(3.0, 30.0), y_rng=(-12.0, 12.0), res=0.05):
    """bird's-eye image of the road plane z = 0 (ego frame) sampled from the camera image (bilinear)."""
    xs = np.arange(x_rng[0], x_rng[1], res) + res / 2
    ys = np.arange(y_rng[0], y_rng[1], res) + res / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = to_cam(gt, np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))
    K = gt["meta"]["K"]
    z = P[:, 2]
    u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    H, W = img.shape[:2]
    ok = (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    out = np.zeros((X.size, 3), np.float32)
    u0, v0 = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    fu, fv = (u[ok] - u0)[:, None], (v[ok] - v0)[:, None]
    I = img.astype(np.float32) / 255
    out[ok] = (I[v0, u0] * (1 - fu) * (1 - fv) + I[v0, u0 + 1] * fu * (1 - fv) + I[v0 + 1, u0] * (1 - fu) * fv
               + I[v0 + 1, u0 + 1] * fu * fv)
    return out.reshape(X.shape + (3,)), (x_rng, y_rng)


def show_ipm(ax, bev, rng, gt, rings, x_zoom=None):
    (x0, x1), (y0, y1) = rng
    ax.imshow(np.flip(np.flip(bev, 0), 1), extent=[y1, y0, x0, x1])
    for layer in ("lane_divider", "road_divider", "centreline"):
        for pl in gt["meta"]["vectors"][layer]:
            ax.plot(pl[:, 1], pl[:, 0], color=LINE_COL[layer], linewidth=0.9 if layer == "centreline" else 1.3,
                    alpha=0.6 if layer == "centreline" else 1.0)
    for r in rings:
        ax.plot(r[:, 1], r[:, 0], color=(0.2, 1.0, 0.3), linewidth=1.0)
    gx, gy = cell_centres(G)
    m = ((gt["lane_divider"] > 0.5) | (gt["road_divider"] > 0.5)) & (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
    for cx, cy in zip(gx[m], gy[m]):
        ax.add_patch(plt.Rectangle((cy - 0.25, cx - 0.25), 0.5, 0.5, fill=False, edgecolor=(1, 1, 1), linewidth=0.4))
    for b in gt["meta"]["boxes"]:
        fp = np.vstack([b["corners"], b["corners"][:1]])
        ax.plot(fp[:, 1], fp[:, 0], color=COL[b["group"]], linewidth=1.2)
    ax.set_xlim(y1, y0)
    ax.set_ylim(x0, x1 if x_zoom is None else x_zoom)
    ax.set_xlabel("y (m, left +)", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color=(1, 1, 1), alpha=0.15, linewidth=0.5)


def figure(src, scene_name, k, out_dir, root):
    sc = next(s for s in src.nusc.scene if s["name"] == scene_name)
    tok = sc["first_sample_token"]
    for _ in range(k):
        tok = src.nusc.get("sample", tok)["next"]
    gt = src.keyframe_gt(tok, G)
    img = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB"))
    rings = drivable_boundaries(src, gt, tok)
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.05, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    for r in rings:
        rr = densify(r)
        rr = rr[(rr[:, 0] > 0) & (rr[:, 0] < 60)]
        if len(rr) > 1:
            draw_polyline_3d(ax, gt, np.c_[rr, np.zeros(len(rr))], (0.2, 1.0, 0.3), 1.0)
    for layer in ("centreline", "lane_divider", "road_divider"):
        for pl in gt["meta"]["vectors"][layer]:
            pl = densify(pl)
            pl = pl[(pl[:, 0] > 0) & (pl[:, 0] < 60)]
            if len(pl) > 1:
                draw_polyline_3d(ax, gt, np.c_[pl, np.zeros(len(pl))], LINE_COL[layer], 0.8 if layer == "centreline" else 1.6)
    for b in gt["meta"]["boxes"]:
        cx, cy, cz = b["center"]
        w, l, h = b["size"]
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        loc = [(sx * l / 2, sy * w / 2, sz * h / 2) for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        xyz = np.array([[cx + c * a - s * bb, cy + s * a + c * bb, cz + d] for a, bb, d in loc])
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            draw_polyline_3d(ax, gt, xyz[[i, j]], COL[b["group"]], 0.9)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"{scene_name} keyframe {k}: GT vectors (yellow lane / orange road divider, cyan centre-lines, green "
                 f"drivable boundary), boxes", fontsize=9)
    bev, rng = ipm(gt, img)
    show_ipm(fig.add_subplot(gs[1]), bev, rng, gt, rings)
    fig.axes[-1].set_title("road plane seen from above (camera image, 5 cm) + GT vectors, 0.5 m divider cells", fontsize=9)
    bev2, rng2 = ipm(gt, img, (4.0, 16.0), (-6.0, 6.0), 0.02)
    show_ipm(fig.add_subplot(gs[2]), bev2, rng2, gt, rings)
    fig.axes[-1].set_title("near road, 2 cm", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"gt_observable_{scene_name}_k{k:02d}.png")
    fig.savefig(p, dpi=80)
    plt.close(fig)
    return p


if __name__ == "__main__":
    out_dir = sys.argv[1]
    picks = [a.split(":") for a in sys.argv[2:]] or [("scene-0103", "12"), ("scene-0061", "8"), ("scene-0916", "8"),
                                                    ("scene-1077", "8"), ("scene-0655", "20"), ("scene-0796", "12")]
    root = os.path.expanduser("~/data/nuscenes")
    src = NuScenesGT(root, "v1.0-mini")
    for name, k in picks:
        print("saved", figure(src, name, int(k), out_dir, root))
