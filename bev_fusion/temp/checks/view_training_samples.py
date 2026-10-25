"""Pictures of training samples: keyframe k with the GT of k-1, k, k+1 (training/gt_bev.py) - "show me your training
data" (Mirko, 2026-10-25).

Per sample one figure:
- the front-camera image at k with the GT projected into it (drivable area green, dividers yellow, centre-lines cyan,
  boxes as 3D wireframes: vehicles red, pedestrians magenta, two-wheelers orange) - if lines miss the painted
  markings, the GT or a transform is wrong
- the BEV GT at k-1, k, k+1, each in the ego frame at its own time (the target frame of the splat), with velocity
  arrows on the boxes at k, the area outside the camera and radar coverage darkened, occluded cells hatched dark blue
- a text strip: times, dt, the ego steps between the keyframes (v, omega per 12 Hz frame)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/view_training_samples.py <out dir>
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
COL = {"vehicle": (0.9, 0.15, 0.15), "pedestrian": (0.85, 0.2, 0.85), "two_wheeler": (1.0, 0.6, 0.1)}


def bev_rgb(gt):
    img = np.zeros((G.nx, G.ny, 3), np.float32)
    img += 0.10
    img = img * (1 - gt["drivable_area"][..., None]) + np.array([0.30, 0.45, 0.30]) * gt["drivable_area"][..., None]
    img = img * (1 - 0.6 * gt["walkway"][..., None]) + 0.6 * np.array([0.35, 0.30, 0.45]) * gt["walkway"][..., None]
    img = img * (1 - gt["ped_crossing"][..., None]) + np.array([0.75, 0.75, 0.75]) * gt["ped_crossing"][..., None]
    img = img * (1 - gt["carpark_area"][..., None] * 0.5) + 0.5 * np.array([0.3, 0.3, 0.5]) * gt["carpark_area"][..., None]
    for layer, c in (("lane_divider", (1.0, 0.85, 0.1)), ("road_divider", (1.0, 0.5, 0.0)), ("centreline", (0.1, 0.9, 0.9))):
        m = gt[layer][..., None]
        img = img * (1 - m) + np.array(c) * m
    for grp, c in COL.items():
        m = gt[grp][..., None]
        img = img * (1 - m) + np.array(c) * m
    occ = gt["occluded"][..., None]
    img = img * (1 - 0.5 * occ) + 0.5 * occ * np.array([0.1, 0.1, 0.5])
    und = gt["undefined"][..., None]
    img = img * (1 - 0.7 * und)
    return np.clip(img, 0, 1)


def show_bev(ax, gt, title, arrows=False):
    ax.imshow(np.flip(np.flip(bev_rgb(gt), 0), 1), extent=[26, -26, 0, 60])
    cam = gt["camera"]
    ax.contour(np.linspace(-26 + 0.25, 26 - 0.25, G.ny)[::-1], np.linspace(0.25, 59.75, G.nx),
               np.flip(cam, 1), levels=[0.5], colors="white", linewidths=0.6)
    if arrows:
        for b in gt["meta"]["boxes"]:
            cx, cy = b["center"][0], b["center"][1]
            if 0 < cx < 60 and -26 < cy < 26 and np.hypot(*b["vel"]) > 0.5:
                ax.arrow(cy, cx, b["vel"][1] * 0.5, b["vel"][0] * 0.5, color="w", width=0.15, head_width=0.8,
                         length_includes_head=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(26, -26)
    ax.set_ylim(0, 60)
    ax.set_xlabel("y (m, left +)", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def project(gt, xyz):
    T = np.linalg.inv(gt["meta"]["T_c2e"])
    P = np.c_[xyz, np.ones(len(xyz))] @ T.T
    z = P[:, 2]
    K = gt["meta"]["K"]
    uv = np.stack([K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1)
    return uv, z


def show_camera(ax, gt, root):
    img = Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB")
    ax.imshow(img)
    gx, gy = cell_centres(G)
    for layer, c, a, sz in (("drivable_area", (0.2, 1.0, 0.2), 0.12, 6), ("lane_divider", (1, 0.9, 0.1), 0.95, 14),
                            ("road_divider", (1, 0.5, 0), 0.95, 14), ("centreline", (0.1, 0.9, 0.9), 0.8, 8)):
        m = gt[layer] > 0.5
        if layer == "drivable_area":
            m &= (gx < 40)
        xyz = np.stack([gx[m], gy[m], np.zeros(m.sum())], 1)
        uv, z = project(gt, xyz)
        ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < img.width) & (uv[:, 1] >= 0) & (uv[:, 1] < img.height)
        ax.scatter(uv[ok, 0], uv[ok, 1], s=sz, color=c, alpha=a, linewidths=0)
    for b in gt["meta"]["boxes"]:
        cx, cy, cz = b["center"]
        if cx < 1:
            continue
        w, l, h = b["size"]
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        xyz = np.stack([cx + c * loc[:, 0] - s * loc[:, 1], cy + s * loc[:, 0] + c * loc[:, 1], cz + loc[:, 2]], 1)
        uv, z = project(gt, xyz)
        if (z <= 1).any():
            continue
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=COL[b["group"]], linewidth=0.8)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis("off")
    ax.set_title("front camera at keyframe k, GT projected", fontsize=9)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    root = os.path.expanduser("~/data/nuscenes")
    src = NuScenesGT(root, "v1.0-mini")
    picks = []
    for name in ("scene-0061", "scene-0103", "scene-1077", "scene-0916"):
        sc = next(s for s in src.nusc.scene if s["name"] == name)
        tok = sc["first_sample_token"]
        for _ in range(8):                                   # a keyframe a few seconds into the scene
            tok = src.nusc.get("sample", tok)["next"]
        picks.append((name, sc["description"], tok))
    for name, desc, tok in picks:
        ts = src.training_sample(tok, G)
        fig = plt.figure(figsize=(19, 6.2))
        gs = fig.add_gridspec(2, 5, height_ratios=[10, 1.6], width_ratios=[2.2, 1, 1, 1, 0.05])
        show_camera(fig.add_subplot(gs[0, 0]), ts["cur"], root)
        for c, key, lab in ((1, "prev", "k-1"), (2, "cur", "k"), (3, "next", "k+1")):
            dt = 0.0 if key == "cur" else ts[key]["meta"]["dt"]
            show_bev(fig.add_subplot(gs[0, c]), ts[key], f"GT at keyframe {lab} (dt = {dt:+.2f} s), own ego frame",
                     arrows=(key == "cur"))
        tx = fig.add_subplot(gs[1, :4])
        tx.axis("off")
        lines = []
        for key in ("prev", "next"):
            st = ts[key]["meta"]["steps"]
            v = [s[1] for s in st]
            w = [s[2] for s in st]
            lines.append(f"{key}: dt {ts[key]['meta']['dt']:+.3f} s in {len(st)} camera steps, ego v {min(v):.1f}-{max(v):.1f} m/s, "
                         f"yaw rate {min(w):+.3f}..{max(w):+.3f} rad/s")
        cur = ts["cur"]
        counts = ", ".join(f"{g}: {sum(1 for b in cur['meta']['boxes'] if b['group'] == g)}" for g in COL)
        lines.append(f"boxes at k: {counts}; camera-covered cells {cur['camera'].mean():.0%}, occluded {cur['occluded'].mean():.1%}, "
                     f"undefined {cur['undefined'].mean():.0%}; grid 120 x 104 at 0.5 m")
        tx.text(0, 0.9, "\n".join(lines), fontsize=9, va="top", family="monospace")
        fig.suptitle(f"training sample - {name}: {desc}", fontsize=11)
        fig.tight_layout()
        p = os.path.join(out_dir, f"training_sample_{name}.png")
        fig.savefig(p, dpi=75)
        plt.close(fig)
        print("saved", p)
