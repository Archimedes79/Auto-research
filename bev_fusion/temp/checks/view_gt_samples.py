"""Sample GT pictures: bounding boxes and the semantic BEV map, per keyframe (Mirko, 2026-10-26: "sample gt for bounding
boxes and semantic BEV map and show them to me").

Per keyframe one figure, all from training/gt_bev.py (the GT the training will see):
1. front camera with the 3D boxes (vehicle red, pedestrian magenta, two-wheeler orange; stationary boxes thin white
   outer line; boxes without lidar / radar points dashed grey = ignored) and the exact map lines (lane divider
   yellow, road divider orange, centre-line cyan), clipped at the near plane
2. BEV semantic map (0.5 m cells): drivable area, walkway, ped crossing, car park, stop line, dividers, centre-lines
   with their direction (arrows every 6 m)
3. BEV objects: the object layers with the box outlines and heading tick, velocity arrows (1 s), occlusion behind boxes
   (blue), ignore cells (yellow), outside camera and radar (dark), camera field of view (white line)
Run from the bev_fusion folder:
    wsl ~/miniconda3/envs/gcar/bin/python temp/checks/view_gt_samples.py <out dir> [scene:keyframe ...]
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
from view_gt_observable import densify, draw_polyline_3d  # noqa: E402

G = Grid()
COL = {"vehicle": (0.95, 0.15, 0.15), "pedestrian": (0.9, 0.2, 0.9), "two_wheeler": (1.0, 0.6, 0.1)}
LCOL = {"lane_divider": (1.0, 0.9, 0.1), "road_divider": (1.0, 0.5, 0.0), "centreline": (0.1, 0.9, 0.9)}
EXT = [26, -26, 0, 60]


def show(ax, rgb):
    ax.imshow(np.flip(np.flip(rgb, 0), 1), extent=EXT, interpolation="nearest")
    ax.set_xlim(26, -26)
    ax.set_ylim(0, 60)
    ax.set_xlabel("y (m, left +)", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def blend(img, m, c, a=1.0):
    m = (a * m)[..., None]
    return img * (1 - m) + np.array(c) * m


def semantic_rgb(gt):
    img = np.full((G.nx, G.ny, 3), 0.12, np.float32)
    img = blend(img, gt["drivable_area"], (0.32, 0.36, 0.40))
    img = blend(img, gt["walkway"], (0.45, 0.35, 0.55), 0.8)
    img = blend(img, gt["carpark_area"], (0.25, 0.30, 0.55), 0.6)
    img = blend(img, gt["ped_crossing"], (0.85, 0.85, 0.85), 0.8)
    img = blend(img, gt["stop_line"], (0.9, 0.2, 0.2), 0.9)
    for layer in ("lane_divider", "road_divider", "centreline"):
        img = blend(img, gt[layer], LCOL[layer])
    return np.clip(img, 0, 1)


def object_rgb(gt):
    img = np.full((G.nx, G.ny, 3), 0.12, np.float32)
    img = blend(img, gt["drivable_area"], (0.30, 0.30, 0.30))
    for grp, c in COL.items():
        img = blend(img, gt[grp], c)
    img = blend(img, gt["occluded"] * (1 - gt["vehicle"]), (0.15, 0.25, 0.75), 0.6)
    img = blend(img, gt["ignore"], (0.95, 0.85, 0.2), 0.8)
    img = img * (1 - 0.75 * gt["undefined"][..., None])
    return np.clip(img, 0, 1)


def box_corners_3d(b):
    cx, cy, cz = b["center"]
    w, l, h = b["size"]
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([cx + c * loc[:, 0] - s * loc[:, 1], cy + s * loc[:, 0] + c * loc[:, 1], cz + loc[:, 2]], 1)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def camera_panel(ax, gt, root):
    img = Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB")
    ax.imshow(img)
    for layer, pls in gt["meta"]["vectors"].items():
        for pl in pls:
            d = densify(pl, 1.0)                     # densify first: long segments may start far behind the car
            keep = (d[:, 0] > -5) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            for run in np.split(np.arange(len(d)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
                if keep[run[0]] and len(run) >= 2:
                    draw_polyline_3d(ax, gt, np.c_[d[run], np.zeros(len(run))], LCOL[layer],
                                     lw=0.9 if layer == "centreline" else 1.4)
    for b in gt["meta"]["boxes"]:
        if b["center"][0] < 0.5 or b["center"][0] > 70:
            continue
        xyz = box_corners_3d(b)
        for i, j in EDGES:
            seg = xyz[[i, j]]
            if b["observed"]:
                if b["stationary"]:
                    draw_polyline_3d(ax, gt, seg, "white", lw=2.2)
                draw_polyline_3d(ax, gt, seg, COL[b["group"]], lw=1.0)
            else:
                draw_polyline_3d(ax, gt, seg, (0.7, 0.7, 0.7), lw=0.8)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis("off")


def semantic_panel(ax, gt):
    show(ax, semantic_rgb(gt))
    for pl in gt["meta"]["vectors"]["centreline"]:
        d = densify(pl, 1.0)
        for i in range(3, len(d) - 1, 6):                      # a direction arrow every 6 m
            a, b = d[i - 1], d[i + 1]
            if 0 < a[0] < 60 and abs(a[1]) < 26:
                ax.annotate("", (b[1], b[0]), (a[1], a[0]),
                            arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8, mutation_scale=9))


def object_panel(ax, gt):
    show(ax, object_rgb(gt))
    ax.contour(np.linspace(-26 + 0.25, 26 - 0.25, G.ny)[::-1], np.linspace(0.25, 59.75, G.nx), np.flip(gt["camera"], 1),
               levels=[0.5], colors="white", linewidths=0.6)
    for b in gt["meta"]["boxes"]:
        cx, cy = b["center"][:2]
        if not (-2 < cx < 62 and abs(cy) < 28):
            continue
        ring = np.vstack([b["corners"], b["corners"][:1]])
        style = dict(color="white", lw=1.0) if b["observed"] else dict(color=(0.95, 0.85, 0.2), lw=0.8, ls="--")
        ax.plot(ring[:, 1], ring[:, 0], **style)
        front = b["corners"][:2].mean(0)
        ax.plot([cy, front[1]], [cx, front[0]], color="white", lw=0.8)
        v = b["vel"]
        if np.hypot(*v) > 0.5:
            ax.annotate("", (cy + v[1], cx + v[0]), (cy, cx), arrowprops=dict(arrowstyle="->", color="cyan", lw=1.0))


def figure(src, scene_name, k, out_dir, root):
    sc = next(s for s in src.nusc.scene if s["name"] == scene_name)
    tok = sc["first_sample_token"]
    for _ in range(k):
        tok = src.nusc.get("sample", tok)["next"]
    gt = src.keyframe_gt(tok, G)
    bx = gt["meta"]["boxes"]
    fig = plt.figure(figsize=(20, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1, 1])
    a0 = fig.add_subplot(gs[0])
    camera_panel(a0, gt, root)
    a0.set_title("front camera: 3D boxes (white rim = stationary, grey dashed = no sensor point -> ignored), map lines", fontsize=9)
    a1 = fig.add_subplot(gs[1])
    semantic_panel(a1, gt)
    a1.set_title("semantic BEV, 0.5 m cells (arrows: lane direction)", fontsize=9)
    a2 = fig.add_subplot(gs[2])
    object_panel(a2, gt)
    a2.set_title("objects: velocity 1 s (cyan), occluded (blue), ignored (yellow)", fontsize=9)
    n = {g: sum(1 for b in bx if b["group"] == g and b["observed"]) for g in COL}
    nst = sum(b["stationary"] for b in bx)
    nig = sum(not b["observed"] for b in bx)
    fig.suptitle(f"{scene_name} keyframe {k}: {sc['description']} | boxes: " + ", ".join(f"{g} {v}" for g, v in n.items())
                 + f", stationary {nst}, ignored {nig}", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, f"gt_sample_{scene_name}_k{k:02d}.png")
    fig.savefig(p, dpi=72)
    plt.close(fig)
    print("saved", p, flush=True)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    root = os.path.expanduser("~/data/nuscenes")
    src = NuScenesGT(root, "v1.0-mini")
    picks = [a.split(":") for a in sys.argv[2:]] or [
        ("scene-0061", "8"), ("scene-0103", "12"), ("scene-0553", "10"), ("scene-0655", "20"), ("scene-0757", "10"),
        ("scene-0796", "12"), ("scene-0916", "8"), ("scene-1077", "8"), ("scene-1094", "10"), ("scene-1100", "8")]
    for name, k in picks:
        figure(src, name, int(k), out_dir, root)
