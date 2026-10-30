"""Pictures of the GT: python -m autogt view OUT_DIR [scene:keyframe ...]  and  python -m autogt sheets OUT_DIR

One figure per keyframe, four panels (all from the generator, nothing re-computed):
1 front camera: 3D boxes (vehicle red, pedestrian magenta, two-wheeler orange; white rim = stationary, fixed in the world;
  grey dashed = no lidar / radar point, ignored) and the exact map lines (lane divider yellow, road divider orange,
  centre-line cyan)
2 semantic BEV 0.5 m: drivable grey, walkway purple, car park blue, crossing light, stop line red, dividers, centre-lines with
  direction arrows
3 objects: footprints with heading, velocity of 1 s (cyan arrows) and speed in km/h, '!' = radar_conflict, 'o' = radar_override,
  occluded cells blue, ignored boxes yellow, outside camera and radar dark
4 motion RELATIVE TO THE STATIC WORLD in m/s (the ego motion removed: the static world sits at the centre of the colour wheel,
  black; only moving things have colour): hue = direction (wheel in the corner), brightness =
  speed to 12 m/s, arrows
`sheets` stacks the ten default samples, three per sheet, with a legend.
"""
import colorsys
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .suite import AutoGT  # noqa: E402

COL = {"vehicle": (0.95, 0.15, 0.15), "pedestrian": (0.9, 0.2, 0.9), "two_wheeler": (1.0, 0.6, 0.1)}
LCOL = {"lane_divider": (1.0, 0.9, 0.1), "road_divider": (1.0, 0.5, 0.0), "centreline": (0.1, 0.9, 0.9)}
EXT = [26, -26, 0, 60]
SAT = 12.0
DEFAULT = [("scene-0061", 8), ("scene-0103", 12), ("scene-0553", 10), ("scene-0655", 20), ("scene-0757", 10),
           ("scene-0796", 12), ("scene-0916", 8), ("scene-1077", 8), ("scene-1094", 10), ("scene-1100", 8)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def densify(pl, step=0.5):
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def draw_3d(ax, gt, xyz, color, lw=1.2, near=0.5):
    """a 3D polyline (ego frame) in the camera image, each segment clipped at the near plane."""
    K = gt["meta"]["K"]
    P = (np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(gt["meta"]["T_c2e"]).T)[:, :3]
    for a, b in zip(P[:-1], P[1:]):
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near or b[2] < near:
            t = (near - a[2]) / (b[2] - a[2])
            c = a + t * (b - a)
            a, b = (c, b) if a[2] < near else (a, c)
        uv = np.array([[K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]] for p in (a, b)])
        ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=lw)


def blend(img, m, c, a=1.0):
    m = (a * m)[..., None]
    return img * (1 - m) + np.array(c) * m


def semantic_rgb(gt):
    nx, ny = gt["drivable_area"].shape
    img = np.full((nx, ny, 3), 0.12, np.float32)
    img = blend(img, gt["drivable_area"], (0.32, 0.36, 0.40))
    img = blend(img, gt["walkway"], (0.45, 0.35, 0.55), 0.8)
    img = blend(img, gt["carpark_area"], (0.25, 0.30, 0.55), 0.6)
    img = blend(img, gt["ped_crossing"], (0.85, 0.85, 0.85), 0.8)
    img = blend(img, gt["stop_line"], (0.9, 0.2, 0.2), 0.9)
    for layer in ("lane_divider", "road_divider", "centreline"):
        img = blend(img, gt[layer], LCOL[layer])
    return np.clip(img, 0, 1)


def object_rgb(gt):
    nx, ny = gt["drivable_area"].shape
    img = np.full((nx, ny, 3), 0.12, np.float32)
    img = blend(img, gt["drivable_area"], (0.30, 0.30, 0.30))
    for grp, c in COL.items():
        img = blend(img, gt[grp], c)
    img = blend(img, gt["occluded"] * (1 - gt["vehicle"]), (0.15, 0.25, 0.75), 0.6)
    img = blend(img, gt["ignore"], (0.95, 0.85, 0.2), 0.8)
    return np.clip(img * (1 - 0.75 * gt["undefined"][..., None]), 0, 1)


def flow_rgb(f):
    sp = np.hypot(f[0], f[1])
    hue = (np.arctan2(f[1], f[0]) + np.pi) / (2 * np.pi)
    v = np.clip(sp / SAT, 0, 1) ** 0.6
    rgb = np.zeros(sp.shape + (3,), np.float32)
    for i, j in zip(*np.nonzero(v > 0.004)):
        rgb[i, j] = colorsys.hsv_to_rgb(hue[i, j], 1.0, v[i, j])
    return rgb


def show(ax, rgb, title):
    ax.imshow(np.flip(np.flip(rgb, 0), 1), extent=EXT, interpolation="nearest")
    ax.set_xlim(26, -26)
    ax.set_ylim(0, 60)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("y (m, left +)", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def corners_3d(b):
    cx, cy, cz = b["center"]
    w, l, h = b["size"]
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([cx + c * loc[:, 0] - s * loc[:, 1], cy + s * loc[:, 0] + c * loc[:, 1], cz + loc[:, 2]], 1)


def figure(gt_gen, name, k, out_dir, total=False, gt=None):
    """one sample figure; out_dir = None returns the matplotlib figure (the caller closes it: no file is written), else a png."""
    sc = next(s for s in gt_gen.nusc.scene if s["name"] == name)
    tok = gt_gen.keyframes(name)[k]
    gt = gt if gt is not None else gt_gen.keyframe(tok)
    bx = gt["meta"]["boxes"]
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.6), gridspec_kw=dict(width_ratios=[2.0, 1, 1, 1]))
    ax[0].imshow(Image.open(os.path.join(gt_gen.dataroot, gt["meta"]["cam_file"])).convert("RGB"))
    for layer, pls in gt["meta"]["vectors"].items():
        for pl in pls:
            d = densify(pl, 1.0)
            keep = (d[:, 0] > -5) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            for run_ in np.split(np.arange(len(d)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
                if keep[run_[0]] and len(run_) >= 2:
                    draw_3d(ax[0], gt, np.c_[d[run_], np.zeros(len(run_))], LCOL[layer], lw=0.9 if layer == "centreline" else 1.4)
    for b in bx:
        if 0.5 < b["center"][0] < 70:
            xyz = corners_3d(b)
            for i, j in EDGES:
                if b["observed"] and b["stationary"]:
                    draw_3d(ax[0], gt, xyz[[i, j]], "white", lw=2.2)
                draw_3d(ax[0], gt, xyz[[i, j]], COL[b["group"]] if b["observed"] else (0.7, 0.7, 0.7), lw=1.0 if b["observed"] else 0.8)
    ax[0].set_xlim(0, 1600)
    ax[0].set_ylim(900, 0)
    ax[0].axis("off")
    ax[0].set_title("front camera: 3D boxes (white rim = stationary, grey = ignored) and map lines", fontsize=9)
    show(ax[1], semantic_rgb(gt), "semantic BEV 0.5 m (arrows: lane direction)")
    for pl in gt["meta"]["vectors"]["centreline"]:
        d = densify(pl, 1.0)
        for i in range(3, len(d) - 1, 6):
            a, b = d[i - 1], d[i + 1]
            if 0 < a[0] < 60 and abs(a[1]) < 26:
                ax[1].annotate("", (b[1], b[0]), (a[1], a[0]), arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8, mutation_scale=9))
    show(ax[2], object_rgb(gt), "objects: velocity 1 s (cyan), km/h, occluded (blue), ignored (yellow)")
    for b in bx:
        cx, cy = b["center"][:2]
        if not (-2 < cx < 62 and abs(cy) < 28):
            continue
        ring = np.vstack([b["corners"], b["corners"][:1]])
        ax[2].plot(ring[:, 1], ring[:, 0], **(dict(color="white", lw=1.0) if b["observed"] else dict(color=(0.95, 0.85, 0.2), lw=0.8, ls="--")))
        front = b["corners"][:2].mean(0)
        ax[2].plot([cy, front[1]], [cx, front[0]], color="white", lw=0.8)
        v = b["vel"]
        if b["observed"] and np.hypot(*v) > 0.5:
            ax[2].annotate("", (cy + v[1], cx + v[0]), (cy, cx), arrowprops=dict(arrowstyle="->", color="cyan", lw=1.0))
            lab = f"{np.hypot(*v) * 3.6:.0f}" + {"radar_conflict": "!", "radar_override": "o"}.get(b["vel_flag"], "")
            ax[2].text(cy, cx + 2.0, lab, color="white", fontsize=7, ha="center")
    # default: relative to the static world (flow - flow_ego = the objects' own world velocity); total=True: the apparent flow
    fl = np.where((gt["flow_weight"] > 0)[None], gt["flow"] if total else gt["velocity"], 0)
    show(ax[3], flow_rgb(fl), "total apparent flow = static world + objects [m/s]" if total
         else "motion relative to the static world [m/s] (static world = black = wheel centre)")
    nx, ny = gt["drivable_area"].shape
    gx = 0.25 + 0.5 * np.arange(nx)[:, None] + 0 * np.arange(ny)[None, :]
    gy = -26 + 0.25 + 0.5 * np.arange(ny)[None, :] + 0 * np.arange(nx)[:, None]
    sel = (np.arange(nx)[:, None] % 6 == 3) & (np.arange(ny)[None, :] % 6 == 3) & (np.hypot(*fl) > 0.3)
    ax[3].quiver(gy[sel], gx[sel], -fl[1][sel], fl[0][sel], color="white", angles="xy", scale_units="xy", scale=6, width=0.003)
    n = 41
    c = (n - 1) / 2
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    wf = np.stack([(ii - c) / c * SAT, (jj - c) / c * SAT]).astype(np.float32)
    wheel = flow_rgb(wf)
    wheel[np.hypot(*wf) > SAT] = 1.0
    w = fig.add_axes([0.925, 0.06, 0.06, 0.12])
    w.imshow(np.flip(np.flip(wheel, 0), 1), interpolation="nearest")
    w.set_xticks([])
    w.set_yticks([])
    w.set_title("0..12 m/s, up = fwd", fontsize=6)
    n_obs = {g: sum(1 for b in bx if b["group"] == g and b["observed"]) for g in COL}
    fig.suptitle(f"{name} keyframe {k}: {sc['description']} | " + ", ".join(f"{g} {v}" for g, v in n_obs.items()) +
                 f", stationary {sum(b['stationary'] for b in bx)}, ignored {sum(not b['observed'] for b in bx)}, "
                 f"velocity flags {sum(b['vel_flag'] in ('radar_conflict', 'radar_override') for b in bx)}", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.92, top=0.92, bottom=0.07, wspace=0.18)
    if out_dir is None:
        return fig
    p = os.path.join(out_dir, f"gt_sample_{name}_k{k:02d}.png")
    fig.savefig(p, dpi=72)
    plt.close(fig)
    print("saved", p, flush=True)
    return p


def run(out_dir, picks=None, dataroot="~/data/nuscenes", version="v1.0-mini", total=False):
    os.makedirs(out_dir, exist_ok=True)
    gen = AutoGT(dataroot, version)
    return [figure(gen, n, int(k), out_dir, total) for n, k in (picks or DEFAULT)]


LEGEND = ["camera: 3D boxes (vehicle red, pedestrian magenta, two-wheeler orange; white rim = stationary, fixed in the world; "
          "grey = no lidar/radar point, ignored); map lines (lane divider yellow, road divider orange, centre-line cyan)",
          "semantic BEV 0.5 m (x forward 0-60 m, y left +-26 m): drivable grey, walkway purple, car park blue, crossing light, "
          "stop line red, dividers, centre-lines with direction arrows",
          "objects: velocity of 1 s (cyan) and km/h, '!' radar conflict, 'o' radar override, occluded blue, ignored yellow; "
          "motion relative to the static world: static world black (wheel centre), hue = direction, brightness = speed (to 12 m/s)"]


def sheets(out_dir):
    imgs = [Image.open(os.path.join(out_dir, f"gt_sample_{n}_k{k:02d}.png")).convert("RGB") for n, k in DEFAULT]
    groups = [imgs[0:3], imgs[3:6], imgs[6:8], imgs[8:10]]
    paths = []
    for gi, grp in enumerate(groups):
        sheet = Image.new("RGB", (grp[0].width, 78 + sum(i.height for i in grp)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((8, 4), f"autoGT overview, sheet {gi + 1} of {len(groups)} - nuScenes mini", fill="black")
        for li, t in enumerate(LEGEND):
            d.text((8, 22 + 18 * li), t, fill=(60, 60, 60))
        y = 78
        for im in grp:
            sheet.paste(im, (0, y))
            y += im.height
        p = os.path.join(out_dir, f"gt_overview_sheet{gi + 1}.png")
        sheet.save(p)
        paths.append(p)
        print("saved", p)
    return paths
