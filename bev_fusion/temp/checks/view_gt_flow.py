"""Pictures of the velocity / flow GT (training/gt_flow.py): per keyframe the camera, the ego-induced flow of the static
world, the object flow, and the total apparent flow. Colour = direction (hue), brightness = speed in m/s (saturates at
SAT), arrows every 4 cells on the total flow. All in m/s, not per frame.
Run from the bev_fusion folder:
    wsl ~/miniconda3/envs/gcar/bin/python temp/checks/view_gt_flow.py <out dir> <method> [scene:keyframe ...]
"""
import colorsys
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "training")
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from gt_flow import instance_tracks, keyframe_flow  # noqa: E402
from lift_pd import Grid  # noqa: E402

G = Grid()
SAT = 12.0
EXT = [26, -26, 0, 60]


def colour(f):
    """(2, nx, ny) flow -> RGB: hue = direction in the ego frame, brightness = speed / SAT."""
    sp = np.hypot(f[0], f[1])
    hue = (np.arctan2(f[1], f[0]) + np.pi) / (2 * np.pi)
    rgb = np.zeros(sp.shape + (3,), np.float32)
    v = np.clip(sp / SAT, 0, 1) ** 0.6
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            if v[i, j] > 0.004:
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


def figure(src, name, k, method, out_dir, root):
    sc = next(s for s in src.nusc.scene if s["name"] == name)
    tok = sc["first_sample_token"]
    for _ in range(k):
        tok = src.nusc.get("sample", tok)["next"]
    gt = src.keyframe_gt(tok, G)
    fl = keyframe_flow(src, gt, tok, instance_tracks(src, sc["token"]), method)
    vx, vy, om = fl["ego"]
    fig, ax = plt.subplots(1, 4, figsize=(21, 6.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1, 1]))
    ax[0].imshow(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB"))
    ax[0].axis("off")
    ax[0].set_title(f"{name} keyframe {k}: ego {vx * 3.6:.0f} km/h, yaw rate {np.degrees(om):+.1f} deg/s", fontsize=9)
    base = np.full((G.nx, G.ny, 3), 0.08, np.float32)
    base[gt["drivable_area"] > 0.5] = 0.18
    unk = (fl["weight"] == 0)
    show(ax[1], np.where(unk[..., None], np.array([0.5, 0.5, 0.1]), colour(fl["flow_ego"])), "flow of the static world (ego motion only)")
    obj = colour(fl["flow_obj"])
    obj[(gt["instance"] >= 0) & (np.hypot(*fl["flow_obj"]) < 0.3)] = (0.35, 0.35, 0.35)       # object at rest: grey
    obj = np.where(unk[..., None], np.array([0.5, 0.5, 0.1]), obj)
    show(ax[2], np.where((gt["instance"] >= 0)[..., None] | unk[..., None], obj, base), "object flow = world velocity (km/h labels); grey = at rest, olive = unknown")
    show(ax[3], colour(fl["flow"]), "total apparent flow = static + object (arrows)")
    gx, gy = cell_centres(G)
    sel = (np.arange(G.nx)[:, None] % 6 == 3) & (np.arange(G.ny)[None, :] % 6 == 3)
    sel &= np.hypot(*fl["flow"]) > 0.3
    ax[3].quiver(gy[sel], gx[sel], -fl["flow"][1][sel], fl["flow"][0][sel], color="white", angles="xy", scale_units="xy", scale=6, width=0.003)
    for bi, b in enumerate(gt["meta"]["boxes"]):
        v = fl["box_vel"][bi]
        cx, cy = b["center"][:2]
        if v is not None and np.hypot(*v) > 1.0 and 0 < cx < 60 and abs(cy) < 26:
            ax[2].text(cy, cx + 2.0, f"{np.hypot(*v) * 3.6:.0f}", color="white", fontsize=7, ha="center")
    # colour wheel
    n = 41                                                       # colour wheel in the same orientation as the BEV panels
    c = (n - 1) / 2
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    wf = np.stack([(ii - c) / c * SAT, (jj - c) / c * SAT]).astype(np.float32)   # [ix forward, iy left]
    wheel = colour(wf)
    wheel[np.hypot(*wf) > SAT] = 1.0
    w = fig.add_axes([0.915, 0.06, 0.07, 0.12])
    w.imshow(np.flip(np.flip(wheel, 0), 1), interpolation="nearest")
    w.set_xticks([]); w.set_yticks([])
    w.set_title(f"0 .. {SAT:.0f} m/s, up = forward", fontsize=6)
    fig.suptitle(f"velocity GT as flow field [m/s], method {method}", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.92, bottom=0.07, wspace=0.18)
    p = os.path.join(out_dir, f"gt_flow_{name}_k{k:02d}.png")
    fig.savefig(p, dpi=72)
    plt.close(fig)
    print("saved", p, flush=True)


if __name__ == "__main__":
    out_dir, method = sys.argv[1], sys.argv[2]
    root = os.path.expanduser("~/data/nuscenes")
    src = NuScenesGT(root, "v1.0-mini")
    picks = [a.split(":") for a in sys.argv[3:]] or [("scene-0061", "8"), ("scene-1077", "8"), ("scene-0103", "12")]
    for name, k in picks:
        figure(src, name, int(k), method, out_dir, root)
