"""One-off (report of experiment t1): block diagram of the trained chain (t1_r0 configuration) -> PNG. No red blocks:
blue = trained, grey = frozen / cached, green = outputs, sand = losses (training only)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(16, 9.5))
ax.set_xlim(0, 160)
ax.set_ylim(0, 95)
ax.axis("off")
C = {"frozen": "#d9d9d9", "train": "#cfe2f3", "out": "#d9ead3", "loss": "#fff2cc", "in": "#ffffff"}


def box(x, y, w, h, text, kind, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=C[kind], ec="0.35", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def arrow(a, b, text=None, dx=0.0):
    (x0, y0, w0, h0), (x1, y1, w1, h1) = a, b
    p0 = (x0 + w0 / 2 + dx, y0) if y1 + h1 < y0 else (x0 + w0, y0 + h0 / 2)
    p1 = (x1 + w1 / 2 + dx, y1 + h1) if y1 + h1 < y0 else (x1, y1 + h1 / 2)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + 1, (p0[1] + p1[1]) / 2, text, fontsize=7.5, color="0.2")


cam = box(2, 84, 46, 8, "CAM_FRONT: current + 2 past frames\n(12 Hz sweeps, stride 2: t, t-0.17 s, t-0.33 s), 256 x 704", "in")
img = box(2, 70, 46, 9, "DualBEV image heads (ResNet-50 + FPN + DepthNet\nwith camera-parameter gate), pre-trained, FROZEN,\n"
          "outputs cached: depth p(z) 118 bins (1-60 m) + context 80 ch", "frozen")
lift = box(2, 55, 46, 10, "Lift PD: per voxel (0.5 m, 8 heights) pull of the context\nat its pixel x depth weight p(z) "
           "(2 grid_sample gathers,\n4x oversampling), heights into channels, 1x1 -> 64 ch", "train")
tmp = box(2, 40, 46, 10, "camera temporal fusion (window memory):\npast frames splatted to t by ego motion (v, omega) +\n"
          "coordinate chaining; unknown cells mass 0", "train")
rad = box(56, 70, 40, 22, "5 radars, all sweeps of the last 0.5 s\nmoved to the camera time (exact poses +\nradial Doppler "
          "motion along the ray)\n-> radar BEV block: per cell count + mean of\nr, ray direction, circular Doppler, RCS, "
          "age\n+ sub-cell offset; 2 convs -> 32 ch", "train")
fus = box(30, 25, 50, 8, "fusion: concat (64 + 32) -> 96 ch (BEVFusion ConvFuser)", "train")
trk = box(30, 12, 50, 9, "BEV trunk: ResNet pyramid + FPN-LSS (width 0.5)\n-> state S(t), grid x 0..60 m, y -26..26 m, 0.5 m",
          "train")
mov = box(92, 40, 64, 14, "TimeMover (shared): velocity head v_hat (1x1 MLP, m/s) ->\nsplat S(t) to t + dt with ego (v, "
          "omega, true keyframe gap)\nand v_hat; mass channel (0 = unknown cell)\n-> per head: refine + FiLM(dt), zero start",
          "train")
sem = box(92, 20, 30, 15, "semantic head at dt:\n5 map areas, 2 line masks,\n2 line distance fields,\n3 object coverages",
          "out")
obj = box(126, 20, 30, 15, "object head at dt (CenterPoint):\nheatmap x 3 groups, offset,\nlog size, (sin, cos) yaw\n"
          "[+ direction class, r1a],\nlog radius, velocity", "out")
dts = box(92, 60, 64, 8, "decoded at dt = -0.5 s, 0, +0.5 s (past AND future keyframe GT)", "out")
los = box(92, 2, 64, 13, "BEVLoss (training only): per dt the GT keyframe of t + dt; masks (camera / radar /\noccluded / "
          "unknown); task groups areas, lines, objects, motion with Kendall\nweights [UW-SO, r1b]; dt weights 1 / 0.5 "
          "with warm-up; aux heads on\nthe camera and radar branches (weight 0.3)", "loss")
arrow(cam, img)
arrow(img, lift, "depth p(z), context")
arrow(lift, tmp, "64 ch per frame")
arrow(tmp, fus)
ax.add_patch(FancyArrowPatch((76, 70), (60, 33.4), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
arrow(fus, trk)
ax.add_patch(FancyArrowPatch((80, 16.5), (92, 47), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
ax.text(83, 33, "S(t)", fontsize=8)
ax.add_patch(FancyArrowPatch((107, 40), (107, 35.4), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
ax.add_patch(FancyArrowPatch((141, 40), (141, 35.4), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
ax.add_patch(FancyArrowPatch((124, 59.6), (124, 54.4), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
ax.add_patch(FancyArrowPatch((124, 19.6), (124, 15.4), arrowstyle="-|>", mutation_scale=12, lw=1, color="0.25"))
for i, (k, t) in enumerate((("frozen", "frozen / cached"), ("train", "trained"), ("out", "outputs"), ("loss", "training only"))):
    ax.add_patch(FancyBboxPatch((2 + 22 * i, 1), 4, 3, boxstyle="round,pad=0.2", fc=C[k], ec="0.35"))
    ax.text(7.5 + 22 * i, 2.5, t, va="center", fontsize=8)
ax.set_title("t1: the trained camera + radar BEV chain (training/model.py, configuration t1_r0; arms of rounds 1-2 in [ ])",
             fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)
