"""One-off (t3 report): block diagram of the camera + radar chain with the outcome of every block exchange (no red blocks).
fill: grey = frozen / cached input, blue = trained block of the reference, green = output / head.
outline of the alternatives: dark green = BETTER, olive = TIE with the reference, purple = WORSE, dashed = not run yet.
Usage: python3 temp/study/draw_architecture_t3.py <out.png>"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 170)
ax.set_ylim(0, 100)
ax.axis("off")
FILL = {"frozen": "#d9d9d9", "train": "#cfe2f3", "out": "#d9ead3", "in": "#ffffff"}
EDGE = {"better": "#1b7837", "tie": "#8c8c00", "worse": "#762a83", "todo": "#777777"}


def box(x, y, w, h, text, kind, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=FILL[kind], ec="0.3", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def alt(x, y, w, text, verdict, h=5.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc="white", ec=EDGE[verdict], lw=2.0,
                                ls="--" if verdict == "todo" else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)


def arrow(a, b, dy=0.0):
    (x0, y0, w0, h0), (x1, y1, w1, h1) = a, b
    p0, p1 = (x0 + w0, y0 + h0 / 2 + dy), (x1, y1 + h1 / 2 + dy)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="0.25"))


ax.text(85, 97.5, "Camera + radar BEV chain (front camera, 60 m x 52 m, 0.5 m cells): reference and the outcome of every block exchange", ha="center", fontsize=11.5,
        weight="bold")
ax.text(85, 94.2, "numbers: cold VAL-1000, mAP2 = vehicle + pedestrian, 3000 steps, one seed; reference 0.339 / 0.324 (two seeds), band 0.054", ha="center", fontsize=8.5, color="0.3")

# top row: camera branch
cam = box(2, 74, 26, 11, "CAM_FRONT, 3 frames\n(t, t-0.17 s, t-0.33 s)\n256 x 704", "in")
img = box(34, 74, 28, 11, "DualBEV image heads\nResNet-50 + FPN + DepthNet\nFROZEN, cached:\ndepth 118 bins + context 80 ch", "frozen")
lift = box(68, 74, 26, 11, "Lift PD (own)\ndepth-weighted pull,\n8 heights, 0.5 m", "train")
red = box(100, 74, 22, 11, "height-into-channels\n1x1 conv 640 -> 64", "train")
tmp = box(128, 74, 38, 11, "camera temporal fusion\nwindow + carried world memory\n(ego-motion warp)", "train")
for a, b in ((cam, img), (img, lift), (lift, red), (red, tmp)):
    arrow(a, b)

# middle: fusion + trunk + heads
fus = box(60, 44, 24, 11, "fusion\nconcat + modality dropout 0.2\n-> 80 ch", "train")
trk = box(90, 44, 34, 11, "BEV encoder: DualBEV ResNet pyramid +\nFPN-LSS, pre-trained, stride-1 detail path,\n96-ch head reduce", "train")
hd = box(130, 44, 36, 11, "light heads (semantic: drivable, walkway,\ndividers, ...; CenterPoint objects; velocity)\nat t-0.5 / t / t+0.5 s via time mover", "out")
arrow(fus, trk)
arrow(trk, hd)
ax.plot([147, 147, 72], [74, 60, 60], color="0.25", lw=1.1)
ax.add_patch(FancyArrowPatch((72, 60), (72, 55.6), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="0.25"))

# bottom row: radar branch
rad = box(2, 44, 24, 11, "5 radars, 0.5 s window,\nmoved to camera time\n(Doppler shift)", "in")
rb = box(30, 44, 26, 11, "radar BEV block\nmeans raster + 2 convs\n(32 ch)", "train")
arrow(rad, rb)
arrow(rb, fus)

# alternatives (outline = verdict)
y0 = 64.5
alt(68, y0, 26, "P (no depth weight): WORSE\n(veh AP@1 -0.04*, ped -0.05*)", "worse")
alt(100, y0, 22, "z-sum collapse (cam only): TIE\nmAP2 0.205 vs 0.210", "tie")
alt(34, y0, 28, "adapter on frozen features\n(cam only): TIE 0.211 vs 0.210", "tie")
alt(128, y0, 18, "no temporal block:\nTIE (0.321)", "tie")
alt(148, y0, 18, "ego warp + ConvGRU:\nTIE (0.327)", "tie")
y1 = 34.5
alt(60, y1, 24, "gate (GaussianCaR): TIE 0.330\nCMX fuser: TIE 0.332", "tie")
alt(90, y1, 16.5, "random init: WORSE\n(walkway -0.055*)", "worse")
alt(107.5, y1, 16.5, "SECOND, width 0.5,\nforward splat: not run", "todo")
alt(30, y1, 26, "pillars (point MLP + scatter-max):\nTIE, best: 0.349 (+0.011)", "tie")
# sensor ablations
alt(2, 30, 24, "radar off (camera only): WORSE\nmAP2 0.210; veh AP@0.5 0.045", "worse")
alt(2, 22.5, 24, "camera off (radar only): WORSE\nmAP2 0.062", "worse")
alt(130, 34.5, 36, "velocity head: 0 m/s predicted in all 3000-step runs\n(AVE = zero baseline); w_obj_vel x10 running", "todo")

# legend
ax.text(2, 10, "fill: grey = frozen / cached, blue = trained, green = outputs      outline of an alternative: ", fontsize=8.5, va="center")
for i, (k, t) in enumerate((("better", "BETTER"), ("tie", "TIE (in the band)"), ("worse", "WORSE"), ("todo", "not run / running"))):
    xx = 78 + i * 22
    ax.add_patch(FancyBboxPatch((xx, 8.5), 3.0, 3.0, boxstyle="round,pad=0.2", fc="white", ec=EDGE[k], lw=2.0, ls="--" if k == "todo" else "-"))
    ax.text(xx + 4.5, 10, t, fontsize=8, va="center")
ax.text(2, 4, "rule: WORSE / BETTER = difference beyond the band (2 x seed spread of the stable metrics = 0.054) AND paired scene bootstrap interval excludes 0; otherwise TIE.", fontsize=8, color="0.3")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
