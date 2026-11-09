"""scratch: build temp/overview/overview.html from template.html (architecture SVG from a spec, images and videos as data URIs).
run inside WSL:  $HOME/miniconda3/envs/gcar/bin/python temp/overview/build.py"""
import base64
import io
import os
from PIL import Image

ROOT = "/mnt/c/Auto-research/bev_fusion"
OUT = f"{ROOT}/temp/overview"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- architecture drawing (spec -> svg)
W, H = 1200, 780
boxes = []   # (id, x, y, w, h, title, lines, cls)
bands = []   # (y, h, label1, label2)
arrows = []  # (points, head)


def box(i, x, y, w, h, title, lines, cls="trained"):
    boxes.append((i, x, y, w, h, title, lines, cls))


bands += [
    (20, 120, "1 Precompute", "precompute.py, caches"),
    (152, 44, "2 Input interface", "interface.py"),
    (208, 428, "3 Network", "model.py slots"),
    (648, 48, "4 Output interface", "interface.py"),
    (708, 64, "5 Post-processing", "extract.py, evaluation"),
]
# layer 1
box("cam", 150, 34, 190, 92, "Front camera", ["3 key frames: now, -1, -2", "(cam_stride 2)"], "input")
box("enc", 356, 34, 250, 92, "Frozen image encoders", ["cached: DualBEV R50 +", "BEVFusion Swin features;", "ResNet-50 stem + layer 1"], "frozen")
box("rad", 622, 34, 190, 92, "5 radars", ["sweeps, ego-compensated", "to the key frame"], "input")
box("geo", 828, 34, 190, 92, "Geometry + visibility", ["calibration, ego steps,", "visibility maps: camera,", "radar fan, fusion area"], "input")
box("gt", 1034, 34, 150, 92, "Ground truth", ["training only:", "autogt v8 maps,", "boxes, flow (m/s)"], "gt")
# layer 2
box("iface", 150, 160, 1034, 28, "", ["batch dict and forward arguments with named shapes, units and frames; contracts are checked on the first batch of every run"], "iface")
# layer 3
box("lift", 150, 222, 190, 76, "Lift x2 + merge", ["depth-weighted (PD) lift", "of both feature sets", "29 ms (lift + lift2)"])
box("cmem", 356, 222, 190, 76, "Camera memory", ["3-frame window, roll", "by the ego step", "13 ms"])
box("lane", 562, 222, 190, 76, "+ Lane branch L4", ["stride 4, lifted with", "the DualBEV depth", "(option, on)"], "opt")
box("rade", 808, 222, 180, 76, "Radar encoder", ["delta slots, C_rad 64", "4 ms"])
box("radm", 1004, 222, 180, 76, "Radar memory", ["temporal block of", "the radar branch"])
box("mot", 150, 312, 400, 64, "Motion block M", ["two camera BEVs + ego steps + radar points ->", "velocity field, 3-level pyramid (strides 4 / 2 / 1) - 4 ms"])
box("fus", 562, 312, 622, 64, "Fusion", ["concat of camera BEV and radar BEV, C_fuse 80, modality dropout - 0.5 ms"])
box("trunk", 562, 390, 300, 64, "Backbone + neck (trunk)", ["DualBEV-pretrained, width 1.0,", "detail 32 -> features S - 11 ms"])
box("mover", 878, 390, 306, 64, "Time mover (shared)", ["moves S to dt -0.5 / 0 / +0.5 s", "6 calls - 17 ms"])
box("hvel", 150, 482, 190, 84, "Velocity from field", ["box velocity read from", "the motion-block field", "(option, on)"], "opt")
box("hmap", 356, 482, 210, 84, "Map head", ["drivable, walkway, car park,", "ped. crossing, lane + road", "divider - 5 ms (3 calls)"])
box("hcl", 582, 482, 210, 84, "1 m centre-line head", ["presence, distance,", "direction at 1 m cells", "(option, on)"], "opt")
box("hobj", 808, 482, 376, 84, "Object head", ["heat, box, heading, moving probability (hurdle);", "3 merged groups: vehicle, pedestrian, two-wheeler", "8 ms (3 calls)"])
box("soft", 150, 580, 1034, 44, "Soft extraction (fixed math, option)", ["dense boxes (peak weight + box per cell) and soft line map; a loss or a later planner reaches every module through it"], "opt")
# layer 4
box("out", 150, 656, 1034, 32, "", ["per time offset dt: maps, object maps, dense boxes, soft lines, velocity field v_hat, features S, temporal state - dense and differentiable"], "iface")
# layer 5
box("sel", 150, 716, 420, 48, "Box selection, line threshold", ["for the metrics and the prototype only"], "post")
box("ev", 586, 716, 280, 48, "Evaluation (BEV suite)", ["maps, boxes, velocity, curves"], "post")
box("plan", 882, 716, 302, 48, "Planner (not part of the system)", ["reads the output interface"], "gt")


def v(x, y0, y1, head=True):
    arrows.append(([(x, y0), (x, y1)], head))


def hline(y, x0, x1):
    arrows.append(([(x0, y), (x1, y)], True))


for x in (245, 481, 717, 923):
    v(x, 126, 160)
for x in (245, 451, 657, 898, 1094):
    v(x, 188, 222)
hline(260, 340, 356)
hline(260, 546, 562)
hline(260, 988, 1004)
v(245, 298, 312)
v(451, 298, 312)
v(657, 298, 312)
v(1094, 298, 312)
v(712, 376, 390)
hline(422, 862, 878)
v(245, 376, 482)
v(1031, 454, 470, head=False)
arrows.append(([(461, 470), (1031, 470)], False))
for x in (461, 687, 996):
    v(x, 470, 482)
for x in (245, 461, 687, 996):
    v(x, 566, 580)
for x in (300, 667, 1030):
    v(x, 624, 656)
for x in (360, 726, 1030):
    v(x, 688, 716)


def svg():
    out = [f'<svg class="arch" viewBox="0 0 {W} {H}" role="img" aria-label="Architecture of the BEV Fusion reference network in five layers">',
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>']
    for (y, h, l1, l2) in bands:
        out.append(f'<rect class="band" x="8" y="{y}" width="1184" height="{h}" rx="6"/>')
        out.append(f'<text class="bl1" x="20" y="{y + 22}">{esc(l1)}</text>')
        out.append(f'<text class="bl2" x="20" y="{y + 39}">{esc(l2)}</text>')
    for pts, head in arrows:
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        out.append(f'<path class="arrow" d="{d}"' + (' marker-end="url(#ah)"' if head else "") + "/>")
    for (i, x, y, w, h, title, lines, cls) in boxes:
        out.append(f'<g class="blk {cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5"/>')
        ty = y + 18
        if title:
            out.append(f'<text class="bt" x="{x + 9}" y="{ty}">{esc(title)}</text>')
            ty += 16
        else:
            ty = y + h / 2 + 4
        for ln in lines:
            out.append(f'<text class="bd" x="{x + 9}" y="{ty}">{esc(ln)}</text>')
            ty += 15
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- media as data uris
def jpg(path, width, q=78):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def vid(path):
    return "data:video/mp4;base64," + base64.b64encode(open(path, "rb").read()).decode()


E = f"{ROOT}/runs/t20_base_s0/eval"
V = f"{ROOT}/results/BEV_result_report_v7/videos/t20_base_s0"
media = {
    "IMG_OV1": jpg(f"{E}/images/scene-0554_029_overview.png", 1150),
    "IMG_OV2": jpg(f"{E}/images/scene-0039_026_overview.png", 1150),
    "IMG_PROBE": jpg(f"{E}/modules/frame_0.png", 760),
    "IMG_CURVES": jpg(f"{ROOT}/runs/t20_base_s0/report/curves.png", 1000),
    "IMG_CONTACT": jpg(f"{OUT}/precision_paint_contact.png", 900),
    "VID_A": vid(f"{V}/scene-0094_overview.mp4"),
    "VID_B": vid(f"{V}/scene-0557_overview.mp4"),
    "VID_C": vid(f"{V}/scene-0097_dashboard.mp4"),
}

html = open(f"{OUT}/template.html", encoding="utf-8").read()
html = html.replace("{{ARCH}}", svg())
for k, val in media.items():
    html = html.replace("{{" + k + "}}", val)
open(f"{OUT}/overview.html", "w", encoding="utf-8").write(html)
print("overview.html", round(len(html) / 1e6, 2), "MB")
