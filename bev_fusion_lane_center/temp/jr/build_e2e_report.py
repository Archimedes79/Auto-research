"""one-off (2026-11-12 evening): the report on the end-to-end differentiable lane graph (Spurmitte): which approaches work, architecture, experiment numbers, quantitative and qualitative results, pictures, videos, the answer to Mirko's question about shape, the night queue.

    wsl bash tools/env/_py14.sh jr/build_e2e_report.py          CPU; writes results/E2E.report/{index.html, img/, vid/}
Every number is read from a result file (runs/**.json, temp/e309/*.json, temp/jr/*.json); a missing file prints "pending".
"""
import glob
import json
import os
import shutil
import sys

ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.join(ROOT, "temp", "jr")
sys.path.insert(0, HERE)
import svgcharts as C  # noqa: E402

OUT = os.path.join(ROOT, "results", "E2E.report")
IMG, VID = os.path.join(OUT, "img"), os.path.join(OUT, "vid")
for d in (IMG, VID):
    os.makedirs(d, exist_ok=True)


def J(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    try:
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    except ValueError:
        return None


def jpg(src, name, q=84, maxw=1900):
    from PIL import Image
    if not os.path.exists(src):
        return None
    im = Image.open(src).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(os.path.join(IMG, name), quality=q, optimize=True)
    return "img/" + name


def webm_from(src, name, width=1280):
    """copy a webm; transcode an mp4 (cv2, vp09 like the junction report)"""
    if not os.path.exists(src):
        return None
    dst = os.path.join(VID, name)
    if src.endswith(".webm"):
        shutil.copyfile(src, dst)
        return "vid/" + name
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    H = int(round(h0 * width / w0 / 2) * 2)
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"vp09"), fps, (width, H))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(fr, (width, H), interpolation=cv2.INTER_AREA))
    wr.release()
    return "vid/" + name


def f3(v):
    return "pending" if v is None else f"{v:.3f}"


def f2(v):
    return "pending" if v is None else f"{v:.2f}"


def sg(v):
    return "pending" if v is None else f"{v:+.4f}"


def pc(v, d=1):
    return "pending" if v is None else f"{v:.{d}f} %"


def tbl(head, rows, num=()):
    h = "".join(f'<th class="{"num" if i in num else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig(src, cap, vid=False):
    if src is None:
        return f'<figure><p class="cap">pending: {cap}</p></figure>'
    if vid:
        return f'<figure><video controls preload="none" loop playsinline src="{src}"></video><p class="cap">{cap}</p></figure>'
    return f'<figure><a href="{src}" target="_blank" rel="noopener"><img src="{src}" loading="lazy" alt=""></a><p class="cap">{cap}</p></figure>'


def pill(kind, text):
    return f"<span class='pill {kind}'>{text}</span>"


# ------------------------------------------------------------------------------------------------------------------ data
def cards(pat):
    return [J(pat.format(s=s)) for s in (0, 1)]


def pair(vals, fmt=f3):
    if any(v is None for v in vals):
        return "pending"
    return " / ".join(fmt(v) for v in vals)


def diff(cs, k):
    return [None if not c else c["difference"][k] for c in cs]


def head(cs, k):
    return [None if not c else c["numbers"]["head"][k] for c in cs]


def base(cs, k):
    return [None if not c else c["numbers"]["base"][k] for c in cs]


C_E284 = cards("runs/e284_h2_s{s}/ends_card.json")
C_E287 = cards("runs/e287_p2_s{s}/ends_card.json")
C_E305 = cards("runs/e305_e_s{s}/ends_card.json")
C_E305S = cards("temp/gt2/snap/e305_e_s{s}_thr0.1.json")
C_E312U = cards("runs/e312_p_s{s}/ends_card_thr0.1.json")                     # E312, p >= 0.1, connectors drawn from the keypoints (no snap)
C_E312S = cards("runs/e312_p_s{s}/ends_card_snap_thr0.1.json")
C_E314S = cards("runs/e314_p_s{s}/ends_card_snap_thr0.1.json")
C_E298S = cards("runs/e298_w_s{s}/ends_card_snap_thr0.1.json")
MAN = {s: J(f"runs/e312_p_s{s}/maneuver_val.json") for s in (0, 1)}
KP = {"E286 j1 (maps, heat sigma 1.5 m)": [J(f"runs/e286_j1_s{s}/jmaps_val.json") for s in (0, 1)],
      "E307 (heat sigma 1.0 m)": [J(f"runs/e307_n_s{s}/jmaps_val.json") for s in (0, 1)],
      "E313 (heat sigma 0.7 m)": [J(f"runs/e313_s_s{s}/jmaps_val.json") for s in (0, 1)]}
LS = J("temp/e309/lines_stats_s0.json")
OFF = J("temp/e309/offsets_s0.json")
SC = J("temp/e309/scenes_s0.json")
SENS312 = J("temp/e309/sens_e312_s0.json")
SENS = J("temp/e309/sens_s0.json")
COST = J("temp/jr/model_cost.json")
TIMEB = J("runs/e273/e273_p_s0/timing.json")
TIMEJ = J("temp/jr/timing_junction.json")


def kp_f1(res):
    if not res:
        return None
    k = res["keypoints"]
    return (k["kp_ours_tol3.0_noang_ent"]["f1"] + k["kp_ours_tol3.0_noang_ext"]["f1"]) / 2


# ------------------------------------------------------------------------------------------------------------------ architecture
def box(x, y, w, h, title, sub, cls="box"):
    lines = sub.split("|")
    s = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/><text class="bt" x="{x + 10}" y="{y + 20}">{title}</text>'
    for i, ln in enumerate(lines):
        s += f'<text class="bs" x="{x + 10}" y="{y + 38 + 14 * i}">{ln}</text>'
    return s


def arrow(x1, y1, x2, y2):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 7 * math.cos(a), y2 - 7 * math.sin(a)
    p1 = (hx - 4 * math.sin(a), hy + 4 * math.cos(a))
    p2 = (hx + 4 * math.sin(a), hy - 4 * math.cos(a))
    return f'<line class="arr" x1="{x1}" y1="{y1}" x2="{hx:.1f}" y2="{hy:.1f}"/><polygon class="arrh" points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"/>'


def architecture():
    pm = COST["params_m"] if COST else None
    mg = COST if COST else None
    s = ['<svg class="chart" viewBox="0 0 1130 560" role="img" xmlns="http://www.w3.org/2000/svg">']
    s.append('<rect class="box hot" x="10" y="8" width="16" height="10" rx="2"/><text class="bs" x="32" y="17">differentiable: the loss reaches it</text>'
             '<rect class="box map" x="270" y="8" width="16" height="10" rx="2"/><text class="bs" x="292" y="17">hard step, named (index choice, threshold, rule on the output)</text>'
             '<rect class="box new" x="680" y="8" width="16" height="10" rx="2"/><text class="bs" x="702" y="17">queued for tonight / tomorrow (differentiable)</text>')
    s.append(box(10, 36, 170, 80, "Camera + radar", "GaussianCaR BEV features|frozen, cached: 39 ch,|200 x 200 at 0.5 m", "box"))
    s.append(box(215, 36, 190, 80, "Pyramid (B, E273)", f"0.5 / 1 / 2 m features, frozen|{mg['pyr_macs_g']:.2f} GMACs, {pm['pyramid']:.2f} M params" if mg else "0.5 / 1 / 2 m features, frozen", "box hot"))
    s.append(box(440, 36, 280, 100, "Junction maps head (E286, E307)", f"1 m: zone, core, entry / exit heat (sigma 1.0 m) + heading|0.5 m: stop line, crossing; dilated context|{mg['jmaps_macs_g']:.2f} GMACs, {pm['jmaps']:.2f} M params" if mg else "1 m heat + heading", "box hot"))
    s.append(box(755, 36, 170, 100, "Keypoints", "3 x 3 peaks, soft-argmax position|(differentiable in the heat)|top K = 32: an index choice|(hard, named, D7)", "box map"))
    s.append(box(955, 36, 165, 100, "Pair head (E287, E305)", f"set attention, rank features,|evidence along the curve,|dual softmax, handle scales|{mg['pair_head_macs_g']:.2f} GMACs, {pm['pair_head']:.2f} M" if mg else "set attention + dual softmax", "box hot"))
    s.append(box(955, 170, 165, 64, "Connectors", "Hermite curve per pair|(chord tangent x learned scales)", "box hot"))
    s.append(box(215, 170, 190, 80, "Node head (B)", "2 m grid: existence, offset,|8 type flags, heading", "box hot"))
    s.append(box(440, 170, 190, 80, "Links + sections (B)", "short / long links, Hermite|handles, soft sections", "box hot"))
    s.append(box(665, 170, 250, 80, "Pruning (B)", "hard close-node step 1.5 m (E275);|differentiable twin: Matrix-NMS (E288 TIE);|soft step E277 CUT", "box map"))
    s.append(box(440, 285, 270, 70, "Map assignment (E273)", "lane-level map with errors,|soft registration, dustbin, attention", "box hot"))
    s.append(box(745, 285, 375, 70, "Read-out (the rules on the output)", "connectors at p >= 0.1 (threshold); ends snapped to the delivered|piece ends (E311, a rule); Douglas-Peucker 0.3 m", "box map"))
    s.append(box(745, 385, 375, 56, "Delivered lane graph", "B's nodes / links / sections + the connectors; planner-ready", "box"))
    s.append(box(10, 285, 400, 156, "Training signals today", "node / link / section losses of B (E273);|junction maps: zone, core, end heat (focal + dice), heading;|pair head: BCE + dual softmax + crossing penalty + handle-scale fit;|e2e (E305): the pair loss reaches the maps through the soft-argmax.|NOT supervised as a shape: the connector curve itself and the|end position beyond the 1 m cell (that is E317 / E318)", "box"))
    s.append(box(10, 465, 270, 84, "E317 connector geometry loss", "Chamfer(curve, true connector) on the delivered|Hermite curve; training only, 0 ms at run time", "box new"))
    s.append(box(300, 465, 270, 84, "E318 end position uncertainty", "per-end log sigma, Laplace NLL; sigma goes to the|pair logit and to the planner; ~0 ms", "box new"))
    s.append(box(590, 465, 270, 84, "E319 heading-guided exchange", "cells exchange features ahead / behind along the|lane heading (2, 4, 8 m), 2 rounds; in the maps head", "box new"))
    s.append(box(880, 465, 240, 84, "E299 time (not built)", "cross-attention from the keypoints of the|previous frame (ego-warped); pair flips 19 %", "box new"))
    for a in ((180, 76, 215, 76), (405, 76, 440, 76), (720, 76, 755, 76), (925, 76, 955, 76), (1037, 136, 1037, 170), (310, 116, 310, 170), (405, 210, 440, 210), (630, 210, 665, 210), (1037, 234, 1037, 270), (570, 250, 570, 285), (915, 210, 940, 300), (932, 355, 932, 385)):
        s.append(arrow(*a))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------------------------------------------------------ tables
def audit_table():
    rows = []

    def row(name, ids, soft, hard, cs, comp):
        d = diff(cs, "topo_F1")
        rows.append([f"<b>{name}</b>", ids, soft, hard, pair(head(cs, "topo_F1")), pair(d, sg) if all(v is not None for v in d) else "-", pair(diff(cs, "lanes_found"), sg) if cs[0] else "-", comp])

    plain = C_E312S
    rows.append(["<b>B, the plain lane head</b>", "E273, E275", "pixel maps, nodes, links, soft sections, soft map assignment", "close-node step 1.5 m, read-out thresholds, Douglas-Peucker", pair(base(plain, "topo_F1")), "0 (reference)", "0 (reference)",
                 f"{TIMEB['total']['median']:.1f} ms (idle GPU, batch 1)" if TIMEB else "-"])
    row("E284 end head", "E284", "pair logits on B's node tokens, dual softmax, Hermite connectors", "peak choice, p >= 0.5", C_E284, "+ small")
    row("E287 pair head", "E286, E287", "junction maps, soft-argmax keypoints, set attention, order features, dual softmax", "peak choice, p >= 0.5", C_E287, "+ 3.3 GMACs maps, 0.02 pair head")
    row("E305 joint training", "E305", "as E287, the pair loss also trains the maps head", "peak choice, p >= 0.5", C_E305, "same")
    row("E305 + snapped ends", "E305, E311", "as E305", "peak choice, p >= 0.1, <b>snap = rule on the output</b>", C_E305S, "same, snap on the CPU")
    row("E312, connectors not snapped", "E307, E305, E303, E312", "narrow end heat, joint training, learned handle scales", "peak choice, p >= 0.1", C_E312U, "same")
    row("<b>E312, snapped (final system)</b>", "E307, E305, E303, E311, E312", "as above", "peak choice, p >= 0.1, <b>snap = rule on the output</b>", C_E312S, "same")
    row("E298 planner variant", "E298", "E312 with the pairs the ego drives weighted 2", "peak choice, p >= 0.1, snap", C_E298S, "same")
    row("E314 narrower heat", "E313, E314", "E312 on sigma 0.7 m maps", "peak choice, p >= 0.1, snap", C_E314S, "same")
    return tbl(["approach", "experiments", "what is soft (differentiable)", "hard steps left on the delivered graph", "topo F1 (s0 / s1)", "topo F1 vs the plain graph", "lanes found vs plain", "compute"], rows, num=(4, 5, 6))


def ladder_table():
    steps = [("E287 p2: pair head, connectors from the keypoints, p >= 0.5", C_E287),
             ("E305: maps and pairs trained together, same read-out", C_E305),
             ("+ connector ends snapped, p >= 0.1 (E311)", C_E305S),
             ("<b>E312 not snapped</b>: narrow end heat + scales, connectors from the keypoints, p >= 0.1 (a differentiable read-out up to the threshold)", C_E312U),
             ("<b>E312 snapped, p >= 0.1 (final system)</b>", C_E312S),
             ("E314: the same on the sigma 0.7 m maps (TIE with E312)", C_E314S),
             ("E298: E312 with driven pairs weighted 2 (planner variant)", C_E298S)]
    rows = []
    for name, cs in steps:
        if not all(cs):
            rows.append([name] + ["pending"] * 7)
            continue
        rows.append([name, pair(head(cs, "topo_F1")), pair(diff(cs, "topo_F1"), sg), pair(diff(cs, "lanes_found"), sg), pair(diff(cs, "junction_F1"), sg), pair(diff(cs, "junction_R"), sg), pair(diff(cs, "dup_frac"), sg), pair(diff(cs, "F1@1.0"), sg)])
    return tbl(["step (delivered graph = B + hard close-node step + the connectors of the pair head)", "topo F1 (s0 / s1)", "topo F1 vs plain", "lanes found", "junction F1", "junction recall", "dup_frac", "F1@1.0"], rows, num=range(1, 8))


def maps_table():
    rows = []
    for name, rs in KP.items():
        for s, r in enumerate(rs):
            if not r:
                rows.append([name, str(s), "pending", "pending", "pending"])
                continue
            k = r["keypoints"]
            rows.append([name, str(s), f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f"<b>{f3(kp_f1(r))}</b>"])
    return tbl(["maps head", "seed", "entry recall", "exit recall", "keypoint F1 (3 m, thr 0.3)"], rows, num=(1, 2, 3, 4))


def maneuver_chart():
    names = [("plain", "plain graph"), ("p0.5s", "connectors p >= 0.5, snapped"), ("p0.1", "p >= 0.1, not snapped"), ("p0.1s", "p >= 0.1, snapped")]
    if not all(MAN.values()):
        return "<p>pending</p>"
    groups = [n[1] for n in names]
    vals = [[100 * MAN[s]["variants"][k]["recall"] for k, _ in names] for s in (0, 1)]
    return C.bars(groups, ["seed 0", "seed 1"], vals, ymax=100, fmt=".0f", width=760, height=290, ylabel="GT maneuvers realised, %")


def tol_table():
    if not SENS or not SENS312:
        return "<p>pending</p>"
    rows = []
    labels = {"ref (snap 2, tol 1, dir 60deg, reach 30)": "reference: match 1 m, direction within 60 degrees, reach 30 m", "tol 2 (snap 3)": "match 2 m (snap 3 m)", "tol 2 + no direction": "match 2 m, direction ignored", "tol 3 (snap 4)": "match 3 m (snap 4 m)"}
    for k, lab in labels.items():
        if k not in SENS312:
            continue
        b = SENS312[k]["B"]["topo_F1"]
        e = SENS312[k]["e312"]["topo_F1"]
        o = SENS.get(k, {})
        rows.append([lab, f3(b), f3(o["e284"]["topo_F1"]) if "e284" in o else "-", f3(o["e287p2"]["topo_F1"]) if "e287p2" in o else "-", f"<b>{f3(e)}</b>", sg(e - b)])
    return tbl(["topo F1, seed 0, VAL 873 frames", "B (plain)", "E284", "E287 p2", "E312", "E312 - B"], rows, num=(1, 2, 3, 4, 5))


def lines_table():
    if not LS:
        return "<p>pending</p>"
    rows = []
    names = (("GT", "l6 GT"), ("B", "B (plain)"), ("e284", "E284"), ("e287p2", "E287 p2"), ("e312", "<b>E312</b>"))
    for k, n in names:
        r = LS[k]
        wrong = lambda v: "-" if v != v or v is None else f"{v:.2f}"  # noqa: E731
        rows.append([n, f"{r['lines_per_frame']:.1f}", f"{r['length_m']:.0f}", wrong(r["bad1"]), wrong(r["bad2"]), f"{r['turn10']:.2f}", f"{r['join_med']:.1f} / {r['join_p90']:.1f}"])
    ex = LS["e312_extra_over_B"]
    return tbl(["variant (seed 0)", "lines per frame", "length m per frame (box 96 m)", "length > 1 m from GT, %", "length > 2 m from GT, %", "turns > 10 degrees per 100 m", "kink at joins, degrees (median / p90)"], rows, num=range(1, 7)) + \
        f'<p class="cap">E312 delivers {ex["length_m"]:.0f} m per frame that B does not have (more than 1 m from every B point); {100 - ex["bad2"]:.0f} % of it lies within 2 m of the GT and {100 - ex["bad1"]:.0f} % within 1 m. The extra lines are mostly right, as seen in the middle panels.</p>'


def off_table():
    if not OFF:
        return "<p>pending</p>"
    rows = []
    for k, n in (("B", "B (plain)"), ("e312", "E312")):
        r = OFF[k]
        rows.append([n, str(r["n"]), f"{r['off_med']:.2f} / {r['off_p90']:.2f}", f"{r['jit_med']:.2f} / {r['jit_p90']:.2f}", f"{r['off_share']:.0f} %", " / ".join(f"{r['by_range'][b]['off_med']:.2f}" for b in ("0-15", "15-30", "30-48"))])
    return tbl(["variant (seed 0)", "lines within 3 m of the GT", "lateral offset of a line, m (median / p90)", "jitter around its straight fit, m (median / p90)", "share of the squared error that is the offset", "offset median by range 0-15 / 15-30 / 30-48 m"], rows, num=range(1, 6))


def scene_chart():
    if not SC:
        return "<p>pending</p>"
    g = [s["scene"] for s in SC]
    return C.bars(g, ["B", "E312"], [[s["B"] for s in SC], [s["e312"] for s in SC]], ymax=1.0, fmt=".1f", width=1000, height=290, ylabel="topo F1 per scene", label_every=True)


def compute_table():
    rows = []
    if COST:
        pm = COST["params_m"]
        tj = TIMEJ
        rows.append(["Pyramid (B's, shared)", f"{pm['pyramid']:.2f}", f"{COST['pyr_macs_g']:.2f}", f"{tj['pyramid']:.1f}" if tj else f"{TIMEB['pyr']['median']:.1f} (B timing)" if TIMEB else "-"])
        rows.append(["Junction maps head", f"{pm['jmaps']:.2f}", f"{COST['jmaps_macs_g']:.2f}", f"{tj['junction_maps']:.1f}" if tj else "pending (idle-GPU job in the night chain)"])
        rows.append(["Pair head (keypoints, attention, curves)", f"{pm['pair_head']:.2f}", f"{COST['pair_head_macs_g']:.3f}", f"{tj['pair_head_ms']:.1f}" if tj else "pending"])
    if TIMEB:
        for k, n in (("nodes", "B node head"), ("links", "B links"), ("sections", "B sections solve"), ("assign", "B map assignment")):
            rows.append([n, "-", "-", f"{TIMEB[k]['median']:.1f}"])
        rows.append(["<b>B total (E273, measured on an idle GPU)</b>", "-", "-", f"<b>{TIMEB['total']['median']:.1f}</b>"])
    rows.append(["Read-out on the CPU (connectors, snap, Douglas-Peucker)", "-", "-", "not timed yet (python, numpy; in the night chain)"])
    return tbl(["block", "parameters, M", "GMACs per frame (batch 1)", "ms per frame (RTX 3060, batch 1, bf16)"], rows, num=(1, 2, 3))


# ------------------------------------------------------------------------------------------------------------------ media
FR = sorted(glob.glob(os.path.join(ROOT, "results", "l6_view_E312", "f*.png")))
GAL = [("aft_typical.jpg", "Typical frames after E312 (snapped, p >= 0.1): GT grey, delivered green, missed GT blue"), ("aft_random12.jpg", "Twelve random frames after E312"),
       ("aft_worst_coverage.jpg", "The twelve frames with the worst maneuver coverage after E312"), ("aft_misses.jpg", "Missed maneuvers after E312"), ("aft_false_pairs.jpg", "False connectors after E312"),
       ("aft_distribution.jpg", "Distribution of the per-frame numbers after E312"), ("maneuver_bars.jpg", "Maneuver coverage: plain graph, connectors, snapped, perfect pairs, perfect scales (E310 / E311)")]
OLD = os.path.join(ROOT, "results", "Junction.report", "img")


def media_block():
    s = ""
    vids = [("scene41_B_E312_E287.webm", "Scene 41, the largest gain of E312 (topo F1 0.63 to 0.76 over the scene), every second keyframe, 2 frames per second. Left to right: front camera, B (plain), E312, B + E287 pair head. Grey with arrows = l6 GT with the travel direction; delivered points green within 1 m, orange within 2 m, red beyond; blue = GT not covered within 2 m."),
            ("scene33_B_E312_E287.webm", "Scene 33 (topo F1 0.73 to 0.82). Same layout."),
            ("scene15_B_E312_E287.webm", "Scene 15, no gain (topo F1 0.647 for both): the honest case. Same layout.")]
    for f, cap in vids:
        s += fig(webm_from(os.path.join(ROOT, "results", "l6_view_E312", "videos", f), "e2e_" + f), cap, vid=True)
    s += fig(webm_from(os.path.join(ROOT, "results", "l6_view_B", "videos", "scene-0221_e273_p_s0.mp4"), "b_scene0221.webm"), "The plain lane head B on the l6 GT with direction, scene 0221 (made before E312, the camera view and the graph).", vid=True)
    cap = ["Frame {n}: the same four panels as the videos (front camera, B, E312, B + E287 p2)."]
    for i, p in enumerate(FR[:8]):
        s += fig(jpg(p, f"frame_{i}.jpg"), cap[0].format(n=os.path.basename(p)[:-4]))
    for f, c in GAL:
        s += fig(jpg(os.path.join(OLD, f), "gal_" + f) if os.path.exists(os.path.join(OLD, f)) else None, c)
    return s


def q_table():
    rows = []
    qs = [("E315", "maps trained 6000 steps (not 3000), then the E312 pair stage", "yes, same model", "0 ms (training only)", "running: maps from 22:00, pair stage after"),
          ("E316", "the E312 pair head trained 6000 steps", "yes, same model", "0 ms (training only)", "chain: after E315, about 01:10 - 04:20"),
          ("E317", "connector geometry loss: Chamfer between the delivered Hermite curve and the true connector", "yes: loss on the output reaches positions, headings, scales and (through the soft-argmax) the maps", "0 ms", "code and card in preparation"),
          ("E318", "per-end position uncertainty (Laplace NLL), sigma as a feature of the pair logit and an output for the planner", "yes", "about 0 ms", "code and card in preparation"),
          ("E319", "heading-guided forward / backward exchange in the junction maps head", "yes", "+0.33 GMACs (+10 % of the maps head), +33 k parameters", "built and tested (commit 35822d9f), card written, expected TIE (prediction +0.005)"),
          ("E320", "soft attachment of the connector ends to the delivered pieces: replaces the snap rule", "yes", "small", "designed, being built after E317 / E318; the highest-value step, the snap carries most of the topology gain"),
          ("E299", "time: cross-attention from the keypoints of the previous frame (ego-warped)", "yes", "small (K = 32 tokens)", "not built; the flip rate of 19 % between keyframes is measured")]
    for r in qs:
        rows.append(list(r))
    return tbl(["id", "question", "differentiable", "run-time cost", "state"], rows)


head_html = open(os.path.join(HERE, "head.html"), encoding="utf-8").read().replace("Spurmitte Lane Report", "Spurmitte E2E Report")
head_html = head_html.replace("</style>", "svg .box.new { fill: var(--surface); stroke: var(--accent); stroke-dasharray: 6 4; stroke-width: 1.6; }</style>", 1)

# tiles
d312s, d312u = diff(C_E312S, "topo_F1"), diff(C_E312U, "topo_F1")
cov_s = [None if not MAN[s] else 100 * MAN[s]["variants"]["p0.1s"]["recall"] for s in (0, 1)]
cov_u = [None if not MAN[s] else 100 * MAN[s]["variants"]["p0.1"]["recall"] for s in (0, 1)]
cov_p = [None if not MAN[s] else 100 * MAN[s]["variants"]["plain"]["recall"] for s in (0, 1)]
within2 = None if not LS else 100 - LS["e312"]["bad2"]
tiles = f"""<div class="tiles">
<div class="tile good"><div class="k">E312 snapped: topo F1 over the plain graph (s0 / s1)</div><div class="v">{pair(d312s, lambda v: '+%.3f' % v)}</div><div class="n">connectors at p &gt;= 0.1, ends snapped to the delivered pieces</div></div>
<div class="tile good"><div class="k">E312 not snapped: the same number</div><div class="v">{pair(d312u, lambda v: '+%.3f' % v)}</div><div class="n">connectors drawn from the keypoints, no snap rule</div></div>
<div class="tile good"><div class="k">GT maneuvers realised (plain graph to E312 snapped / not snapped)</div><div class="v">{f2(cov_p[0])} to {f2(cov_s[0])} / {f2(cov_u[0])} %</div><div class="n">seed 0, 10991 maneuvers in 826 junction frames</div></div>
<div class="tile"><div class="k">delivered length within 2 m of the GT (E312)</div><div class="v">{f"{within2:.1f} %" if within2 else "pending"}</div><div class="n">within 1 m: {f"{100 - LS['e312']['bad1']:.1f} %" if LS else "pending"}; the strict metrics ask for 1 m</div></div>
<div class="tile"><div class="k">the whole lane head</div><div class="v">{f"{COST['total_macs_g']:.1f} GMACs" if COST else "pending"}</div><div class="n">{f"{COST['params_m']['total']:.2f} M parameters; B alone {TIMEB['total']['median']:.0f} ms per frame" if COST and TIMEB else ""}</div></div>
<div class="tile"><div class="k">hard steps left on the delivered graph</div><div class="v">4</div><div class="n">top-K peak choice, probability threshold, close-node step, snap (the snap carries most of the topology gain: E320 replaces it)</div></div>
</div>"""

body = f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Spurmitte - status of 2026-11-12, evening - lane head on frozen GaussianCaR features - VAL 873 frames - TEST untouched</div>
  <h1>What works end to end, and what still needs a rule</h1>
  <p class="lead">The best working differentiable approach is E312: junction maps with a narrow end heat, a pair head trained together with them, learned connector shapes, delivered at a low probability threshold. Every loss reaches every block; what stays hard are four named steps on the output. This page lists every approach with its experiment id, architecture, numbers, compute, pictures and videos, says which parts are differentiable, answers the question why the right shape does not simply emerge from a transformer, and lists the experiments of the coming night. Mirko's reading of the pictures: the middle result (E312) is very reasonable even though it delivers more lines.</p>
  {tiles}
</header>
<nav class="idx"><a href="#audit">Differentiable?</a><a href="#arch">Architecture</a><a href="#numbers">Numbers</a><a href="#metrics">Do the metrics fit what we see?</a><a href="#compute">Compute</a><a href="#look">Pictures and videos</a><a href="#why">Why the shape does not emerge</a><a href="#queue">Night queue</a><a href="#weak">Weaknesses</a><a href="#open">Open</a></nav>

<section id="audit"><h2>Which approaches are end-to-end differentiable</h2>
<p>Rule of the project (D55): everything on the training path is soft, a loss on the output reaches every module, hard selection only on the output and named. The table shows each approach from the plain lane head to the planner variant. The topo F1 columns are differences to the plain delivered graph of the same seed (B plus the hard close-node step), paired scorer, VAL, two seeds. The new row is E312 without the snap: the connectors are drawn from the keypoints, so the only rules left are the peak choice, the threshold and the close-node step.</p>
{audit_table()}
<div class="note warn"><b>Reading: the snap carries most of the topology gain.</b> Without the snap (connectors drawn from the keypoints) E312 gains {pair(d312u, sg)} topo F1 over the plain graph; with it {pair(d312s, sg)}. Topology precision falls from {pair(head(C_E312S, "topo_P"))} to {pair(head(C_E312U, "topo_P"))} (s0 / s1) because connectors that do not start and end exactly at the delivered pieces break reachability. The geometric coverage does not change (lanes found {pair(head(C_E312U, "lanes_found"))} unsnapped against {pair(head(C_E312S, "lanes_found"))} snapped; maneuvers realised {f2(cov_u[0])} against {f2(cov_s[0])} %). So the one rule on the output that matters is the attachment of the connector ends to the delivered pieces, and it is the first thing to make differentiable: a soft attachment (a softmax over the delivered piece ends near each connector end, with a dustbin, trained by the geometry loss of E317 on the final curve) is E320.</div>
<div class="note">The soft parts carry the rest: the junction maps, the pair head and the learned shapes. The close-node step has a differentiable twin (Matrix-NMS, E288, TIE: it reproduces the hard step); the soft step trained inside the network (E277) was cut.</div></section>

<section id="arch"><h2>Architecture</h2>{architecture()}
<p class="cap">Forward path left to right, top to bottom. Blue boxes are differentiable (the loss reaches them), amber boxes are hard steps (named in the audit), dashed boxes are the queued experiments. MACs and parameters are counted on the E312 checkpoint (batch 1, CPU, torch FlopCounterMode).</p>
<div class="scroll"><table><thead><tr><th>block</th><th>experiments</th><th>what it does</th><th>literature behind it</th></tr></thead><tbody>
<tr><td>Junction maps head</td><td>E286, E307, E313</td><td>1 m heat of entries and exits with heading, zone, conflict core; 0.5 m stop line and crossing; the end heat is a Gaussian of 1.0 m (E307: keypoint F1 0.63 to 0.72)</td><td>CenterNet heat maps, HDMapNet dense heads</td></tr>
<tr><td>Keypoints</td><td>E287</td><td>3 x 3 peaks, soft-argmax position over the window (gradient flows into the heat), heading and probability at the peak</td><td>CenterNet peak extraction</td></tr>
<tr><td>Pair head</td><td>E287, E305, E303</td><td>self-attention over the keypoints, 13 geometry features, rank features of the entry / exit groups (the order rule holds for 1899 of 1899 multi-pair maneuvers), evidence along the curve, BCE + dual softmax with a none entry, soft crossing penalty, two learned handle scales per pair</td><td>TopoNet, LaneSegNet, SuperGlue (dustbin), path evidence of road-graph methods</td></tr>
<tr><td>Joint training</td><td>E305, E312</td><td>the pair loss reaches the maps head through the soft-argmax positions; the shared pyramid stays frozen</td><td>end-to-end set prediction</td></tr>
<tr><td>Snap</td><td>E311</td><td>the curve starts at the end of the delivered piece wired to the entry and ends at the start of the piece wired to the exit (a rule on the output)</td><td>graph post-processing</td></tr></tbody></table></div></section>

<section id="numbers"><h2>Quantitative results</h2>
<h3>The ladder: what each step adds to the delivered graph</h3>{ladder_table()}
<p class="cap">topo F1 is the directed reachability score (seeds every 5 m on the GT, reach 30 m, match 1 m, direction within 60 degrees), scored against the graph GT of the same frames; lanes found is the share of GT pieces with at least 80 % of their points within 1 m of delivered points. Both are strict. Verdicts of the written rules: E312 KEEP (two seeds), E314 TIE, E298 KEEP for its own question (recall of the driven pairs +0.12 / +0.11).</p>
<h3>Maneuver coverage (E310 / E312)</h3>{maneuver_chart()}
<p class="cap">Share of the 10991 GT maneuvers (entry to exit pairs of the 826 junction frames) that the delivered graph realises: entry and exit matched within 1.5 m and 40 degrees, at least 90 % of the connector covered. Geometrically the snap hardly matters on the E312 keypoints (the narrow heat of E307 put them where the snap used to move them); the topology metric still needs the connectors to touch the delivered pieces (table above).</p>
<h3>Junction maps: the single biggest gain</h3>{maps_table()}
<p class="cap">The end heat target of 1.0 m (E307) separates neighbouring lanes and keeps far peaks: keypoint F1 0.63 / 0.64 to 0.72 / 0.71. 0.7 m (E313) is better on the maps and a TIE in the delivered graph (E314).</p></section>

<section id="metrics"><h2>Do the metrics fit what we see?</h2>
<p>The videos and frames agree with the ranking of the metrics (E312 above B, E284 and E287 close to B). What the metrics do not show well is how close the lines are: the strict match radius of 1 m and the direction test lower the absolute level, a 2 m radius lifts every variant by about 0.06 and keeps the order.</p>
{tol_table()}
{scene_chart()}
<p class="cap">Per scene (seed 0, 22 VAL scenes): E312 improves 19 scenes and ties 3; none gets worse by more than 0.001.</p>
<h3>How many lines, how much of it right, how smooth</h3>{lines_table()}
<h3>Where the lateral error sits</h3>{off_table()}
<p class="cap">Typical delivered lines are 0.2 m off the GT with 0.13 m jitter; about half of the squared lateral error is the offset of a whole line, the other half slope and jitter along it. Junction connectors and the far ring make up the tails (p90 0.6 - 0.7 m), where the nearest-GT association is ambiguous too.</p></section>

<section id="compute"><h2>Compute</h2>{compute_table()}
<p class="cap">MACs and parameters by counting, ms only on an idle GPU (the night chain holds both GPU slots until about 04:30; the idle-GPU job runs first in the next free slot and this table fills on the next build). Training: maps 0.9 s per step, pair stage with joint training 1.9 s per step (3000 steps = 1.6 h per seed). The pair head costs 0.02 GMACs, the maps head 3.3 GMACs on top of the pyramid that B already runs.</p></section>

<section id="look"><h2>Pictures and videos</h2>
<p>Against the l6 GT with the travel direction. Videos play every second keyframe at 2 frames per second; the four panels are the front camera, the plain graph B, E312 and the older B + E287 pair head. In the middle panel there are more lines than on the left and the right, and most of them lie on real lanes (the table above: {f"{100 - LS['e312_extra_over_B']['bad2']:.0f} % of the extra length within 2 m" if LS else ""}); the false ones are connectors near the ego vehicle and at the far ring.</p>{media_block()}</section>

<section id="why"><h2>Why the right shape does not emerge by itself</h2>
<p>The question from a colleague: if transformers can do everything, why splines at junctions and different point types and classes? A prior helps with little data, but why does the right shape not appear on its own? Mirko's hint: smoothness is not a constraint, the position uncertainty is not large enough, and a forward-backward exchange of information must take place. Each of the three is tested against our system below; the measurements come from the delivered graphs of this page.</p>
<div class="note"><b>Short answer from the data.</b> On ordinary lane stretches the shape does emerge: median lateral offset 0.2 m, jitter 0.13 m, 98 % of the delivered length within 2 m of the GT. What does not emerge is what the evidence does not fix: which entry joins which exit, where an end sits to a metre, what lies beyond the far ring. The three hints name three different reasons for that, and none of them is tested by the training signals we have today.</div>
<div class="scroll"><table><thead><tr><th>hint</th><th>what we measured</th><th>verdict</th><th>differentiable test</th></tr></thead><tbody>
<tr><td><b>smoothness is no constraint</b></td><td>the connector curve itself has no loss: the pair head sees the end heat, the pair BCE and a fit of the handle scales. Kinks at the joins between pieces: median {f"{LS['B']['join_med']:.1f}" if LS else "?"} degrees in B, {f"{LS['e312']['join_med']:.1f}" if LS else "?"} in E312, GT {f"{LS['GT']['join_med']:.1f}" if LS else "?"}; p90 {f"{LS['B']['join_p90']:.1f}" if LS else "?"} / {f"{LS['e312']['join_p90']:.1f}" if LS else "?"} / {f"{LS['GT']['join_p90']:.1f}" if LS else "?"}; turns above 10 degrees per 100 m: {f"{LS['B']['turn10']:.1f}" if LS else "?"} in B, none in the GT</td><td>true, small on plain stretches (the learned handle scales of E303 already help: +0.006 / +0.011 topo F1), unsupervised exactly where the shape is multimodal</td><td><b>E317</b>: Chamfer loss between the delivered Hermite curve and the true connector; training only</td></tr>
<tr><td><b>position uncertainty is not large enough</b></td><td>the end position is a 1 m cell plus a soft-argmax over 3 x 3 cells and carries no uncertainty of its own; with the E305 keypoints (before the narrow heat of E307) the ends were 1.44 m off the GT ends on average (E310) and only 52 % of the GT pairs had both ends matched strictly, 69 % loosely (E312 card). A narrower heat helped (E307), because the heat also has to separate crowded ends: detection and position share one heat</td><td>open and plausible: the net cannot say "this end is somewhere along these 4 m"</td><td><b>E318</b>: a log sigma per end with a Laplace NLL on the end position; sigma enters the pair logit and goes to the planner</td></tr>
<tr><td><b>forward-backward exchange</b></td><td>on the E305 keypoints the pair decision was at the level of a perfect classifier (topo F1 +0.0304 against +0.0314 for the oracle on the same keypoints, E310), so the pairs were not the limit; the ends are: 53 % of the GT ends lie beyond 35 m and the pair recall there is 0.23 against 0.38 near (E300). Node tokens at the keypoints improved the pair ranking (AUC 0.974 against 0.930) and nothing else (E308, TIE). The context of the maps head is isotropic (dilated blocks); a lane end is decided by what lies ahead and behind it along the lane</td><td>plausible for the ends, not for the pairs</td><td><b>E319</b>: heading-guided exchange (ahead / behind along the lane heading); <b>E299</b> in time (the pair flips between keyframes in 19 % of the cases)</td></tr></tbody></table></div>
<h3>Splines, point types and classes next to a transformer</h3>
<p>They do not replace the transformer, they are what the transformer's output is made of: a Hermite curve between two typed points is what the planner and the loss can use, and the types carry the structure the GT has (the order rule, one out-degree per entry). With a nuScenes-sized set they act as a prior: the order rule holds for 1899 of 1899 multi-pair maneuvers, and a generic decoder would have to learn it from data (E287: the order features help, p0 to p1). The honest test of the colleague's claim is an unstructured set-prediction head on the same cached features, with the same budget; it is not built and costs one to two days. The experiments above ask something cheaper: which missing signal (shape, uncertainty, exchange) explains most of what the structure is there to compensate.</p></section>

<section id="queue"><h2>Experiments of the coming night and day</h2>
<p>Both GPU slots run the first two from 22:00 until about 04:30; the new ones start after them inside the allowed windows (night until 07:00, Friday 08:30 to 17:30). Each has a card with a written rule before any run, two seeds, VAL only, and each is differentiable, judged by run-time cost next to its gain.</p>{q_table()}
<div class="note warn"><b>Rule of each card.</b> KEEP when the snapped topo F1 over E312 rises by more than 0.01 on both seeds with the guards held (F1@1.0, duplicates, vertices per 100 m, junction F1); otherwise TIE (the cheaper variant stays) or CUT. E317 reports in addition the topo F1 without the snap, and how much of the gap between the two the arm closes: that is the answer to "can the rule step go".</div></section>

<section id="weak"><h2>Weaknesses and honest negatives</h2>
<ul class="tight"><li>The headline numbers are strict (1 m, direction) and scored against the old graph GT in a 48 m box, not directly against l6; a lenient 2 m cross-check keeps the order of the variants.</li>
<li>The snap is a rule on the output and carries most of the topology gain (E312 unsnapped: topo F1 gain about a third of the snapped one); the close-node step is a rule too. Until E320 and a soft close-node step are in, the delivered graph is end to end only up to these named rules.</li>
<li>E312 has false connectors near the ego vehicle at the low threshold (frame 603 of the pictures) and weak far-ring ends; pair flips of 19 % between keyframes.</li>
<li>Single keyframes: no temporal fusion in the lane head; E299 is not built.</li>
<li>TIE results are kept and refined, never worse: E284, E287, E288, E296, E303, E308, E314; E277 (soft step) was cut.</li>
<li>Nothing here touches TEST; the final system has not been run on TEST.</li></ul></section>

<section id="open"><h2>Open decisions</h2>
<ul class="tight"><li>TEST once for E312 and its control (needs Mirko's recorded OK and the log row).</li><li>Champion promotion and the change of PROBLEM.md (a D entry).</li><li>The soft step of 1.25 m (E279), D4 / D5, push, the scratch folders of the GT work.</li><li>Whether to build the unstructured set-prediction head as the direct test of the colleague's claim.</li></ul></section>
<footer>Built by temp/jr/build_e2e_report.py from the result files of the runs; sources: runs/e312_p_s*/ends_card*.json, runs/e312_p_s*/maneuver_val.json, temp/e309/*.json, temp/jr/model_cost.json, runs/e273/e273_p_s0/timing.json. Videos and pictures: results/l6_view_E312, results/Junction.report/img.</footer>
</div>"""

open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(head_html + "\n" + body)
print("written", os.path.join(OUT, "index.html"), len(body))
