"""one-off: build the report of the junction thread (D82 / D83, E285 - E303): approach, architecture, ground truth, numbers, qualitative results, pictures and videos.

    wsl bash tools/env/_py14.sh build_junction_report.py [--media]       CPU; --media transcodes the pictures / videos in temp/jr/media (made by make_media.sh) into the report folder
writes results/Junction.report/{index.html (local, relative links), page.html (body for an artifact), img/, vid/}
Every number is read from the result files of the runs (runs/**.json) and of the scratch analyses (temp/gt2, temp/e29x); a missing file prints "pending".
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.join(ROOT, "temp", "jr")
sys.path.insert(0, HERE)
import svgcharts as C  # noqa: E402

OUT = os.path.join(ROOT, "results", "Junction.report")
IMG, VID = os.path.join(OUT, "img"), os.path.join(OUT, "vid")
for d in (IMG, VID):
    os.makedirs(d, exist_ok=True)
MEDIA = "--media" in sys.argv


def J(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    try:
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    except ValueError:                                                                                              # an empty / half written result file: treated as missing
        return None


def jpg(src, name, q=84, maxw=1900):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(os.path.join(IMG, name), quality=q, optimize=True)
    return "img/" + name


def webm(src, name, width=1280, crf=36):
    import cv2
    dst = os.path.join(VID, name)
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return "vid/" + name
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
    print(name, round(os.path.getsize(dst) / 1e6, 2), "MB", flush=True)
    return "vid/" + name


def f3(v):
    return "pending" if v is None else f"{v:.3f}"


def f2(v):
    return "pending" if v is None else f"{v:.2f}"


def sg(v):
    return "pending" if v is None else f"{v:+.4f}"


def tbl(head, rows, num=()):
    h = "".join(f'<th class="{"num" if i in num else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig(src, cap, vid=False, poster=None):
    if vid:
        return f'<figure><video controls preload="none" loop playsinline {"poster=%s" % chr(34) + poster + chr(34) if poster else ""} src="{src}"></video><p class="cap">{cap}</p></figure>'
    return f'<figure><a href="{src}" target="_blank" rel="noopener"><img src="{src}" loading="lazy" alt=""></a><p class="cap">{cap}</p></figure>'


# ------------------------------------------------------------------------------------------------------------------ data
E286 = {(a, s): J(f"runs/e286_{a}_s{s}/jmaps_val.json") for a in ("j0", "j1", "j2", "jj") for s in (0, 1)}
E307 = {s: J(f"runs/e307_n_s{s}/jmaps_val.json") for s in (0, 1)}
E313 = {s: J(f"runs/e313_s_s{s}/jmaps_val.json") for s in (0, 1)}                                   # E313: heat sigma 0.7 m                                   # E307: the dilated arm with heat sigma 1.0 m
CBASE = J("temp/gt2/dummy_runs/baseline_c2_jmaps_val.json") or J("temp/gt2/dummy_runs/baseline_c_jmaps_val.json")
E287 = {(a, s): (J(f"runs/e287_{a}_s{s}/ends_head.json"), J(f"runs/e287_{a}_s{s}/ends_card.json")) for a in ("p0", "p1", "p2", "q2") for s in (0, 1)}
E287.update({("e308", s): (J(f"runs/e308_t_s{s}/ends_head.json"), J(f"runs/e308_t_s{s}/ends_card.json")) for s in (0, 1)})   # E308: p2 + the node tokens of B
E284 = {s: (J(f"runs/e284_h2_s{s}/ends_head.json"), J(f"runs/e284_h2_s{s}/ends_card.json")) for s in (0, 1)}
THR = {}                                                                                                         # E309: the delivery threshold sweep (scratch json of temp/gt2/thr_sweep.py)
for _p in sorted(glob.glob(os.path.join(HERE, "..", "gt2", "thr", "*.json"))):
    _d = J(_p)
    if _d:
        THR[(_d["run"], _d["thr"])] = _d
import glob as _glob
RULE = {}
for _p in sorted(_glob.glob(os.path.join(ROOT, "runs/e287/rule_control_thr0.5_*.json"))):                           # strict one-to-one matching, keypoint threshold 0.5 (critic fix of 2026-11-12)
    RULE.update(json.load(open(_p)))
GATES = J("temp/gt2/gates_e285_val.json")
JST = J("temp/gt2/jstats_val.json")
JST2 = J("temp/gt2/jstats2_val.json")
JPAINT = J("temp/gt2/jpainted_val.json")
JDEG = J("temp/gt2/jdeg_val.json")
CMP = J("temp/gt2/cmp_l4_l6.json")
REP288 = {s: (J(f"runs/e288/replay_s{s}.json"), J(f"runs/e288/replay_s{s}_refine.json")) for s in (0, 1)}
REP296 = {s: (J(f"runs/e296/replay_s{s}.json"), J(f"runs/e296/replay_s{s}_refine.json")) for s in (0, 1)}
PROBE = {s: J(f"runs/e289/shift_probe_s{s}.json") for s in (0, 1)}
PROBE2 = {s: J(f"runs/e289/shift_probe2_s{s}.json") for s in (0, 1)}
FLIP = {s: J(f"runs/e290/pair_flip_e284_h2_s{s}.json") for s in (0, 1)}
DEPTH = J("runs/e293/depth_probe_s0.json")
SPLIT = {s: (J("runs/e300/splits_thr0.5_e284_h2_s0_e284_h2_s1.json") or {}).get(f"e284_h2_s{s}") for s in (0, 1)}


def kp_f1(res, who="ours", tol="3.0"):
    """mean of the entry and exit F1 at 3 m, thr 0.3, distance only (the card's number); who = ours | B"""
    if not res:
        return None
    k = res["keypoints"]
    a, b = k[f"kp_{who}_tol{tol}_noang_ent"]["f1"], k[f"kp_{who}_tol{tol}_noang_ext"]["f1"]
    return (a + b) / 2


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


C_BASE = kp_f1(CBASE, "B")
ARM_SCORE = {a: mean([kp_f1(E286[(a, s)]) for s in (0, 1)]) for a in ("j0", "j1", "j2", "jj")}
BEST = max((a for a in ARM_SCORE if ARM_SCORE[a] is not None), key=lambda a: ARM_SCORE[a], default="j1")

# ------------------------------------------------------------------------------------------------------------------ architecture (inline SVG, by code)


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
    s = ['<svg class="chart" viewBox="0 0 1130 470" role="img" xmlns="http://www.w3.org/2000/svg">']
    s.append(box(10, 20, 170, 74, "Camera + radar", "GaussianCaR BEV features|frozen, cached: 39 ch,|200 x 200 at 0.5 m", "box map"))
    s.append(box(220, 20, 190, 74, "Pyramid (B, E273)", "0.5 / 1 / 2 m features,|frozen (E286 j*) or|trained jointly (jj)", "box"))
    s.append(box(450, 20, 270, 100, "Junction maps head (E286)", "1 m: zone, conflict core,|entry / exit heat + heading|0.5 m: stop line, crossing|context: dilated blocks (dil) / attention (attn)", "box hot"))
    s.append(box(760, 20, 170, 74, "Keypoints", "3 x 3 peaks, soft-argmax|(the one hard step,|named: index choice)", "box"))
    s.append(box(960, 20, 160, 100, "Pair head v2 (E287)", "tokens + self-attention|geometry + rank features|(order rule) + evidence|along the curve; crossing|penalty; dual softmax", "box hot"))
    s.append(box(960, 150, 160, 64, "Connectors", "Hermite curve per pair|(chord tangent; E303:|learned handle scales)", "box"))
    s.append(box(220, 170, 190, 90, "Node head (B)", "2 m grid: existence, offset,|8 type flags, heading,|section start", "box"))
    s.append(box(450, 170, 190, 90, "Links + sections (B)", "short / long links, Hermite|handles, soft sections|(soft absorbing walk)", "box"))
    s.append(box(680, 170, 230, 90, "Pruning", "hard close-node step 1.5 m;|Matrix-NMS R 1.75 s 0.1 = the same,|differentiable (E288 / E292); x0.9", "box"))
    s.append(box(450, 300, 270, 74, "Map assignment (E273)", "lane-level map with errors; soft|registration, dustbin, cross-attention|SD level with lane counts: E291", "box map"))
    s.append(box(760, 300, 360, 74, "Delivered lane graph", "B's nodes / links / sections + the head's connectors|output stage: Douglas-Peucker 0.3 m (14.3 vertices / 100 m budget)", "box hot"))
    s.append(box(10, 330, 400, 120, "Ground truth on the lane side (E285, D82 / D83)", "lane GT l6 (GtGenerator, store 9ad1d4ce30) -> junction GT layer:|zone, conflict core, arms with lane counts and ranks, entry / exit points,|order-preserving pairs (driven = tier 1), painted evidence (stop lines,|crossings from the map expansion), visibility; targets by location only", "box map"))
    for a in ((180, 57, 220, 57), (410, 57, 450, 57), (720, 57, 760, 57), (930, 57, 960, 57), (1040, 120, 1040, 150), (315, 94, 315, 170), (410, 215, 450, 215), (640, 215, 680, 215)):
        s.append(arrow(*a))
    s.append(arrow(910, 215, 940, 330 if False else 300))
    s.append(arrow(1040, 214, 1040, 300))
    s.append(arrow(585, 260, 585, 300))
    s.append(arrow(410, 390, 450, 390 - 20))
    s.append('<text class="bs" x="10" y="14">forward path: soft up to the keypoint read-out and the delivered thresholds (D55); every box has a gradient report in its self-test</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------------------------------------------------------ numbers
def e286_rows():
    rows = []
    cb = CBASE
    if cb:
        rows.append(["<b>c</b> B's own stop / exit event heat", f3(cb["keypoints"]["kp_B_tol3.0_noang_ent"]["recall"]), f3(cb["keypoints"]["kp_B_tol3.0_noang_ent"]["precision"]), f3(cb["keypoints"]["kp_B_tol3.0_noang_ext"]["recall"]),
                     f3(cb["keypoints"]["kp_B_tol3.0_noang_ext"]["precision"]), f"<b>{f3(C_BASE)}</b>", "-", "-", "-"])
    for a in ("j0", "j1", "j2", "jj"):
        for s in (0, 1):
            r = E286[(a, s)]
            if not r:
                rows.append([f"{a} seed {s}"] + ["pending"] * 8)
                continue
            k = r["keypoints"]
            rows.append([f"<b>{a}</b> seed {s}", f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ent"]["precision"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["precision"]),
                         f"<b>{f3(kp_f1(r))}</b>", f3(r["zone_iou"]), f3(r["core_iou"]), f"{f2(r['stop_prec'])} / {f2(r['stop_rec'])}"])
    for s in (0, 1):
        r = E307[s]
        if r:
            k = r["keypoints"]
            rows.append([f"<b>E307: j1 with heat sigma 1.0 m</b> seed {s}", f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ent"]["precision"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["precision"]),
                         f"<b>{f3(kp_f1(r))}</b>", f3(r["zone_iou"]), f3(r["core_iou"]), f"{f2(r['stop_prec'])} / {f2(r['stop_rec'])}"])
    for s_ in (0, 1):
        r = E313[s_]
        if r:
            k = r["keypoints"]
            rows.append([f"<b>E313: heat sigma 0.7 m</b> seed {s_}", f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ent"]["precision"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["precision"]),
                         f"<b>{f3(kp_f1(r))}</b>", f3(r["zone_iou"]), f3(r["core_iou"]), f"{f2(r['stop_prec'])} / {f2(r['stop_rec'])}"])
    return tbl(["arm", "entry R", "entry P", "exit R", "exit P", "kp_f1", "zone IoU", "core IoU", "stop line P / R"], rows, num=range(1, 9))


def split_chart():
    r = E286.get((BEST, 0)) or E286.get(("j1", 0))
    if not r or not CBASE:
        return "<p>pending</p>"
    ours, base = r["range_city_split"], CBASE.get("range_city_split")
    if not base:
        return "<p>pending</p>"
    groups, vo, vb = [], [], []
    for lab, nm in (("within 20 m", "lt20"), ("20 - 35 m", "20to35"), ("beyond 35 m", "gt35")):
        for typ in ("ent", "ext"):
            groups.append(f"{typ} {lab}")
            vo.append(ours[f"range_ours_{typ}_{nm}"]["recall"])
            vb.append(base[f"range_B_{typ}_{nm}"]["recall"])
    for city in ("boston", "singapore"):
        for typ in ("ent", "ext"):
            groups.append(f"{typ} {city}")
            vo.append(ours[f"city_ours_{typ}_{city}"]["recall"])
            vb.append(base[f"city_B_{typ}_{city}"]["recall"])
    ps = r["painted_split"]
    for typ in ("ent", "ext"):
        groups += [f"{typ} painted", f"{typ} unpainted"]
        vo += [ps[f"ours_{typ}"]["painted_recall"], ps[f"ours_{typ}"]["unpainted_recall"]]
        vb += [CBASE["painted_split"][f"B_{typ}"]["painted_recall"], CBASE["painted_split"][f"B_{typ}"]["unpainted_recall"]]
    return C.bars(groups, ["B's event heat (control)", f"junction maps {BEST} (seed 0)"], [vb, vo], ymax=1.0, width=1120, height=300, label_every=True, ylabel="recall of the GT ends (thr 0.3, 3 m)")


def e287_rows():
    rows = []
    for s in (0, 1):
        h, c = E284[s]
        if h:
            n = c["numbers"]["head"] if c else {}
            d = c["difference"] if c else {}
            rows.append([f"E284 end head (node tokens) seed {s}", f3(h["ent_recall"]), f3(h["ext_recall"]), f3(h["pair_recall"]), f3(h["pair_precision"]), f3(n.get("topo_F1")), sg(d.get("topo_F1")), f3(n.get("F1@1.0")), f3(n.get("lanes_found"))])
    if RULE:
        for run, v in RULE.items():
            rows.append([f"rule control on the keypoints of {run} (ceiling {f3(v['ceiling_both_ends_matched'])})", "-", "-", f3(v["rule"]["recall"]), f3(v["rule"]["precision"]), "-", "-", "-", "-"])
    for a in ("p0", "p1", "p2", "e308"):
        for s in (0, 1):
            h, c = E287[(a, s)]
            if not h:
                rows.append([f"<b>{a}</b> seed {s}"] + ["pending"] * 8)
                continue
            n = c["numbers"]["head"] if c else {}
            d = c["difference"] if c else {}
            rows.append([f"<b>{a}</b> seed {s}", f3(h["ent_recall"]), f3(h["ext_recall"]), f3(h["pair_recall"]), f3(h["pair_precision"]), f3(n.get("topo_F1")), sg(d.get("topo_F1")), f3(n.get("F1@1.0")), f3(n.get("lanes_found"))])
    return tbl(["head", "entry recall", "exit recall", "pair recall", "pair precision", "topo F1 (graph + connectors)", "vs plain graph", "F1@1.0", "lanes found"], rows, num=range(1, 9))


def depth_chart():
    if not DEPTH:
        return "<p>pending</p>"
    ks = list(DEPTH["by_depth_m"].keys())
    xs = list(range(len(ks)))
    true = [DEPTH["by_depth_m"][k]["heat_ge_0_3"] for k in ks]
    false = [DEPTH["false_pair_control_by_depth_m"][k]["heat_ge_0_3"] for k in ks]
    herm = [DEPTH["by_depth_m"][k]["hermite_within_1m"] for k in ks]
    paint = [DEPTH["by_depth_m"][k]["paint_within_1m"] for k in ks]
    return C.bars([k.replace("-inf", "+") + " m" for k in ks], ["lane heat >= 0.3 on the TRUE connector", "lane heat >= 0.3 on FALSE candidate paths", "Hermite (chord tangent) within 1 m", "painted evidence within 1 m"], [true, false, herm, paint], ymax=1.0, width=1120, height=320, label_every=True,
                  ylabel="share of the points at this depth from the entry")


def e288_table():
    rows = []
    for text in ("close:1.5@0.5", "plain@0.5", "mnms:1.5@0.5", "mnms:2.0@0.5", "mnms:1.75s0.2@0.5", "mnms:1.75s0.1@0.5"):
        r = [text]
        for s in (0, 1):
            found = None
            for src in REP288[s]:
                if src:
                    for row in src["rows"]:
                        if row["spec"] == text:
                            found = row
            r += [f3(found["topo_F1"]), f3(found["junction_F1"]), f3(found["lanes_found"]), f2(found["vert100_dp"])] if found else ["-"] * 4
        rows.append(r)
    return tbl(["read-out", "s0 topo F1", "s0 junction F1", "s0 lanes", "s0 vert/100m", "s1 topo F1", "s1 junction F1", "s1 lanes", "s1 vert/100m"], rows, num=range(1, 9))


def e296_table():
    rows = []
    for text in ("close:1.5@0.5", "close:1.5x@0.5", "close:1.5x0.7@0.5", "close:1.5x0.9@0.5", "mnms:1.75s0.1@0.5", "mnms:1.75s0.1x@0.5"):
        r = [text]
        for s in (0, 1):
            found = None
            for src in list(REP296[s]) + list(REP288[s]):
                if src:
                    for row in src["rows"]:
                        if row["spec"] == text:
                            found = row
            r += [f3(found["topo_F1"]), f3(found["junction_F1"]), f3(found["junction_R"]), f3(found["dup_frac"])] if found else ["-"] * 4
        rows.append(r)
    return tbl(["read-out", "s0 topo F1", "s0 junction F1", "s0 junction R", "s0 dup", "s1 topo F1", "s1 junction F1", "s1 junction R", "s1 dup"], rows, num=range(1, 9))


def split_table():
    s0 = SPLIT.get(0)
    if not s0:
        return "<p>pending</p>"
    s1 = SPLIT.get(1)
    rows = []
    for k, v in s0["pairs"].items():
        rows.append([k, v["n"], f3(v["recall"]), f3(s1["pairs"][k]["recall"]) if s1 and k in s1["pairs"] else "-"])
    return tbl(["split of the true pairs", "n", "pair recall seed 0", "seed 1"], rows, num=(1, 2, 3))


def tiles():
    t = []
    best = ARM_SCORE.get(BEST)
    t.append(("good" if best and best > 0.5 else "", f3(best), f"kp_f1 of the junction maps ({BEST}, mean of 2 seeds)", f"B's own event heat: {f3(C_BASE)}"))
    p2 = mean([(E287[("p2", s)][0] or {}).get("pair_recall") for s in (0, 1)])
    t.append(("", f3(p2), "pair recall of the pair head p2 (E287)", "E284 end head: 0.29 / 0.29 (precision 0.74); rule control 0.36 (precision 0.34); ceiling of the keypoints 0.43"))
    t.append(("good", f"{GATES['mono'] if GATES else '-'} / {GATES['multi'] if GATES else '-'}", "multi-pair maneuvers that keep the order", "GT, VAL 873 frames (identity in 66 %)"))
    t.append(("", "38 %", "of the junction zone is the conflict core", "the zone is about 2.6 times the area where paths cross, merge or diverge"))
    t.append(("good", "+0.057", "topo F1 over the plain graph (E312: narrower end heat, handle scales, snapped connector ends, p >= 0.1, seed 0; seed 1 +0.062)", "the pair head alone, connectors from the keypoints at p >= 0.5: +0.006 (E287)"))
    t.append(("good", "TIE", "Matrix-NMS = the hard close-node step", "differentiable: R 1.75, sigma 0.1"))
    return "".join(f'<div class="tile {c}"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>' for c, v, k, n in t)


# ------------------------------------------------------------------------------------------------------------------ media
def media_block():
    if not MEDIA:
        return "<p>pending (run with --media)</p>"
    out = []
    srcs = sorted(glob.glob(os.path.join(HERE, "media", "frames", "*.png")))
    names = {}
    for p in srcs:
        names[os.path.basename(p)] = jpg(p, os.path.basename(p).replace(".png", ".jpg"))
    caps = J(os.path.join(HERE, "media", "captions.json")) or {}
    for fn, rel in names.items():
        out.append(fig(rel, caps.get(fn, fn)))
    vids = sorted(glob.glob(os.path.join(HERE, "media", "videos", "*.mp4")))
    vh = []
    for p in vids:
        nm = os.path.basename(p).replace(".mp4", ".webm")
        rel = webm(p, nm)
        vh.append(fig(rel, caps.get(os.path.basename(p), os.path.basename(p)), vid=True))
    return "".join(out) + (f'<div class="vids">{"".join(vh)}</div>' if vh else "")


def gt_images():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "media", "gt", "*.png"))):
        out.append(fig(jpg(p, "gt_" + os.path.basename(p).replace(".png", ".jpg")), os.path.basename(p)))
    return "".join(out)


def common_table():
    """the pair heads on the SAME ends: the output of temp/gt2/pair_diag_common.py (candidates whose two ends are GT ends that all three systems found, strict one-to-one matching)"""
    import re
    rows = []
    for s in (0, 1):
        p = os.path.join(HERE, "..", "gt2", f"common_s{s}.log")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        n_common = re.search(r"with both ends found by all \d+ systems: (\d+)", txt)
        for m in re.finditer(r"^(\S+)\s+candidates\s+(\d+), true\s+(\d+), AUC ([\d.]+), at logit >= 0: recall ([\d.]+) precision ([\d.]+); precision at the recall of \S+ \(([\d.]+)\): ([\d.]+)", txt, re.M):
            name = {"e284": "E284 end head", "e287": "E287 p2", "e308": "E308 (p2 + node tokens)"}[m.group(1)[:4]]
            rows.append([f"{name}, seed {s}", n_common.group(1) if n_common else "-", m.group(2), f3(float(m.group(4))), f3(float(m.group(5))), f3(float(m.group(6))), f3(float(m.group(8)))])
    return tbl(["head", "GT pairs with both ends found by all", "candidates", "AUC", "recall at p >= 0.5", "precision at p >= 0.5", "precision at the recall of the E284 head"], rows, num=range(1, 7))


def thr_table():
    """E309: the delivery threshold; every number is a difference to the plain delivered graph of the same seed (the head's own ends_card at 0.5 for the first row of a run)"""
    rows = []
    keys = ("lanes_found", "junction_F1", "junction_R", "dup_frac", "F1@1.0")
    for run in sorted({k[0] for k in THR}):
        for thr in sorted({k[1] for k in THR if k[0] == run}, reverse=True):
            d = THR[(run, thr)]
            n, b = d["numbers"]["head"], d["numbers"]["base"]
            rows.append([run, f"{thr:.1f}", str(d["curves"]), f3(n["topo_F1"]), sg(n["topo_F1"] - b["topo_F1"])] + [sg(n[k] - b[k]) for k in keys])
    if not rows:
        return "<p>pending</p>"
    return tbl(["run", "delivery threshold", "curves", "topo F1", "vs plain graph", "lanes found", "junction F1", "junction recall", "dup_frac", "F1@1.0"], rows, num=range(1, 10))


def errors_block(dirname="errors", prefix="err_", label="", maneuver=True):
    """the failure figures of temp/jr/fail_gallery.py (and maneuver coverage) and the error videos of one read-out; dirname = temp/jr/media/<dirname>"""
    ed = os.path.join(HERE, "media", dirname)
    caps = [("distribution", f"{label}Distribution over the 873 VAL frames: coverage of the GT lanes by the delivered graph (left), true pairs delivered per frame (middle), pair probability of true and false candidates (right; log scale)."),
            ("random12", f"{label}Twelve random junction frames: the delivered graph against the GT (green near GT, red with no GT, blue GT not covered)."),
            ("misses", f"{label}Largest junctions, most missed pairs. Left GT lanes with ends; middle the delivered graph against the GT; right the pairs (green true, red false, blue dashed missed)."),
            ("worst_coverage", f"{label}Lowest coverage: whole arms of the junction are missing from the delivered graph."),
            ("false_pairs", f"{label}Most false connectors."),
            ("typical", f"{label}A typical junction frame.")]
    out = []
    for n, c in caps:
        pth = os.path.join(ed, n + ".png")
        if os.path.exists(pth):
            out.append(fig(jpg(pth, prefix + n + ".jpg", maxw=1800), c))
    for pth in sorted(glob.glob(os.path.join(ed, "*_errors.webm"))):
        n = os.path.basename(pth)
        shutil.copy(pth, os.path.join(VID, prefix + n))
        out.append(fig("vid/" + prefix + n, f"{label}Video: " + n.replace("_errors.webm", "") + ", three panels per keyframe: GT lanes with ends, delivered graph against GT, pairs.", vid=True))
    return "".join(out)


def maneuver_block():
    """E310 / E311: maneuver coverage per variant (runs/<run>/maneuver_val.json of evaluation/jmaneuver.py) as a table and a bar chart"""
    rows, bars = [], []
    for run, label in (("e308_t_s0", "E308 seed 0"), ("e305_e_s0", "E305 seed 0"), ("e312_p_s0", "E312 seed 0"), ("e312_p_s1", "E312 seed 1"), ("e314_p_s0", "E314 seed 0"), ("e314_p_s1", "E314 seed 1")):
        d = J(f"runs/{run}/maneuver_val.json")
        if not d:
            continue
        for v, r in d["variants"].items():
            rows.append([label, v, f"{r['realised']} / {r['maneuvers']}", f3(r["recall"]), f3(r["junction_precision"]), str(r["frames_none"])])
            bars.append((label + " " + v, r["recall"]))
    if not rows:
        return "<p>pending</p>"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, ax = plt.subplots(figsize=(11, 4.2))
    cols = ["#7f7f7f" if "plain" in b else ("#ff7f0e" if "oracle" in b else ("#006400" if b.endswith("s") else "#2ca02c")) for b, _ in bars]
    ax.bar(range(len(bars)), [v for _, v in bars], color=cols)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b for b, _ in bars], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("share of GT maneuvers realised")
    ax.set_title("maneuver coverage of the delivered graph (grey plain, green connectors, dark green snapped ends, orange perfect pairs)", fontsize=9)
    fig_.tight_layout()
    pth = os.path.join(HERE, "media", "maneuver_bars.png")
    fig_.savefig(pth, dpi=80)
    plt.close(fig_)
    return tbl(["run", "delivered graph", "maneuvers realised", "share", "junction precision", "frames with none"], rows, num=range(2, 6)) + fig(jpg(pth, "maneuver_bars.jpg"), "Maneuver coverage per variant (evaluation/jmaneuver.py).")


def ladder_table():
    """the ladder of the thread: the delivered graph's gain over the plain graph, step by step, every cell read from the card of the run (both seeds)"""
    def cd(path):
        return J(path)

    steps = [("E287 p2: pair head, connectors from the keypoints, p >= 0.5", "runs/e287_p2_s{s}/ends_card.json"),
             ("E305: maps and pairs trained together, same read-out", "runs/e305_e_s{s}/ends_card.json"),
             ("+ connector ends snapped, p >= 0.1 (E311)", "temp/gt2/snap/e305_e_s{s}_thr0.1.json"),
             ("E303: + learned handle scales (E303 pairs on the E286 maps)", "runs/e303_s_s{s}/ends_card_snap_thr0.1.json"),
             ("E312 with chord curves instead of the learned scales (the same pairs)", "runs/e312_p_s{s}/ends_card_snap_chord_thr0.1.json"),
             ("<b>E312: + the narrower end heat E307 (heat sigma 1.0 m), scales, snapped, p >= 0.1 (the final system)</b>", "runs/e312_p_s{s}/ends_card_snap_thr0.1.json"),
             ("E314: the same on the sigma 0.7 m maps of E313 (TIE with E312)", "runs/e314_p_s{s}/ends_card_snap_thr0.1.json"),
             ("E298: E312 with the driven pairs weighted higher (the planner variant; driven pair recall +0.12 / +0.11)", "runs/e298_w_s{s}/ends_card_snap_thr0.1.json")]
    rows = []
    for name, pat in steps:
        cs = [cd(pat.format(s=s)) for s in (0, 1)]
        if not all(cs):
            rows.append([name] + ["pending"] * 6)
            continue
        def cell(k):
            return " / ".join(sg(c["difference"][k]) for c in cs)
        rows.append([name, " / ".join(f3(c["numbers"]["head"]["topo_F1"]) for c in cs), cell("topo_F1"), cell("lanes_found"), cell("junction_F1"), cell("junction_R"), cell("dup_frac")])
    return tbl(["step (the delivered graph = B + the hard close-node step + the connectors of the pair head)", "topo F1 (seed 0 / seed 1)", "topo F1 vs the plain graph", "lanes found", "junction F1", "junction recall", "dup_frac"], rows, num=range(1, 7))


def snap_table():
    """E311: the snapped connectors, every run / threshold scored by temp/gt2/snap_score.py (differences to the plain graph of the same seed); next to them the same pairs drawn from the keypoints (E309 sweep or the card of the run) when stored"""
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "gt2", "snap", "*.json"))):
        d = J(f)
        if not d:
            continue
        n, b = d["numbers"]["head"], d["numbers"]["base"]
        un = THR.get((d["run"], d["thr"]))
        if un is None:
            for nm in (f"ends_card_thr{d['thr']}.json", "ends_card.json"):                                         # the unsnapped card of this threshold (runs/<run>/, written by the scorer itself since 2026-11-12) or the card of record
                c = J(f"runs/{d['run']}/{nm}")
                if c and abs(c.get("thr", 0.5) - d["thr"]) < 1e-9 and not c.get("snap"):
                    un = c
                    break
        un_t = sg(un["numbers"]["head"]["topo_F1"] - un["numbers"]["base"]["topo_F1"]) if un else "-"
        rows.append([d["run"], f"{d['thr']:.1f}", str(d["curves"]), f3(n["topo_F1"]), sg(n["topo_F1"] - b["topo_F1"]), un_t, sg(n["lanes_found"] - b["lanes_found"]), sg(n["junction_F1"] - b["junction_F1"]), sg(n["junction_R"] - b["junction_R"]),
                     sg(n["dup_frac"] - b["dup_frac"]), sg(n["F1@1.0"] - b["F1@1.0"]), sg(n["vert100_dp"] - b["vert100_dp"])])
    rows.sort(key=lambda r: (r[0], -float(r[1])))
    if not rows:
        return "<p>pending</p>"
    return tbl(["run", "p >=", "curves", "topo F1", "vs plain graph", "(connectors from the keypoints)", "lanes found", "junction F1", "junction recall", "dup_frac", "F1@1.0", "vertices / 100 m"], rows, num=range(1, 12))


exec(open(os.path.join(HERE, "report_text.py"), encoding="utf-8").read())      # the sections (LEAD, STATUS, ...) written by hand, using the helpers above

head = open(os.path.join(HERE, "head.html"), encoding="utf-8").read().replace("Spurmitte Lane Report", "Spurmitte Junction Report")
body = f'''<div class="wrap">
<header class="top">
  <div class="eyebrow">Spurmitte - the junction thread of 2026-11-11 (D82 / D83) - lane head on frozen GaussianCaR features - VAL 873 frames - TEST untouched</div>
  <h1>Junctions: a ground truth with everything, junction maps, entry-to-exit pairs</h1>
  <p class="lead">{LEAD}</p>
  <div class="tiles">{tiles()}</div>
</header>
<nav class="idx"><a href="#status">Status board</a><a href="#approach">Approach</a><a href="#arch">Architecture</a><a href="#gt">Ground truth</a><a href="#maps">Junction maps</a><a href="#pairs">Pairs</a><a href="#prune">Pruning, stability, depth</a><a href="#fail">Where it fails</a><a href="#look">Pictures and videos</a><a href="#weak">Weaknesses</a><a href="#next">Next</a><a href="#versions">Versions</a></nav>
<section id="status"><h2>Status board</h2>{STATUS}</section>
<section id="approach"><h2>Approach: from the brief to experiments</h2>{APPROACH}</section>
<section id="arch"><h2>Architecture, with the literature behind each block</h2>{architecture()}{ARCH}{ARCH2}</section>
<section id="gt"><h2>Ground truth: what the layer holds and what it measured</h2>{GT}{gt_images()}</section>
<section id="maps"><h2>Junction maps (E286)</h2>{MAPS}{e286_rows()}{MAPS3}{MAPS4}<h3>Where the gain is: range, city, paint</h3>{split_chart()}{MAPS2}</section>
<section id="pairs"><h2>Entry-to-exit pairs (E287, E284, rule control, E300)</h2>{PAIRS}{e287_rows()}<h3>Where the pair head fails (E300, E284 head)</h3>{split_table()}{PAIRS2}<h3>The same ends, the same pairs: the E284 head against the new heads</h3>{common_table()}{PAIRS3}<h3>E309: the delivery threshold of the pairs (differences to the plain delivered graph)</h3>{thr_table()}{PAIRS4}<h3>The ladder of the thread: what each step adds to the delivered graph</h3>{ladder_table()}{PAIRS6}{DRIVEN}<h3>E311: snap the connector ends to the delivered lane pieces (differences to the plain delivered graph)</h3>{snap_table()}{PAIRS5}</section>
<section id="prune"><h2>Pruning, stability, how far into the junction</h2>{PRUNE}<h3>E288: the close-node step as a smooth operator</h3>{e288_table()}<h3>E296: protect the special types</h3>{e296_table()}<h3>E293: evidence and interpolation by depth into the junction</h3>{depth_chart()}{PRUNE2}</section>
<section id="fail"><h2>Where it fails, and what the numbers say about it</h2>{FAIL}{FAIL2}{maneuver_block()}<h3>After: E312 (narrower end heat, handle scales, snapped ends, p >= 0.1)</h3>{errors_block('errors_snap_e312', 'aft_', 'After (E312, snapped, p >= 0.1). ')}<h3>Before: E308 pairs, connectors from the keypoints, p >= 0.5</h3>{errors_block('errors', 'err_', 'Before (E308, p >= 0.5). ')}</section>
<section id="look"><h2>Pictures and videos</h2>{LOOK}{media_block()}</section>
<section id="weak"><h2>Weaknesses, honest negatives, what might be overlooked</h2>{WEAK}</section>
<section id="next"><h2>Next steps</h2>{NEXT}</section>
<section id="versions"><h2>Versions and how to rebuild</h2>{VERSIONS}</section>
<footer>{FOOTER}</footer>
</div>'''
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(head + "\n" + body)
open(os.path.join(OUT, "page.html"), "w", encoding="utf-8").write(head + "\n" + body)
print("written", os.path.join(OUT, "index.html"), len(body))
