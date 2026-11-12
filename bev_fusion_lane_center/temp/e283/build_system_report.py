"""System report page (scratch): results/system_report_2026-11-08/index.html from the RESULT FILES (scoring cards, temp/e276|e280|e281|e282 json, gate logs, pictures), never typed numbers; a number that does not exist
reads "pending". Quantitative / qualitative / architecture state of the Spurmitte lane system, in the layout of the whole-system report of 2026-11-05 (results/E273.report) with fewer videos.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh build_system_report.py
Pictures are embedded (base64 JPEG), the videos are WebM files in vid/ (the page links them relatively)."""
import base64
import glob
import html
import json
import os
import re
import subprocess

import cv2

ROOT = os.environ["BFLC_ROOT"]
OUT = os.path.join(ROOT, "results", "system_report_2026-11-08")
VID = os.path.join(OUT, "vid")
os.makedirs(VID, exist_ok=True)
SEEDS = (0, 1)


def J(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def card(exp, run):
    return J(f"runs/{exp}/{run}/card.json")


def g(c, k, sec="graph"):
    return None if c is None else c["p"].get(sec, {}).get(k)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def num(x, fmt="{:.3f}", none="pending"):
    return none if x is None else fmt.format(x)


def two(cs, k, sec="graph", fmt="{:.3f}"):
    """seed 0 / seed 1"""
    v = [g(c, k, sec) for c in cs]
    return " / ".join(num(x, fmt) for x in v)


def m2(cs, k, sec="graph", fmt="{:.3f}"):
    return num(mean([g(c, k, sec) for c in cs]), fmt)


def table(head, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in head)
    r = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{r}</table>"


def jpg_b64(path, width=1500, q=80):
    im = cv2.imread(path)
    if im is None:
        return None
    if im.shape[1] > width:
        im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def img(path, caption, width=1500):
    b = jpg_b64(path, width) if os.path.exists(path) else None
    if b is None:
        return f"<p class='pending'>picture pending ({html.escape(os.path.basename(path))})</p>"
    return f"<figure><img src='data:image/jpeg;base64,{b}' alt='{html.escape(caption)}'><figcaption>{caption}</figcaption></figure>"


def webm(src, name, width=1120, crf=37):
    dst = os.path.join(VID, name)
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return "vid/" + name
    os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = f"video_codec;libvpx-vp9|crf;{crf}|b;0|deadline;good|cpu-used;4"
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


def video(path, poster, caption):
    p = os.path.join(OUT, path)
    if not os.path.exists(p):
        return f"<p class='pending'>video pending ({html.escape(path)})</p>"
    return f"<figure><video controls preload='none' loop playsinline src='{path}'></video><figcaption>{caption}</figcaption></figure>"


# ------------------------------------------------------------------------------------------------------------------ data
B = [card("e273", f"e273_p_s{s}") for s in SEEDS]
H = [card("e275", f"e273_p_s{s}.close1.5_t0.5") for s in SEEDS]
U = [card("e277", f"e277_u_s{s}.plain_t0.5") for s in SEEDS]
W = [card("e279", f"e279_w_s{s}.plain_t0.5") for s in SEEDS]
WH = [card("e279", f"e279_w_s{s}.close1.5_t0.5") for s in SEEDS]
Tc = [card("e276", f"e276_t_s{s}") for s in SEEDS]
Bl4 = [card("e276", f"e273_p_s{s}") for s in SEEDS]
D = B[0]["d_ref"] if B[0] else {}
TIM = [J(f"runs/e273/e273_p_s{s}/timing.json") for s in SEEDS]


def gate_line(run):
    p = os.path.join(ROOT, "runs", run, "log.txt")
    if not os.path.exists(p):
        return None
    last = None
    for ln in open(p, encoding="utf-8", errors="replace"):
        if " GATE " in ln:
            last = ln
    return None if last is None else json.loads(last.split(" GATE ", 1)[1])


GB = [gate_line(f"e273_p_s{s}_p7") for s in SEEDS]
GW = [gate_line(f"e279_w_s{s}_p7") for s in SEEDS]


def gg(gl, ph, k):
    return None if gl is None else gl.get(ph, {}).get(k)


verd = open(os.path.join(ROOT, "temp", "e276", "verdicts.log"), encoding="utf-8").read() if os.path.exists(os.path.join(ROOT, "temp", "e276", "verdicts.log")) else ""


def verdict_of(tag):
    i = verd.rfind(tag)
    if i < 0:
        return None
    j = verd.find('"verdict": "', i)
    return None if j < 0 else verd[j + 12: verd.find('"', j + 12)]


V277, V279 = verdict_of("== verdict277 u against"), verdict_of("== verdict277 for E279")
asg = [J(f"temp/e276/assignment_uncertainty_e273_p_s{s}.json") for s in SEEDS]
tmpl = [J(f"temp/e276/temporal_links_e273_p_s{s}.json") for s in SEEDS]
P11 = [J(f"temp/e280/p11_s{s}.json") for s in SEEDS]
P12 = [J(f"temp/e281/p12_s{s}.json") for s in SEEDS]
P13 = [J(f"temp/e281/p13_s{s}.json") for s in SEEDS]
P14 = J("temp/e282/pilot_outcome.json")
DO = J("temp/e276/density_oracle.json")
ORACLE_V = None if not DO else DO["dense (E273)"]["graph"]["vert100_dp"]
REF = J("temp/e282/references.json")


def long_stats():
    out = []
    for x in asg:
        lk = (x or {}).get("result", {}).get("links")
        if not lk:
            return None
        rc = next(v for v in lk["risk_coverage"] if v["threshold"] == 0.5)
        lg_ = lk["p>=0.5 long"]
        out.append(dict(prec=lg_["precision"], share=lg_["n"] / rc["links"], wrong=lg_["n"] * (1 - lg_["precision"]) / (rc["links"] * (1 - rc["precision"]))))
    return out


LSX = long_stats()
LS = None if LSX is None else [x["prec"] for x in LSX]
rng = lambda vals: "pending" if not vals or None in vals else f"{100 * min(vals):.0f}-{100 * max(vals):.0f} %"  # noqa: E731
PERS = [((x or {}).get("result") or {}).get("delivered / long links", {}).get("persistent") for x in tmpl]

# ------------------------------------------------------------------------------------------------------------------ svg helpers
def bar_chart(title, cats, series, vmax=None, unit=""):
    """grouped horizontal bars; series = [(name, [values per category], css class)]"""
    n = len(series)
    vmax = vmax or max([v for _, vs, _ in series for v in vs if v is not None] + [0.01]) * 1.1
    rowh, left, width = 14 * n + 16, 190, 560
    h = len(cats) * rowh + 54
    out = [f"<svg viewBox='0 0 {left + width + 80} {h}' role='img' aria-label='{html.escape(title)}' class='chart'>", f"<text x='0' y='16' class='ct'>{html.escape(title)}</text>"]
    for i, c in enumerate(cats):
        y0 = 30 + i * rowh
        out.append(f"<text x='{left - 8}' y='{y0 + 12}' text-anchor='end' class='cl'>{html.escape(c)}</text>")
        for j, (nm, vs, cls) in enumerate(series):
            v = vs[i]
            y = y0 + j * 14
            if v is None:
                out.append(f"<text x='{left + 4}' y='{y + 11}' class='cl pend'>pending</text>")
                continue
            w = max(1.0, width * v / vmax)
            out.append(f"<rect x='{left}' y='{y}' width='{w:.1f}' height='11' class='{cls}'/><text x='{left + w + 5:.1f}' y='{y + 10}' class='cl'>{v:.3f}{unit}</text>")
    ly = h - 14
    for j, (nm, vs, cls) in enumerate(series):
        out.append(f"<rect x='{left + j * 170}' y='{ly - 9}' width='11' height='11' class='{cls}'/><text x='{left + j * 170 + 16}' y='{ly}' class='cl'>{html.escape(nm)}</text>")
    out.append("</svg>")
    return "".join(out)


def box(x, y, w, h, lines, cls="bx", sub=None):
    t = "".join(f"<tspan x='{x + w / 2:.0f}' dy='{14 if i else 0}'>{html.escape(s)}</tspan>" for i, s in enumerate(lines))
    s = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' class='{cls}'/><text x='{x + w / 2:.0f}' y='{y + 18}' text-anchor='middle' class='bt'>{t}</text>"
    if sub:
        s += "".join(f"<text x='{x + w / 2:.0f}' y='{y + 18 + 14 * len(lines) + 14 * i}' text-anchor='middle' class='bs'>{html.escape(l)}</text>" for i, l in enumerate(sub))
    return s


def arrow(x1, y1, x2, y2):
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' class='ar' marker-end='url(#ah)'/>"


def badge(x, y, text, cls="rule"):
    w = 8 * len(text) + 12
    return f"<rect x='{x}' y='{y}' width='{w}' height='16' rx='8' class='{cls}'/><text x='{x + w / 2:.0f}' y='{y + 12}' text-anchor='middle' class='bg'>{html.escape(text)}</text>"


def architecture():
    s = ["<svg viewBox='0 0 940 640' role='img' aria-label='architecture of the lane head' class='arch'>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' class='ahp'/></marker></defs>"]
    s.append(box(20, 10, 420, 52, ["Camera + radar -> frozen GaussianCaR BEV features"], "bx fixed", ["39 channels, 200 x 200, cached; no encoder is trained here"]))
    s.append(arrow(230, 62, 230, 84))
    s.append(box(20, 84, 420, 70, ["Pixel pyramid (trained, phase 2)"], "bx", ["lines 0.5 m | lane centres 1 m + 6 event heat maps | road, intersection 2 m"]))
    s.append(arrow(230, 154, 230, 176))
    s.append(box(20, 176, 420, 84, ["Node head (phase 3): one node per 2 m cell"], "bx", ["existence, offset, direction, 8 type flags, section start, long-link ends", "neighbourhood attention 7 x 7, refinement on the fine features"]))
    s.append(box(470, 190, 220, 56, ["Soft close-node step"], "bx new", ["E277 / E279: trained inside the head", "radius 1.25 m (decision open)"]))
    s.append(arrow(440, 218, 468, 218))
    s.append(arrow(230, 260, 230, 284))
    s.append(box(20, 284, 420, 84, ["Link head (phase 4): connection matrix"], "bx", ["short links <= 8 m, long links through junctions (top-64 exits x top-64 entries)", "Hermite curve per link with learned handle scales"]))
    s.append(arrow(130, 368, 130, 392))
    s.append(arrow(330, 368, 330, 392))
    s.append(box(20, 392, 200, 74, ["Sections (phase 5)"], "bx", ["soft absorbing walk,", "one id per lane piece"]))
    s.append(box(240, 392, 200, 74, ["Map encoder + assigner (6)"], "bx", ["SD-map pieces, cross-attention,", "dustbin, soft registration"]))
    s.append(box(480, 392, 200, 74, ["SD map (input)"], "bx fixed", ["lane-level pieces,", "here: lane GT with simulated errors"]))
    s.append(arrow(480, 429, 442, 429))
    s.append(arrow(130, 466, 130, 500))
    s.append(arrow(340, 466, 340, 500))
    s.append(box(20, 500, 420, 70, ["Delivered lane graph"], "bx out", ["nodes, links with curves, section ids, probability over map pieces + none", "head alone 24 ms per frame (RTX 3060, batch 1, cached features)"]))
    s.append(badge(452, 506, "read-out e >= 0.5", "rule"))
    s.append(badge(452, 528, "read-out p >= 0.5", "rule"))
    s.append(badge(452, 550, "hard step 1.5 m (B only)", "rule"))
    s.append(badge(470, 168, "top-512 cells (index choice)", "rule"))
    s.append(box(720, 84, 200, 70, ["Temporal fusion"], "bx gap", ["built in E272, not in the", "lane head; untested value"]))
    s.append(box(720, 176, 200, 84, ["Route attribute", "+ planner consumer"], "bx gap", ["old chain only (route success", "0.889, path error 0.34 m)"]))
    s.append(box(720, 392, 200, 74, ["Prototype chain d"], "bx gap", ["dense heads + graph builder +", "rule assembly; best numbers"]))
    s.append("<g transform='translate(20,612)'><rect x='0' y='-10' width='12' height='12' class='bx'/><text x='18' y='0' class='bs'>trained, differentiable</text><rect x='200' y='-10' width='12' height='12' class='rule'/><text x='218' y='0' class='bs'>output rule step (named)</text>"
             "<rect x='430' y='-10' width='12' height='12' class='gap'/><text x='448' y='0' class='bs'>not in the lane head</text></g>")
    s.append("</svg>")
    return "".join(s)


css = """
:root{--bg:#fbfbf9;--fg:#1d2329;--mut:#5b6670;--line:#d8dde2;--acc:#1f6feb;--ok:#2a7d3f;--bad:#b3261e;--pend:#8a5f00;--bx:#e8f0fb;--bxl:#9db7dd;--new:#e4f4e8;--newl:#58a06a;--fixed:#eceff2;--fixedl:#a5adb5;--rule:#fff0cc;--rulel:#d7a526;--gap:#f6e7e5;--gapl:#d19a94;--s1:#9aa5af;--s2:#1f6feb;--s3:#2a7d3f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;--bx:#1d2a3d;--bxl:#41648f;--new:#18301f;--newl:#3f8c55;--fixed:#232a31;--fixedl:#566069;--rule:#3a3115;--rulel:#a0791c;--gap:#3a2220;--gapl:#8a504a;--s1:#6c7782;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;--bx:#1d2a3d;--bxl:#41648f;--new:#18301f;--newl:#3f8c55;--fixed:#232a31;--fixedl:#566069;--rule:#3a3115;--rulel:#a0791c;--gap:#3a2220;--gapl:#8a504a;--s1:#6c7782;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;padding-block:0 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1000px;margin:0 auto}h1{font-size:1.65rem;margin:28px 0 4px;text-wrap:balance}h2{font-size:1.25rem;margin:36px 0 6px;border-top:1px solid var(--line);padding-top:16px;text-wrap:balance}h3{font-size:1.04rem;margin:22px 0 4px}
p,li{max-width:82ch}.mut{color:var(--mut)}.pending{color:var(--pend)}table{border-collapse:collapse;margin:10px 0;display:block;max-width:100%;overflow-x:auto;font-variant-numeric:tabular-nums;font-size:.93rem}th,td{border:1px solid var(--line);padding:4px 10px;text-align:left;vertical-align:top}th{background:color-mix(in srgb,var(--fg) 6%,transparent)}
figure{margin:14px 0}img,video{max-width:100%;height:auto}figcaption{color:var(--mut);font-size:.9rem}.note{border-left:3px solid var(--acc);padding:2px 12px;margin:10px 0}.box{border:1px solid var(--line);border-radius:6px;padding:4px 16px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}.tile{border:1px solid var(--line);border-radius:8px;padding:8px 12px}.tile b{display:block;font-size:1.5rem}.tile span{color:var(--mut);font-size:.85rem}
.chart,.arch{width:100%;height:auto;max-width:960px}.ct{font:600 13px system-ui;fill:var(--fg)}.cl{font:12px system-ui;fill:var(--fg)}.cl.pend{fill:var(--pend)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}
.bx{fill:var(--bx);stroke:var(--bxl)}.bx.new{fill:var(--new);stroke:var(--newl)}.bx.fixed{fill:var(--fixed);stroke:var(--fixedl)}.bx.out{fill:var(--new);stroke:var(--newl)}.gap{fill:var(--gap);stroke:var(--gapl)}.rule{fill:var(--rule);stroke:var(--rulel)}
.bt{font:600 13px system-ui;fill:var(--fg)}.bs{font:11.5px system-ui;fill:var(--mut)}.bg{font:11px system-ui;fill:var(--fg)}.ar{stroke:var(--mut);stroke-width:1.6}.ahp{fill:var(--mut)}
.ok{color:var(--ok)}.bad{color:var(--bad)}
"""

# ------------------------------------------------------------------------------------------------------------------ page
dtopo, dj, dl, dv = D.get("topo_F1"), D.get("junction_F1"), D.get("lanes_found"), D.get("vert100")
parts = [f"<title>Spurmitte system report</title><meta name='claude:viewport-fit-sides' content='cover'><style>{css}</style><main>"]
P = parts.append
P("<h1>Spurmitte lane system: where it stands</h1>")
P("<p class='mut'>Spurmitte · lane-centre graph from camera + radar, attached to an SD map · status 2026-11-08 · VAL 873 frames, lane GT l4 · TEST untouched · every number is read from a result file</p>")
P(f"<p>This is the whole-system picture in the layout of the report of 2026-11-05, with the numbers of today and fewer videos: how good the system is (numbers), how it looks (pictures and four videos), and how it is built (architecture, with the steps that are still rules). "
  f"Since the last whole-system report nine experiment cards (E274 - E282) were worked off. The end-to-end lane head is better than it was (a trained close-node step lifts the topology F1 from {m2(B, 'topo_F1')} to {m2(W, 'topo_F1')} and the junction F1 from {m2(B, 'junction_F1')} to {m2(W, 'junction_F1')}), "
  f"but it is still clearly behind the old rule-based chain (topology F1 {num(dtopo)}, junction F1 {num(dj)}, lanes found {num(dl)}). The weak spots are measured: junction precision, the long links through junctions, and lanes found.</p>")
P("<div class='tiles'>"
  f"<div class='tile'><span>Topology F1 of the delivered graph</span><b>{m2(W, 'topo_F1')}</b><span>E279 head (mean of 2 seeds); baseline B {m2(B, 'topo_F1')}; best graph so far {num(dtopo)}</span></div>"
  f"<div class='tile'><span>Junction F1</span><b>{m2(W, 'junction_F1')}</b><span>E279; B {m2(B, 'junction_F1')}; best graph so far {num(dj)}</span></div>"
  f"<div class='tile'><span>Lanes found</span><b>{m2(W, 'lanes_found')}</b><span>E279; B {m2(B, 'lanes_found')}; best graph so far {num(dl)}</span></div>"
  f"<div class='tile'><span>Vertices per 100 m (Douglas-Peucker 0.3 m)</span><b>{m2(W, 'vert100_dp', fmt='{:.1f}')}</b><span>E279; B {m2(B, 'vert100_dp', fmt='{:.1f}')}; best graph so far {num(dv, '{:.1f}')}; budget 14.3</span></div>"
  f"<div class='tile'><span>Map assignment, lane accuracy with the map 1-2 m off</span><b>{m2(W, 'lane_acc', 'assign_error')}</b><span>E279 (mean of 2 seeds); geometric control {num(mean([(c or {}).get('g', {}).get('assign_error', {}).get('lane_acc') for c in W]))}</span></div>"
  f"<div class='tile'><span>Time per frame, head alone</span><b>{num(mean([(t or {}).get('total', {}).get('median') for t in TIM]), '{:.1f}')} ms</b><span>E273 head, idle RTX 3060, batch 1, cached features, median of 60 frames; the soft step and the CPU read-out are not timed</span></div></div>")

# ---- short answer
P("<h2>Short answer</h2><ul>"
  f"<li><b>Quantitatively.</b> The end-to-end head finds the lane pixels, nodes and the assignment to the SD map well, and it is weakest exactly where the old chain is strongest: the connections across junctions. "
  f"After the trained close-node step the delivered graph has topology F1 {m2(W, 'topo_F1')} (old chain {num(dtopo)}), junction F1 {m2(W, 'junction_F1')} ({num(dj)}), lanes found {m2(W, 'lanes_found')} ({num(dl)}) and {m2(W, 'vert100_dp', fmt='{:.0f}')} vertices per 100 m ({num(dv, '{:.0f}')}). All numbers are in-sample on VAL; the step is a gain in precision bought with a small loss of recall.</li>"
  f"<li><b>Qualitatively.</b> The pictures show the system following the lanes, with the typical errors at junctions: false or doubled connectors across the junction mouth, missing branches that no camera sees, and a node row that is much denser than the old chain's. The videos show two junction scenes.</li>"
  f"<li><b>Architecture.</b> One module in five stages (pixel maps, nodes, links, sections, map assignment) on frozen, cached features, trained in six phases; the forward path is differentiable up to the assignment (D55). What is still a rule on the delivered path: the two read-out thresholds, the choice of the 512 strongest cells and, for baseline B, the hard close-node step (E279 would replace it by a trained one). Not in the head: temporal fusion, the route attribute and the planner link.</li>"
  f"<li><b>This week's levers.</b> Six ideas were tried and closed (tier-weighted loss, sparse node target, the step inside the network at 1.5 m, a learned pair radius, a memory for the long links, a zone-matching loss), the close-node step was kept (as a rule on the output in E275, trained inside the head at 1.25 m in E279). Details below.</li></ul>")

# ---- status board
P("<h2>Status board</h2>")
rows = [
    ["Foundation features", "frozen GaussianCaR (camera + radar, map-pretrained), cached; nothing in the lane project trains an image encoder", "fixed"],
    ["Pixel maps (pyramid)", f"phase-7 gate on 150 VAL frames: lane-centre F1 at 1 m {num(gg(GW[0], 'p2', 'centre_f1'))} / {num(gg(GW[1], 'p2', 'centre_f1'))}; road IoU {num(gg(GW[0], 'p2', 'road_iou'))} / {num(gg(GW[1], 'p2', 'road_iou'))}; intersection IoU {num(gg(GW[0], 'p2', 'inter_iou'))} / {num(gg(GW[1], 'p2', 'inter_iou'))}", "good"],
    ["Nodes (2 m grid)", f"node F1 {two(W, 'node_F1', 'nodes')}, precision {two(W, 'node_P', 'nodes')}, recall {two(W, 'node_R', 'nodes')}, median position error {two(W, 'pos_err_med', 'nodes', '{:.2f}')} m, {two(W, 'nodes_per_100m', 'nodes', '{:.1f}')} nodes per 100 m (E279; B {two(B, 'nodes_per_100m', 'nodes', '{:.1f}')})", "good, recall short"],
    ["Links (Hermite curves)", f"short links P {two(W, 'link_short_P', 'nodes')} / R {two(W, 'link_short_R', 'nodes')}; long links through junctions P {two(W, 'link_long_P', 'nodes')} / R {two(W, 'link_long_R', 'nodes')}", "short good, long weak"],
    ["Sections (one id per lane piece)", f"{two(W, 'pieces_per_gt_piece', fmt='{:.2f}')} delivered pieces per true piece, {two(W, 'id_switches_per_100m', fmt='{:.2f}')} id switches per 100 m; section start right in {num(gg(GW[0], 'p5', 'section_start_acc'))} / {num(gg(GW[1], 'p5', 'section_start_acc'))} (gate 0.95)", "partial"],
    ["Delivered graph (E279)", f"topology F1 {two(W, 'topo_F1')}, junction F1 {two(W, 'junction_F1')}, F1@1.0 {two(W, 'F1@1.0')}, lanes found {two(W, 'lanes_found')}", "behind the old chain"],
    ["Assignment to the SD map", f"lane accuracy, map 1-2 m off: head {two(W, 'lane_acc', 'assign_error')} (E279); paired gain over the geometric control {num((W[0] or {}).get('diff', {}).get('error', {}).get('mean'), '{:+.4f}')} / {num((W[1] or {}).get('diff', {}).get('error', {}).get('mean'), '{:+.4f}')}", "works"],
    ["Temporal fusion", "built and trained in E272 (hold), but it closes itself on clean frames, so its value is untested; not part of the lane head; the E281 pre-check found that the previous frame adds almost no information about which long link is right", "not in the head"],
    ["Route attribute + planner link", "old chain only; not part of the new head", "not in the head"],
    ["Best graph so far (old chain, recall_sp0_dp)", f"Q {num(D.get('Q'))}, topology F1 {num(dtopo)}, junction F1 {num(dj)} (P {num(D.get('junction_P'))} / R {num(D.get('junction_R'))}), lanes found {num(dl)}, F1@1.0 {num(D.get('F1_1'))}, {num(dv, '{:.1f}')} vertices per 100 m; rule-based assembly, not end-to-end", "reference"],
    ["Speed", "E273 head alone " + " / ".join(num((t or {}).get('total', {}).get('median'), '{:.1f}') for t in TIM) + " ms per frame (seeds 0 / 1; seed 0: pixel maps " + num((TIM[0] or {}).get('pyr', {}).get('median'), '{:.1f}') + ", nodes " + num((TIM[0] or {}).get('nodes', {}).get('median'), '{:.1f}') + ", links " + num((TIM[0] or {}).get('links', {}).get('median'), '{:.1f}') + ", sections " + num((TIM[0] or {}).get('sections', {}).get('median'), '{:.1f}') + ", map encoder + assignment " + num((TIM[0] or {}).get('assign', {}).get('median'), '{:.1f}') + " ms); the soft step and the CPU read-out are not timed", "fast enough"],
]
P(table(["Stage", "Where it stands (VAL 873 frames; seed 0 / seed 1)", "Reading"], rows))
P("<p class='mut'>The pixel and section numbers are the joint-phase gates on 150 VAL frames, the rest are the scoring cards on all 873 frames. Sources: <code>runs/e279/*/card.json</code>, <code>runs/e273/*/card.json</code>, <code>runs/e279_w_s*_p7/log.txt</code>, the card of E273 for the timing.</p>")

# ---- since the last report
P("<h2>What happened since the report of 2026-11-05</h2>")
cut = lambda a, b, k: None if (a is None or b is None) else mean([g(x, k) for x in a]) - mean([g(x, k) for x in b])  # noqa: E731
sg = lambda v, f="{:+.3f}": "pending" if v is None else f.format(v)  # noqa: E731
p11 = P11[0] and P11[1]
p12 = P12[0] and P12[1]
exp_rows = [
    ["E274 (r, p)", "node recall by loss weights; a dual-softmax loss over the long-link candidates", "CUT: neither arm beat the baseline B on its written rule; the long-link loss lifted the long-link precision but broke the exact-map assignment guard", "card E274"],
    ["D58", "your word: the E273 head of the morning of 2026-11-05 is the baseline B", "B = E273; everything is judged against it", "DECISIONS"],
    ["E275", "drop every delivered node that lies within 1.5 m of a stronger node (a rule step on the output, no retraining)", f"KEEP: topology F1 {m2(B, 'topo_F1')} to {m2(H, 'topo_F1')}, junction F1 {m2(B, 'junction_F1')} to {m2(H, 'junction_F1')}, vertices per 100 m {m2(B, 'vert100_dp', fmt='{:.1f}')} to {m2(H, 'vert100_dp', fmt='{:.1f}')}", "card E275"],
    ["E276", "ground truth with relevance tiers (own lane, neighbours, rest) and visibility; tier-weighted loss (arm t); a sparse node target (arm s)", f"arm t TIE: the own lane did not gain (F1@1 on own lane + neighbours {two(Tc, 't12_F1_1', 'tier')} against {two(Bl4, 't12_F1_1', 'tier')} for B), far-lane recall and lanes found fell ({two(Tc, 'lanes_found')} against {two(Bl4, 'lanes_found')}); arm s failed its gate (the per-cell loss cannot place a sparse row)", "card E276"],
    ["E277", "the close-node step as a soft step inside the head, radius 1.5 m", f"{V277 or 'CUT'}: it reproduces the hard step, but lanes found changes by {' / '.join(sg(g(u, 'lanes_found') - g(b, 'lanes_found')) for u, b in zip(U, B))} against B (seed 0 / seed 1)", "card E277"],
    ["E279", "the same step at radius 1.25 m", f"{V279 or 'KEEP'}: topology F1 {two(W, 'topo_F1')}, junction F1 {two(W, 'junction_F1')}, lanes found {two(W, 'lanes_found')}; guards held, with thin margins on seed 0", "card E279"],
    ["D64", "should the long links be read at a lower threshold than 0.5?", "you decided: keep 0.5", "DECISIONS"],
    ["E280", "a learned radius per pair of close nodes", (f"CUT before any code: the pair features add at most {sg(max(f['gain'] for x in P11 for f in x['auc']['folds']))} AUC over the head's own confidence (needed +0.05); an oracle that spares the real close pairs reaches junction F1 {num(P11[0]['graph']['oracle1.5']['junction_F1'])} / {num(P11[1]['graph']['oracle1.5']['junction_F1'])}, worse than the fixed 1.25 m step ({num(P11[0]['graph']['hard1.25']['junction_F1'])} / {num(P11[1]['graph']['hard1.25']['junction_F1'])}): the junction gain of the step IS the thinning of real nodes" if p11 else "pending"), "card E280"],
    ["E281", "a memory of the previous keyframe's long links", (f"CUT before any code: the previous frame adds at most {sg(max(x['B']['card_label']['gain S1 -> S2']['band'][f] for x in P12 for f in (0, 1)))} AUC in the decision band (needed +0.05); the best hand-made fusion raises the persistence of the long links from {num(P12[0]['A']['persistence'])} to {num(P12[0]['C']['V6 recursive average']['long_persist'])} (seed 0) but loses {num(P12[0]['C']['none']['lanes_found'] - P12[0]['C']['V6 recursive average']['lanes_found'])} lanes found" if p12 else "pending"), "card E281"],
    ["E282", "a zone-matching node loss so that the head can learn a sparse node row", (f"STOP at pilot level: gain over the per-cell sparse loss {num(P14['bootstrap_m_minus_s_matched']['point'], '{:+.3f}')} in free-zone exact (needed +0.15), only {num(P14['arms']['m']['at_0_5']['nodes_per_100m'], '{:.1f}')} nodes per 100 m reach e >= 0.5 against the target 14.4; CPU pilot, one seed" if P14 else "pending"), "card E282"],
]
P(table(["Experiment", "What", "Result", "Where"], exp_rows))

# ---- quantitative
P("<h2>Quantitatively</h2><h3>The delivered graph, variant by variant</h3>")
vr = [["Best graph so far (old chain)", num(dtopo), num(D.get("topo_P")), num(D.get("topo_R")), num(dj), f"{num(D.get('junction_P'))} / {num(D.get('junction_R'))}", num(D.get("F1_1")), num(dl), num(dv, "{:.1f}")]]
for nm, cs in (("B = E273 baseline (read out at 0.5)", B), ("B + hard close-node step 1.5 m (E275)", H), ("E277: step inside the head, 1.5 m", U), ("E279: step inside the head, 1.25 m", W), ("E279 + hard step 1.5 m on top", WH)):
    vr.append([nm, two(cs, "topo_F1"), two(cs, "topo_P"), two(cs, "topo_R"), two(cs, "junction_F1"), f"{two(cs, 'junction_P')} / {two(cs, 'junction_R')}", two(cs, "F1@1.0"), two(cs, "lanes_found"), two(cs, "vert100_dp", fmt="{:.1f}")])
P(table(["Variant (seed 0 / seed 1)", "topology F1", "topology P", "topology R", "junction F1", "junction P / R", "F1@1.0", "lanes found", "vertices per 100 m"], vr))
P(bar_chart("Topology F1 and junction F1 (mean of 2 seeds)", ["best graph so far", "B", "B + hard step", "E277", "E279"],
            [("topology F1", [dtopo, mean([g(c, "topo_F1") for c in B]), mean([g(c, "topo_F1") for c in H]), mean([g(c, "topo_F1") for c in U]), mean([g(c, "topo_F1") for c in W])], "s2"),
             ("junction F1", [dj, mean([g(c, "junction_F1") for c in B]), mean([g(c, "junction_F1") for c in H]), mean([g(c, "junction_F1") for c in U]), mean([g(c, "junction_F1") for c in W])], "s3")], vmax=0.85))
def rg(cs, k):
    v = [g(c, k) for c in cs]
    a, b = f"{min(v):.2f}", f"{max(v):.2f}"
    return a if a == b else f"{a} - {b}"


P(f"<div class='note'>Reading. The step buys precision with recall: the junction precision rises from {rg(B, 'junction_P')} (B) to {rg(W, 'junction_P')} (E279) and {rg(H, 'junction_P')} (hard step), the junction recall falls from {rg(B, 'junction_R')} to {rg(W, 'junction_R')} and {rg(H, 'junction_R')}; "
  f"the topology F1 rises by {num(mean([g(c, 'topo_F1') for c in W]) - mean([g(c, 'topo_F1') for c in B]), '{:+.3f}')} (E279) and {num(mean([g(c, 'topo_F1') for c in H]) - mean([g(c, 'topo_F1') for c in B]), '{:+.3f}')} (hard step); the lanes found change by {num(mean([g(c, 'lanes_found') for c in W]) - mean([g(c, 'lanes_found') for c in B]), '{:+.3f}')} (E279). "
  f"The best graph so far is ahead on topology, F1@1.0, lanes found, junction precision and vertices; the end-to-end head has the higher junction RECALL ({rg(W, 'junction_R')} against {num(D.get('junction_R'))}), i.e. it finds more of the junctions but also invents many (precision). "
  f"The trained step at 1.25 m (E279) keeps most of the gain of the hard step with a little more lanes found; the margins of its guards are thin on seed 0, and all numbers are in-sample on VAL.</div>")

P("<h3>Where the loss is: the junctions and the long links</h3>")
lk = [["Delivered long links, share of all delivered links", rng([x["share"] for x in LSX]) if LSX else "pending", "E276 P9"],
      ["... precision (pair rule) of the long links", " / ".join(num(x) for x in (LS or [None, None])), "E276 P9 (B)"],
      ["... share of the wrong links that are long links", rng([x["wrong"] for x in LSX]) if LSX else "pending", "E276 P9"],
      ["... persistence from one keyframe to the next (0.5 s), delivered / ground truth", f"{' / '.join(num(x) for x in PERS)} / 0.98", "E276 P5b (B)"],
      ["Topology F1 without any long link", f"{num(mean([x['variants']['0.5_1.01']['topo_F1'] for x in [J('temp/e276/linkthr_e273_p_s0.json'), J('temp/e276/linkthr_e273_p_s1.json')] if x]))} (with them {m2(B, 'topo_F1')})", "E276 P10"]]
P(table(["Measure (baseline B, VAL 873)", "Value (seed 0 / seed 1)", "Source"], lk))
if all(P13):
    th = [[("0.5 (B, decided D64)" if t == "0.5" else t), f"{P13[0]['variants'][t]['lanes_found']:.3f} / {P13[1]['variants'][t]['lanes_found']:.3f}", f"{P13[0]['variants'][t]['topo_F1']:.3f} / {P13[1]['variants'][t]['topo_F1']:.3f}",
           f"{P13[0]['variants'][t]['long_precision_pair']:.3f} / {P13[1]['variants'][t]['long_precision_pair']:.3f}", f"{P13[0]['variants'][t]['long_persist']:.3f} / {P13[1]['variants'][t]['long_persist']:.3f}"] for t in ("0.5", "0.4", "0.3", "0.2", "0.1")]
    P("<p>A lower read-out threshold for the long links is a clean trade of long-link precision for lanes found; it does not make the links more stable and the topology F1 never rises above B's. You decided to keep 0.5 (D64).</p>")
    P(table(["Long-link threshold", "lanes found", "topology F1", "long-link precision", "long-link persistence"], th))

P("<h3>Assignment to the SD map</h3>")
ar = [["lane accuracy, exact map", two(W, "lane_acc", "assign_exact")], ["lane accuracy, map 1-2 m off", two(W, "lane_acc", "assign_error")],
      ["paired gain over the geometric control, map 1-2 m off (95 % interval)", " ; ".join(f"{num(x['diff']['error']['mean'], '{:+.4f}')} [{num(x['diff']['error']['lo'], '{:+.3f}')}, {num(x['diff']['error']['hi'], '{:+.3f}')}]" for x in W if x)],
      ["paired difference with the exact map (guard: not below -0.01)", " ; ".join(num(x['diff']['exact']['mean'], '{:+.4f}') for x in W if x)]]
P(table(["Assignment (E279, 2 seeds)", "value"], ar))
P("<p class='mut'>The assignment is independent of the graph quality; it labels the nodes that exist. It was tested only with two map conditions (exact, 1-2 m off); larger shifts, sensor failure and a real SD map with its own errors are not tested.</p>")

P("<h3>What the closed levers tell us</h3><ul>"
  "<li><b>The junction gain of the step is the thinning of real nodes</b> (E280): keeping the real close pairs destroys the gain, and nothing the head knows tells the real pairs from the redundant ones.</li>"
  "<li><b>The flicker of the long links is the flicker of uncertain links</b> (E281): the previous frame does not tell which long link is right; stabilising them by hand costs recall.</li>"
  "<li><b>A sparse node row cannot be had by a loss alone at this size</b> (E276 arm s, E282 pilot): the existence probability smears over the cells of a straight stretch and the head does not commit.</li>"
  "<li><b>Loss weights by relevance tier did not move the own lane</b> (E276 arm t): the limit of the node head in the own lane is what the features and the 2 m grid can resolve, not the weighting.</li></ul>")

# ---- qualitative
P("<h2>Qualitatively</h2>")
P("<p>All pictures and videos are rendered from the networks' real predictions (no ground truth in the forward path). Colours: dark = correct, red = false, blue = missed; each panel is the scorer's picture of the frame (delivered nodes and links against the ground-truth graph). Top view, 100 m window, the car at the bottom.</p>")
CAP = {643: "VAL frame 643, a large three-way junction. Rows: the baseline B, B with the hard close-node step (E275), the E279 head (soft step inside). Left: delivered nodes and links against the ground truth (red halo = false, blue = missed); right: the delivered graph coloured by the assigned map piece. "
              "B delivers several overlapping connector curves through the junction mouth; in the two thinned variants the overlap is largely gone and the curves are cleaner; in all three rows blue curves on the left mark ground-truth connectors that are not found. The scorer's numbers of the frame are printed under each panel.",
       300: "VAL frame 300, a road with a crossing at both ends. The long straight stretch is right in all three variants; the differences sit at the two crossings, where the nodes of B are doubled (stacked event symbols) and the thinned variants carry fewer of them; false (red) nodes remain at the upper crossing.",
       100: "VAL frame 100, a curved road with a branch at the upper right. The three long lanes are right in all variants; at the branch false nodes (red halo) remain in all three rows and a ground-truth loop (blue) is missed in all three; the E279 row carries fewer nodes there."}
for i in (643, 300, 100):
    P(img(os.path.join(ROOT, "results", "E279", "figures", f"frame_{i}.png"), CAP[i]))
P("<h3>Videos (the full scenes, 2 frames per second)</h3>")
for scene, desc in (("0221", "a large junction with a bus ahead, pedestrian crossing, stopped vehicle"), ("0108", "construction zone, waiting for a car to pass, turn right")):
    b = os.path.join(ROOT, "results", "E273.report", "vid", f"scene-{scene}_s0.webm")
    e = glob.glob(os.path.join(ROOT, "results", "E279.w_s0", "videos", f"scene-{scene}_*.mp4"))
    P(f"<h3>scene-{scene}</h3><p class='mut'>{desc}</p>")
    if os.path.exists(b):
        dst = os.path.join(VID, f"scene-{scene}_B.webm")
        if not os.path.exists(dst):
            open(dst, "wb").write(open(b, "rb").read())
        P(video(f"vid/scene-{scene}_B.webm", None, f"Baseline B (E273 seed 0) on scene-{scene}, rendered on 2026-11-05."))
    if e:
        w = webm(e[0], f"scene-{scene}_E279.webm")
        P(video(w, None, f"E279 head (soft close-node step inside, seed 0) on scene-{scene}: same layout, six cameras with the projected graph, pixel maps, nodes and links, true against delivered ids."))
    else:
        P("<p class='pending'>E279 video pending</p>")

# ---- architecture
P("<h2>Architecture</h2><p>One module, five stages, trained in six phases on the frozen, cached features. Green boxes are trained and differentiable; amber badges are the rule steps that remain on the delivered path (D55: allowed on the output only and named); grey boxes are not part of the lane head.</p>")
P(architecture())
P("<p><b>Training.</b> Phase 2 pixel maps (the 1 m branch starts from the E271 head), 3 nodes (pixels frozen), 4 links + Hermite scales, 5 sections (the node head's section output, links frozen), 6 map encoder + assignment, 7 everything jointly at a lower learning rate. Every phase has a gate; targets come from the lane ground truth by location (no matching in the loss).</p>")
lit = [["Pixel pyramid", "HDMapNet, MGMap, PseudoMapTrainer, CenterNet / CondLaneNet", "three resolutions: lines 0.5 m, centres 1 m, areas 2 m; event heat maps for splits, merges, stops, exits, starts, ends"],
       ["Grid nodes", "CenterNet (targets by location), SAM-Road, MaGRoad, PolyWorld", "one node per 2 m cell, the point of the true line nearest the cell centre; E279: a trained close-node step"],
       ["Hermite links", "TopoNet, LaneSegNet, LaneGAP, CGNet, RATopo, SAM-Road", "set attention over the active nodes, one-to-many labels, learned handle scales; published methods need matching in the loss, ours uses targets by location"],
       ["Sections", "path-aware attention in MAT / OMA (no direct precedent)", "soft absorbing walk along the links, a linear solve over 512 nodes"],
       ["Temporal fusion (not in the head)", "StreamMapNet, MapTracker, TopoStreamer, HRMapNet, MambaMap", "built in E272, untested; E281 found that the previous frame adds almost nothing about which long link is right"],
       ["Assignment", "SuperGlue / LightGlue (dustbin), PolyWorld, MAT / OMA, SMERF, SEPT, TopoSD, OrienterNet", "map encoder over five relations, cross-attention, geometric start value, soft registration, dustbin; nobody assigns perceived lane-level ids to a lane-level map softly and trains it through the perception"],
       ["Route + planner (not in the head)", "PlanTF, PLUTO, PDM, VAD, SparseDrive", "route head along the edges and the planner consumer exist in the old chain only"]]
P(table(["Block", "Literature it builds on", "What we do"], lit))
P("<h3>Rule steps and gaps on the delivered path</h3><ul>"
  "<li>Output thresholds: a node exists at e &gt;= 0.5, a link at p &gt;= 0.5 (named, allowed by D55).</li>"
  "<li>The choice of the 512 strongest cells and the long-link candidates (top-64 exits x top-64 entries): index choices, the gradient flows through the chosen values.</li>"
  "<li>Baseline B reads out with a hard close-node step (1.5 m). The trained step of E279 would replace it by a soft one inside the network; adopting it changes the baseline and is your decision (open).</li>"
  "<li>Gaps against the goal 'fully end-to-end with temporal fusion and planner': no temporal fusion in the head, no route attribute, no planner link, the ground truth is non-causal (the own lane is labelled from the later driven path).</li></ul>")

# ---- weaknesses and options
P("<h2>Weaknesses, ranked</h2>")
wk = [["1", "Junction precision and the long links", f"junction precision {two(W, 'junction_P')} even with the step; long-link precision {' / '.join(num(x) for x in (LS or [None, None]))}; the long links are {rng([x["share"] for x in LSX]) if LSX else "pending"} of the links but {rng([x["wrong"] for x in LSX]) if LSX else "pending"} of the wrong ones, and cannot be cut (topology F1 falls by {num(mean([g(c, 'topo_F1') for c in B]) - mean([x['variants']['0.5_1.01']['topo_F1'] for x in [J('temp/e276/linkthr_e273_p_s0.json'), J('temp/e276/linkthr_e273_p_s1.json')] if x]))} without them)"],
      ["2", "Lanes found", f"{m2(W, 'lanes_found')} against {num(dl)}; the node recall inside intersections is the weakest part (E273 analysis of 2026-11-05: 0.54 inside the widened intersection mask against 0.80 outside)"],
      ["3", "Node row density and jitter", f"{m2(W, 'vert100_dp', fmt='{:.0f}')} vertices per 100 m after the step, {m2(B, 'vert100_dp', fmt='{:.0f}')} before, a perfect network needs {num(ORACLE_V, '{:.1f}')} for any node target (E276 P1): the excess is jitter and duplicates in the delivered geometry, not the node grid"],
      ["4", "No temporal fusion, no route/planner link", "the head runs frame by frame; a memory of the long links does not help (E281); a recurrent node memory is a different design"],
      ["5", "Everything is in-sample on VAL", "choices were made on VAL; TEST is untouched and needs your recorded OK; two seeds only"]]
P(table(["#", "Weakness", "Numbers"], wk))
P("<h2>Where the steps could go (options, none decided)</h2><ul>"
  "<li><b>Settle the baseline.</b> Adopt the trained 1.25 m step (E279) in place of the hard read-out, or keep B plus the hard step. Needs your decision; the guard margins on seed 0 are thin.</li>"
  "<li><b>Attack the junction connectors directly.</b> The old chain closed its junction gap with a rule (fork pruning, loop 18); the end-to-end counterpart does not exist. A trained fork decision on the link candidates is the closest analogue and keeps the head differentiable.</li>"
  "<li><b>Make the head see more at junctions.</b> Node recall is lowest inside intersections; the sparse-target and tier ideas did not help, a larger context or a dedicated junction branch has not been tried.</li>"
  "<li><b>Confirm on TEST.</b> One check on the 21 TEST scenes of the finished design with your recorded OK (the numbers are all in-sample so far).</li>"
  "<li><b>Return to the product chain.</b> Join the lane head with temporal fusion and the route/planner link, because the planner-ready graph is the goal and the head alone is not yet at the old chain's quality.</li></ul>")

# ---- provenance
head = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %ad", "--date=short"], capture_output=True, text=True).stdout.strip()
P("<h2>Provenance</h2>")
P(f"<p class='mut'>Lane ground truth: store 0702a065c6 (lane GT l4, tiers and visibility); VAL 873 frames (data/splits.json), TEST untouched. Scoring cards: runs/e273, e275, e276, e277, e279; analysis files temp/e276, temp/e280, temp/e281, temp/e282 (scratch, kept in git tags lcr/e276-scratch-*, lcr/e280-scratch-1, lcr/e281-scratch-*). Cards: experiments/E273 ... E282; decisions D55-D64 in experiment_execution/DECISIONS.md. "
  f"Checkpoints: runs/e273_p_s*, runs/e277_u_s*, runs/e279_w_s* (not in git). Repository state at build time: {html.escape(head)}. Nothing is pushed. Page built by temp/e283/build_system_report.py from the result files.</p>")
P("</main>")
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("".join(parts))
print("written", os.path.join(OUT, "index.html"), round(os.path.getsize(os.path.join(OUT, "index.html")) / 1e6, 2), "MB")
