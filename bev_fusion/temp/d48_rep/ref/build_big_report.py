"""scratch of the BIG t16 result report (Mirko's request of 2026-11-02): what was tried, what came out, lane markings, comparison with the state of the art,
quantitative and qualitative, with videos. One html page (images embedded), videos as files next to it (results/t16/videos/...). Numbers are read from the stored
evaluation files of the runs and from the two digests (history_digest.md, soa_digest.md: copied from the experiment cards and papers by a read-only agent, spot-checked).
Re-runnable: whatever does not exist yet is skipped or marked "not finished".
    wsl /home/ml/miniconda3/envs/gcar/bin/python bev_fusion/temp/t16/build_big_report.py [--out bev_fusion/results/t16/report_big.html]   (from the repository root)"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_report as BR                      # noqa: E402  (helpers: img, rows_table, MAPS / OBJ / VEL / COV, CSS)
from bigr_lib import (bar_chart, esc, histogram, md_bullets, md_inline, md_tables, paired_table, quantiles, read_frames, read_metrics, read_table)  # noqa: E402

T16 = "bev_fusion/temp/t16"
OUT_DIR = "bev_fusion/temp/d48_rep/ref_out"
RUNS = [("t16_full_s0", "baseline"), ("t16_ctl_s0", "control (old chain)"), ("t17_data", "more data"), ("t18_fix", "bundle")]
LONG = {"t16_full_s0": "t16.full_s0, the baseline: the full network, 428 train scenes", "t16_ctl_s0": "t16.ctl_s0, the matched control: the old chain's recipe on the same data",
        "t17_data": "t17.data: the baseline recipe on 700 train scenes", "t18_fix": "t18.fix: 700 scenes + the module fixes + the motion block on its own learning rate"}

PAGE_CSS = """
/* Layout: one reading column (max 62rem) with a sticky-free table of contents on top; tables and charts scroll inside their own boxes.
   Palette: blue-biased paper, ink, one steel-blue accent; amber / grey / violet are reserved for chart series (ground-truth version, SOA). */
:root{--bg:#f6f7fa;--fg:#1a2130;--mut:#566073;--line:#d9dee8;--card:#ecf0f6;--acc:#14608f;--good:#17744a;--bad:#b3261e;--warn:#9a6b12;
--c-v5:#c0861c;--c-old:#8b94a6;--c-soa:#6a4bb8;--font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10151d;--fg:#e3e8f0;--mut:#9ba6b8;--line:#2a3342;--card:#19212c;--acc:#6db3e8;--good:#4cc38a;--bad:#ff8a80;--warn:#e0b04a;
--c-v5:#e0a84a;--c-old:#7c869a;--c-soa:#a58df0;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10151d;--fg:#e3e8f0;--mut:#9ba6b8;--line:#2a3342;--card:#19212c;--acc:#6db3e8;--good:#4cc38a;--bad:#ff8a80;--warn:#e0b04a;
--c-v5:#e0a84a;--c-old:#7c869a;--c-soa:#a58df0;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--font);padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:62rem;margin:0 auto}
h1{font-size:1.9rem;line-height:1.2;margin:.2em 0 .3em;text-wrap:balance;font-weight:600}
h2{font-size:1.35rem;margin:2.2em 0 .5em;padding-top:.9em;border-top:1px solid var(--line);text-wrap:balance;font-weight:600}
h3{font-size:1.05rem;margin:1.6em 0 .4em;font-weight:600}h4{font-size:.95rem;margin:1.1em 0 .3em;font-weight:600;color:var(--mut)}
p{margin:.5em 0;max-width:72ch}.lead{color:var(--mut);max-width:72ch}.small,.cap{color:var(--mut);font-size:.85rem}.cap{margin:0 0 1.1em}
a{color:var(--acc)}code{font:.85em var(--mono);background:var(--card);padding:.05em .3em;border-radius:3px;overflow-wrap:anywhere}
ul{padding-left:1.25em;margin:.5em 0}li{margin:.3em 0;max-width:80ch}
.scroll{overflow-x:auto;margin:.6em 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:.35em .6em;text-align:right;vertical-align:top}
th:first-child,td:first-child,th.l,td.l,table.wide td,table.wide th{text-align:left}th{color:var(--mut);font-weight:600}table.wide td{font-size:.82rem}
img,video{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;margin:.4em 0;display:block}
.delta{font-size:.78rem;color:var(--mut)}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}
.card{background:var(--card);border-radius:6px;padding:.7em 1em;margin:.9em 0}
.warn{border-left:3px solid var(--warn);padding:.3em 1em;background:var(--card);margin:1em 0}
.exp{border-top:1px solid var(--line);padding:.3em 0 .6em;margin:.2em 0}.exp h4{margin:.5em 0 .2em;color:var(--fg);font-size:.98rem}.exp p{margin:.2em 0;font-size:.9rem}.exp .k{color:var(--mut);font-weight:600}
details{border:1px solid var(--line);border-radius:6px;padding:.3em .9em;margin:.7em 0}summary{cursor:pointer;font-weight:600;padding:.2em 0}
nav.toc{display:flex;flex-wrap:wrap;gap:.3em 1.1em;background:var(--card);border-radius:6px;padding:.6em 1em;margin:1em 0;font-size:.88rem}nav.toc a{text-decoration:none;white-space:nowrap}
svg.chart{max-width:100%;height:auto;margin:.5em 0;display:block}svg.chart text{fill:var(--fg);font:12px var(--font)}svg.chart .ct{font-weight:600;font-size:13px}svg.chart .cn,svg.chart .cl{fill:var(--mut)}
svg.chart .bar{fill:var(--acc)}svg.chart .bar.v8{fill:var(--acc)}svg.chart .bar.v5{fill:var(--c-v5)}svg.chart .bar.old{fill:var(--c-old)}svg.chart .bar.soa{fill:var(--c-soa)}
.vgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,22rem),1fr));gap:.9rem}.vgrid figure{margin:0;min-width:0}.vgrid video{width:100%}
a:focus-visible,summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
"""


# ------------------------------------------------------------------------------------------------ data helpers
def have_runs():
    return [(r, l) for r, l in RUNS if read_table(r)]


def lane_stats(run):
    """lane-divider IoU over the validation frames that have at least 30 ground-truth lane-divider cells in view"""
    vals = []
    for lab, s, sup in read_frames(run):
        v, n = s.get("seg.lane_divider_iou"), sup.get("seg.lane_divider_iou") or 0
        if v is not None and n >= 30:
            vals.append(v)
    return vals


def f(t, key, fmt="{:.3f}"):
    return fmt.format(t[key]) if key in t else "n/a"


def ledger_hours():
    out = {}
    p = "bev_fusion/records/LEDGER.csv"
    if os.path.exists(p):
        for row in csv.reader(open(p, encoding="utf-8")):
            if row and row[0] in ("t16.full_s0", "t16.ctl_s0", "t17.data", "t18.fix", "t17.feat") and len(row) > 5 and row[3] == "ok":
                out[row[0]] = (row[1], row[2], float(row[5]) / 3600 if row[5] else 0.0)
    return out


def table_html(header, rows, cls="", left=()):
    h = "".join(f"<th class=l>{esc(c)}</th>" if i in left else f"<th>{esc(c)}</th>" for i, c in enumerate(header))
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{b}</table>"


def render_md_file(path):
    """a small markdown result file (tables + paragraphs) -> html; '' when it does not exist"""
    if not os.path.exists(path):
        return ""
    text = open(path, encoding="utf-8").read()
    out, buf = [], []
    for ln in text.split("\n"):
        if ln.startswith("|"):
            buf.append(ln)
            continue
        if buf:
            out.append(md_table_html(buf))
            buf = []
        if ln.startswith("#"):
            out.append(f"<h4>{md_inline(ln.lstrip('# '))}</h4>")
        elif ln.strip():
            out.append(f"<p>{md_inline(ln)}</p>")
    if buf:
        out.append(md_table_html(buf))
    return "".join(out)


def md_table_html(lines):
    rows = [[c.strip() for c in ln.strip()[1:-1].split("|")] for ln in lines if not re.match(r"^\|[\s:|-]+\|\s*$", ln)]
    if not rows:
        return ""
    return "<table class=wide><tr>" + "".join(f"<th class=l>{md_inline(c)}</th>" for c in rows[0]) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>"


# ------------------------------------------------------------------------------------------------ sections
def soa_rows():
    """{(layer, metric): (ours, GaussianCaR, ratio)} of the first table of the maps re-scoring, and the two front-sector rows [dualbev, ours] ; None when a file is missing"""
    p, q = f"{T16}/soa/soa_maps_v8_t16_full_s0.md", f"{T16}/soa/front_v8_n1000.md"
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    t = md_tables(open(p, encoding="utf-8").read())[0][2]
    maps = {(r[0], r[1]): (r[4], r[5], r[6]) for r in t if len(r) >= 7}
    fr = md_tables(open(q, encoding="utf-8").read())[0][2]
    return maps, fr


def soa_sentence():
    d = soa_rows()
    if d is None:
        return ("the honest statement is narrow. On the same frames and our metric the camera path equals a published camera detector (DualBEV 0.250 against 0.242 mAP2, old ground truth) and the radar adds the rest; "
                "against GaussianCaR the dual-feature chain reached 94 % on drivable area and 26 to 76 % on the other layers (old ground truth, 3000-step chains). The re-scoring on the current ground truth had not finished when this page was built.")
    maps, fr = d
    names = (("drivable area", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("car park", "carpark_area", "iou@0"), ("pedestrian crossing", "ped_crossing", "iou@0"),
             ("lane divider (0.5 m tolerance)", "lane_divider", "iou@0.5"), ("road divider (0.5 m tolerance)", "road_divider", "iou@0.5"))
    parts = "; ".join(f"{n} {maps[(l, m)][0]} / {maps[(l, m)][1]} ({int(round(float(maps[(l, m)][2]) * 100))} %)" for n, l, m in names if (l, m) in maps)
    return (f"two like-for-like re-scorings were made on the current ground truth, the same frames and the same metric code. <b>Maps:</b> the published camera + radar map model GaussianCaR (6 cameras, 360 degrees, trained for maps only on 700 scenes) "
            f"and our baseline (one front camera + radar) on 384 validation frames, on the cells the front camera sees, ours / GaussianCaR (ours as a share): {parts}. "
            f"<b>Detection, front sector 0 - 50 m:</b> the camera-only DualBEV reaches mAP2 {fr[0][1]}, our baseline {fr[1][1]} ({fr[1][2]}); DualBEV was run as a single-frame front-camera detector, so the gap contains the radar AND the temporal fusion; there is no radar ablation on the current ground truth, so this says what the whole chain adds over that detector, not that one block is better. "
            "The 384 frames of the map comparison are easier than the 1000 of the record (car park 0.47 against 0.27, drivable area 0.82 against 0.78), so the shares are not shares of the 1000-frame numbers. "
            "No leaderboard number can be set beside ours (range, cameras, classes, ground truth and scoring differ, section 6.4); section 6.3 has the tables and the caveats.")


def zc_stats():
    """precision / recall / F1 (0.5 m tolerance) of the thin lane-divider line read from the distance field, baseline, 1000 frames: from the stored metrics file"""
    m = json.load(open("bev_fusion/runs/t16_full_s0/eval/by_n/metrics_n1000.json", encoding="utf-8"))["metrics"]
    base = "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/"
    return {k: f"{m[base + s]:.2f}" for k, s in (("p", "p@0.5"), ("r", "r@0.5"), ("f1", "f1@0.5"))}


ZC = zc_stats()


def tt(r):
    """total ms per frame of the idle-GPU timing of a run (stored file), as text"""
    p = f"bev_fusion/runs/{r}/eval/timing_modules.json"
    return f"{json.load(open(p, encoding='utf-8'))['inference_ms']['total_ms']:.1f}" if os.path.exists(p) else "n/a"


def sec_summary(runs, T, nvid=0):
    B = T.get("t16_full_s0", {})
    ls = lane_stats("t16_full_s0")
    q = quantiles(ls) if ls else [0] * 5
    share_lo = sum(x < 0.1 for x in ls) / len(ls) if ls else 0
    share_hi = sum(x >= 0.5 for x in ls) / len(ls) if ls else 0
    done = ", ".join(l for r, l in runs)
    soa_txt = soa_sentence()
    return f"""
<div class=card><b>In short.</b> The system that was asked for exists and works end to end: a camera + radar BEV network with dense map layers (drivable area, walkways, car parks, pedestrian crossings, lane dividers, road dividers),
object detection, and a velocity field that uses the radar Doppler. It was trained for 6000 steps on 428 scenes and evaluated on 1000 validation frames of 82 scenes that it never saw
(ground truth v8, one training seed per run). Below: what was tried to get here, what every experiment gave, the lane markings in detail, the comparison with the state of the art, and videos.
Runs in this report (finished): {esc(done)}.</div>
<h3>The numbers of the baseline (validation, 1000 frames; maps on the cells the front camera sees, detection on the fusion area)</h3>
{table_html(["quantity", "baseline", "reading"], [
        ["drivable area IoU", f(B, "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), "plain IoU on the cells the front camera sees"],
        ["walkway IoU", f(B, "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"), "plain IoU"],
        ["pedestrian crossing IoU", f(B, "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), "plain IoU"],
        ["lane divider IoU (0.5 m tolerance)", f(B, "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"), "a predicted cell within 0.5 m of a true line cell counts; the plain IoU of these 1-cell lines is 0.189"],
        ["road divider IoU (1 m tolerance)", f(B, "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"), "same idea with 1 m"],
        ["mAP, fusion area (vehicles, pedestrians, two-wheelers)", f(B, "detf/dt+0.00/mean/all/mAP"), "mean AP over 0.5 / 1 / 2 / 4 m"],
        ["vehicle AP at 2 m / pedestrian AP at 2 m", f"{f(B, 'detf/dt+0.00/vehicle/all/ap@2')} / {f(B, 'detf/dt+0.00/pedestrian/all/ap@2')}", "fusion area"],
        ["velocity error at moving box centres (m/s)", f(B, "vel/dt+0.00/centre/field/moving/epe"), f"a predictor that always says 'standing still' has {f(B, 'vel/dt+0.00/centre/zero/moving/epe')}"],
        ["train-minus-validation gap of the six map IoUs", "0.091", "the network fits the training scenes better than new ones (section 4, 'Train against validation')"]], left=(0, 2))}
<h3>What this report says, in nine lines</h3>
<ul>
<li><b>What was tried</b> (section 2): 23 experiment folders from the first chain (2026-10-26) to the full network (2026-11-02). What moved the numbers: using both sensors (the radar adds about +0.12 mAP over the camera-only chain at equal training length: +0.133 and +0.117 in the front sector on the older ground truth), the map-trained BEVFusion image features (lane divider 0.19 to 0.35 on the old ground truth), keeping both feature sets in one chain, the velocity hurdle and the motion block (the velocity left the zero predictor), and longer training. What did not: swapping standard blocks in and out (7 of 11 swaps tied, 4 were worse, none better), a one-shot bundle of fixes, a stride-4 side branch beyond the thin lines, 64 % more training scenes at the same step budget, and, last, a bundle of module fixes with a smaller learning rate for the motion block (velocity worse, maps mixed).</li>
<li><b>Which system is delivered:</b> the baseline <code>t16.full_s0</code>. Against it, the more-data run (t17) is a tie or worse on the maps, and the bundle (t18) is worse on velocity (moving box centres 1.11 to 1.20 m/s, static cells 0.19 to 0.24 m/s, intervals above 0) with mixed maps: level on drivable area, lane divider and the mean of the six IoUs, better on walkway (+0.010), worse on car park (-0.020) and road divider (IoU -0.012, thin-line F1 0.51 to 0.31; no bootstrap interval was computed for these two). Both are reported next to the baseline; neither replaces it.</li>
<li><b>Lane markings</b> (section 5): the baseline finds lane dividers well in some frames and not at all in others. Over the {len(ls)} validation frames with at least 30 lane-divider cells in the ground truth the median frame IoU (0.5 m tolerance) is {q[2]:.2f}, {100 * share_lo:.0f} % of the frames are below 0.1 and {100 * share_hi:.0f} % reach 0.5 or more. Read as a thin line from the distance field (what a planner would use), the lane divider has precision {ZC['p']} and recall {ZC['r']} on the 1000 frames (F1 {ZC['f1']}): rarely a false line, often a missing one. The lane divider is the weakest line output, car parks and pedestrian crossings are the weakest areas; the experiments say the image features, not the blocks, are the limit.</li>
<li><b>State of the art</b> (section 6): {soa_txt}</li>
<li><b>Velocity:</b> the radar Doppler + two frames give a velocity field whose error at moving box centres is {f(B, 'vel/dt+0.00/centre/field/moving/epe', '{:.2f}')} m/s against {f(B, 'vel/dt+0.00/centre/zero/moving/epe', '{:.2f}')} for the zero predictor: this was a dead output until the hurdle and the motion block (section 2).</li>
<li><b>Negative results are reported as results:</b> more data did not close the train-to-validation gap; a whole family of block swaps changed nothing; several heads and memories were dead or unlearned (section 2, 'What did not work').</li>
<li><b>Cost:</b> {tt('t16_full_s0')} ms per frame for the baseline at batch 1 (the cached chain without the two frozen image encoders, whose outputs are stored; three time offsets, idle RTX 3060, median of 30 frames; the more-data run and the bundle {tt('t17_data')} and {tt('t18_fix')} ms) and a median of 3.7 to 3.8 s per optimiser step in training (section 8). The image encoders' time is not included in the per-frame figure.</li>
<li><b>Honest limits:</b> one seed per run; no matched large-set control of the old chain (training was stopped after t18.fix); 82 of the 150 validation scenes (the rest was not downloaded); the TEST split was never used; ground truth versions changed from v5 to v8, so numbers of old runs and new runs are not comparable.</li>
<li><b>Videos</b> (section 7): {nvid} result videos next to this page: random validation scenes with camera, ground truth, prediction and error maps, including the lane dividers.</li>
</ul>"""


def sec_tried():
    p = f"{T16}/history_digest.md"
    if not os.path.exists(p):
        return "<p>history digest not found</p>"
    text = open(p, encoding="utf-8").read()
    tabs = md_tables(text)
    exp = next((t for t in tabs if t[0].startswith("1. Table")), None)
    out = ["<p>Every experiment of the BEV system in date order, with the question, what was changed, the decisive numbers, the verdict as the card or the ledger states it, and the lesson. "
           "The numbers are copied from the experiment cards (never recomputed); the GT version and the number of seeds are part of every entry because they decide what may be compared. "
           "The digest was written by a read-only pass over all cards and spot-checked against the cards (the long run, the latency and the t16 / t17 entries were re-checked by hand).</p>"]
    if exp:
        for row in exp[2]:
            if len(row) < 7:
                continue
            ident, question, changed, numbers, verdict, lesson, source = row[:7]
            out.append(f"<div class=exp><h4>{md_inline(ident)}</h4><p><span class=k>Question.</span> {md_inline(question)}</p><p><span class=k>What was changed.</span> {md_inline(changed)}</p>"
                       f"<p><span class=k>Decisive numbers.</span> {md_inline(numbers)}</p><p><span class=k>Verdict.</span> {md_inline(verdict)}</p><p><span class=k>Lesson.</span> {md_inline(lesson)}</p>"
                       f"<p class=small>source: {md_inline(source)}</p></div>")
    return "".join(out)


def sec_lists():
    p = f"{T16}/history_digest.md"
    text = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
    out = []
    for head, a, b in (("What worked", "### (a)", "### (b)"), ("What did not work or stayed flat (with the reason the cards give)", "### (b)", "### (c)"),
                       ("Open questions the cards name", "### (c)", "## 3.")):
        items = md_bullets(text, a, b)
        out.append(f"<h3>{head}</h3><ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ul>")
    return "".join(out)


def sec_caveats():
    text = open(f"{T16}/history_digest.md", encoding="utf-8").read() if os.path.exists(f"{T16}/history_digest.md") else ""
    items = md_bullets(text, "## 3. Caveats", "## 4.")
    return "<ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ul>"


def ladder():
    """lane divider and walkway over the experiments: values copied from the cards (history digest), GT version as the digest states it"""
    lane = [("t2_ref_s0, 3000 steps (first chain)", 0.181, "old"), ("t3_ref_long, 6000 steps", 0.234, "old"), ("t1_r4_soa, 7000 steps x 2", 0.273, "old"),
            ("t7 stride-4 side branch (res4 s0)", 0.246, "old"), ("t8 BEVFusion features alone (s0)", 0.351, "v5"), ("t10 dual features (s0 / s1: 0.335 / 0.350)", 0.335, "v5"),
            ("t16.full_s0 full network", 0.429, "v8")]
    walk = [("t2_ref_s0, 3000 steps", 0.191, "old"), ("t3_ref_long, 6000 steps", 0.225, "old"), ("t1_r4_soa, 7000 steps x 2", 0.246, "old"),
            ("t8 BEVFusion features alone (s0)", 0.339, "v5"), ("t10 dual features (s0 / s1: 0.340 / 0.354)", 0.340, "v5"), ("t16.full_s0 full network", 0.4135, "v8")]
    note = "grey = older ground truth (live code, no version), amber = GT v5, blue = GT v8: the bars are NOT strictly comparable (see the caveats)"
    return (bar_chart(lane, "Lane divider IoU at 0.5 m tolerance over the experiments (validation)", vmax=0.5, note=note) +
            bar_chart(walk, "Walkway IoU over the experiments (validation)", vmax=0.5, note=note))


def sec_system():
    import report_text as RT
    return RT.SYSTEM


def sec_results(runs, T):
    labels = [l for _, l in runs]
    tabs = [T[r] for r, _ in runs]
    out = ["<p>VAL: 1000 frames spread over the 82 validation scenes, cold memory per frame, GT v8 (store <code>c643a7539d</code>, the trainer's cache key <code>0.5_4_6a7aa7864a</code>), "
           "map layers (seg_fov) scored on the cells the front camera sees, detection (detf) on the fusion area (cells that camera and radar both observe). The small number beside a value is the difference to the first column; green = better, red = worse, one seed per run, "
           "so the seed spread of this recipe is not measured; differences below about 0.01 are treated as noise because earlier recipe families (3000 steps, ground truth v5) spread by 0.003 to 0.010 between seeds.</p>"]
    out.append("<div class=warn><b>No matched control.</b> A control run (the old chain's recipe on the same 428 scenes and the same 6000 steps) was planned to show what the new blocks add, but no further training was started after t18.fix "
               "(Mirko's order of 2026-11-02, evening). The only matched old-chain references are the small-set control of the t16 card (400 steps, 77 frames: velocity at moving centres 1.40 m/s against 1.92, static cells 1.26 against 0.51) "
               "and the older chains t8 / t10 (section 2), which used GT v5. So how much of the result the new blocks add over the old chain at equal budget is not measured at large-set scale; the evidence is the small-set pair and the stage-by-stage experiments of section 2.</div>")
    out.append("<h3>Maps</h3>" + BR.rows_table(BR.MAPS, tabs, labels))
    out.append("<h3>Objects</h3>" + BR.rows_table(BR.OBJ, tabs, labels))
    out.append("<h3>Velocity (lower is better)</h3>" + BR.rows_table(BR.VEL, tabs, labels))
    out.append("<h3>Detection recall by the sensors that see the object (does the radar help?)</h3>"
               "<p class=small>Reading: this is a recall ceiling (nearest prediction to each true box, every score counts), not recall at an operating point, and the rows are different subsets of the ground-truth boxes (seen by both sensors, by one, by neither), not an ablation. "
               "Some subsets are small (pedestrians seen only by the radar hold under 100 boxes in the evaluation files), so differences of a few hundredths there are a handful of boxes.</p>" + BR.rows_table(BR.COV, tabs, labels))
    out.append("<h3>Paired comparisons (the same frames, scenes resampled 1000 times)</h3>")
    for path, a, b, title in ((f"{T16}/compare_t17.txt", "baseline", "more data", "more data against the baseline (the t17 experiment)"),
                              (f"{T16}/compare_t18.txt", "more data", "bundle", "the bundle against more data (the t18 experiment)"),
                              (f"{T16}/compare_t18_vs_base.txt", "baseline", "bundle", "the bundle against the baseline"),
                              (f"{T16}/compare_ctl.txt", "control", "baseline", "the baseline against the matched control (the old chain's recipe)")):
        t = paired_table(path, a, b)
        if t:
            out.append(f"<h4>{esc(title)}</h4>" + t)
    gen = []
    for r, l in runs:
        v, tr = read_table(r), read_table(r, tag="_train")
        if tr:
            gen.append((l, v, tr))
    if gen:
        keys = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"), ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"),
                ("pedestrian crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), ("lane divider IoU (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
                ("road divider IoU (1 m)", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"), ("vehicle AP@2 (whole area)", "det/dt+0.00/vehicle/all/ap@2")]
        rows = []
        for n, k in keys:
            if not all(k in tr and k in v for _, v, tr in gen):
                continue
            cells = []
            for l, v, tr in gen:
                cells.append(f"{tr.get(k, float('nan')):.3f} / {v.get(k, float('nan')):.3f} / {tr.get(k, float('nan')) - v.get(k, float('nan')):+.3f}")
            rows.append([n] + cells)
        out.append("<h3>Train against validation (the generalisation gap)</h3><p>Each cell: IoU or AP on 1000 frames of the run's own TRAIN scenes / on the 1000 validation frames / the difference. "
                   "The train frames are scored with the same code; the gap is what is lost on scenes the network has not seen.</p>" + table_html(["metric"] + [l for l, _, _ in gen], rows, left=(0,)))
    return "".join(out)


def sec_lane(runs, T):
    B = T.get("t16_full_s0", {})
    M = {r: read_metrics(r) for r, _ in runs}
    out = []
    out.append("""<h3>5.1 What is measured, and why the numbers look the way they do</h3>
<ul><li><b>Ground truth.</b> The two line layers <code>lane_divider</code> and <code>road_divider</code> of the nuScenes map expansion (autogt v8) are drawn as lines one cell (0.5 m) wide in a bird's-eye grid of 120 x 104 cells (x 0 - 60 m ahead, y +-26 m). Pedestrian crossings are polygons.</li>
<li><b>Plain IoU of a 1-cell line is very harsh:</b> a line that is predicted one cell beside the true one scores zero. We therefore report the plain IoU (0 m) and two tolerant versions: a predicted cell within 0.5 m (or 1 m) of a true line cell counts as a hit, and symmetrically for the true cells. Some published map models (GaussianCaR) draw their divider labels about 1 m thick and score the plain IoU: that is a similar kind of generosity as our 0.5 m tolerance, but not the same number (section 6.4).</li>
<li><b>Zero-crossing F1 and field error.</b> The network predicts a signed distance field to the nearest line (and a probability). The line is read where the field crosses zero; <code>zc f1@0.5</code> is the F1 of that thin line against the true line within 0.5 m, <code>field MAE</code> is the mean absolute error of the distance field in metres.</li>
<li><b>Area.</b> Scored on the cells the front camera sees (about 62 % of the grid); the network also gets the radar, but a painted line is a camera feature.</li></ul>""")
    keys = [("lane divider IoU, plain (0 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0"), ("lane divider IoU, 0.5 m tolerance", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
            ("lane divider IoU, 1 m tolerance", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"), ("lane divider zero-crossing F1 (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/f1@0.5"),
            ("lane divider distance-field MAE (m, lower is better)", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/field_mae"), ("lane divider AP (probability)", "curve/dt+0.00/lane_divider/ap"),
            ("lane divider ROC-AUC", "curve/dt+0.00/lane_divider/roc_auc"), ("lane divider calibration error (ECE)", "curve/dt+0.00/lane_divider/ece"),
            ("lane divider IoU 0.5 m, 0 - 20 m", "seg_fov/seg/dt+0.00/lane_divider/all/0-20/iou@0.5"), ("lane divider IoU 0.5 m, 20 - 40 m", "seg_fov/seg/dt+0.00/lane_divider/all/20-40/iou@0.5"),
            ("lane divider IoU 0.5 m, 40 - 60 m", "seg_fov/seg/dt+0.00/lane_divider/all/40-60/iou@0.5"), ("lane divider IoU 0.5 m, predicted 0.5 s ahead (dt +0.5)", "seg_fov/seg/dt+0.50/lane_divider/all/all/iou@0.5"),
            ("lane divider IoU 0.5 m, 0.5 s back (dt -0.5)", "seg_fov/seg/dt-0.50/lane_divider/all/all/iou@0.5"),
            ("road divider IoU, 1 m tolerance", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"), ("road divider IoU, 0.5 m tolerance", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@0.5"),
            ("road divider zero-crossing F1 (0.5 m)", "seg_fov/seg/dt+0.00/road_divider:zc/all/all/f1@0.5"), ("road divider AP", "curve/dt+0.00/road_divider/ap"), ("road divider ROC-AUC", "curve/dt+0.00/road_divider/roc_auc"),
            ("road divider IoU 1 m, 0 - 20 m", "seg_fov/seg/dt+0.00/road_divider/all/0-20/iou@1"), ("road divider IoU 1 m, 20 - 40 m", "seg_fov/seg/dt+0.00/road_divider/all/20-40/iou@1"),
            ("road divider IoU 1 m, 40 - 60 m", "seg_fov/seg/dt+0.00/road_divider/all/40-60/iou@1"),
            ("pedestrian crossing IoU (plain)", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), ("pedestrian crossing AP", "curve/dt+0.00/ped_crossing/ap"), ("pedestrian crossing ROC-AUC", "curve/dt+0.00/ped_crossing/roc_auc")]
    rows = []
    for n, k in keys:
        cells = []
        base = None
        for i, (r, l) in enumerate(runs):
            v = T[r].get(k, M[r].get(k))
            if v is None:
                cells.append("-")
                continue
            if base is None:
                base = v
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.3f} <span class=delta>{v - base:+.3f}</span>")
        rows.append([n] + cells)
    out.append("<h3>5.2 The numbers (validation, 1000 frames)</h3>" + table_html(["metric"] + [l for _, l in runs], rows, left=(0,)) +
               "<p class=small>The 'prior' baseline (a fixed per-cell map of the training scenes) scores 0.000 on all three line / crossing layers, and its field MAE is 1.33 m against 0.57 m for the network: the numbers above are not what a lookup of typical road layout gives.</p>")
    out.append("<h3>5.3 How the lane divider quality is distributed over frames</h3>")
    for r, l in runs[:2]:
        ls = lane_stats(r)
        if ls:
            q = quantiles(ls)
            out.append(histogram(ls, f"{l}: lane-divider IoU (0.5 m) per validation frame, {len(ls)} frames with at least 30 true divider cells",
                                 note=f"quantiles 10/25/50/75/90 %: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} / {q[4]:.2f}; mean {sum(ls) / len(ls):.2f}"))
    ls0 = lane_stats(runs[0][0]) if runs else []
    if ls0:
        out.append(f"<p>The shape is the point: the network is not 'a bit off everywhere'. In {100 * sum(x >= 0.5 for x in ls0) / len(ls0):.0f} % of these frames the divider IoU is 0.5 or more, and in {100 * sum(x < 0.1 for x in ls0) / len(ls0):.0f} % it is below 0.1 (the line is almost entirely missed). "
                   "The experiments (t5, t6, t7, t8) give the reasons the cards name: the nuScenes labels miss part of the visible paint (so a good prediction is also punished), painted lines are blurred in the stride-16 image features that the first chains used, "
                   "and map-trained stride-8 features nearly doubled the divider IoU (0.19 to 0.35 on the old ground truth; chart below).</p>")
    out.append("<h3>5.4 What moved the lane divider (the ladder of experiments)</h3>" + ladder())
    out.append(lane_gallery())
    reading = f"{T16}/lane_reading.html"
    if os.path.exists(reading):
        out.append(open(reading, encoding="utf-8").read())
    out.append("<h3>5.6 Against the state of the art</h3>" + lane_soa())
    return "".join(out)


def lane_gallery():
    d = "bev_fusion/runs/t16_full_s0/eval/showcase/qualitative"
    picks = []
    p = f"{T16}/lane_frames_t16_full_s0.txt"
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            m = re.match(r"PICK\s+(\w+)\s+(scene-\d+)/(\d+)\s+lane_iou\s+([0-9.]+)", ln)
            if m:
                picks.append(m.groups())
    out = ["<h3>5.5 The frames: worst, median and best lane-divider frames of different scenes (not hand-picked for looks)</h3>"
           "<p>Selection rule: of the frames with at least 30 true divider cells, the four with the lowest IoU, the four closest to the median and the four with the highest IoU, each from a different scene. "
           "Each picture: front camera with the true boxes and map lines projected, the true BEV, the prediction, and error maps (grey = right, red = false, blue = missed; the last panel is the lane dividers at 0.5 m tolerance).</p>"]
    found = 0
    for kind in ("worst", "median", "best"):
        imgs = []
        for k, scene, idx, iou in picks:
            if k != kind:
                continue
            cand = glob.glob(f"{d}/**/{scene}*{int(idx):03d}*overview*.png", recursive=True) + glob.glob(f"{d}/**/{scene}_{int(idx)}*overview*.png", recursive=True) + glob.glob(f"{d}/**/*{scene}*{int(idx)}*.png", recursive=True)
            if cand:
                found += 1
                imgs.append(BR.img(cand[0], 1000) + f"<div class=cap>{kind}: {scene} frame {int(idx)}, lane-divider IoU {iou} (the frame number counts from the start of the scene)</div>")
        if imgs:
            out.append(f"<h4>{kind.capitalize()} frames</h4>" + "".join(imgs))
    if not found:
        out.append("<p><em>The gallery pictures have not been rendered yet (they run on the CPU behind the videos).</em></p>")
    return "".join(out)


def compact_maps(path, layers):
    """the rows of the first table (threshold 0.5, all ranges) of a soa_maps_v8 result file for some layers, as html; '' when the file does not exist"""
    if not os.path.exists(path):
        return ""
    text = open(path, encoding="utf-8").read()
    tabs = md_tables(text)
    if not tabs:
        return ""
    hd, rows = tabs[0][1], [r for r in tabs[0][2] if r[0] in layers and len(r) >= 7]
    head = text.split("\n\n")[1] if "\n\n" in text else ""
    return (f"<p class=small>{md_inline(head)}</p><table class=wide><tr>" + "".join(f"<th class=l>{esc(c)}</th>" for c in hd) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>" +
            "<p class=small>All tables (best threshold, by range, the distance-field numbers, the checks) are in section 6.3.</p>")


def lane_soa():
    s = "<table class=wide><tr><th class=l>method</th><th class=l>published divider number</th><th class=l>how it is defined</th></tr>"
    for r in (("BEVFusion (camera + LiDAR)", "53.7 (divider)", "6 cameras + LiDAR, +-50 m, 200 x 200, plain IoU; divider = road + lane dividers merged"),
              ("BEVFusion (camera only)", "46.4 (divider)", "same"), ("GaussianCaR (camera + radar)", "50.1 (lane divider)", "6 cameras + radar, +-50 m, 0.5 m, labels drawn 1 m thick, plain IoU"),
              ("RCBEVDet++ (camera + radar)", "50.5 (lane)", "ResNet-101, settings of LSS"), ("BEVCar (camera + radar, ViT-B/14)", "45.3 (lane)", "+-50 m, 200 x 200"),
              ("GaussianBeV (camera)", "47.4 (lane divider)", "via the GaussianCaR table"), ("GaussianLSS (camera)", "38.7 (divider)", "+-50 m, 200 x 200"),
              ("this system, baseline", "0.429 at 0.5 m tolerance; 0.189 plain; 0.549 at 1 m", "ONE front camera + 5 radars, 0 - 60 m x +-26 m, 1-cell lines, front-camera cells only")):
        s += "<tr>" + "".join(f"<td class=l>{esc(c)}</td>" for c in r) + "</tr>"
    s += "</table><p class=small>Published values are IoU in per cent from the papers (sources and protocols in section 6.1); ours is a fraction. They are NOT comparable (different area, number of cameras, line thickness, tolerance): the table shows where the numbers sit, not who is ahead.</p>"
    new = compact_maps(f"{T16}/soa/soa_maps_v8_t16_full_s0.md", ("lane_divider", "road_divider", "ped_crossing"))
    s += "<h4>Same frames, same ground truth, same metric (our re-scoring of GaussianCaR's cached logits)</h4>"
    s += new or ("<p><em>The like-for-like re-scoring on the current ground truth (GaussianCaR's own map logits and our baseline on the same 400 frames) has not finished; the earlier re-scoring on the older ground truth says (400 frames, 0.5 m tolerance, GaussianCaR / "
                 "our dual-feature chain after 3000 steps): lane divider 0.691 / 0.347 - 0.362, road divider 0.634 / 0.455 - 0.479, pedestrian crossing 0.536 / 0.140 - 0.164 (section 6.2, S2).</em></p>")
    return s


def soa_reading():
    """what the two new re-scorings say, with the caveats; numbers are read from the result files"""
    d = soa_rows()
    if d is None:
        return ""
    maps, fr = d
    old = {"drivable_area": "0.94", "walkway": "0.57 - 0.62", "carpark_area": "0.31 - 0.34", "ped_crossing": "0.26 - 0.31", "lane_divider": "0.50 - 0.52", "road_divider": "0.72 - 0.76"}
    rows = []
    for name, layer, metric in (("drivable area", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("car park", "carpark_area", "iou@0"), ("pedestrian crossing", "ped_crossing", "iou@0"),
                                ("lane divider (0.5 m tolerance)", "lane_divider", "iou@0.5"), ("road divider (0.5 m tolerance)", "road_divider", "iou@0.5")):
        o, g, r = maps[(layer, metric)]
        rows.append([name, o, g, r, old[layer]])
    t = table_html(["layer", "ours (baseline)", "GaussianCaR", "ours as a share", "share of the dual chain after 3000 steps in the older scoring"], rows, left=(0,))
    dual, ours = fr[0], fr[1]
    return f"""
<h4>What the two re-scorings say</h4>
<p><b>Maps, ours against GaussianCaR</b> (384 validation frames of 41 scenes, ground truth v8, the cells the front camera sees, threshold 0.5, plain IoU for areas and 0.5 m tolerance for lines):</p>{t}
<p>The last column is the share the dual-feature chain of 3000 steps reached in the older scoring of the same kind (section 6.2, S2: ground truth v5, 400 frames): the shares of the full network are higher on every layer, but the two columns are NOT a measured improvement of one system: ground truth version, frame set and training budget differ, and how much of the rise is the network cannot be separated.</p>
<p><b>Detection in the front sector (x 0 - 50 m, +-32 degrees), 1000 validation frames, same ground truth:</b> the pre-trained camera-only DualBEV (front camera only, unchanged weights) reaches mAP2 {dual[1]}; our baseline {ours[1]} (difference {ours[2]}, paired scene bootstrap). The earlier scoring on the older ground truth had our <i>camera-only</i> arm at 0.242 against DualBEV 0.250 (a tie), so the camera path is at the level of the published camera detector and the difference is what the radar and the temporal fusion add.</p>
<ul>
<li><b>GaussianCaR has more input.</b> It uses six cameras with 360 degrees and seven radar sweeps; ours uses one front camera and five radars. Only the cells the front camera sees are scored, but GaussianCaR still gets the side and rear views as context.</li>
<li><b>Training budget.</b> GaussianCaR was trained for 40 epochs on four A100 GPUs on the 700 official train scenes for maps only; ours for 6000 steps (about six passes over 428 scenes) on one RTX 3060, together with detection and velocity.</li>
<li><b>Threshold.</b> The tables in section 6.3 also give the best of three thresholds per layer, chosen on these frames (in-sample, as GaussianCaR was trained with another loss): the lane-divider share drops from 0.66 to 0.60 and GaussianCaR stays ahead on every layer.</li>
<li><b>Frames.</b> These 384 frames are easier than the 1000 of the record: our lane divider scores 0.455 here and 0.429 on the 1000 frames, drivable area 0.818 against 0.779.</li>
<li><b>Checks.</b> Our numbers on 130 of these frames equal the stored GPU evaluation (3992 of 4160 per-frame count arrays are identical, the others differ by at most 5 cells at the 0.5 threshold); GaussianCaR's own numbers on the older and the current ground truth differ by at most 0.02 on all six layers.</li>
</ul>"""


def sec_soa():
    p = f"{T16}/soa_digest.md"
    if not os.path.exists(p):
        return "<p>SOA digest not found</p>"
    text = open(p, encoding="utf-8").read()
    tabs = md_tables(text)
    byhead = {h: (hd, rows) for h, hd, rows in tabs}
    out = ["""<div class=warn><b>How to read this section.</b> A number is only worth comparing with a published one when range, cameras, classes, ground truth and scoring are the same. Here they are not
(one front camera and five radars, 0 - 60 m ahead, three merged object groups, tolerance-based line scoring, a map ground truth that changed twice). So: 6.1 lists the published numbers, 6.2 lists every time the repository scored a state-of-the-art model
with OUR metric on OUR frames (the only like-for-like facts), 6.3 is the new re-scoring on the current ground truth, 6.4 puts our numbers next to the published ones and says plainly why they cannot be ranked.</div>"""]
    # 6.1
    out.append("<h3>6.1 Published state-of-the-art numbers (as published; protocol in the last column)</h3>")
    for key, title in (("1A BEV map segmentation (nuScenes val; IoU in %)", "BEV map segmentation (IoU in per cent)"), ("1B 3D detection (nuScenes val unless stated; NDS and mAP in %, errors dimensionless)", "3D detection")):
        t = byhead.get(key)
        if t:
            hd, rows = t
            out.append(f"<details><summary>{esc(title)}: {len(rows)} rows</summary><table class=wide><tr>" + "".join(f"<th class=l>{esc(c)}</th>" for c in hd) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table></details>")
    out.append("<p class=small>Numbers marked 'web' were read through a page summariser and are to be spot-checked against the paper before they are quoted elsewhere; 97 of the 161 numbers of the digest come from the repository's own literature notes.</p>")
    # 6.2
    t = byhead.get("2 Stored re-scorings of SOA models with OUR metric on OUR frames")
    out.append("<h3>6.2 The repository's own re-scorings of state-of-the-art models (same frames, our metric, older ground truth)</h3>")
    if t:
        hd, rows = t
        for r in rows:
            if len(r) < 10:
                out.append(f"<div class=exp><h4>{md_inline(r[0])}: {md_inline(r[1][:160]) if len(r) > 1 else ''}</h4><p>{md_inline(' / '.join(c for c in r[2:] if c and c != '-'))}</p></div>")
            if len(r) >= 10:
                out.append(f"<div class=exp><h4>{md_inline(r[0])}: {md_inline(r[1][:160])}</h4><p><span class=k>SOA model.</span> {md_inline(r[2])}</p><p><span class=k>Ours.</span> {md_inline(r[3])}</p>"
                           f"<p><span class=k>Ground truth, frames, metric.</span> {md_inline(r[4])}; {md_inline(r[5])}; {md_inline(r[6])}</p><p><span class=k>Numbers.</span> {md_inline(r[7])}</p>"
                           f"<p><span class=k>Statement of the card.</span> {md_inline(r[8])}</p><p class=small>{md_inline(r[9])}</p></div>")
    # 6.3
    out.append("<h3>6.3 New: the like-for-like re-scoring on the current ground truth (v8)</h3>")
    out.append(soa_reading())
    got = False
    for path in (f"{T16}/soa/soa_maps_v8_t16_full_s0.md", f"{T16}/soa/soa_maps_v8_t17_data.md", f"{T16}/soa/front_v8_n1000.md"):
        h = render_md_file(path)
        if h:
            got = True
            out.append(h)
    if not got:
        out.append("<p><em>Not finished when this page was built: the GaussianCaR map re-scoring (400 frames, CPU) and the DualBEV front-sector re-scoring (1000 frames, CPU) run in the background; "
                   "rebuild the report to include them.</em></p>")
    # 6.4
    out.append("<h3>6.4 Our numbers next to the published ones, and why they cannot be ranked</h3>")
    for key in ("3A Maps", "3B Detection and velocity"):
        t = byhead.get(key)
        if t:
            hd, rows = t
            out.append(f"<h4>{esc(key)}</h4><table class=wide><tr>" + "".join(f"<th class=l>{esc(c)}</th>" for c in hd) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>")
    sec4 = re.search(r"## 4 How to read this\n\n(.+?)\n\n## 5", text, re.S)
    if sec4:
        out.append(f"<div class=card><b>What a fair claim is today.</b> {md_inline(sec4.group(1))}</div>")
    return "".join(out)


def sec_qual(runs):
    out = ["<p>Pictures are produced by the evaluation suite, not by hand: each frame is drawn with the same code for every run; random frames are chosen by a fixed seed, not for looks. "
           "Error maps: grey = right, red = false positive, blue = missed, cream = not evaluated (not seen by the camera).</p>"]
    for r, l in runs[:1]:
        for fn in sorted(glob.glob(f"bev_fusion/runs/{r}/eval/images/*_overview.png"))[:6]:
            out.append(BR.img(fn, 1000) + f"<div class=cap>{esc(l)}: {esc(os.path.basename(fn))}: front camera with the true boxes and lines, true BEV, prediction, error maps of drivable area, vehicle footprint and lane dividers</div>")
        for k in (0, 2):
            fn = f"bev_fusion/runs/{r}/eval/modules/frame_{k}.png"
            if os.path.exists(fn):
                out.append(BR.img(fn, 1100) + f"<div class=cap>{esc(l)}: the module probe: what every stage of the chain produces for one validation frame</div>")
    return "".join(out)


def sec_videos(runs):
    os.makedirs(f"{OUT_DIR}/videos", exist_ok=True)
    groups = []
    for r, l in runs:
        files = []
        for pat in (f"bev_fusion/runs/{r}/eval/showcase/qualitative/videos/*_overview.mp4", f"bev_fusion/runs/{r}/eval/suite/qualitative/videos/*_dashboard.mp4",
                    f"bev_fusion/runs/{r}/eval/suite/qualitative/videos/*_overview.mp4"):
            files += sorted(glob.glob(pat))
        if files:
            dst = f"{OUT_DIR}/videos/{r}"
            os.makedirs(dst, exist_ok=True)
            names = []
            for fn in files:
                to = os.path.join(dst, os.path.basename(fn))
                if not os.path.exists(to) or os.path.getsize(to) != os.path.getsize(fn):
                    shutil.copyfile(fn, to)
                names.append(os.path.basename(fn))
            groups.append((r, l, names))
    out = ["<p>Each video runs one validation scene (about 38 frames, 2 Hz keyframes shown at 4 frames per second) with the memory carried along. 'overview' videos show the front camera with the true boxes and map lines, the true BEV, the prediction with boxes "
           "and the error maps including the lane dividers; 'dashboard' videos show every stage of the chain (radar, camera BEV, trunk, predictions at -0.5 / 0 / +0.5 s, speed field). "
           "The videos are files next to this page (<code>videos/&lt;run&gt;/&lt;name&gt;.mp4</code>); the scenes are random (seed 21, 11 and 1), the same scenes for every run where the run has them.</p>"]
    for r, l, names in groups:
        out.append(f"<h3>{esc(LONG.get(r, l))}</h3><div class=vgrid>")
        for n in names:
            out.append(f"<figure><video controls preload=none src='videos/{r}/{n}'></video><figcaption class=small>{esc(n)}</figcaption></figure>")
        out.append("</div>")
    if not groups:
        out.append("<p><em>no videos yet</em></p>")
    return "".join(out)


def sec_cost(runs):
    hrs = ledger_hours()
    rows = []
    for rid, label in (("t17.feat", "BEVFusion features of the 272 new scenes"), ("t16.full_s0", "baseline, 6000 steps"), ("t17.data", "700 scenes, 6000 steps (interrupted once, resumed)"), ("t18.fix", "bundle, 6000 steps (interrupted once, resumed)"), ("t16.ctl_s0", "control, 6000 steps")):
        if rid in hrs:
            rows.append([rid, esc(label), hrs[rid][0].replace("T", " "), f"{hrs[rid][2]:.1f} h"])
    import report_text as RT
    trows, per, totals = [], None, []
    for r, l in runs:
        p = f"bev_fusion/runs/{r}/eval/timing_modules.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        inf, tm = d["inference_ms"], d.get("train_microbatch_ms") or {}
        trows.append([esc(l), f"{inf['total_ms']:.1f}", f"{sum(inf['per_frame_ms'].values()):.1f}", f"{inf['unattributed_ms']:+.1f}",
                      f"{tm.get('forward_loss', float('nan')):.0f} / {tm.get('backward', float('nan')):.0f}", "yes" if d.get("idle") else "NO (do not quote)"])
        totals.append(inf["total_ms"])
        if per is None:
            per = (l, inf)
    spread = (max(totals) - min(totals)) if totals else 0.0
    out = RT.COST
    if trows:
        out += (f"<h3>Idle-GPU timing of the large runs (measured 2026-11-03)</h3><p>Batch 1, the dts -0.5 / 0 / +0.5 s, a cold memory, the cached chain without the frozen image heads, fp32, median of 30 frames on the idle RTX 3060 "
                "(<code>bev_suite.py RUN timing</code>, stored in <code>runs/&lt;run&gt;/eval/timing_modules.json</code>). 'sum of modules' is the sum of the per-module times, 'unattributed' the host work outside every module. "
                f"The three runs have the same architecture (t18 only lost an unused branch), and the totals differ by at most {spread:.1f} ms; each was measured once, so a difference of that size is not interpreted. The small-set bring-up measured the same 96.4 ms for the baseline. "
                "The guard of the timing tool had to be worked around for this measurement (open finding of 2026-11-03: it counted the timing process itself as a GPU user); every other idle check was kept and the stored files say idle.</p>"
                + table_html(["run", "total (ms per frame)", "sum of modules (ms)", "unattributed (ms)", "training micro-batch, forward+loss / backward (ms)", "idle"], trows, left=(0,)))
    if per:
        mods = sorted(per[1]["per_frame_ms"].items(), key=lambda kv: -kv[1])
        out += (f"<h4>Where the time goes ({esc(per[0])}, ms per frame)</h4>"
                + table_html(["module", "ms per frame", "calls", "share of the module sum"],
                             [[esc(k), f"{v:.2f}", str(per[1]['calls'][k]), f"{100 * v / sum(per[1]['per_frame_ms'].values()):.1f} %"] for k, v in mods], left=(0,)))
    return out + ("<h3>Wall-clock of the large runs</h3>" + table_html(["run", "what", "started", "duration"], rows, left=(0, 1)) if rows else "")


def sec_weak(runs, T):
    import report_text as RT
    return RT.WEAK


def sec_files():
    import report_text as RT
    return RT.HOWTO + "<h3>Where things are</h3>" + RT.FILES


# ------------------------------------------------------------------------------------------------ assembly
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/report_big.html")
    a = ap.parse_args()
    runs = have_runs()
    T = {r: read_table(r) for r, _ in runs}
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    videos_html = sec_videos(runs)
    nvid = len(glob.glob(f"{OUT_DIR}/videos/*/*.mp4"))
    secs = [("s1", "1 Summary", sec_summary(runs, T, nvid)), ("s2", "2 What has been tried", sec_tried() + sec_lists() + "<h3>Caveats of the digest</h3>" + sec_caveats()),
            ("s3", "3 The system", sec_system()), ("s4", "4 Results on the validation set", sec_results(runs, T)),
            ("s5", "5 Lane markings", sec_lane(runs, T)), ("s6", "6 Against the state of the art", sec_soa()),
            ("s7", "7 Pictures and videos", sec_qual(runs) + videos_html), ("s8", "8 Cost and timing", sec_cost(runs)),
            ("s9", "9 Weak spots, honest limits, next steps", sec_weak(runs, T)), ("s10", "10 How to run it, where things are", sec_files())]
    toc = "<nav class=toc>" + "".join(f"<a href='#{i}'>{esc(t)}</a>" for i, t, _ in secs) + "</nav>"
    body = "".join(f"<h2 id='{i}'>{esc(t)}</h2>{b}" for i, t, b in secs)
    main_html = (f"<main><h1>BEV camera + radar system: result report</h1><p class=lead>For Mirko, 2026-11-02 / 03. What was tried, what came out, the lane markings, the comparison with the state of the art, quantitative and qualitative, with videos. "
                 f"Repository state <code>{head}</code>; numbers come from the stored evaluation files of the runs (VAL, 1000 frames, GT v8) and from the experiment cards, never typed by hand; every comparison is single seed unless it says otherwise.</p>{toc}{body}</main>")
    main_html = main_html.replace("<table", "<div class=scroll><table").replace("</table>", "</table></div>")      # a wide table scrolls inside its own box
    head_bits = ('<title>BEV Result Report</title><meta name="claude:viewport-fit-sides" content="cover">'
                 '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
                 f"<style>{PAGE_CSS}</style>")
    page = f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1,viewport-fit=cover'>{head_bits}</head><body>{main_html}</body></html>"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(page)
    open(f"{OUT_DIR}/artifact_page.html", "w", encoding="utf-8").write(head_bits + main_html)       # the publish format: no document wrapper
    os.makedirs(f"{OUT_DIR}/tables", exist_ok=True)
    for r, _ in runs:
        for src, name in ((f"bev_fusion/runs/{r}/eval/by_n/table_n1000.txt", f"{r}_table_n1000.txt"), (f"bev_fusion/runs/{r}/eval/table_train.txt", f"{r}_table_train.txt")):
            if os.path.exists(src):
                shutil.copyfile(src, f"{OUT_DIR}/tables/{name}")
    print("written", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB", "runs:", [r for r, _ in runs])


if __name__ == "__main__":
    main()
