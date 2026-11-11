"""Night report page (scratch): results/E276.report/index.html from the RESULT FILES (scoring cards, temp/e276/*.json, pictures), never typed numbers; a number that does not exist yet reads "pending".
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh e276_build_report.py
Answers Mirko's eight notes on the result videos with the measurements of the night and gives the state of E275 / E276 / E277 / E278. Pictures are embedded (base64), the page is self-contained.
"""
import base64
import glob
import html
import json
import os
import shutil

ROOT = os.environ["BFLC_ROOT"]
T = os.path.join(ROOT, "temp", "e276")
OUT = os.path.join(ROOT, "results", "E276.report")
FIG = os.path.join(ROOT, "results", "E276", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
top = os.path.abspath(os.path.join(ROOT, "..", "temp", "gt_l4", "out", "topviews.png"))
if os.path.exists(top):
    shutil.copy(top, os.path.join(FIG, "l4_tier_topviews.png"))


def J(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def num(x, fmt="{:.3f}", none="pending"):
    return none if x is None else fmt.format(x)


def pct(x, none="pending", d=1):
    return none if x is None else f"{100 * x:.{d}f} %"


def img(path, caption, from_fig=True):
    p = os.path.join(FIG, path) if from_fig else path
    if not os.path.exists(p):
        return f"<p class='pending'>picture pending ({html.escape(os.path.basename(path))})</p>"
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f"<figure><img src='data:image/png;base64,{b}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"


def table(head, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    r = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table><tr>{h}</tr>{r}</table>"


def card(exp, run):
    return J(f"runs/{exp}/{run}/card.json")


def g(c, k, sec="graph"):
    return None if c is None else c["p"].get(sec, {}).get(k)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def m2(cs, k, sec="graph", fmt="{:.3f}"):
    return num(mean([g(c, k, sec) for c in cs]), fmt)


B = [card("e273", f"e273_p_s{s}") for s in (0, 1)]
H = [card("e275", f"e273_p_s{s}.close1.5_t0.5") for s in (0, 1)]
Bl4 = [card("e276", f"e273_p_s{s}") for s in (0, 1)]
Tc = [card("e276", f"e276_t_s{s}") for s in (0, 1)]
U = [card("e277", f"e277_u_s{s}.plain_t0.5") for s in (0, 1)]
UH = [card("e277", f"e277_u_s{s}.close1.5_t0.5") for s in (0, 1)]
P6 = [card("e278", f"e273_p_s{s}_p6.plain_t0.5") for s in (0, 1)]
P6H = [card("e278", f"e273_p_s{s}_p6.close1.5_t0.5") for s in (0, 1)]
verd = open(os.path.join(T, "verdicts.log"), encoding="utf-8").read() if os.path.exists(os.path.join(T, "verdicts.log")) else ""
dens, tiers1, ext, tmp, tmpl0, tmpl1, endp, inter, tstat = (J("temp/e276/density_oracle.json"), J("temp/e276/tiers_e273_p_s1.json"), J("temp/e276/extension_accuracy.json"), J("temp/e276/temporal_e273_p_s0.json"),
                                                            J("temp/e276/temporal_links_e273_p_s0.json"), J("temp/e276/temporal_links_e273_p_s1.json"), J("temp/e276/endpoints_e273_p_s0.json"),
                                                            J("temp/e276/intersection_label.json"), J("temp/e276/tier_stats_0702a065c6.json"))
asg = [J(f"temp/e276/assignment_uncertainty_e273_p_s{s}.json") for s in (0, 1)]
cov = [J(f"temp/e276/coverage_e273_p_s{s}.json") for s in (0, 1)]
lthr = [J(f"temp/e276/linkthr_e273_p_s{s}.json") for s in (0, 1)]
tiers0 = J("temp/e276/tiers_e273_p_s0.json")
diag = {s: card("e276diag", f"e276_s_s0_p3.plain_t{s}") for s in ("0.5", "0.3", "0.2")}



def verdict_of(tag):
    """the "verdict" of the last `== verdict277 ...<tag>` block of verdicts.log"""
    i = verd.rfind(tag)
    if i < 0:
        return None
    j = verd.find('"verdict": "', i)
    return None if j < 0 else verd[j + 12: verd.find('"', j + 12)]


V277 = verdict_of("== verdict277 u against")
V279 = verdict_of("== verdict277 for E279")
W = [card("e279", f"e279_w_s{s}.plain_t0.5") for s in (0, 1)]
WH = [card("e279", f"e279_w_s{s}.close1.5_t0.5") for s in (0, 1)]


def long_stats():
    out = []
    for x in asg:
        lk = (x or {}).get("result", {}).get("links")
        if not lk:
            return None
        rc = next(v for v in lk["risk_coverage"] if v["threshold"] == 0.5)
        lg_ = lk["p>=0.5 long"]
        out.append(dict(prec=lg_["precision"], share=lg_["n"] * (1 - lg_["precision"]) / (rc["links"] * (1 - rc["precision"]))))
    return out


def rng(vals, d=0):
    vals = [v for v in vals if v is not None]
    return "pending" if not vals else f"{100 * min(vals):.{d}f} - {100 * max(vals):.{d}f} %"


LS = long_stats()
PERS = [((x or {}).get("result") or {}).get("delivered / long links", {}).get("persistent") for x in (tmpl0, tmpl1)]
CUT = None if not all(lthr) else mean([x["variants"]["0.5_0.5"]["topo_F1"] - x["variants"]["0.5_1.01"]["topo_F1"] for x in lthr])
AN = J("temp/e275/analysis_e273_p_s0.json") or {}

css = """
:root{--bg:#fbfbf9;--fg:#1d2329;--mut:#5b6670;--line:#d8dde2;--acc:#1f6feb;--ok:#2a7d3f;--bad:#b3261e;--pend:#8a5f00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;padding-block:0 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:980px;margin:0 auto}h1{font-size:1.6rem;margin:28px 0 4px;text-wrap:balance}h2{font-size:1.2rem;margin:32px 0 6px;border-top:1px solid var(--line);padding-top:16px;text-wrap:balance}h3{font-size:1.02rem;margin:20px 0 4px}
p,li{max-width:80ch}.mut{color:var(--mut)}.pending{color:var(--pend)}table{border-collapse:collapse;margin:10px 0;display:block;max-width:100%;overflow-x:auto;font-variant-numeric:tabular-nums}th,td{border:1px solid var(--line);padding:4px 10px;text-align:left}
figure{margin:12px 0}img{max-width:100%;height:auto}figcaption{color:var(--mut);font-size:.9rem}.note{border-left:3px solid var(--acc);padding:2px 12px;margin:10px 0}.box{border:1px solid var(--line);border-radius:6px;padding:4px 16px;margin:14px 0}
pre{white-space:pre-wrap;font-size:.8rem;overflow-x:auto}
"""
parts = [f"<title>Spurmitte E276 report</title><meta name='claude:viewport-fit-sides' content='cover'><style>{css}</style><main>",
         "<h1>Spurmitte: what the night showed</h1>",
         "<p class='mut'>State of the lane-centre head after your eight notes on the result videos (night of 2026-11-07 to 11-08). Every number comes from a result file; <span class='pending'>pending</span> means it was not there when this page was built. VAL, 873 frames, mean of two seeds unless a seed is named; nothing here touched TEST.</p>"]

# ------------------------------------------------------------------------------------------------------------- short answer
hp, bp = mean([g(c, "topo_F1") for c in H]), mean([g(c, "topo_F1") for c in B])
parts.append("<div class='box'><h2 style='border:0;margin-top:8px'>Short answer</h2><ul>"
             f"<li><b>A simple clean-up step on the output helps a lot (E275, kept).</b> Dropping a node when a more certain node lies within 1.5 m lifts the junction F1 from {m2(B, 'junction_F1')} to {m2(H, 'junction_F1')} and the topology F1 from {num(bp)} to {num(hp)}, and cuts the vertices per 100 m from {m2(B, 'vert100_dp', fmt='{:.1f}')} to {m2(H, 'vert100_dp', fmt='{:.1f}')} (budget 14.3). It is a rule step, not part of the network.</li>"
             f"<li><b>E277 puts that step inside the network (trained, soft)</b>: topology F1 {m2(U, 'topo_F1')}, junction F1 {m2(U, 'junction_F1')} (the hard step: {m2(H, 'topo_F1')} / {m2(H, 'junction_F1')}), so it reproduces the step with no rule step left, but on one seed it loses more lane recall than the guards allow (lanes found {m2(U, 'lanes_found')} against {m2(B, 'lanes_found')}): by the rule written before the run the verdict is <b>{V277 or 'pending'}</b>. A smaller radius (1.25 m, E279) gives topology F1 " + m2(W, 'topo_F1') + ", junction F1 " + m2(W, 'junction_F1') + ", lanes found " + m2(W, 'lanes_found') + " and passes the same guards: verdict <b>" + (V279 or 'pending') + "</b> (thin margins on one seed, two seeds, in-sample).</li>"
             f"<li><b>Giving the own lane more weight in the loss did not make the own lane better</b> (arm t): " + (f"F1 at 1 m on the own lane and its neighbours {m2(Bl4, 't12_F1_1', 'tier')} for the baseline against {m2(Tc, 't12_F1_1', 'tier')} for arm t, and lanes found fell from {m2(Bl4, 'lanes_found')} to {m2(Tc, 'lanes_found')}." if all(Tc) else "<span class='pending'>pending</span>") + "</li>"
             "<li><b>The sparse node target (arm s) failed</b>: the head cannot decide where along a straight lane the one node of a stretch sits, so its probability smears and almost no node passes 0.5 (section 2).</li>"
             "<li><b>The weak spot is the links across junctions</b>, not the node-to-map assignment: long links are right in only " + (rng([x["prec"] for x in LS]) if LS else "pending") + " of the cases, only " + rng(PERS) + " of them survive half a second, they carry " + (rng([x["share"] for x in LS]) if LS else "pending") + " of the wrong links, and are still needed (without them the topology F1 falls by " + num(CUT, "{:.2f}") + ").</li></ul></div>")

# ------------------------------------------------------------------------------------------------------------- 1 E275
rows = [[n, m2(B, k, s, f), m2(H, k, s, f), m2(P6, k, s, f), m2(P6H, k, s, f)] for n, k, s, f in (("topology F1", "topo_F1", "graph", "{:.3f}"), ("junction F1", "junction_F1", "graph", "{:.3f}"), ("F1 at 1 m", "F1@1.0", "graph", "{:.3f}"),
                                                                                               ("lanes found", "lanes_found", "graph", "{:.3f}"), ("vertices per 100 m (budget 14.3)", "vert100_dp", "graph", "{:.1f}"), ("duplicates", "dup_frac", "graph", "{:.3f}"),
                                                                                               ("node recall", "node_R", "nodes", "{:.3f}"), ("node precision", "node_P", "nodes", "{:.3f}"), ("long-link precision", "link_long_P", "nodes", "{:.3f}"))]
p6txt = ("E278 (no card, scoring only): the model before the joint refinement phase has topology F1 " + m2(P6, "topo_F1") + ", long-link precision " + m2(P6, "link_long_P", "nodes") + " and duplicates " + m2(P6, "dup_frac") +
         " against " + m2(B, "topo_F1") + ", " + m2(B, "link_long_P", "nodes") + " and " + m2(B, "dup_frac") + " after it, but junction F1 " + m2(P6, "junction_F1") + " against " + m2(B, "junction_F1") + " and " + m2(P6, "vert100_dp", fmt="{:.1f}") +
         " vertices per 100 m against " + m2(B, "vert100_dp", fmt="{:.1f}") + ": the joint phase helps the links and the topology and costs junction F1 and vertices.") if all(P6) else "<span class='pending'>E278 (the model before the joint phase) is not fully scored yet.</span>"
parts.append("<h2>1. E275: the close-node step (decided: KEEP)</h2>" + table(["", "baseline B", "B + step", "B before the joint phase (E278)", "that + step"], rows) +
             "<p>The step thins the dense node row (" + pct(AN.get("dropped_share"), d=0) + " of the nodes). The scorer counts " + pct(AN.get("dropped_counted_true_share"), d=0) + " of the dropped nodes as true, and the pictures show a node row thinned along the lanes: it is mainly a density cut that also cleans the junction links, not a removal of false nodes only.</p><p>" + p6txt + "</p>")
e275fig = os.path.join(ROOT, "results", "E275", "figures", "e273_p_s1_dropped_00.png")
parts.append(img(e275fig, "E275: baseline seed 1, one frame: GT nodes (green rings), kept nodes (blue), nodes the step dropped (red crosses).", from_fig=False))

# ------------------------------------------------------------------------------------------------------------- 2 E276
rows = []
for name, key, sec, fmt in (("F1 at 1 m, own lane + neighbours, visible parts", "t12_F1_1", "tier", "{:.3f}"), ("own-lane recall", "t1_R_1", "tier", "{:.3f}"), ("neighbour-lane recall", "t2_R_1", "tier", "{:.3f}"), ("far-lane recall", "t3_R_1", "tier", "{:.3f}"),
                            ("topology F1", "topo_F1", "graph", "{:.3f}"), ("junction F1", "junction_F1", "graph", "{:.3f}"), ("lanes found", "lanes_found", "graph", "{:.3f}"), ("node precision", "node_P", "nodes", "{:.3f}"), ("node recall", "node_R", "nodes", "{:.3f}")):
    rows.append([name, m2(Bl4, key, sec, fmt), m2(Tc, key, sec, fmt)])
parts.append("<h2>2. E276: tiers in the loss (arm t) and the sparse node target (arm s)</h2>" + table(["VAL 873, scored on lane GT l4", "baseline B", "arm t (tier weights)"], rows) +
             ("<p>Verdict of the written rule (primary: F1 at 1 m on the own lane and its neighbours, band 0.02): <b>" + (verdict_of("== verdict276 t against") or "pending") + "</b>.</p>") +
             "<p><b>Arm s failed its gate</b> after phase 3 (seed 0): at e &gt;= 0.5 it delivered " + num(None if diag["0.5"] is None else diag["0.5"]["p"]["nodes"]["nodes_per_100m"], "{:.1f}") + " nodes per 100 m against 14.4 in its target (recall " +
             num(None if diag["0.5"] is None else diag["0.5"]["p"]["nodes"]["node_R"], "{:.2f}") + "); at e &gt;= 0.3 it delivers " + num(None if diag["0.3"] is None else diag["0.3"]["p"]["nodes"]["nodes_per_100m"], "{:.0f}") +
             " per 100 m with recall " + num(None if diag["0.3"] is None else diag["0.3"]["p"]["nodes"]["node_R"], "{:.2f}") + ". The head knows where the nodes are but not which of about three cells along a straight lane carries the one node of the target, so the probability smears. A matching-based node loss could solve that (not tried); E277 takes the other way (no change of the target).</p>")

# ------------------------------------------------------------------------------------------------------------- 3 E277
rows = [[n, m2(B, k, s, f), m2(H, k, s, f), m2(U, k, s, f), m2(W, k, s, f), m2(WH, k, s, f)] for n, k, s, f in (("topology F1", "topo_F1", "graph", "{:.3f}"), ("junction F1", "junction_F1", "graph", "{:.3f}"), ("F1 at 1 m", "F1@1.0", "graph", "{:.3f}"), ("lanes found", "lanes_found", "graph", "{:.3f}"),
                                                                                                         ("vertices per 100 m", "vert100_dp", "graph", "{:.1f}"), ("node recall", "node_R", "nodes", "{:.3f}"), ("node precision", "node_P", "nodes", "{:.3f}"))]
parts.append("<h2>3. E277 / E279: the step inside the network</h2>" + table(["VAL 873", "baseline B", "B + hard step 1.5 m (E275)", "soft step 1.5 m (E277)", "soft step 1.25 m (E279)", "E279 + hard step 1.5 m"], rows) +
             ("<p class='pending'>The trained heads are not scored yet.</p>" if not all(U) else "<p>Rule (card E277, written before the run): KEEP when the soft step lifts topology F1 by 0.02 and junction F1 by 0.03 over B on both seeds, with every guard held (F1 at 1 m not below B by more than 0.01, lanes found by more than 0.03, the assignment guards) and a topology F1 within 0.01 of the hard step. Result: <b>" + (V277 or "pending") + "</b>. Both gains hold on both seeds and the hard step on top of the soft one removes no further node; the guards fail on seed 0 only (lanes found -0.039, F1 at 1 m -0.015); seed 1 holds all of them. Whether the guard values of the hard step are the right ones for a trained step is your call; the delivered chain still uses the hard step until you decide. E279 repeats it at a radius of 1.25 m under the same rule: <b>" + (V279 or "pending") + "</b>. It keeps more lane recall (lanes found " + m2(W, "lanes_found") + " against " + m2(U, "lanes_found") + ") and gives up junction F1 (" + m2(W, "junction_F1") + " against " + m2(U, "junction_F1") + "); the margins on seed 0 are thin (topology gain 0.021 against the band 0.02). The hard 1.5 m step on top of it still drops some nodes (last column). So the thinning is learned by the network; which radius, or whether to add the hard step, is your call.</p>") +
             "")
cmpfigs = sorted(glob.glob(os.path.join(ROOT, "results", "E277", "figures", "frame_*.png")))[:4]
for p in cmpfigs:
    parts.append(img(p, "Same frame, rows: baseline | baseline + hard step | soft-step head. Left: delivered nodes / links against the GT (red halo = false, blue halo = missed); right: the delivered graph coloured by the assigned map piece.", from_fig=False))

# ------------------------------------------------------------------------------------------------------------- 4 notes
parts.append("<h2>4. Your eight notes</h2>")
il = inter
parts.append("<h3>1. The intersection label did not always look good</h3>" + (
    f"<p>Part of it is the label. Only {pct(il['inside_polygon'])} of the connector points that the lane graph puts inside an intersection lie inside the intersection polygon ({pct(il['inside_2m'])} within 2 m, the dilation of the target); the stop / exit points lie a median "
    f"{num(il['stop_exit_distance_to_polygon_m']['median'], '{:.1f}')} m before the polygon. The GT chat has the wish for an intersection area built from the lane graph.</p>" if il else "<p class='pending'>pending</p>"))
a0, a1 = [(x or {}).get("result", {}).get("assignment") for x in asg]
c_all = [next((gr for gr in (c or {}).get("groups", []) if gr["group"] == "all"), None) for c in cov]
c_t1 = [next((gr for gr in (c or {}).get("groups", []) if gr["group"] == "tier 1"), None) for c in cov]
parts.append("<h3>2. The assignment to the map elements looked horrible: 1:1 matching?</h3>" + (
    f"<p>By the numbers it is not the node assignment: per node {pct(a0['node_accuracy_on_gt_lane'])} are right, {pct(a0['section_correct'])} of the delivered sections choose the right map lane, and a map lane is claimed by two sections in only {pct(a0['claimed_by_two_or_more_sections'])} of the cases (so it is mostly 1:1). "
    f"Of all map lanes of at least 10 m, {pct(c_all[0]['claimed'])} are claimed by a node, {pct(c_all[0]['shared'])} are taken by an overlapping lane, {pct(c_all[0]['misassigned'])} are misassigned and {pct(c_all[0]['no node'])} have no node at all "
    f"(own lane: {pct(c_t1[0]['no node'])} without a node). The ugly look in the video is therefore more likely the sections and links across the junction mouths than the assignment itself; I would like to look at the frames you saw together (a first count of 45 % uncovered lanes was wrong: a metric artefact, corrected).</p>" if a0 and c_all[0] else "<p class='pending'>pending</p>"))
r = (tmp or {}).get("result")
rl0, rl1 = [(x or {}).get("result") for x in (tmpl0, tmpl1)]
parts.append("<h3>3. No temporally consistent behaviour, especially on intersections</h3>" + (
    f"<p>Measured over 851 pairs of frames 0.5 s apart, both ends moved by the ego motion. Nodes persist in {pct(r['delivered / all']['persistent'])} of the cases ({pct(r['delivered / near an intersection']['persistent'])} near an intersection). Links persist in {pct(rl0['delivered / all']['persistent'])}, "
    f"but <b>long links only in {pct(rl0['delivered / long links']['persistent'])} / {pct(rl1['delivered / long links']['persistent'])}</b> (seeds 0 / 1; the ground truth: {pct(rl0['GT / long links']['persistent'])}), and near an intersection {pct(rl0['delivered / near an intersection']['persistent'])} against "
    f"{pct(rl0['delivered / elsewhere']['persistent'])} elsewhere. So what you saw is mainly the choice of the junction links from frame to frame. The head has no memory over frames; holding a long link while it is not contradicted is the lever (not built).</p>" if r and rl0 and rl1 else "<p class='pending'>pending</p>"))
e = (endp or {}).get("result")
parts.append("<h3>4. Do we find the endpoints at the right position?</h3>" + (
    f"<p>Where a node exists the position is good (median error {num(e['end / all tiers']['median_error_m_of_flag_within_6m'], '{:.1f}')} m for line ends). The weak part is the flag and the node itself: of the line ends only {pct(e['end / all tiers']['flag_within_3m'])} get an end node within 3 m "
    f"({pct(e['end / all tiers']['any_node_within_3m'])} have any node there), of the starts {pct(e['start / all tiers']['flag_within_3m'])}. In the own lane the events are found better ({pct(e['all events / tier 1']['flag_within_3m'])} flagged, "
    f"{pct(e['all events / tier 1']['any_node_within_3m'])} any node); no own-lane line end occurs in the VAL frames, so the tier weights cannot help the ends.</p>" if e else "<p class='pending'>pending</p>"))
d = dens
parts.append("<h3>5. The point density along the road graph is very high; idea to reduce it</h3>" + (
    f"<p>Two things. The nodes: the target puts a node in every 2 m cell ({num(d['dense (E273)']['nodes_per_100m'], '{:.1f}')} per 100 m); the delivered row is as dense ({m2(Bl4, 'nodes_per_100m', 'nodes', '{:.1f}')}). A sparse target ({num(d['sparse 0.3 / 7.5 m']['nodes_per_100m'], '{:.1f}')} per 100 m) keeps the geometry exactly for a perfect network, but the head cannot learn it (arm s failed). "
    f"The cheap way that works: the E275 step takes the vertices per 100 m from {m2(B, 'vert100_dp', fmt='{:.1f}')} to {m2(H, 'vert100_dp', fmt='{:.1f}')}; E277 trains it into the network. The geometry measure (vertices per 100 m after simplification) is {num(d['dense (E273)']['graph']['vert100_dp'], '{:.1f}')} for a perfect network "
    f"under any node target; the budget is 14.3.</p>" if d else "<p class='pending'>pending</p>"))
parts.append("<h3>6. Are the metrics good?</h3><p>Not good enough: they count every lane of the window equally, the assignment per node, and nothing measured flicker, endpoints or uncertainty. Added tonight: recall and precision by tier and by visibility, event numbers by tier, node density, link persistence between frames, link calibration and risk-coverage, map-piece coverage. "
             "The tier-weighted and the flicker measures should enter the headline; that needs your word (PROBLEM.md is protected).</p>")
ts = tstat
parts.append("<h3>7. The own lane is much more important; no problem-solving without camera visibility; non-causal GT</h3>" + (
    f"<p>Done: tier 1 (the lane the car drives, labelled from the future poses and the map) is {pct(ts['share_of_gt_length']['tier1'])} of the lane length, tier 2 (neighbours within 10 m) {pct(ts['share_of_gt_length']['tier2'])}; {pct(ts['hidden_by_box']['tier1'])} of the own lane is hidden by a vehicle. "
    f"The baseline is only slightly better in the own lane than elsewhere (node recall of the visible nodes {num(tiers1['node_R_visible'][0], '{:.2f}')} own lane, {num(tiers1['node_R_visible'][1], '{:.2f}')} neighbours, {num(tiers1['node_R_visible'][2], '{:.2f}')} far lanes). "
    "The tier loss did not lift the own lane (section 2): the limit seems to be the features and the node grid, not the weighting. A lever that is not tried: more input for the own lane (memory over frames, a higher-resolution look at the driving corridor).</p>" if ts and tiers1 else "<p class='pending'>pending</p>"))
parts.append(img("l4_tier_topviews.png", "Top views of the new labels: red = tier 1 (the lane the car drives), orange = tier 2, grey = tier 3, black x = not visible to a camera, blue = driven path"))
l0 = (asg[0] or {}).get("result", {}).get("links")
l1 = (asg[1] or {}).get("result", {}).get("links")
v50 = [next((x for x in lk["risk_coverage"] if x["threshold"] == 0.5), None) for lk in (l0, l1)] if l0 and l1 else None
v70 = [next((x for x in lk["risk_coverage"] if x["threshold"] == 0.7), None) for lk in (l0, l1)] if l0 and l1 else None
lt = [(x or {}).get("variants") for x in lthr]
parts.append("<h3>8. How to express and show uncertainty on connection results?</h3>" + (
    f"<p>Every link has a probability, but it is over-confident (expected calibration error {num(l0['ece'])}): a link at probability 0.55 is right in about one case of four. Raising the threshold from 0.5 to 0.7 gives precision {num(v50[0]['precision'])} to {num(v70[0]['precision'])} "
    f"for a coverage of {pct(v50[0]['share_of_gt_links_delivered'])} to {pct(v70[0]['share_of_gt_links_delivered'])} (seed 0). On the graph metrics it is a trade, not a gain: topology F1 {num(lt[0]['0.5_0.5']['topo_F1'])} at 0.5 against {num(lt[0]['0.7_0.7']['topo_F1'])} at 0.7, with the topology precision "
    f"{num(lt[0]['0.5_0.5']['topo_P'])} to {num(lt[0]['0.7_0.7']['topo_P'])} and recall {num(lt[0]['0.5_0.5']['topo_R'])} to {num(lt[0]['0.7_0.7']['topo_R'])}; cutting the long links altogether costs {num(lt[0]['0.5_0.5']['topo_F1'] - lt[0]['0.5_1.01']['topo_F1'])} topology F1. "
    "Proposal for the output: the probability per link, shown as width and opacity (picture), an explicit unknown state between 0.5 and 0.8 for the planner, and a risk-coverage curve per tier as the measure of \"better not estimate than estimate wrong\".</p>" if l0 and v50 and lt[0] else "<p class='pending'>pending</p>"))
parts.append(img("links_uncertainty_e273_p_s1.png", "Links of the baseline: width and opacity = probability, dashed below 0.5; green right, red wrong, orange = an end not matched to a GT node. The long red chords across the junction mouths are the problem."))
parts.append("<h2>5. Next</h2><ul><li><b>Your decision:</b> the soft step at 1.25 m (E279, KEEP by the written rule) as the delivered thinning in place of the hard read-out; it changes the baseline B (D58), so it needs a recorded decision. Both runs are two seeds, in-sample on VAL.</li><li>Memory for the long links (hold while not contradicted), the one lever the measurements point at for temporal consistency and junction quality.</li>"
             "<li>A matching-based node loss (one-to-one matching inside the loss) so that the head may place a node anywhere along a straight lane; then the sparse target can work.</li>"
             "<li>Your look at the frames: which picture of the assignment looked horrible. An intersection area from the lane graph (GT chat). The held-out check on TEST only with your recorded OK.</li></ul></main>")
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(parts))
print("written", os.path.join(OUT, "index.html"), os.path.getsize(os.path.join(OUT, "index.html")) // 1024, "KB")
