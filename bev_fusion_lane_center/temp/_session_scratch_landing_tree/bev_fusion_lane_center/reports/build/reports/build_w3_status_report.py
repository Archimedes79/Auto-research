"""Week-3 report on the lane-centre system (Spurmitte) on top of the frozen GaussianCaR features (27 Oct - 2 Nov 2026), HTML builder.
Run after fig_w3_status.py:  tools/env/_py14.sh reports/build/reports/build_w3_status_report.py
Numbers of the systems come from the metric cards runs/eval/<name>/<split>.json; the other numbers are quoted from the summaries named next to them
(reports/source/Condensed_Record_2026-11-01.md, reports/source/weekly/morning_2026-11-01_loop22.md, training/architecture.md).
Output: reports/results/week3/Week3_Status_Report_EN.html (PDF: reports/build/reports/print_weekly.ps1, then the html is deleted)."""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
OUTD = f"{R}/reports/results/week3"
FIG = f"{OUTD}/figures/"
OUT = f"{OUTD}/Week3_Status_Report_EN.html"
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))


def card(name, split="val"):
    return json.load(open(f"{R}/runs/eval/{name}/{split}.json"))["metrics"]


def m(name, key, split="val", d=3):
    return f"{card(name, split)[key]:.{d}f}"


def svg(d, path, caption):
    b = base64.b64encode(open(path, "rb").read()).decode()
    d.raw(f'<figure><img src="data:image/svg+xml;base64,{b}"><figcaption>{caption}</figcaption></figure>')


d = W.Doc("Spurmitte on frozen GaussianCaR: where we are", "Lane-centre graph from camera + radar, built on the cached, frozen GaussianCaR BEV features: results of the week, the prototype, the limits of the design",
          "Research report - week 3 (27 October - 2 November 2026)",
          "Mirko Meuter  |  prepared with Claude  |  numbers from stored result files, split and denominator named in every table. Continues the earlier "
          "<i>Week3_Loops18-20</i> (27 October) and the <i>Status after loop 21</i> (31 October).")

d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")

# ------------------------------------------------------------------ 1
d.h1("1  At a glance", new_page=False)
d.stats([
    (m("recall_sp0_dp", "lanes_found"), "lanes found, VAL 873", f"prototype recall_sp0_dp, Q {m('recall_sp0_dp', 'Q')}; control {m('control', 'lanes_found')}"),
    (m("recall_sp0_dp", "Q", "test"), "Q on TEST (once)", f"lanes found {m('recall_sp0_dp', 'lanes_found', 'test')}, R@1 far {m('recall_sp0_dp', 'R@1_far', 'test')}; control Q 0.712, lanes found 0.762"),
    (f"{m('recall_sp0_dp', 'vert100', d=1)}", "node ids per 100 m", f"half of the control's {m('control', 'vert100', d=1)} at a higher Q"),
    ("about 21 ms", "lane graph per frame", "estimate; the control needs 628 ms; extraction of the frozen features (about 0.4 s per frame) is outside this time"),
    ("12 / 24", "rule steps on the main path", "stages of the delivered chain without a gradient path"),
    ("+0.24 F1", "foundation effect (E27)", "map-pretrained GaussianCaR against detection-pretrained CRN, same head: the largest single effect of the project"),
])
d.h2("In five sentences")
d.ul([
    "Spurmitte reads the <b>frozen, cached GaussianCaR BEV features</b> (camera + radar, trained for map segmentation) and builds a sparse lane-centre graph with forks and merges, in travel direction, attached to an SD-map route.",
    "The <b>prototype of record</b>, <code>recall_sp0_dp</code>, has the best Q of all sparse systems (VAL 0.742) and the best lanes found among them (0.781) at about half the control's vertex count; on TEST it keeps its Q lead (+0.015 against the control, lower bound just 0.000) but still trails on lanes found and R@1 far.",
    "The week's experiments closed more questions than they opened: spurs kept and the seed ring each gave coverage back <b>without training</b>; recall weight 6, the Poisson set loss, range thresholds and the two-seed union did not pass their written rules.",
    "Coverage at range is mostly lost <b>in the chain, not in the network</b>; the strongest systems at range are now blocked by precision at range.",
    "Because the features are frozen and 12 of 24 stages are rule steps, the design has a ceiling that is set by the features and by hand-set assembly rules (section 6).",
])

# ------------------------------------------------------------------ 2
d.h1("2  How it works on the frozen GaussianCaR features", new_page=True)
d.p("Goal (PROBLEM.md): a planner-ready lane-centre graph without an HD map, lane centre lines with forks and merges, direction, fused over time, assigned to the SD-map route with confidence; very good <b>and</b> very fast; better short graphs than false ones. Data: nuScenes, 255 scenes, scene-disjoint TRAIN 8,414 / VAL 873 / TEST 833 frames.")
d.ul([
    "<b>Input (frozen, cached):</b> GaussianCaR camera + radar BEV features, decoder PCA 32 + 7 map logits = 39 channels on a 200 x 200 grid of 0.5 m cells (+-50 m). Extracting them costs about 0.4 s per frame and happens once; the timed chain starts from the cache.",
    "<b>Dense lane head (frozen, cached):</b> a U-Net with heat map and direction field, trained separately; training rows use out-of-fold maps. Its outputs join the features: 42 channels.",
    "<b>Builder D7 (learned, matching-free losses):</b> a stem pyramid initialised from the control run, a lattice of seed candidates every 5 m with learned validity and snap, rays read with a survival product, three children per tip, graph-biased attention over the tips, a junction head (split / merge heat), a link head between forward and backward ends.",
    "<b>Assembly (rule steps, no gradient):</b> alive filter, path union, clean-up of bubbles and spurs, junction anchoring, fork relocation, fork pruning, piece cut, optional Douglas-Peucker output stage.",
    "<b>After the graph:</b> a route head (SD-map attribute along the edges) and a planner consumer, trained on stored tokens.",
])
svg(d, f"{FIG}w3_arch_lane.svg", "The delivered chain, drawn from its code (<code>archdraw.py</code>): blue = learned network, grey = frozen / cached, orange = a choice inside the network, red = post-processing without a gradient path.")
d.callout("Why the frozen foundation is the centre of the design", "Swapping the foundation was the largest effect of the project: map-pretrained GaussianCaR against detection-pretrained CRN with the same 2.7 M U-Net head lifts F1@1 m from 0.624 to 0.867 on VAL (+0.24; TEST 0.598 to 0.835, E27). Later loops found the same from the other side: the shared stem carries the skill, and the fidelity curve is flat from 2 training units (E122): the features, not the builder, set the level.", "")

# ------------------------------------------------------------------ 3
d.h1("3  Where the systems stand", new_page=True)
rows = []
for nm, lab, ms in (("champion", "champion (D7 f1 + fork pruning 0.3)", "24"), ("recall", "recall (rc1 + pruning 0.4)", "20.8 (E253, without recursion)"),
                    ("fast_recall", "fast_recall (j1rc, 192 / 96)", "20 - 22"), ("recall_sp0_dp", "<b>recall_sp0_dp (prototype)</b>", "about 21 lane graph, about 27 full chain (estimates)"),
                    ("control", "control (E94 trace network)", "628")):
    rows.append([lab, m(nm, "Q"), m(nm, "topo_F1"), m(nm, "lanes_found"), m(nm, "R@1_far"), m(nm, "J"), m(nm, "vert100", d=1), ms])
d.table(rows, header=["system (registry name), VAL 873 frames", "Q", "topo F1", "lanes found", "R@1 far", "J", "node ids / 100 m", "ms per frame"], num=(1, 2, 3, 4, 5, 6), hl=(3,), idcol=0)
d.p("Source: metric cards <code>runs/eval/&lt;name&gt;/val.json</code> (cached outputs, <code>evaluate.py</code>). Q = S without the sparsity term; node ids per 100 m is the delivered graph's vertex count (D25: the basis of the 14.3 budget is still an open decision; <code>*_dp</code> systems are counted after Douglas-Peucker 0.2 m). ms: stored timing; the prototype's numbers are estimates (measured without the time recursion plus its cost).", "small")
d.fig(FIG + "w3_lane_systems.png", "VAL, 873 frames: the prototype (orange) has the highest Q and the best lanes found of the sparse systems; the 628 ms control still leads on lanes found and R@1 far.")
d.h2("TEST: the one check of the loop")
d.table([["recall_sp0_dp (prototype)", m("recall_sp0_dp", "Q", "test"), m("recall_sp0_dp", "lanes_found", "test"), m("recall_sp0_dp", "R@1_far", "test"), m("recall_sp0_dp", "topo_F1", "test"), m("recall_sp0_dp", "vert100", "test", d=1)],
         ["control E94 (same frames)", "0.712", "0.762", "0.776", "0.722", "22.0"]],
        header=["TEST, 833 frames", "Q", "lanes found", "R@1 far", "topo F1", "node ids / 100 m"], num=(1, 2, 3, 4, 5), hl=(0,))
d.p("Paired against the control: Q +0.015 [+0.000, +0.029] (the lower end only just excludes 0), lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012], 43 % fewer vertices (the prototype is counted after its output stage, the control without one). The written prediction (Q 0.735 - 0.74) hit. Control numbers from <i>morning_2026-11-01_loop22.md</i>.", "small")
d.h2("Known gaps, ranked (PROBLEM.md)")
d.table([["1", "Coverage at range: lanes found and R@1 far below the control; loss weights are spent", "this week: spurs kept, seed ring (+0.015 R@1 far); blocked by precision at range"],
         ["2", "Connectivity: topo F1 -0.022 against the control", "fork pruning gave +0.021 (loop 18); no new lever"],
         ["3", "No planner-level metric that responds (path error floor 0.27 m)", "open"],
         ["4", "A map with 1 - 3 m error: registration recovers about half the gap", "E238 replicated the gain on two new map draws (loop 19)"],
         ["5", "Recall variants with cameras off (Q -0.038 for j1rc)", "open"]], header=["#", "gap", "state"], idcol=0)

# ------------------------------------------------------------------ 4
d.h1("4  The week's experiments", new_page=True)
d.p("Every row is decided by the rule written before the run (<code>decide.py</code> or the card); VAL = 873 frames unless marked. Numbers from the experiment summaries and ledger rows; the condensed record of all key experiments is <i>reports/source/Condensed_Record_2026-11-01.md</i>.")
d.table([
    ["E236", "does the Q ranking survive a matched vertex budget?", TIE("TIE") + " recall vs champion", "at 14.3 vertices Q: champion 0.7306, recall 0.7331, fast 0.7206, fast_recall 0.7355; the recall-over-champion edge shrinks to +0.0025"],
    ["E240", "what is the measurement floor (four paired seeds)?", TIE("TIE"), "seed SD of Q 0.0030 / 0.0020; j1rc - j1eq: Q +0.0062 (inside the band), lanes found +0.0596 and R@1 far +0.0475 with the same sign in all four pairs: the recall weight is a coverage lever, not a Q lever"],
    ["E241", "can a run be killed early from its learning curve?", INFO("not adopted"), "the lanes-found gap has its final sign at 25 % of the fine-tune, Q does not separate"],
    ["E237", "does the damaged exact-distance training map cost anything?", TIE("TIE"), "the stored map breaks its 0.5 m bound in 51 % of the frames (0.7 % above 1 m); fine-tuning on the exact map changes Q by -0.0017 (two seeds); D14: new trainings use the exact map"],
    ["E243 / E253", "timing on a quiet machine", TIE("TIE"), "j1 18.5, f1 18.8, j1rc 21.1, rc1 20.8, f1 + recursion 24.9 ms; the recall weight costs +2.65 ms, the recursion +6.1 ms; the loaded-machine run was invalid"],
    ["E244 / E255", "set likelihood instead of the hinge; recall clamp 8 m", NEG("CUT"), "Poisson loss at matched vertices: Q 0.6520 against 0.7241 / 0.7252, lanes found 0.551 against 0.675 / 0.687"],
    ["E246", "range-specific admission thresholds", NEG("KILL"), "in-sample ceiling R@1 far +0.0059 against a gate of +0.015"],
    ["E257", "output stage Douglas-Peucker 0.2 m (D20)", OK("KEEP"), "fast_recall 14.98 to 9.86, champion 14.83 to 9.52 node ids per 100 m; Q -0.0003 / -0.0005"],
    ["E258 - E260", "assembly profile and clean-up ablations", INFO("measured"), "relocation off: Q -0.011, J -0.05; anchoring off: Q within +-0.001 on fast_recall; dead-code removal leaves the graph hashes identical on 873 frames"],
    ["E265", "does the clean-up's spur rule delete true far lanes?", OK("KEEP"), "lanes found +0.0115 (4 checkpoints, every interval above 0), R@1 far +0.0065, Q +0.0039"],
    ["E266", "seed lattice to the box edge (ring at +-47.5 m)", OK("KEEP") + " with two rule deviations", "R@1 far +0.0152 mean over 4 checkpoints, Q -0.0016; gate G0 missed by a hair and overridden by the lead, both stated in the record"],
    ["E261 / E262", "recall weight 6", NEG("CUT"), "held out: no admissible operating point in 7 of 20 halves, Q -0.0075"],
    ["E264", "union of two seeds' hypothesis trees", INFO("not adopted"), "lanes found +0.047, R@1 far +0.040 on both pairs, but false length 0.0635 above the 0.062 guard"],
    ["E267", "ring + spurs stacked", NEG("CUT"), "lanes found +0.022, R@1 far +0.027, but false length +0.006 and P@1 far -0.015 miss their guards"],
    ["E268", "the one TEST check of the prototype", OK("confirmed"), "see section 3; the prediction hit"],
], header=["id", "question", "verdict", "key numbers"], widths=["9%", "24%", "13%", "54%"], idcol=0)
d.callout("What the week says about gap 1", "Coverage at range is mostly lost <b>in the chain, not in the network</b>: the spur rule and the seed lattice at the box edge each gave coverage back without any training. The strongest systems at range (the stack: R@1 far 0.798; the union: lanes found 0.786) are blocked by <b>precision at range</b>, not by coverage. Prediction record of the loop: 4 of 6 hit; misses E261 (+0.022 vs +0.013) and E267.", "ok")

# ------------------------------------------------------------------ 5
d.h1("5  Example: the prototype on real frames", new_page=True)
d.p("One command, CPU, cached network outputs: <code>lane_prototype.py recall_sp0_dp --scenes 23 7 --video</code>. It writes, per frame, the lane centre lines in the ego frame (x forward, y left, in travel direction, with vertex ids, confidence and successor edges; planner format in <code>tools/prototype/README.md</code>) and a results video: <code>video/Week3_lane_prototype_recall_sp0_dp.mp4</code> (80 VAL frames of two scenes; not in git).")
d.fig(FIG + "w3_lane_frame.png", "A frame of the video (VAL scene 23, Boston seaport): red = prediction, green = ground truth; the camera image is shown for orientation only; the panel states the numbers of the shown 80 frames (Q 0.737, lanes found 0.76, 11.5 vertices per 100 m).")
d.fig(FIG + "w3_lane_poster.png", "The poster frame of the same video.", width=70)

# ------------------------------------------------------------------ 6
d.h1("6  What the frozen foundation and the rule steps limit", new_page=True)
d.p("The design has two structural limits, both visible in the architecture picture of section 2.")
d.h2("Frozen, cached features")
d.ul(["No gradient from the lane losses ever reaches GaussianCaR or the dense lane head; they are stored arrays. The builder was tuned on out-of-fold dense maps, so folding the dense head into the builder needs care with train rows that would see their own labels.",
      "The cache is what makes the loop cheap (hours of GPU saved per experiment) and the chain fast; the delivered system from raw sensors is not 21 ms, because extraction takes about 0.4 s per frame.",
      "Foundation effect and ceiling: +0.24 F1 for the map-pretrained foundation (E27), flat fidelity curve of the builder from 2 units (E122): further gains from the builder side are small against what the features allow."])
d.h2("Rule steps in the delivered graph (12 of 24 stages)")
d.table([["fork pruning", "keeps one branch at forks without junction-head support", "largest measured gain of any rule step: J 0.265 to 0.409, Q 0.701 to 0.735 (E215; vertices 13.5 to 14.8, so part of the Q gain is vertex budget)"],
         ["fork relocation", "moves a fork vertex to the nearest junction-heat peak", "off: Q -0.011, J -0.054 on fast_recall (E258)"],
         ["junction anchoring", "splits the nearest edge at junction-heat peaks", "off: Q -0.0004, J -0.0042 on fast_recall; the first rule to drop"],
         ["clean-up of bubbles and spurs", "removes duplicates under 2.5 m and spurs under 5 m", "spur half switched off (E265): lanes found +0.0115; the bubble half stays"],
         ["Douglas-Peucker output stage", "simplifies polylines to 0.2 m", "champion 14.8 to 9.5 node ids per 100 m at Q -0.0005 (E257)"],
         ["path union, alive filter, seed selection, child choice, beam top-M, time-recursion match, piece cut", "greedy and hard selections inside and after the network", "path union is 16 % of the assembly time; the time recursion costs +6.1 ms and its value is stability, not the single-frame score"]],
        header=["rule step", "what it does", "measured value"], widths=["24%", "26%", "50%"], idcol=0)
d.p("Each of these has a differentiable stand-in listed in the project's gap table (soft top-k, a learned continuation head, an edge-keep head, a recurrent token state); each has to reproduce the reference before it counts, and fork pruning is the one to beat first.", "small")

# ------------------------------------------------------------------ 7
d.h1("7  Open and next", new_page=False)
d.h2("For Mirko")
d.ul(["<b>D4:</b> promote <code>recall_sp0_dp</code> as the lane prototype of record? (after route success on spur-kept graphs and idle timing in the next lane loop)",
      "<b>D25:</b> the vertex-count basis of the 14.3 budget (node ids against per-piece Douglas-Peucker); <b>D18:</b> per-metric noise bands",
      "<b>Is Douglas-Peucker (D20) the accepted sparsification</b> of the delivered graph, or must the vertex budget come out of the network?"])
d.p("<b>Next:</b> lane loop 23, a precision step at range (range-dependent threshold for edge seeds, or the union with a wider vertex merge), idle timing of ring / union / prototype, route success on spur-kept graphs before any champion change.")

d.h1("Appendix  Sources", new_page=False)
d.ul(["metric cards <code>runs/eval/&lt;name&gt;/{val,test}.json</code>, <code>models/registry.json</code>, <code>evaluation/golden.json</code>, <code>training/architecture.md</code>, <code>PROBLEM.md</code>",
      "<code>reports/source/Condensed_Record_2026-11-01.md</code> (key experiments K1 - K26, reviewed against primary sources), <code>reports/source/weekly/morning_2026-11-01_loop22.md</code>"])
d.save(OUT)
