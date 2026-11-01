"""Week-3 status report (27 Oct - 2 Nov 2026), HTML builder. Run after fig_w3_status.py:  tools/env/_py14.sh reports/build/reports/build_w3_status_report.py
Numbers of the lane systems come from the metric cards runs/eval/<name>/<split>.json, the BEV board from bev_fusion/records/BOARD.md; the other numbers are quoted from the
summaries named next to them (reports/source/Condensed_Record_2026-11-01.md, reports/source/weekly/morning_2026-11-01_loop22.md, bev_fusion/records/morning_2026-11-01.md).
Output: reports/results/week3/Week3_Status_Report_EN.html (PDF: reports/build/reports/print_weekly.ps1, then the html is deleted)."""
import base64
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
REPO = os.path.dirname(R)
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


gr = json.load(open(f"{REPO}/temp/g1_route_j1.json"))
f1 = json.load(open(f"{REPO}/temp/gradient_report_f1.json"))

d = W.Doc("Week 3: where we are", "Lane-centre graph (Spurmitte) and the camera + radar BEV system: results, the road to end-to-end, the state of the work",
          "Research report - week 3 (27 October - 2 November 2026)",
          "Mirko Meuter  |  prepared with Claude  |  numbers from stored result files, split and denominator named in every table. This report continues the earlier "
          "<i>Week3_Loops18-20</i> (written 27 October) and the <i>Status after loop 21</i> (31 October).")

d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")

# ------------------------------------------------------------------ 1
d.h1("1  At a glance", new_page=False)
d.stats([
    (m("recall_sp0_dp", "lanes_found"), "lanes found, VAL 873", f"prototype recall_sp0_dp; Q {m('recall_sp0_dp', 'Q')} at {m('recall_sp0_dp', 'vert100', d=1)} node ids per 100 m; control {m('control', 'lanes_found')}"),
    (m("recall_sp0_dp", "Q", "test"), "Q on TEST (once)", f"lanes found {m('recall_sp0_dp', 'lanes_found', 'test')}, R@1 far {m('recall_sp0_dp', 'R@1_far', 'test')}; control TEST Q 0.712 / lanes found 0.762"),
    ("0.566", "velocity skill, moving cars (BEV)", "0.000 before, vehicle AVE 2.385 to 1.189 m/s; cost: front mAP2 0.361 to 0.347"),
    ("12 / 0", "rule steps on the main path", "lane chain 12 of 24 stages, BEV system 0 of 14: the end-to-end gap is the lane chain's"),
    ("19.4 GB", "moved off a full C:", "C: free 4.1 GB at the start; nothing deleted for good"),
    ("101 / 101", "tensors with gradient", "all builder tensors learn; the route loss reaches the builder but orthogonally"),
])
d.h2("In five sentences")
d.ul([
    "The lane system has a <b>prototype of record</b>, <code>recall_sp0_dp</code>: it finds more lanes than any earlier sparse system at the best Q, in about 21 ms for the lane graph (a 628 ms control reaches the same Q); on TEST it keeps its Q lead but still trails the control on lanes found at range.",
    "The week's experiments closed more questions than they opened: spurs kept and the seed ring each gave coverage back without training; recall weight 6, the Poisson set loss, range thresholds and the two-seed union did not pass their written rules.",
    "The BEV system (camera + radar) got a layered, fully differentiable design (0 rule steps) and a working velocity head after 28 of 30 earlier runs had a dead one; its prototype runs as one command.",
    "For the end-to-end lane chain a set-up was prepared and measured (contract, gradient report, experiment card E269 after three critic passes); no model was trained for it.",
    "The disk was full, the research tools were simplified and gated more narrowly, and the repository was pushed on 26 October; decisions wait for Mirko (section 9).",
])

# ------------------------------------------------------------------ 2
d.h1("2  The lane-centre system: where it stands", new_page=True)
d.p("Goal (PROBLEM.md): a planner-ready lane-centre graph with forks and merges, direction and confidence, attached to an SD-map route, from camera + radar, fully differentiable, very good <b>and</b> very fast. Data: nuScenes, 255 scenes, scene-disjoint TRAIN 8,414 / VAL 873 / TEST 833 frames; TEST once per finished design.")
rows = []
for nm, lab, ms in (("champion", "champion (D7 f1 + fork pruning 0.3)", "24"), ("recall", "recall (rc1 + pruning 0.4)", "20.8 (E253, without recursion)"), ("fast_recall", "fast_recall (j1rc, 192 / 96)", "20 - 22"),
                    ("recall_sp0_dp", "<b>recall_sp0_dp (prototype)</b>", "about 21 lane graph, about 27 full chain (estimate)"), ("control", "control (E94 trace network)", "628")):
    rows.append([lab, m(nm, "Q"), m(nm, "topo_F1"), m(nm, "lanes_found"), m(nm, "R@1_far"), m(nm, "J"), m(nm, "vert100", d=1), ms])
d.table(rows, header=["system (registry name), VAL 873 frames", "Q", "topo F1", "lanes found", "R@1 far", "J", "node ids / 100 m", "ms per frame"], num=(1, 2, 3, 4, 5, 6), hl=(3,), idcol=0)
d.p("Source: metric cards <code>runs/eval/&lt;name&gt;/val.json</code> (cached outputs, <code>evaluate.py</code>). Q = S without the sparsity term; node ids per 100 m is the delivered graph's vertex count (D25: the budget basis of 14.3 is still an open decision; <code>*_dp</code> systems are counted after Douglas-Peucker 0.2 m). ms: stored timing, the prototype's numbers are estimates (measured without the time recursion plus its cost).", "small")
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

# ------------------------------------------------------------------ 3
d.h1("3  The week's lane experiments", new_page=True)
d.p("Every row is decided by the rule written before the run (<code>decide.py</code> or the card); VAL = 873 frames unless marked. Numbers from the experiment summaries and ledger rows; the full condensed record of all key experiments is <i>reports/source/Condensed_Record_2026-11-01.md</i>.")
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
    ["E268", "the one TEST check of the prototype", OK("confirmed"), "see section 2; the prediction hit"],
], header=["id", "question", "verdict", "key numbers"], widths=["9%", "24%", "13%", "54%"], idcol=0)
d.callout("What the week says about gap 1", "Coverage at range is mostly lost <b>in the chain, not in the network</b>: the spur rule and the seed lattice at the box edge each gave coverage back without any training. The strongest systems at range (the stack: R@1 far 0.798; the union: lanes found 0.786) are blocked by <b>precision at range</b>, not by coverage. Prediction record of the loop: 4 of 6 hit; misses E261 (+0.022 vs +0.013) and E267.", "ok")

# ------------------------------------------------------------------ 4
d.h1("4  Example: the prototype on real frames", new_page=True)
d.p("One command, CPU, cached network outputs: <code>lane_prototype.py recall_sp0_dp --scenes 23 7 --video</code>. It writes, per frame, the lane centre lines in the ego frame (x forward, y left, in travel direction, with vertex ids, confidence and successor edges; the format is the planner format of <code>tools/prototype/README.md</code>) and a results video. The example video of this week is in <code>video/Week3_lane_prototype_recall_sp0_dp.mp4</code> (80 VAL frames of two scenes; not in git).")
d.fig(FIG + "w3_lane_frame.png", "A frame of the video (VAL scene 23, Boston seaport): red = prediction, green = ground truth; the camera image is shown for orientation only; the panel states the numbers of the shown 80 frames (Q 0.737, lanes found 0.76, 11.5 vertices per 100 m).")
d.fig(FIG + "w3_lane_poster.png", "The poster frame of the same video.", width=70)

# ------------------------------------------------------------------ 5
d.h1("5  The BEV system (camera + radar)", new_page=True)
d.p("The second problem of the repository: a camera + radar BEV perception system on nuScenes with semantic maps, objects, velocity and a time mover (outputs at any dt). Since D23 its trainings run through the same launcher, ledger and code tags as the lane problem. This week it was rebuilt into five layers (precompute, input interface, network slots, output interface, post-processing).")
svg(d, f"{FIG}w3_arch_bev.svg", "The BEV system in five layers, drawn from its code (<code>archdraw.py</code>); 14 stages on the main path: 8 learned, 2 fixed-differentiable, 4 frozen, 0 rule steps.")
d.h2("The board: arms against the base")
txt = open(f"{REPO}/bev_fusion/records/BOARD.md", encoding="utf-8").read().splitlines()
rowsb = []
for l in txt:
    if re.match(r"\| (t8_ctl|t8_bevfeat|t10_dual) ", l):
        c = [x.strip() for x in l.strip("|").split("|")]
        rowsb.append([c[0]] + c[2:-1])
head = next(l for l in txt if l.startswith("| arm "))
cols = [x.strip().replace("maps: ", "").replace("objects: ", "").replace("motion: ", "").replace("time: ", "") for x in head.strip("|").split("|")][2:-1]
d.table(rowsb, header=["arm"] + cols, num=tuple(range(1, len(cols) + 1)), widths=None, hl=(0,))
d.p("Source: <code>bev_fusion/records/BOARD.md</code>, made from the stored <code>metrics.json</code> files (cold VAL, 1000 frames of 82 scenes, GT signature 0.5_4_15efbde2d1). Base row = value, arm rows = change; ^ better and v worse than the band in every seed. Band per column: max(0.02, 2 x seed spread).", "small")
d.fig(FIG + "w3_bev_board.png", "Change against the base t8_ctl: the BEVFusion image features (t8_bevfeat) lift every map layer by 0.06 - 0.22 but halve the object detection; using both feature sets (t10_dual) keeps the map gains and loses almost nothing on objects (mAP -0.016, inside the band).")
d.h2("The velocity head is alive (t11)")
d.p("In the velocity session's table 28 of 30 BEV runs with an effective batch of 3 had a dead velocity output (skill 0 on moving cars). The fix: a moving / static logit next to the velocity regression, the regression only on moving centres, a decode gate; 1000 steps from <code>t10_dual_s0</code>. Source: <i>bev_fusion/records/morning_2026-11-01.md</i>.")
d.table([["vehicle velocity skill on moving cars", "0.000", "0.000", "0.470", "<b>0.566</b>"], ["vehicle AVE (m/s)", "2.385", "2.376", "1.445", "<b>1.189</b>"],
         ["vehicle error on static cars (m/s)", "0.029", "0.029", "0.336", "0.284"], ["drivable IoU / walkway IoU / lane divider IoU@0.5", "0.756 / 0.340 / 0.335", "0.757 / 0.354 / 0.350", "0.759 / 0.345 / 0.343", "0.757 / 0.345 / 0.343"],
         ["front mAP2 (0 - 50 m)", "0.361", "0.364", "0.346", "0.347 (-0.014 [-0.020, -0.006])"], ["ms per frame (batch 1, cached chain)", "89.8 / 92.1", "", "", "90.7 / 89.8"]],
        header=["cold VAL, 1000 frames", "t10_dual_s0", "t10_dual_s1", "B' (batch 3)", "B2 (accumulation 2)"], num=(1, 2, 3, 4), hl=())
d.p("Verdict by the card's rule: V1 (learns) for B2. One seed only; the seed twin is the next run. Cost: front mAP2 -0.014 (vehicle AP@0.5 0.254 to 0.207) and a static-car error of 0.28 m/s. <code>t10_dual</code> held on its seed twin (maps within 0.02, front mAP2 0.361 / 0.364). The B2 prototype: <code>python evaluation/bev_prototype.py runs/t11_vel_hurdle_acc2 --scenes scene-0003 5 --video</code>.", "small")
d.fig(FIG + "w3_bev_frame.png", "A frame of the BEV dashboard video (<code>video/Week3_bev_dashboard_scene-0003.mp4</code>, scene-0003, B2).")
d.callout("Honest limits", "The +0.5 s forecast of the prototype uses the true ego motion from the ground truth; boxes are decoded from score 0.05; pedestrian velocity is still near zero (skill 0.014); the velocity fix was trained for 1000 steps from one checkpoint. Failures and the SD map of the lane problem are synthetic; the BEV numbers are front camera + all radars only.", "warn")

# ------------------------------------------------------------------ 6
d.h1("6  Toward an end-to-end lane chain", new_page=True)
d.p("Mirko's target: the lane chain must become network-only. The BEV system's design points the way: a dense, differentiable output interface with hard selection only for the metric. The two systems side by side:")
d.table([["main path", "24 stages: 8 learned, 2 fixed-differentiable, 2 frozen, <b>12 rule steps</b>", "14 stages: 8 learned, 2 fixed-differentiable, 4 frozen, <b>0 rule steps</b>"],
         ["rule steps", "seed selection, child choice, beam top-M, time-recursion match, alive filter, path union, clean-up, anchoring, relocation, fork pruning, piece cut, Douglas-Peucker", "only outside the main path: line-cell threshold, box selection (metric side)"],
         ["training", "builder, dense head, route head and consumer trained in separate pieces; the one joint attempt (route loss through the builder) was negative three times", "one trainer for the whole chain, Kendall task balancing, sequence training with a carried memory"],
         ["front end", "frozen GaussianCaR features and a frozen dense head, both cached", "frozen DualBEV image heads cached; trainable image features decided (D26)"]],
        header=["", "lane chain (registry systems)", "BEV system"], idcol=0)
svg(d, f"{FIG}w3_arch_lane.svg", "The lane chain, drawn from its code: orange = a choice inside the network, red = post-processing without a gradient path.")
d.h2("Measured this week: who gets gradient from the loss")
d.p(f"Gradient report of the champion builder (<code>e2e_gradient_report.py</code>, 2 VAL frames, CPU float32, the trainer's loss; indicative): <b>no tensor is silent</b> ({sum(x['tensors'] for x in f1['params'])} of {sum(x['tensors'] for x in f1['params'])} parameter tensors get gradient). The end-to-end gap is therefore before and after the network, not inside it: the route head trains on stored token files, the assembly runs after the network, the GaussianCaR features and the dense head are frozen. On the 42 input channels the frozen GaussianCaR channels get a mean gradient of about 9e-6 per channel against 3e-4 for the dense-head heat: the signal a trainable front end would receive is small but nonzero.")
d.fig(FIG + "w3_gradient_route.png", "G1(1) of experiment E269: the route loss alone through the old joint path (builder e123_D7_r3_j1, 2 TRAIN frames, CPU). The route loss reaches the builder strongly through the node tokens, almost at right angles to the builder's own objective; it gives exactly zero to node positions, validity, link logits, junction head and link head.")
d.h2("Experiment card E269 and what three critic passes did to it")
d.p("The first idea (soft graph tokens: a route head on the beam nodes) went through three fresh critic passes, each returning REVISE. They found that the head interface described did not exist (no validity input, no link bias), that the route-success bootstrap the rule needs does not exist, that the first cost estimate was half the real one, and that a cheaper test of the same mechanism exists. The card is now the <b>straight-through proxy</b>: the old head and evaluator unchanged, only a gradient path from the route loss to the node positions; arms a (position path), c (E232's detached control), d (the route gradient on the tokens scaled to 0.1, added after the measurement above), two seeds each, about 8 slot-hours. The full soft-node design is a follow-up (E270) written only if the proxy shows an effect. Nothing is scheduled; the next step is code (the straight-through term, the route-success bootstrap).")
d.callout("Why this matters", "The old premise &quot;the route loss cannot reach the builder&quot; was wrong; the measured picture is a <b>loud, near-orthogonal gradient</b> on the shared features (up to 6 times the builder's own on the child tokens, 20 times on the node tokens), which is a second candidate explanation of the three negative joint fine-tunes. Arm d tests exactly that.", "warn")
d.h2("What was taken over from the BEV system")
d.table([["dense differentiable output interface, selection only for the metric", "contract of <code>Builder.forward</code> (<code>p16_interface.py</code>, self-test; not yet called by the live chain)"],
         ["slot tables and a gradient test", "<code>e2e_gradient_report.py</code> (self-test); slot tables come with the first rule-step replacement"],
         ["one coverage rule for loss, evaluation, execution (D36 / D37)", "later; the BEV chat owns the rule"],
         ["trainable front end, modality dropout (D26)", "step 2: the builder on the BEV trunk features, frozen first; risk: GaussianCaR features carry the lane skill (+0.24 F1, E27), BEV maps are weaker"],
         ["world-aligned temporal memory, sequence training", "step 3: replace the hard-match time recursion"],
         ["loss machinery: task balancing, gradient-conflict cosines", "the cosines are in the report; balancing later"]], header=["BEV concept", "use in the lane chain"], idcol=0)

# ------------------------------------------------------------------ 7
d.h1("7  How the work is organised", new_page=True)
d.p("Several Claude chats work on one repository; the research system <code>autoresearch/</code> gives them a launcher with a code tag per run, a ledger, written decision rules, evaluation and training suites, and a gate on the truth files. This week's changes:")
d.table([["D16", "run state is derived from run records, not synced between three stores; the tools shrank from 1623 to 1426 lines; a fresh critic found 8 defects in the first version, 7 fixed with tests", OK("deployed")],
         ["D21", "the gate hook protects only the truth files and itself (problem, TEST log, golden numbers, champion entries, <code>project.json</code>, hook script and settings); other tool changes need a passing self-test and a critic pass", OK("deployed")],
         ["D22 - D24", "tool fixes with self-tests; BEV system under the launcher (shared GPU slots); duplicates removed", OK("done")],
         ["D26", "dense head first for lanes and centre lines; image features become trainable", INFO("decided")],
         ["D28 / D29 / D30", "overnight prototype round; one GT version with tags; one image region for all pipelines", INFO("decided")],
         ["D32", "restructuring target: four layers, two loops, four roles; first steps merged and smoke-tested", INFO("in progress")],
         ["D36 / D37 / D38", "coverage rule (camera loss only where the camera sees; lanes are camera-only), fusion results scored on the fusion area only, front-only rig and footprint share", INFO("decided; BEV chat implements")],
         ["git rule", "scratch is never committed by hand; promote a script at its second use; push reviewed units only; best three networks kept per problem (<code>keep3.py</code>)", OK("in force")]],
        header=["decision", "what changed", "state"], idcol=0)
d.h2("Fresh eyes found real things")
d.ul(["The numbers review of the condensed record (about 420 numbers against primary sources) found a wrong &quot;tie at matched vertices&quot; sentence, a missing vertex-count caveat on the fork-pruning gain (Q 0.701 to 0.735 came with 13.5 to 14.8 vertices) and unlabelled denominators (route success is counted over 809 TEST / 823 VAL frames, not 833 / 873); all corrected.",
      "Three critic passes on E269 (section 6) turned a &quot;re-wiring&quot; into either a small proxy or a 4 - 5 day build, before any GPU time was spent.",
      "The tools review passes found run.py exit-code, hold-marker and provenance gaps; fixes are committed, a few remain on the review chat's list."])

# ------------------------------------------------------------------ 8
d.h1("8  Operations: disk and repository", new_page=False)
d.table([["C: free space", "4.1 GB at the start of the clean-up, 36.4 GB after the move below, 55.5 GB at the last check (other chats cleaned too)"],
         ["lane run folders", "19.4 GB (615 files of 92 run folders) moved to <code>D:\\Auto-research\\lane_archive_2026-11-01</code>, verified by size, nothing deleted; 7.85 GB stay on C: (registered systems, the D7 lineage, control, live inputs). Record: <code>experiment_execution/history/lane_archive_2026-11-01/</code>"],
         ["golden check", "all six cached entries reproduced after the move (champion, champion_noprune, control, fast, fast_recall, recall; tolerance 0.0005); the one uncached entry needs the GPU"],
         ["WSL image", "<code>ext4.vhdx</code> 318 GB on C: for 279 GB used inside; compaction alone returns about 40 GB; one script prepared for Mirko (<code>purge_and_compact_2026-11-01.ps1</code>): delete the archive (19.4 GB), the D5 caches (32 GB) and the second-wave list (13.25 GB), then compact; needs an administrator shell and a quiet WSL"],
         ["repository", "main (183 commits) and the 25 <code>lcr/*</code> run tags were pushed on 26 October; later commits are local; the push rule of 1 November asks for a reviewed state first"]],
        header=["", "state"], idcol=0)

# ------------------------------------------------------------------ 9
d.h1("9  Decisions open and next steps", new_page=False)
d.h2("For Mirko")
d.ul(["<b>D4:</b> promote <code>recall_sp0_dp</code> as the lane prototype of record? (after route success on spur-kept graphs and idle timing in the next lane loop)",
      "<b>D25:</b> the vertex-count basis of the 14.3 budget (node ids against per-piece Douglas-Peucker); <b>D18 / D19:</b> per-metric noise bands, the machine lock between chats",
      "<b>Purge and compaction:</b> run the prepared script when no chat trains; decide whether the failure-series caches <code>pert</code> and <code>p17</code> (74.6 GB) go if the robustness line stops",
      "<b>E269:</b> the card is committed (arms a, c, d); a fourth look is cheap but not required; building the straight-through term needs a critic code pass because <code>p18_joint.py</code> is shared",
      "<b>Is Douglas-Peucker (D20) the accepted layer-5 sparsification</b>, or must the vertex budget come out of the network?"])
d.h2("Next, in order")
d.ul(["lane loop 23: a precision step at range (range-dependent threshold for edge seeds or the union with a wider vertex merge), idle timing of ring / union / prototype, route success on spur-kept graphs",
      "E269 build: straight-through term, route-success bootstrap, arm flag; then G1(2 - 4) and four-to-six trainings on the shared GPU",
      "BEV system: seed twin of B2, the long run with hurdle and accumulation 2, then the D36 coverage rule in loss and evaluation",
      "tools: reproducibility block, hold marker for stopped runs, hygiene list"])

d.h1("Appendix  Sources", new_page=False)
d.ul(["metric cards <code>runs/eval/&lt;name&gt;/{val,test}.json</code>; <code>models/registry.json</code>; <code>evaluation/golden.json</code>",
      "<code>reports/source/Condensed_Record_2026-11-01.md</code> (key experiments K1 - K26, reviewed against primary sources); <code>reports/source/weekly/morning_2026-11-01_loop22.md</code>",
      "<code>bev_fusion/records/{BOARD.md, morning_2026-11-01.md}</code>; <code>bev_fusion/training/architecture.md</code>; <code>training/architecture.md</code>",
      "<code>experiments/E269_soft-graph-tokens-route-loss/experiment.md</code>; <code>experiment_execution/plans/spurmitte_e2e_setup_2026-11-01.md</code>; gradient reports <code>temp/g1_route_j1.json</code>, <code>temp/gradient_report_f1.json</code>",
      "<code>experiment_execution/{DECISIONS.md, HANDOVER.md}</code>; <code>tools/maintenance/PURGE_2026-10-31.md</code>"])
d.save(OUT)
