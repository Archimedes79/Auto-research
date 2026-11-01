"""Week-3 framework report (27 Oct - 2 Nov 2026): the BEV system, the road to an end-to-end lane chain, the tools and the disk work - everything that is NOT the Spurmitte results
(those are in build_w3_status_report.py). HTML builder; the figures were made by the first combined version of fig_w3_status.py (git commit c0734ec) and lie in reports/results/week3_framework/.
Run:  tools/env/_py14.sh reports/build/reports/build_w3_framework_report.py, then print_weekly.ps1.  No experiment is run: every number is read from a stored file or quoted from the summary named next to it
(bev_fusion/records/BOARD.md, bev_fusion/records/morning_2026-11-01.md, temp/g1_route_j1.json, temp/gradient_report_f1.json, the decision records)."""
import base64
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
REPO = os.path.dirname(R)
OUTD = f"{R}/reports/results/week3_framework"
FIG = f"{OUTD}/figures/"
LANE_SVG = f"{R}/reports/results/week3/figures/w3_arch_lane.svg"
OUT = f"{OUTD}/Week3_Framework_Report_EN.html"
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))


def svg(d, path, caption):
    b = base64.b64encode(open(path, "rb").read()).decode()
    d.raw(f'<figure><img src="data:image/svg+xml;base64,{b}"><figcaption>{caption}</figcaption></figure>')


f1 = json.load(open(f"{REPO}/temp/gradient_report_f1.json"))
n_tensors = sum(x["tensors"] for x in f1["params"])

d = W.Doc("Week 3: the framework behind Spurmitte", "The camera + radar BEV system, the road to an end-to-end lane chain, the research tools and the disk work",
          "Research report - week 3, part 2 (27 October - 2 November 2026)",
          "Mirko Meuter  |  prepared with Claude  |  numbers from stored result files, split and denominator named. The Spurmitte results of the week are in the separate "
          "<i>Week3_Status_Report_EN</i>; nothing here was run for this report.")

d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")

d.h1("1  At a glance", new_page=False)
d.stats([
    ("0.566", "velocity skill, moving cars (BEV)", "0.000 before; vehicle AVE 2.385 to 1.189 m/s; cost: front mAP2 0.361 to 0.347"),
    ("12 / 0", "rule steps on the main path", "lane chain 12 of 24 stages, BEV system 0 of 14"),
    (f"{n_tensors} / {n_tensors}", "builder tensors with gradient", "the route loss reaches the builder strongly but almost orthogonally"),
    ("3 x REVISE", "critic passes on experiment card E269", "the first design needed new head code; a cheaper proxy exists"),
    ("19.4 GB", "moved off a full C:", "nothing deleted for good; C: free 4.1 GB at the start"),
    ("8 defects", "found by a fresh critic in the simplified tools", "7 fixed with tests (D16)"),
])
d.h2("In five sentences")
d.ul([
    "The <b>BEV system</b> (camera + radar, nuScenes) was rebuilt into five layers with 0 rule steps on its main path, and its velocity head, dead in 28 of 30 earlier runs, learns after a small fix.",
    "For an <b>end-to-end lane chain</b> a set-up was prepared and measured: a contract for the builder, a gradient report, and an experiment card (E269) that went through three critic passes; no model was trained for it.",
    "The measurement overturned an assumption: the route loss <b>does</b> reach the builder (through the node tokens), strongly and almost at right angles to the builder's own objective; node positions, validity and links get exactly zero.",
    "The <b>research tools</b> were simplified (run state derived from run records), the gate hook narrowed to the truth files, and a layered restructuring started.",
    "The disk was full: 19.4 GB of old lane run files were archived to D:, a purge and compaction script waits for Mirko, and main was pushed on 26 October.",
])

d.h1("2  The BEV system (camera + radar)", new_page=True)
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
d.table(rowsb, header=["arm"] + cols, num=tuple(range(1, len(cols) + 1)), hl=(0,))
d.p("Source: <code>bev_fusion/records/BOARD.md</code>, made from the stored <code>metrics.json</code> files (cold VAL, 1000 frames of 82 scenes, GT signature 0.5_4_15efbde2d1). Base row = value, arm rows = change; ^ better and v worse than the band in every seed. Band per column: max(0.02, 2 x seed spread).", "small")
d.fig(FIG + "w3_bev_board.png", "Change against the base t8_ctl: the BEVFusion image features (t8_bevfeat) lift every map layer by 0.06 - 0.22 but roughly halve object detection; using both feature sets (t10_dual) keeps the map gains and loses almost nothing on objects (mAP -0.016, inside the band).")
d.h2("The velocity head is alive (t11)")
d.p("In the velocity session's table 28 of 30 BEV runs with an effective batch of 3 had a dead velocity output (skill 0 on moving cars). The fix: a moving / static logit next to the velocity regression, the regression only on moving centres, a decode gate; 1000 steps from <code>t10_dual_s0</code>. Source: <i>bev_fusion/records/morning_2026-11-01.md</i>.")
d.table([["vehicle velocity skill on moving cars", "0.000", "0.000", "0.470", "<b>0.566</b>"], ["vehicle AVE (m/s)", "2.385", "2.376", "1.445", "<b>1.189</b>"],
         ["vehicle error on static cars (m/s)", "0.029", "0.029", "0.336", "0.284"], ["drivable IoU / walkway IoU / lane divider IoU@0.5", "0.756 / 0.340 / 0.335", "0.757 / 0.354 / 0.350", "0.759 / 0.345 / 0.343", "0.757 / 0.345 / 0.343"],
         ["front mAP2 (0 - 50 m)", "0.361", "0.364", "0.346", "0.347 (-0.014 [-0.020, -0.006])"], ["ms per frame (batch 1, cached chain)", "89.8 / 92.1", "", "", "90.7 / 89.8"]],
        header=["cold VAL, 1000 frames", "t10_dual_s0", "t10_dual_s1", "B' (batch 3)", "B2 (accumulation 2)"], num=(1, 2, 3, 4))
d.p("Verdict by the card's rule: V1 (learns) for B2. One seed only; the seed twin is the next run. Cost: front mAP2 -0.014 (vehicle AP@0.5 0.254 to 0.207) and a static-car error of 0.28 m/s. <code>t10_dual</code> held on its seed twin (maps within 0.02, front mAP2 0.361 / 0.364). The B2 prototype: <code>python evaluation/bev_prototype.py runs/t11_vel_hurdle_acc2 --scenes scene-0003 5 --video</code>.", "small")
d.fig(FIG + "w3_bev_frame.png", "A frame of the BEV dashboard video (<code>video/Week3_bev_dashboard_scene-0003.mp4</code>, scene-0003, B2).")
d.callout("Honest limits", "The +0.5 s forecast of the prototype uses the true ego motion from the ground truth; boxes are decoded from score 0.05; pedestrian velocity is still near zero (skill 0.014); the velocity fix was trained for 1000 steps from one checkpoint. The numbers are front camera + all radars only.", "warn")

d.h1("3  Toward an end-to-end lane chain", new_page=True)
d.p("Mirko's target: the lane chain must become network-only. The BEV system's design points the way: a dense, differentiable output interface with hard selection only for the metric. The two systems side by side:")
d.table([["main path", "24 stages: 8 learned, 2 fixed-differentiable, 2 frozen, <b>12 rule steps</b>", "14 stages: 8 learned, 2 fixed-differentiable, 4 frozen, <b>0 rule steps</b>"],
         ["rule steps", "seed selection, child choice, beam top-M, time-recursion match, alive filter, path union, clean-up, anchoring, relocation, fork pruning, piece cut, Douglas-Peucker", "only outside the main path: line-cell threshold, box selection (metric side)"],
         ["training", "builder, dense head, route head and consumer trained in separate pieces; the one joint attempt (route loss through the builder) was negative three times", "one trainer for the whole chain, Kendall task balancing, sequence training with a carried memory"],
         ["front end", "frozen GaussianCaR features and a frozen dense head, both cached", "frozen DualBEV image heads cached; trainable image features decided (D26)"]],
        header=["", "lane chain (registry systems)", "BEV system"], idcol=0)
svg(d, LANE_SVG, "The lane chain, drawn from its code: orange = a choice inside the network, red = post-processing without a gradient path.")
d.h2("Measured this week: who gets gradient from the loss")
d.p(f"Gradient report of the champion builder (<code>e2e_gradient_report.py</code>, 2 VAL frames, CPU float32, the trainer's loss; indicative): <b>no tensor is silent</b> ({n_tensors} of {n_tensors} parameter tensors get gradient). The end-to-end gap is therefore before and after the network, not inside it: the route head trains on stored token files, the assembly runs after the network, the GaussianCaR features and the dense head are frozen. On the 42 input channels the frozen GaussianCaR channels get a mean gradient of about 9e-6 per channel against 3e-4 for the dense-head heat: the signal a trainable front end would receive is small but nonzero.")
d.fig(FIG + "w3_gradient_route.png", "G1(1) of experiment E269: the route loss alone through the old joint path (builder e123_D7_r3_j1, 2 TRAIN frames, CPU). The route loss reaches the builder strongly through the node tokens, almost at right angles to the builder's own objective; it gives exactly zero to node positions, validity, link logits, junction head and link head.")
d.h2("Experiment card E269 and what three critic passes did to it")
d.p("The first idea (soft graph tokens: a route head on the beam nodes) went through three fresh critic passes, each returning REVISE. They found that the head interface described did not exist (no validity input, no link bias), that the route-success bootstrap the rule needs does not exist, that the first cost estimate was half the real one, and that a cheaper test of the same mechanism exists. The card is now the <b>straight-through proxy</b>: the old head and evaluator unchanged, only a gradient path from the route loss to the node positions; arms a (position path), c (E232's detached control), d (the route gradient on the tokens scaled to 0.1, added after the measurement above), two seeds each, about 8 slot-hours. The full soft-node design is a follow-up (E270) written only if the proxy shows an effect. Nothing is scheduled.")
d.callout("Why this matters", "The old premise &quot;the route loss cannot reach the builder&quot; was wrong; the measured picture is a <b>loud, near-orthogonal gradient</b> on the shared features (up to 6 times the builder's own on the child tokens, 20 times on the node tokens), a second candidate explanation of the three negative joint fine-tunes.", "warn")
d.h2("What was taken over from the BEV system")
d.table([["dense differentiable output interface, selection only for the metric", "contract of <code>Builder.forward</code> (<code>p16_interface.py</code>, self-test; not yet called by the live chain)"],
         ["slot tables and a gradient test", "<code>e2e_gradient_report.py</code> (self-test); slot tables come with the first rule-step replacement"],
         ["one coverage rule for loss, evaluation, execution (D36 / D37)", "later; the BEV chat owns the rule"],
         ["trainable front end, modality dropout (D26)", "step 2: the builder on the BEV trunk features, frozen first; risk: GaussianCaR features carry the lane skill (+0.24 F1, E27), BEV maps are weaker"],
         ["world-aligned temporal memory, sequence training", "step 3: replace the hard-match time recursion"],
         ["loss machinery: task balancing, gradient-conflict cosines", "the cosines are in the report; balancing later"]], header=["BEV concept", "use in the lane chain"], idcol=0)

d.h1("4  How the work is organised", new_page=True)
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
      "Three critic passes on E269 (section 3) turned a &quot;re-wiring&quot; into either a small proxy or a 4 - 5 day build, before any GPU time was spent.",
      "The tools review passes found run.py exit-code, hold-marker and provenance gaps; fixes are committed, a few remain on the review chat's list."])

d.h1("5  Operations: disk and repository", new_page=False)
d.table([["C: free space", "4.1 GB at the start of the clean-up, 36.4 GB after the move below, 55.5 GB at the last check (other chats cleaned too)"],
         ["lane run folders", "19.4 GB (615 files of 92 run folders) moved to <code>D:\\Auto-research\\lane_archive_2026-11-01</code>, verified by size, nothing deleted; 7.85 GB stay on C: (registered systems, the D7 lineage, control, live inputs). Record: <code>experiment_execution/history/lane_archive_2026-11-01/</code>"],
         ["golden check", "all six cached entries reproduced after the move (champion, champion_noprune, control, fast, fast_recall, recall; tolerance 0.0005); the one uncached entry needs the GPU"],
         ["WSL image", "<code>ext4.vhdx</code> 318 GB on C: for 279 GB used inside; compaction alone returns about 40 GB; one script prepared for Mirko (<code>purge_and_compact_2026-11-01.ps1</code>): delete the archive (19.4 GB), the D5 caches (32 GB) and the second-wave list (13.25 GB), then compact; needs an administrator shell and a quiet WSL"],
         ["repository", "main (183 commits) and the 25 <code>lcr/*</code> run tags were pushed on 26 October; later commits are local; the push rule of 1 November asks for a reviewed state first"]],
        header=["", "state"], idcol=0)

d.h1("6  Decisions open and next steps", new_page=False)
d.ul(["<b>Purge and compaction:</b> run the prepared script when no chat trains; decide whether the failure-series caches <code>pert</code> and <code>p17</code> (74.6 GB) go if the robustness line stops",
      "<b>D18 / D19:</b> per-metric noise bands, the machine lock between chats",
      "<b>E269:</b> the card is committed (arms a, c, d); building the straight-through term needs a critic code pass because <code>p18_joint.py</code> is shared",
      "BEV system: seed twin of B2, the long run with hurdle and accumulation 2, then the D36 coverage rule in loss and evaluation",
      "tools: reproducibility block, hold marker for stopped runs, hygiene list"])
d.h1("Appendix  Sources", new_page=False)
d.ul(["<code>bev_fusion/records/{BOARD.md, morning_2026-11-01.md}</code>; <code>bev_fusion/training/architecture.md</code>; <code>training/architecture.md</code>",
      "<code>experiments/E269_soft-graph-tokens-route-loss/experiment.md</code>; <code>experiment_execution/plans/spurmitte_e2e_setup_2026-11-01.md</code>; gradient reports <code>temp/g1_route_j1.json</code>, <code>temp/gradient_report_f1.json</code>",
      "<code>experiment_execution/{DECISIONS.md, HANDOVER.md}</code>; <code>tools/maintenance/PURGE_2026-10-31.md</code>"])
d.save(OUT)
