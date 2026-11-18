"""Status report after loop 21 (2026-10-31): HTML builder.  Run in gcar:  tools/env/_py14.sh reports/build/reports/build_status21_report.py   (after temp/loop21/fig_status_frames.py and fig_status_chart.py)
Numbers of the systems come from the metric cards through reports/source/weekly/s21_numbers.json (written by fig_status_chart.py); numbers of earlier reviewed summaries are quoted with their experiment id.
The review section is read from reports/source/weekly/s21_review.json (written by hand from the reviewer's findings).
Output: reports/weekly/Status_after_Loop21_EN.html  (PDF: reports/build/reports/print_weekly.ps1)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
F = "reports/source/figures/"
OUT = R + "/reports/weekly/Status_after_Loop21_EN.html"
NUM = json.load(open(R + "/reports/source/weekly/s21_numbers.json"))["cards"]
REV = json.load(open(R + "/reports/source/weekly/s21_review.json"))
FR = json.load(open(R + "/reports/source/figures/s21_frames.json"))
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))


def n(name, k, d=3):
    return f"{NUM[name][k]:.{d}f}"


d = W.Doc("Status after loop 21: where the lane graph stands",
          "Review of the loop-21 experiments, the best system and how it works, the numbers, qualitative results with the intermediate network outputs, a verdict and the next steps.",
          "Status report · after loop 21 · 31 October 2026",
          "Camera + radar BEV features (nuScenes) → sparse lane graph → map assignment → planner. Numbers are validation (VAL, 873 frames, 22 scenes) unless marked TEST. Earlier reports: Week 1 - 3 and the loop-18 report (reports/weekly/).")

# ------------------------------------------------------------------ 1 at a glance
d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")
d.h1("1  At a glance", new_page=False)
d.stats([
    (f"{n('recall_dp', 'Q')}", "Q of the best system (recall_dp)", f"at {n('recall_dp', 'vert100', 1)} vertices per 100 m after the output stage; the control E94 has {n('control', 'Q')} at {n('control', 'vert100', 1)} (node ids)"),
    ("15.5 → 10.4", "node ids per 100 m, recall → recall_dp", f"Douglas-Peucker output stage 0.2 m (E257): Q {n('recall', 'Q')} → {n('recall_dp', 'Q')}; a re-count of the same geometry: on the budget's own basis the count is unchanged (reviewer, section 2)"),
    ("−149 lines", "assembly code removed (E260)", "11 options gone, 31 → 24 parameters, graphs bit-identical (sha256 over all 873 frames)"),
    (f"{n('recall_dp', 'lanes_found', 2)} / {n('control', 'lanes_found', 2)}", "lanes found, recall_dp / control", f"R@1 far {n('recall_dp', 'R@1_far', 2)} / {n('control', 'R@1_far', 2)}: the open gap of the project has not moved since loop 18"),
    ("20 - 22 ms", "fast_recall chain per frame", "against 628 ms for the control; measured in loop 18, not re-measured in loop 21 (machine shared, N18)"),
    ("0 TEST uses", "in loop 21", "the one TEST check (E230, loop 18) covered champion, j1 and the recall network rc1 (TEST S 0.747, J 0.463, lanes found 0.724; control 0.702 / 0.404 / 0.762); nothing with the output stage was scored on TEST"),
])
d.h2("In four sentences")
d.ul([
    "<b>Loop 21 made the system simpler and its output smaller, not better.</b> Quality (Q about 0.74) has been at the same level since loop 18; what changed is the graph the system hands over (one third fewer nodes for the same geometry, so the same Q by construction) and a smaller, hash-verified assembly.",
    "<b>The best system is the recall variant of the differentiable builder D7 with fork pruning and the new output stage</b> (registry <code>recall_dp</code>); the fast variant <code>fast_recall_dp</code> is within noise on Q and 30 times faster than the control.",
    "<b>Against the slow control (E94) we match Q and win on junctions (J) and speed, and still lose on coverage</b> (lanes found, far recall, connectivity): the gap named first in PROBLEM.md is unchanged.",
    f"<b>The experiments were conducted correctly in their design;</b> {REV['one_line']}",
])

# ------------------------------------------------------------------ 2 review
d.h1("2  Were the experiments conducted correctly?", new_page=False)
d.p("Two independent checks: a fresh <code>ar-reviewer</code> (it was given the experiment folders, the code diffs and the rules, not the author's conclusions) and my own re-run of the stored numbers through the one evaluation entry point. "
    "Result of the reviewer, in short:")
d.callout("Verdict of the reviewer", REV["verdict"], REV.get("kind", "ok"))
TAG = dict(ok=OK, neg=NEG, tie=TIE, info=INFO)
d.table([[a, b, TAG[c.split("|")[0]](c.split("|", 1)[1])] for a, b, c in REV["rows"]], header=["item", "finding", "status"], widths=["26%", "58%", "16%"])
d.h2("What this means for the claims below")
d.ul(REV["consequences"])

# ------------------------------------------------------------------ 3 experiments
d.h1("3  Experiments in brief", new_page=False)
d.p("Loops 18 - 20 are in the week-3 report; they built the champion and measured the noise floor. Loop 21 (CPU only, on cached network outputs, because another chat trains on the GPU) asked one question: <i>can the system be simpler and faster without losing quality?</i>")
d.table([
    ["E257", "Douglas-Peucker output stage on the assembled graph", "eps 0.3 m: 15.0 → 9.0 node ids per 100 m, Q −0.0008 (paired scene bootstrap of −0.0014 … −0.0001); eps 0.2 m (chosen after the sweep, because the champion misses the F1@0.5 guard by 0.0003 at 0.3): fast_recall 15.0 → 9.9, champion 14.8 → 9.5, Q −0.0003 / −0.0005. The oracle-basis count is unchanged (12.2 / 11.3)", TIE("KEEP by pooled rule, adopted as D20; caveats")],
    ["E258", "profile of the assembly; remove a stage", "relocation 21 %, anchoring 16 %, union 16 %, clean-up 11 %, pruning 6 %; relocation stays (J −0.05 without it); anchoring off is free for the fast family on four seeds (Q −0.0011 … +0.0005) but costs the champion Q −0.0076", INFO("entry fast_recall_na")],
    ["E259", "clean-up ablation (spur / bubble)", "without it +0.44 … +0.73 vertices per 100 m on all seven models; lanes found +0.02 only because short stubs survive; stays", TIE("clean-up stays")],
    ["E260", "remove dead options of assemble()", "11 options, 149 lines removed; hashes of 6 systems identical; golden check, 9 + 18 tests, code review: no defect", OK("KEEP")],
    ["loop-21 timing", "E258 step 1b, E259", "machine load 4 - 15 during the runs: wall time and CPU time both unusable; only relative effects of about 25 % of the assembly visible. No ms claim is made", NEG("no valid timing")],
], header=["id", "question", "result", "decision"], widths=["8%", "24%", "52%", "16%"])
d.p("<b>Earlier loops (for orientation).</b> Loop 18: fork pruning closed the junction gap (J 0.265 → 0.409, Q 0.701 → 0.735), rough localisation and map trust, numba assembly (chain 17.5 ms). Loop 19: seed floor (Q SD 0.0025), recall weight is a recall lever, not a quality lever. "
    "Loop 20: a Poisson set loss, range-binned thresholds and a clamp all failed at a matched vertex budget; the map offsets of the ground truth do not explain the gap.", "small")

# ------------------------------------------------------------------ 4 architecture
d.h1("4  The best system and why it is built this way", new_page=True)
d.fig(F + "p18_arch.png", "The chain (drawing of loop 18; unchanged in loop 21 except that an optional output stage simplify_dp now follows the assembly). Grey = frozen or cached, blue = learned and differentiable, green = memory, orange = new in loop 18, hatched = rule step at inference.")
d.h2("How it works, stage by stage")
d.table([
    ["1  Foundation", "Camera + radar → BEV features (GaussianCaR): 200 × 200 cells of 0.5 m, ±50 m, 32 + 7 channels. Frozen, cached: the lane project trains nothing below this level.", "GaussianCaR (K61), nuScenes (K66); alternative CRN (K60); dense-segmentation convention SN-notes"],
    ["2  Dense head", "Lane heat and heading on the grid (U-Net / FCN-type head). Its maps are an input of the builder (next to the BEV features) and of the map registration; it is a pixel-based head.", "U-Net / FCN (K62)"],
    ["3  Builder D7 (graph-based, differentiable)", "A lattice of seeds (288; 192 in the fast variant) every few metres; each seed reads the features along a teardrop of 12 rays with ≤5 rings (loop-16 design, bilinear sampling, no argmax), gets a survival-type validity along its ray (path validity π), spawns child nodes (a tree), and a junction head predicts split and merge heat maps. Losses are matching-free: distance fields, point-to-polyline both ways, a soft-recall term (weight 3 in the recall variants).", "seeds and growth: RoadTracer (K14), DAGMapper (K11), iCurb (K50); verification along a line: L-CNN / HAWP (K40); discrete-time hazard with censoring (K41); multiple hypotheses (K38, K132); junction keypoints: TopoPoint (K31), TopoLogic (K30); matching-free set losses: Chamfer (K115)"],
    ["4  Assembly (rule step, no gradient)", "Path union in descending validity (snap to an edge within 0.5 m, reuse a vertex within 1.5 m), spur / bubble clean-up, junction anchoring (a heat peak forks the nearest edge), fork relocation, <b>fork pruning</b> (a fork without junction support keeps one branch), numba kernels.", "path-level graph: LaneGAP (K12); junction metric: RNTR (K47); K180 - K191 are own steps of loops 16 - 18"],
    ["5  Output stage (new, D20)", "Douglas-Peucker with 0.2 m on every polyline; ends and shared nodes are kept, so topology is unchanged and no vertex moves more than 0.2 m. It drops redundant nodes on straight runs: the geometry is the same, the node count is lower.", "Douglas-Peucker (K25, 1973); ground-truth compaction by DP (K142); dynamic pivots: PivotNet (K121)"],
    ["6  Downstream", "Temporal memory (hold / learned gate), rough localisation and trust of the SD map, route head along the edges, planner consumer. These work on the unsimplified vertices (<code>aux=True</code>).", "StreamMapNet / MapTracker (K45), OrienterNet (K99), TopoSD (K73), NavMapFusion (K74)"],
], header=["stage", "what it does", "literature basis (register id)"], widths=["15%", "50%", "35%"])
d.p("Why this shape: a <b>sparse graph</b> is what the planner and the map assignment need, and a short graph is cheaper to assign and to plan on; a false lane is worse than a missing short one. "
    "The builder is <b>differentiable up to the graph</b> (no argmax, no matching on the training path), the assembly is a rule step because merging hypotheses by proximity has no good differentiable form yet; "
    "the dense head and the foundation stay frozen so that the lane project trains in short runs on one 12 GB GPU. "
    "Facts about the papers are as recorded in <code>literature/register.md</code>; they were not re-read for this report.", "small")
d.callout("Which variant is which",
          f"<b>recall_dp</b>: the 288-seed builder with recall weight, fork pruning 0.4 and the output stage (Q {n('recall_dp', 'Q')}, lanes found {n('recall_dp', 'lanes_found', 2)}); "
          f"<b>fast_recall_dp</b>: 192 / 96 builder, recall weight (Q {n('fast_recall_dp', 'Q')}, 20 - 22 ms); <b>champion_dp</b>: the earlier champion with the stage (Q {n('champion_dp', 'Q')}, 24 ms). "
          "The registered <code>champion</code> and <code>control</code> entries are unchanged; the stage is off by default.", "")

# ------------------------------------------------------------------ 5 numbers
d.h1("5  Numerical results", new_page=True)
rows = []
for nm, lab in (("control", "control E94 (trace network)"), ("champion", "champion"), ("champion_dp", "champion_dp"), ("recall", "recall"), ("recall_dp", "recall_dp"),
                ("fast_recall", "fast_recall"), ("fast_recall_dp", "fast_recall_dp"), ("fast_recall_na", "fast_recall_na (no anchoring)")):
    rows.append([lab, n(nm, "Q"), n(nm, "topo_F1"), n(nm, "lanes_found"), n(nm, "R@1_far"), n(nm, "J"), n(nm, "S"), n(nm, "vert100", 2)])
d.table(rows, header=["system (registry name)", "Q", "topo F1", "lanes found", "R@1 far", "J", "S", "vertices / 100 m"], num=range(1, 8), hl=(4, 6))
d.p("VAL 873 frames, <code>evaluate.py</code> metric cards (single run per row; Q seed SD 0.0025, band 0.010). Vertices: node-id count for the rows without <code>_dp</code>, count after <code>simplify_dp</code> (0.2 m) for <code>_dp</code> rows; "
    "<b>S of a _dp row is not comparable with S of the others</b> (S rewards the lower count), Q is. The control's count is its raw node count, not simplified: a DP stage on the control would lower it too, so \"half the vertices\" is not a claim of this report. "
    "Speed (loop 18, idle GPU): control 628 ms, champion 24 ms (17.5 without time recursion), fast_recall 20 - 22 ms; the assembly alone about 7.5 - 7.8 ms idle (E253).", "small")
d.fig(F + "s21_numbers.png", "Left: Q against node ids per 100 m. The grey curve merges vertices (a larger vertex radius) and removes geometry: it loses 0.12 Q at 12 vertices. The blue curve (DP output stage) only removes nodes on straight runs; the two are not a like-for-like quality curve (reviewer). Right: the quality metrics of control and the three _dp systems.")
d.h2("Reading the table")
d.ul([
    f"<b>Q:</b> all systems lie within {NUM['recall']['Q'] - NUM['champion']['Q']:.3f} of each other (control {n('control', 'Q')}, champion {n('champion', 'Q')}, recall {n('recall', 'Q')}); the seed SD is 0.0025; recall over champion is +0.0054, but only +0.0025 at a matched vertex budget (E236); champion against control is a tie.",
    f"<b>The stage costs nothing on the metric,</b> by construction (same geometry, fewer nodes): Q changes by {NUM['recall_dp']['Q'] - NUM['recall']['Q']:+.4f} (recall), {NUM['fast_recall_dp']['Q'] - NUM['fast_recall']['Q']:+.4f} (fast_recall), {NUM['champion_dp']['Q'] - NUM['champion']['Q']:+.4f} (champion) for 33 - 36 % fewer node ids; lanes found, R@1 far, J are unchanged to the third decimal.",
    f"<b>Against the control:</b> topo F1 −{NUM['control']['topo_F1'] - NUM['recall']['topo_F1']:.3f}, lanes found −{NUM['control']['lanes_found'] - NUM['recall']['lanes_found']:.3f}, R@1 far −{NUM['control']['R@1_far'] - NUM['recall']['R@1_far']:.3f} for recall; J +{NUM['recall']['J'] - NUM['control']['J']:.3f}. fast_recall pays more coverage (lanes found −{NUM['control']['lanes_found'] - NUM['fast_recall']['lanes_found']:.3f}) for 30 times the speed.",
    "<b>TEST (once, E230, loop 18, 833 frames, from the TEST log):</b> champion S 0.748, J 0.422, lanes found 0.653; recall network rc1 + pruning 0.4: S 0.747, Q 0.727, J 0.463, lanes found 0.724 at 16.6 vertices; control S 0.702, J 0.404, lanes found 0.762 at 22.0 vertices. "
    "<b>Route success on TEST (head along the edges, two seeds) favours the control:</b> 0.894 / 0.868 against 0.852 / 0.833 for rc1. The systems with the output stage were not scored on TEST; their network is the one tested in E230.",
])
d.h2("The output stage in detail (E257, fast_recall)")
sw = json.load(open(R + "/reports/source/weekly/s21_numbers.json"))["sweep"]["fast_recall"]
d.table([[("none" if a == "none" else a + " m"), f"{b:.2f}", f"{c:.4f}", f"{c - sw[0][2]:+.4f}"] for a, b, c in sw], header=["DP tolerance", "vertices / 100 m", "Q", "dQ"], num=(1, 2, 3), hl=(2,))
d.p("The tolerance 0.2 m is the one that keeps every guard of the card for both systems (the champion misses the F1@0.5 guard by 0.0003 at 0.3 m); the choice of 0.2 was made from this sweep on VAL, so it is selected on the evaluation set (no separate tuning split; the effect is a monotone trade-off, not a fitted optimum).", "small")

# ------------------------------------------------------------------ 6 qualitative
d.h1("6  Qualitative results", new_page=True)
d.p("Six VAL frames, three per city, chosen by <b>rank</b> of the per-frame F1@1.0 of recall_dp inside the city (quantiles 0.85, 0.5, 0.15; a different scene each; not hand-picked for beauty). "
    "Left: front camera with the graphs projected onto the road plane (ground truth green, prediction red); the camera image is for orientation: the lane project itself works on the cached BEV features that the frozen foundation computed from camera and radar. Right: ego-frame graphs, prediction red over ground truth green, triangles = splits, diamonds = merges.")
d.fig(F + "s21_frames_boston.png", f"Boston seaport (frames {FR['boston']['frames']}, recall_dp F1@1.0 {', '.join(f'{x:.2f}' for x in FR['boston']['f1_recall_dp'])}). The junction frames show the typical error: surplus short branches and cluttered forks inside intersections; the third frame (a truck and a car ahead) is one where recall_dp beats the control (0.81 against 0.65).")
d.fig(F + "s21_frames_singapore.png", f"Singapore One-North (frames {FR['singapore']['frames']}, recall_dp F1@1.0 {', '.join(f'{x:.2f}' for x in FR['singapore']['f1_recall_dp'])}). Straight roads are near-perfect; at the intersection (middle) forks are found but extra stubs appear; in the curve (bottom) the lines are partly doubled.")
d.p("What the images show honestly: (1) on straight and gently curved roads the graph is almost exact; in these six frames recall_dp has 24 - 85 vertices and the control 62 - 225 (the control is not simplified); "
    "(2) intersections are the failure region: split / merge marks are in about the right places, but the fork structure is noisy (extra stubs, duplicated lanes), which is what the J and topo F1 numbers say; "
    "(3) the camera overlay is a check of the geometry (the projection uses the calibration, ground at the vehicle plane), not part of the evaluation.")

# ------------------------------------------------------------------ 7 intermediate
d.h1("7  Inside the network", new_page=True)
d.p("The intermediate outputs of <code>recall</code> for the median frame of each city (rank 0.5 above). Stages 1 - 3 are inputs from the frozen foundation, 4 - 5 are what the differentiable builder predicts, 6 - 7 are the rule-based assembly and the output stage.")
d.fig(F + "s21_inter_singapore.png", "Singapore frame 242 (a fork with a side road). 1: the BEV features already show the lane structure but also clutter (bright blobs are other things). 2 - 3: the dense head turns it into lane heat and heading (opposite directions get different hues). 4: the builder's hypothesis tree: only nodes with path validity above tau 0.7 (coloured lines, 237 of 576 here) are used; the rest (grey) is discarded. 5: the junction head peaks on the actual splits and merges (green marks = ground truth). 6 - 7: assembled graph and the same graph after the DP stage: 71 → 55 vertices, same shape, same F1@1.0.")
d.fig(F + "s21_inter_boston.png", "Boston frame 696 (a four-way junction). The dense head finds the parallel lanes and the cross street (2 - 3); the junction head (5) has heat near the ground-truth forks, several of them weak; the assembled graph (6) has the right skeleton and extra short branches at the junction; the DP stage (7) removes the redundant intermediate vertices on the straight parts (132 → 85) and leaves the topology as it was.")
d.p("Two things to notice. The DP stage only removes vertices on straight runs; it cannot remove the spurious stubs at junctions (that is the assembly's job, E259 shows that the clean-up already does what it can). And the heat map and the direction field are accurate where the features are (straight roads) and diffuse in the clutter at the image border, which is why far recall (> 40 m) is the weakest metric.")

# ------------------------------------------------------------------ 8 verdict
d.h1("8  Final verdict", new_page=True)
d.table([
    ["Quality", "Q about 0.74 at about 10 node ids (about 12 on the budget basis) per 100 m; equal to the 628 ms control within the noise band; plateau since loop 18", INFO("on a plateau")],
    ["Speed", "fast_recall 20 - 22 ms per frame against 628 ms (loop 18); no valid new timing in loop 21 (the relative stage savings of E258 / E259 are provisional)", INFO("not re-measured")],
    ["Sparsity", "−33 … −36 % node ids at ≤ −0.0005 Q (E257); on the budget's own basis (per-piece DP count) 12.2 / 11.3 before and after, so no gain there; node-id graphs remain for the route / map pipeline", TIE("smaller output, same content")],
    ["Simplicity", "assembly options 31 → 24, −149 lines, identical graphs (E260); one more optional stage (D20)", OK("improved")],
    ["Coverage at range", f"lanes found {n('recall_dp', 'lanes_found', 2)} / R@1 far {n('recall_dp', 'R@1_far', 2)} against {n('control', 'lanes_found', 2)} / {n('control', 'R@1_far', 2)}; loss weights, clamp, range thresholds and the Poisson loss are all spent", NEG("open gap")],
    ["Connectivity", f"topo F1 {n('recall_dp', 'topo_F1', 3)} against {n('control', 'topo_F1', 3)}; fork structure inside intersections noisy (section 6)", NEG("open gap")],
    ["Robustness / map", "map registration + trust works on new error draws (E238); recall variants lose Q 0.038 with cameras off (known gap 5)", INFO("unchanged")],
    ["Route success", "TEST (E230): control 0.894 / 0.868 against 0.852 / 0.833 for the recall network, 0.84 - 0.86 for the champion; not re-measured in loops 19 - 21 and not on the simplified graphs", NEG("control ahead")],
    ["Planner-level", "no planner metric responds (path error floor 0.27 m)", NEG("not verified")],
], header=["axis", "state", "verdict"], widths=["16%", "68%", "16%"])
d.callout("Overall", "The system is a good, fast, sparse lane-graph builder that is as good as its slow predecessor on the headline Q and better on junction score and speed, and it is clearly worse on coverage at range and on connectivity. "
          "Loop 21 improved form (a smaller delivered graph, a simpler verified assembly) and not substance. More loss and threshold tweaks on the frozen features have been tried and are closed; the next real gain has to come from a different place (see below).", "warn")

# ------------------------------------------------------------------ 9 next
d.h1("9  Next steps")
d.ul([
    "<b>Idle-window timing</b> (E253 driver) of <code>fast_recall_na</code>, the <code>*_dp</code> systems and the stage ranking of E259: the only way to turn the relative savings into ms. Needs the GPU chat to pause (declined for now); a machine lease between chats is open as D19.",
    "<b>Route and map on simplified graphs:</b> the route head and map assignment still use the node-id graph (<code>aux=True</code>); to adopt the stage in the whole chain the route scores must be carried to the simplified polylines and route success re-measured.",
    "<b>Attack the coverage gap with something new</b> (GPU needed): a seed ensemble of cached outputs as a ceiling test first (CPU), then candidates from PLAN.md; the loss and threshold family is closed.",
    "<b>TEST and the output stage:</b> the networks behind recall_dp and champion_dp were scored on TEST in E230; the stage itself has not been. A TEST check is needed only if the stage becomes the product default, and then only with Mirko's OK and a row in the TEST log; all loop-21 numbers are VAL.",
    "<b>Decisions waiting for Mirko:</b> D4 (which fast variant), D5 (purge), D17 (loop list in project.json: loop 21 is not in it, so the loop table does not count its runs), D18 (per-metric noise bands), D19 (machine lease).",
    "<b>Correct the records the reviewer found</b> (small): the E258 timing header and E259 slips, the stale HANDOVER lines, a stored result file for the vertex-basis probe (done: E257 <code>vertex_basis.txt</code>); the PROBLEM.md sentence on the vertex count needs a grant naming a decision (it compares a node-id count with the 14.3 budget). Vectorise the DP stage only if it stays in the product.",
])

d.h1("Appendix  Experiment index of loop 21")
d.table([
    ["E257", "DP output stage", TIE("KEEP (pooled), D20 adopted; caveats"), "experiments/E257_dp-output-simplification"],
    ["E258", "assembly profile, remove a stage", INFO("entry fast_recall_na"), "experiments/E258_assembly-profile-and-speedup"],
    ["E259", "clean-up ablation", TIE("clean-up stays"), "experiments/E259_assembly-cleanup-ablation"],
    ["E260", "dead-option removal", OK("KEEP"), "experiments/E260_assembly-dead-code-removal"],
], header=["id", "question", "outcome", "folder"], widths=["8%", "34%", "20%", "38%"])
d.p("Figures: <code>temp/loop21/fig_status_frames.py</code>, <code>fig_status_chart.py</code>; the numbers file <code>reports/source/weekly/s21_numbers.json</code> is written from the metric cards. The mp4 of recall (<code>Lane_graph_recall_val.mp4</code>) shows two full scenes.", "small")
d.save(OUT)
