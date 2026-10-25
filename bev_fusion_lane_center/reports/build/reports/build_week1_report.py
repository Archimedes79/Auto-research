"""Week 1 report (loops 1-9, 2026-10-11 .. 2026-10-17): HTML builder.  Run in gcar:  tools/env/_py14.sh build_week1_report.py
Numbers come from reports/source/weekly/digest_week1.md (every number there carries its source); keep both in step.
Output: reports/weekly/Week1_Loops1-9_Report_EN.html  (PDF: reports/build/print_weekly.ps1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

F = "reports/source/figures/"
A = "reports/source/figures/"
OUT = W.ROOT + "/reports/weekly/Week1_Loops1-9_Report_EN.html"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))

d = W.Doc("From a frozen camera-radar foundation to a perception-first lane graph with map assignment",
          "Week 1 of the lane-centre-line graph project, loops 1 to 9: what was built, what worked, what did not, and which numbers can be trusted.",
          "Week 1 · Loops 1 - 9 · 11 - 17 October 2026",
          "Camera + radar BEV features (nuScenes) → lane-centre-line graph → map assignment → planner. Numbers are TEST unless marked VAL or DEV.")

# ------------------------------------------------------------------ contents + at a glance
d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")
d.h1("1  At a glance", new_page=False)
d.p("The task: from six cameras and radar sweeps alone, produce the <b>lane centre lines around the car as a graph</b> (with splits and merges), assign them to a navigation map "
    "that carries the route, <b>transfer the route to the extracted lanes</b> and hand the result to a planner, robustly enough that sensor failures and map errors do not silently produce a wrong route. "
    "In nine loops the work moved from a detection-pretrained foundation and a dense distance field to a perception-first graph pipeline with a learned assignment and training-free extension and tracking.")
d.stats([
    ("+0.243", "F1@1 m from the foundation swap", "CRN 0.624 → GaussianCaR 0.867 (VAL); TEST 0.598 → 0.835. Larger than any decoder or loss change"),
    ("0.876", "lane F1@1 m, TEST, no map", "grid + refiner + temporal head (VAL 0.899). fold heads: pixel F1 0.932 on own training scenes, 0.820 on TEST (final head 0.959 / 0.856); a scene shift nothing removed"),
    ("0.881 → 0.839", "route success, map error 0 → 3 m", "assignment after perception on 6 m edges (loop 8); loop 9: 0.876 / 0.879 / 0.876 / 0.858"),
    ("0.867 → 0.259", "graph F1 at night", "perception fails silently: blur 0.433, three front cameras off 0.648"),
    ("0.663 → 0.847", "route success, per-road map bias 2 m", "training against map faults (loop 9); wrong-map route flagged 60.7 → 4.2 m"),
    ("0.308 m", "reference path error at 20 m", "SD route + perceived lateral offset; SD centre line alone 0.949 m"),
])
d.h2("What we learned this week")
d.ul([
    "<b>The foundation decides.</b> Swapping the frozen detection-pretrained CRN for the map-pretrained GaussianCaR gave +0.24 F1; no head, loss or read-out change came close.",
    "<b>A dense grid head with a rule-based tracer beat a sparse DETR-style extractor</b> by 0.06 F1 and 0.18 route F; refiner, learned links and temporal fusion added little. "
    "The ceiling is a seen-to-unseen scene shift (cause inferred: the frozen foundation) (fold heads: 0.932 on training scenes, 0.820 on TEST).",
    "<b>A map helps most after registration.</b> A head trained on exact poses is hurt by a 1 m error (worse than no map, E40) and without registration the benefit shrinks with the error. With registration the route transfers by rule-based association (0.923 at 1 m).",
    "<b>Perception fails silently under sensor loss</b> and head-side training cannot restore lost information. The answer is layers: monitor, hold registration, carry the graph forward, fall back to map lanes.",
    "<b>A navigation map carries road and branch, not lane.</b> In this implementation, assigning after perception on 6 m edges reaches 0.88 against 0.73 for the pixel SD head (0.71 when that head's map is registered first); the comparison also changes front end, features, ensemble and training data. Retraining against map faults was worth more than the re-registration we tried.",
    "<b>Several results are upper bounds.</b> The SD map is simulated from the same lane map as the ground truth, all failures are synthetic, and some thresholds were fitted on TEST frames.",
])
d.callout("How to read 'end to end' in this week",
          "Week 1 ended with a modular pipeline whose <i>output</i> is a differentiable interface for a planner. Tracer, registration and tracks are not differentiable, and joint training with a planner loss (E86c) gave no clear gain (mixed: better without a map and with three cameras off, worse at night and on turning rows). "
          "Making the builder itself differentiable is the topic of week 2.", "warn")

# ------------------------------------------------------------------ setup
d.h1("2  Task, data and how to read the numbers")
d.p("<b>Output.</b> Per nuScenes keyframe a lane-centre-line graph in the ego BEV frame (x right, y forward, ±50 m, 200 × 200 cells of 0.5 m) with explicit splits and merges; "
    "assigned to a map graph that carries the navigation goal; the goal transferred to the extracted lanes and handed to a planner with trust signals. "
    "End-to-end module output, no manual refinement; only the map input may be edited by hand.")
d.p("<b>Data.</b> The first three nuScenes blobs: Boston and Singapore, day and dry only (no night or rain scene). Split by scene: TRAIN 212 scenes / 8,414 frames, VAL 22 / 873, TEST 21 / 833 (809 with a route label; blur, extrinsics and radar-off exist for every second TEST frame = 417, 409 with a route label). "
    "255 scenes in all; the '323 scenes' in older texts is wrong. VAL decides, TEST reports. The TEST rows are not contiguous: always index with <code>splits.json['test']</code>.")
d.p("<b>Foundation.</b> Frozen camera-radar BEV network; only small heads are trained. Loop 1: CRN (detection-pretrained, 128 channels at 0.8 m cells). Loops 2 - 9: GaussianCaR (map-segmentation-pretrained; 32 PCA decoder channels + 7 map logits = 39 channels at 0.5 m; about 0.4 s per frame on the RTX 3060).")
d.h3("Metrics")
d.table([
    ["F1@1 m", "Precision / recall of dense 0.5 m points; a point is correct if the nearest point of the other set is within 1 m."],
    ["Lanes found, false length", "Share of ground-truth lane pieces with ≥ 80 % of points within 1 m; predicted length more than 1.5 m from any ground-truth point."],
    ["Split / merge F1 @6 m", "Location match of nodes with ≥ 2 successors / predecessors within 6 m."],
    ["Route F", "APLS-like comparison of directed shortest-path lengths between sampled reachable pairs (≤ 40 m)."],
    ["Route success (loop 5 on)", "Per frame, the ground-truth route is covered (share within 1.5 m of a point flagged on-route) and precisely covered (share of flagged points within 1.5 m of the route), both ≥ 0.8. The route label is the lane sequence actually driven in the next ≤ 6 s."],
    ["Map error σ", "Rigid shift σ per axis plus yaw 2 σ degrees applied to the map; one evaluation condition is a 'cell'."],
    ["Lateral error, reference path", "Planner proxies: lateral error of a route polyline followed at true speed (loop 5); deviation of the reference path from the driven trajectory at 10 / 20 / 30 m (loop 8)."],
], header=["Metric", "Meaning"], widths=["24%", None])
d.callout("Five rules for reading any number in this report",
          "<ol style='margin:0;padding-left:5mm'>"
          "<li><b>VAL, DEV or TEST.</b> TEST (21 scenes) is meant to be used once, but was not: finalist and ablation runs exist on TEST, and the monitors and thresholds of loop 5 (E64, E67, E69, E70 - E72) were fitted on TEST frames, so those results are in-sample.</li>"
          "<li><b>Noise.</b> Scene-level intervals are about ±0.035 (F1), ±0.05 (route F) and ±0.03 (route success). Dense-head seed SD 0.0025 VAL / 0.0006 TEST; DEV score SD 0.0011 - 0.0014. Most TEST runs are single seeds; later work put the route-head seed spread at up to 0.03 on clean frames and 0.05 - 0.09 under failure, so headline differences of single route-head runs (E60, E65d, v2) are not interpretable below that.</li>"
          "<li><b>The SD map is simulated</b> from the lane map that also gives the ground truth, and the SD route equals the driven route. Extension, reference path and exact-pose numbers are upper bounds or circular; the 'oracle' labels (0.913) are not an upper bound.</li>"
          "<li><b>Everything adverse is synthetic:</b> sensor failures, map errors, drift, jumps. Open loop only.</li>"
          "<li><b>Loop 1 has no test set;</b> it tuned and reported on the same validation data and is not comparable to later numbers.</li></ol>")

# ------------------------------------------------------------------ story
d.h1("3  The week as a story")
d.table([
    ["1", "11 - 13 Oct", "How far do frozen camera + radar BEV features carry lane centre lines? (E15 - E26)", "Distance-field U-Net reaches F1@1 m 0.670 (1,706-sample validation set, no test set). The field, not the line forming, is the limit.", TIE("mixed")],
    ["2", "13 Oct", "Which foundation, which feature level, grid or sparse? (E27 - E33)", "Map-pretrained GaussianCaR: +0.243. Grid + tracer beats a sparse DETR-style extractor.", OK("positive")],
    ["3", "13 - 14 Oct", "Bring the grid head to maturity; refine, link, map prior, temporal (E30, E37 - E43)", "Final system TEST F1 0.876, route F 0.744. Seen-to-unseen shift caps everything.", TIE("mixed")],
    ["4", "14 Oct", "Better topology and registration: SAM-Road, LaneGAP, TopoLogic, RAVE ideas (E44 - E56)", "Learned key points: split / merge F1@6 m 0.34 / 0.38 → 0.55 / 0.61. Foundation read-out variants fail.", OK("positive")],
    ["5", "14 - 15 Oct", "Route transfer, end-to-end head, temporal tracking, robustness (E57 - E72)", "Route transfer by rule-based association (0.926 at exact pose, an upper bound); perception fails silently; layered cover.", OK("positive")],
    ["6", "15 Oct", "Systematic fine-tuning of the route head (E73 - E76)", "Auxiliary trust task + weight averaging: exact-map route success 0.849 → 0.920.", OK("positive")],
    ["7", "15 - 16 Oct", "Simulated SD map, temporal match, ADASIS quality (E77 - E80)", "SD head 0.733 (geometric 0.46). A training-free average gave more lane F1 than a learned memory.", TIE("mixed")],
    ["8", "16 - 17 Oct", "Perception first, assignment after (E81 - E86)", "Assignment on 6 m edges: 0.881 / 0.879 / 0.868 / 0.839. A protocol bug was found and fixed.", OK("positive")],
    ["9", "17 Oct", "The biggest issues of loop 8: map bias, abstention (E87 - E90)", "Per-road bias 2 m 0.663 → 0.847; wrong-map flag 60.7 → 4.2 m.", OK("positive")],
], header=["Loop", "Dates", "Question", "Answer", ""], widths=["5%", "10%", "34%", "45%", "6%"], idcol=0)
d.p("<span class='small'>The journal has no entries for loops 1 - 2; their dates and requests are inferred from file times. The data download started on 11 October at 12:12. "
    "Lidar experiments E01 - E14 were deleted on 13 October at the user's request and are not part of the project.</span>")

# ------------------------------------------------------------------ loop 1
d.h1("4  Loop 1: how far do frozen camera + radar features carry lane centre lines? (E15 - E26)")
d.p("<b>Built.</b> CRN features → dense decoder that predicts the distance to the nearest centre line → threshold + Zhang-Suen skeleton → polylines. "
    "Decoders: FCN 0.10 M, U-Net (base 64) 3.38 M, CNT-Net 4.78 M (ConvNeXt V2 + four-layer transformer on 25 × 25 tokens). Map nodes every 3 m with synthetic drift, grid alignment and a Kalman drift tracker; a point-by-point tracer (E26).")
d.table([
    ["E17  FCN → U-Net 64", "FCN 0.264 on Mini → 0.493 with all data; U-Net 64 0.613", "more data and a bigger decoder help most (U-Net +0.12 over FCN at default line forming, +0.09 tuned; 26x more data and 10x more steps)"],
    ["E19  threshold 0.5 → 0.15 m", "U-Net 0.613 → 0.670 (P 0.750, R 0.606); smoothing hurts", "default threshold costs ~0.06 F1"],
    ["E18  CNT-Net", "0.507 (0.583 tuned), about 0.11 below U-Net", "negative"],
    ["E20, E23, E25", "temporal field fusion 0.675 vs 0.670; ensemble 0.650; coordinate channels +0.131 on a short-trained FCN", "no gain / not confirmed with full training"],
    ["E24  range", "recall 0.81 (0 - 10 m) → 0.50 (40 - 48 m); precision 0.87 → 0.66", "far field is the weak part"],
    ["E21, E22  map assignment", "route-node lane assignment 79.2 % → 91.9 %; Kalman tracker node error 1.19 → 0.83 m at large drift", "positive"],
    ["E26  tracer", "perfect field: tracer 0.997 / skeleton 0.995; predicted field: tracer 0.693 vs skeleton 0.670; 3 m nodes 0.692 with 175 nodes", "sparsification costs at most 0.3 - 0.5 % F1, but the field is the bottleneck"],
], header=["Experiment", "Result (F1@1 m, 1,706-sample validation set)", "Reading"], idcol=0)
d.callout("Verdict " + TIE("mixed"),
          "A frozen detection-pretrained CRN is the limit: about 0.30 F1 is lost in the field, not in line forming. The next lever is a map-pretrained foundation. "
          "<b>Caveat:</b> loop 1 has no test set; it tunes and reports on a 1,706-sample validation set (= the later VAL and TEST scenes together; E26 reports on the held-out odd half) and is not comparable to loop 2+ numbers. E17's 'x1.9' compared 81 Mini with 1,706 validation samples and 10x more steps.")
d.fig("reports/source/figures/R17_viz_est_unet64.png", "CRN U-Net 64: predicted field and extracted lines against the ground truth (loop 1).", 56)

# ------------------------------------------------------------------ loops 2-3
d.h1("5  Loops 2 - 3: GaussianCaR, grid against sparse, refiner and map prior (E27 - E43)")
d.h2("The foundation swap and the dense head")
d.p("Dense head = U-Net 2.7 M (heat map σ 0.75 m + direction, focal BCE, AdamW one-cycle, fp16) on the 39 GaussianCaR channels, plus a rule-based tracer that builds the directed graph. "
    "The sparse alternative E32 (LaneFormer 4.3 M, 96 queries × 11 points, Hungarian matching) was trained for comparison.")
d.table([
    ["E27  CRN features", "0.647", "0.624", "0.598"],
    ["E27  GaussianCaR features", "0.926", "0.867", "0.835"],
    ["E31  grid graph, tracer (route F VAL / TEST 0.765 / 0.702)", "-", "0.879", "0.851"],
    ["E31  grid graph, skeleton (route F 0.735 / 0.672)", "-", "0.854", "0.821"],
    ["E32  sparse-only LaneFormer (route F 0.587 / 0.519)", "0.880", "0.821", "0.791"],
    ["E37  final dense head (6,000 steps, 8,414 rows)", "0.959", "0.885", "0.856"],
], header=["Pixel F1@1 m unless stated", "TRAIN_EVAL", "VAL", "TEST"], num=(1, 2, 3), hl=(1, 5))
d.p("Decoder feature levels (VAL / TEST): PCA-32 of the decoder 0.874 / 0.840, fused BEV feature (PCA-32) 0.772 / 0.743, seven logits only 0.779 / 0.738 (E28). "
    "A three-times false-alarm weight (E29) only recalibrates (+0.003). Training the E37 head took <b>14 attempts</b>: loss jumps with decoder BN variance blowing up 100 - 800 ×, bf16 giving NaN; "
    "the fix was lr 6e-4, Adam β2 0.98, gradient clipping and a roll-back guard.")
d.h2("Refiner, links, map prior, temporal (loop 3)")
d.table([
    ["Grid graph (E37 head)", "0.891 / 0.867", "0.782 / 0.722", "0.286 / 0.390 (VAL)"],
    ["+ cascade refiner (E39, 0.76 M)", "0.892 / 0.869", "0.796 / 0.736", "gain 0.017 on seen vs 0.006 on VAL scenes"],
    ["+ learned link head (E42)", "-", "0.799 vs 0.796", "ties the rule links (negative)"],
    ["+ temporal A, previous features warped (E41)", "0.899 / 0.876 (dense head 0.895 / 0.869)", "stability 0.971 / 0.961", "B (propagated graph as proposals) neutral or worse"],
    ["Map prior, exact pose (E40)", "0.958 / 0.940 (dense head 0.962 / 0.943)", "0.824 / 0.777", "no-map control 0.887 / 0.860 (dense head)"],
    ["Map prior, 1 m noise, unaligned / aligned", "0.830 / 0.915 (VAL)", "-", "unaligned is worse than no map"],
    ["<b>E43 final system, no map (TEST [VAL])</b>", "<b>0.876</b> [0.899]", "<b>0.744</b> [0.799]", "split / merge 0.342 / 0.380 [0.308 / 0.417]; ≤ 41 ms chain (upper bound, GPU shared, without foundation)"],
    ["E43 final with exact-pose map", "0.939", "0.778", "stability 0.976"],
], header=["Component", "F1@1 m VAL / TEST", "Route F VAL / TEST", "Note"], num=(), hl=(6,), idcol=0)
d.fig(A + "E43_e43_ladder.png", "E43 build-up (VAL): grid graph → refiner → links → map prior → temporal variants → final system with and without map.", 100)
d.p("<b>Grid ceilings (E38, VAL).</b> Relaxed proposals cover 87.5 / 94.1 / 98.2 % of the ground truth within 1 / 1.5 / 2.5 m; the ceiling F1 if perfectly snapped is 0.934 / 0.970. "
    "With an oracle field the rule assembler reaches route F 0.838 (VAL) / 0.823 (TEST): the topology is limited by the rule-based assembly and the piece set (annotation conventions are a possible reason, not tested).")
d.h2("The seen-to-unseen shift")
d.fig(A + "E30_e30_decomposition.png", "E30: the same three fold heads on own training scenes (0.932), held-out training scenes (0.899), VAL (0.851) and TEST (0.820). Left: the drop. Right: it is not a sampling effect of 22 scenes.", 100)
d.p("212 distinct training scenes matter more than the number of rows: 212 seen scenes give 0.849 / 0.844 (VAL-B / TEST), 11 seen scenes 0.814 / 0.800, 11 unseen scenes 0.786 / 0.776. "
    "Weight decay + dropout (0.860) and mirror augmentation (0.849) lower VAL (baseline 0.872); a wider read-out, instance norm and a second foundation did not remove the gap either (loop 4). "
    "A plausible cause is that the foundation was trained on the training scenes' map labels (inferred, not tested). The gap is the same in both cities.")
d.callout("Verdict " + TIE("mixed"),
          "Grid + tracer adopted, sparse-only dropped. There is no under-fitting; the loss is 0.033 head over-fitting plus a further 0.048 (VAL) / 0.079 (TEST; about 0.03 of it is attributed to harder TEST scenes) seen-to-unseen shift that nothing on the head side removes. "
          "<b>Bug:</b> the first E30 split by scene parity produced one scene; fixed by scene rank and rerun. The E30 chain uses fold heads (0.885 / 0.856 is E37): do not mix.")
d.fig(A + "E31_e31_pred.png", "E31: directed lane graph over the ground truth (VAL).", 84)

# ------------------------------------------------------------------ loop 4
d.h1("6  Loop 4: topology and registration (E44 - E56)")
d.p("Request: 'better foundation, sparsification; consider SAM-Road, LaneGAP, SeqGrowGraph, TopoLogic; RAVE is interesting for localisation'. The grid graph has many fewer splits and merges than the truth (1.8 / 2.1 vs 2.6 / 2.9 per sample).")
d.ul([
    "<b>E44 key-point head</b> (SAM-Road style): two heat maps for split and merge nodes (σ 1.5 m), NMS 3.5 m, then a geometry-preserving cut and prune of the graph.",
    "<b>E53 Bezier connectors</b> from split to merge key points (no new weights, accepted by mean dense heat).",
    "<b>E48 learned registration</b> (RAVE-like FFT cost volume, 7 yaw × 9 × 9 shifts) after the correlation search.",
    "E45 TopoLogic-style link scoring, E46 LaneGAP path view, E49 - E52 read-out variants (PCA 96, instance norm, GaussianCaR + CRN, node-conditioned head), E54 paired scene bootstrap, E55 seeds.",
])
d.table([
    ["Grid graph (E43)", "0.342 / 0.380", "0.744", "0.876", "-"],
    ["+ key points, default cut + prune (E44)", "0.552 / 0.607", "0.733", "0.876", "split +0.210 [+0.133, +0.291], merge +0.227 [+0.184, +0.267], route F −0.011 [−0.017, −0.005]"],
    ["+ Bezier connectors, heat 0.5 (E53)", "0.714 / 0.637", "0.719", "0.875", "route F −0.025 [−0.033, −0.018]"],
    ["Oracle key points (GT nodes)", "0.694 / 0.752", "-", "-", "55 % / 42 % of confirmed GT nodes have no branch piece (VAL diagnosis)"],
    ["with exact-pose map prior + connectors", "0.717 / 0.643", "0.750", "0.937", "without connectors: map prior vs none F1 +0.063, route F +0.033 (bootstrap); with connectors +0.062 / +0.031"],
], header=["TEST", "Split / merge F1@6 m", "Route F", "F1@1 m", "Paired bootstrap (95 %)"], hl=(1,), idcol=0, widths=["30%", "14%", "9%", "9%", None])
d.p("<b>Registration (E48, VAL, 300 samples):</b> median position error at σ 1 m: correlation search 0.54 m, learned chain 0.43 m (yaw −38 %); downstream head F1 0.915 / 0.904 → 0.922 / 0.912 (exact pose 0.962 / 0.943). The floor of about 0.4 m is far above RAVE's 0.13 m. "
    "<b>Path view (E46):</b> path recall / precision / F1 0.428 / 0.557 / 0.484, node insertion lowers path F1 to 0.459. "
    "<b>No gain:</b> PCA-96 read-out 0.864 / 0.834 (reference 0.872 / 0.838), instance norm 0.869 / 0.840, GaussianCaR + CRN 0.865 / 0.836, node-conditioned head 0.898 / 0.873 vs 0.895 / 0.869. "
    "<b>Seeds (E55):</b> dense head TEST 0.838 / 0.837 / 0.838.")
d.fig(A + "E44_e44_examples.png", "E44: graph before and after node insertion, with ground truth and key points.", 58)
d.callout("Verdict " + OK("positive"),
          "Learned key points (split / merge F1 ×1.6 on TEST) and optional Bezier connectors (recall-first switch) are adopted; the cost is 0.011 - 0.025 route F, and the node gains over the previous system lie far outside their intervals (lower bounds +0.13 / +0.18 for key points, +0.27 / +0.21 for connectors); the merge gain of the connectors over the key-point default is only just outside (lower bound +0.003). "
          "Foundation read-out changes do not help; the shift persists. Intervals cover scene sampling only, not seeds or VAL-selected thresholds. A review round corrected the wording of the E55 comparison.")

# ------------------------------------------------------------------ loop 5
d.h1("7  Loop 5: route transfer, an end-to-end head, tracking and robustness (E57 - E72)")
d.p("New task: the map graph carries the navigation goal; assign the extracted lanes to it and <b>transfer the route</b>; hand the module output to a planner without manual refinement; track in time; "
    "make it robust ('nuScenes is not the real world'); and answer whether a vision-language-action model is needed.")
d.h2("What happens when a sensor fails (E57)")
d.fig(A + "E57_e57_bars.png", "E57: system output under synthetic sensor perturbations (TEST). Corruptions are applied before the foundation: front camera black, three forward cameras black, night (images × 0.15 + noise), fog, blur (σ 3 px), radar nearly off (3 points), extrinsics jitter 3°.", 100)
d.p("Graph F1@1 m: clean 0.867, front camera off 0.759, three forward cameras off 0.648, night 0.259, fog 0.815, blur 0.433, radar off 0.816, extrinsics 0.695. At night recall falls 0.84 → 0.16 and route F 0.721 → 0.206.")
d.h2("Route transfer with rule-based association (E59, E61, E62)")
d.fig(A + "E59_e59_curves.png", "E59: lane-id association, route F1 and route success against pose noise, with and without graph-ICP registration.", 100)
d.table([
    ["exact pose", "0.926", "-", "-"],
    ["no registration", "0.640", "0.305", "0.153"],
    ["graph-ICP registration", "0.923", "0.883", "0.782"],
    ["tracked registration (E61)", "0.920", "0.920", "0.918"],
    ["per-frame registration (E61)", "0.845", "0.723", "0.608"],
], header=["Route success (TEST) at pose noise σ", "1 m", "2 m", "3 m"], num=(1, 2, 3), hl=(2, 3))
d.p("No map: 0.218. The oracle assignment reaches 0.926, so the 7.4 % misses are extraction errors, not association. Removing route lanes from the map: 0.776 / 0.501 / 0.315. A changed goal: 0.801 vs 0.875 at 1 m. "
    "<b>Planner proxy (E62, 686 frames, 1 m noise):</b> the registered map route has lateral error 0.25 m against 0.60 m for the perceived-lane route, 0.53 m straight ahead and 0.60 m for the map as given; the perceived route is no better than straight (+0.07 [−0.13, +0.26]).")
d.h2("An end-to-end route head and the layers of protection (E60, E64 - E72)")
d.ul([
    "<b>E60 route-conditioned head</b> (U-Net + 4 map channels, route heat-map output): exact map lane F1 0.900 / route success 0.800 (tau 0.5; 0.906 at tau 0.6); 1 m 0.871 / 0.705; no map 0.857 / 0.363; <b>wrong map 0.857 / 0.277 (it trusts its route input)</b>. "
    "E65d (fine-tuned on perturbed features, 2.74 M parameters, ~5 ms): 0.923 / 0.849.",
    "<b>Monitor E64:</b> six output signals + logistic regression, AUROC 0.953 (scene-disjoint 0.935). Mean lateral error: perception route 1.03 m → registered 0.43 → gated 0.40 (oracle 0.31).",
    "<b>Remedies for the failures (E65):</b> real-feature fine-tuning lifts night graph F1 0.259 → 0.451 (0.454 with the final recipe), blur 0.433 → 0.540, three cameras 0.648 → 0.661; extrinsics and radar are not repairable (+0.00); feature augmentation within ±0.016; AdaBN +0.044 night.",
    "<b>Hold registration (E67):</b> lateral error in the failure phase: map as given 1.32 m, per-frame 0.60, tracked 0.45, <b>hold 0.39</b>, oracle gate 0.37, floor 0.26 (draw 0 of five; five-draw means 1.61 / 0.64 / 0.83 / 0.44 / 0.39 for as given / per-frame / dropped / tracked / hold).",
    "<b>Carry the graph forward (E68):</b> F1 0.651 → 0.741, night 0.293 → 0.688, no better ahead of the car from about 7 s. <b>Map lanes under held registration (E69)</b> 0.887, circular because the ground truth is the map.",
    "<b>Jumps (E70 - E72):</b> a restart with a wide search when the window score drops cuts the first-1.5 s error after a 3 / 6 / 10 m jump from 0.58 / 1.54 / 2.84 to 0.25 / 0.25 / 0.55 m. An ungated restart adds +0.36 m [+0.20, +0.54]. All three are in-sample (thresholds fitted on TEST).",
    "<b>City transfer (E58):</b> Boston head 0.878 / 0.777 (Boston / Singapore), Singapore head 0.793 / 0.837; 400 / 1,600 target rows recover 63 % / 90 % of the gap to the specialist. "
    "<b>VLA review:</b> not needed on the evidence reviewed (language adds +0.4 driving score inside a noise of ±0.95).",
])
d.fig(A + "E67_e67_timeline.png", "E67: lateral error over time around a sensor failure for map route as given, per-frame registration, tracked registration, tracked + monitor gate and tracked + oracle gate.", 100)
d.callout("Verdict " + OK("positive"),
          "Association and route flag are rule-based (no trained weights) on top of the learned graph; 0.926 at exact pose is an upper bound (the map is the ground-truth source), 0.923 at 1 m. Hand the planner the registered <b>map</b> route, because perception fails silently. Layered cover: monitor → hold registration → carry the graph forward (about 5 s) → map lanes; restart and verify after jumps. "
          "<b>Negatives:</b> augmentation, an ID vote, Viterbi assignment, E66's planner uses the route as a cue but does not follow a changed one, and verification can replace a correct registration by a wrong one (10 better / 4 worse of 504 runs). Four review passes (the first with two reviewers: 24 + ~15 issues; then ~250 and ~550 claims checked) corrected the 'every choice on VAL' statement: E64 and E67 - E72 are in-sample.")

# ------------------------------------------------------------------ loops 6-7
d.h1("8  Loops 6 - 7: systematic fine-tuning and the simulated SD map (E73 - E80)")
d.h2("Loop 6: a clean protocol for fine-tuning the route head")
d.p("<b>DEV protocol without TEST:</b> the foundation was re-run on every third VAL row for the seven corruptions (291 rows per type); the DEV score is the mean of eight numbers. One factor at a time from E60: data, learning rate, steps, trainable layers, L2-SP, EMA, loss weights, "
    "an auxiliary <i>map-trust</i> head (is the map usable here?), soups and WiSE-FT.")
d.fig(F + "e73_dev_scores.png", "E73: DEV score per run; the band is the baseline mean ±2 SD. Seeds 0.8232 / 0.8230 / 0.8256 (SD 0.0014).", 54)
d.table([
    ["R0 baseline (three seeds)", "0.823", "clean-only control 0.765 (−0.059); first block only 0.814; BN only 0.786; WiSE-FT with E60 0.812"],
    ["trust head weight 1.0", "0.834", "joint map-trust head"],
    ["<b>soup of six trust runs = v2</b>", "<b>0.842</b>", "uniform weight average; chosen on DEV with a per-type guard"],
], header=["E73 (DEV)", "Score", "Note"], num=(1,), hl=(2,))
d.table([
    ["E60", "0.900 / 0.800", "-", "-"],
    ["E65d", "0.923 / 0.849", "0.869 / 0.742", "0.849 / 0.261"],
    ["<b>v2 (E73)</b>", "<b>0.935 / 0.920</b>", "0.868 / 0.791", "0.846 / 0.198"],
    ["geometric transfer (E59)", "- / 0.926", "- / 0.923", "-"],
], header=["TEST lane F1 / route success", "exact map", "1 m", "wrong map"], hl=(2,))
d.p("<b>Registered first (E74):</b> v2 route success 0.915 / 0.896 / 0.842 at 1 / 2 / 3 m, lane F1 0.902 / 0.901 / 0.891 (E65d: route success 0.852 / 0.848 / 0.794, lane F1 0.894 / 0.894 / 0.887; geometric transfer 0.923 / 0.883 / 0.782). "
    "<b>Alternative goal (E75):</b> v2 0.828 against driven 0.887 (geometric 0.805 / 0.887; E65d 0.699 / 0.805). <b>Agreement score (E76)</b> (the head's own map-free lane probability over the map lanes) separates exact from wrong map with AUROC 0.996 (night 0.815, blur 0.885); a learned scalar trust only 0.819. "
    "v2 is worse than E65d at night (0.687 vs 0.729) and trusts a wrong map more.")
d.callout("Protocol note " + TIE("honest"),
          "The first final candidate (a soup of three weight-1.0 seeds, DEV 0.8381) had already been run on TEST when the DEV guard (no type may lose > 0.10 route success; night 0.652 vs 0.802) dropped it. Its TEST numbers stay in the table.", "warn")
d.h2("Loop 7: the simulated SD map")
d.p("An ADASIS-like SD map was derived from the lane graph: one centre line per carriageway with a lane count, the route as the centred driven path, Douglas-Peucker 0.5 - 2 m, lane count wrong with probability 0.1, missing side roads, rigid pose error, a five-channel raster. "
    "<b>Geometric baseline</b> (graph-ICP, follow the ego lane along the successors) gets only <b>0.461</b> route success exact (0.449 / 0.450 / 0.425 registered; 0.555 with a 150 m chain cap): the SD map alone is lane-ambiguous.")
d.fig(F + "e77_sd_sim.png", "E77: how the SD map is derived from the lane map.", 100)
d.table([
    ["exact SD map", "0.884", "0.733"],
    ["registered, 1 / 2 / 3 m", "0.869 / 0.869 / 0.866", "0.714 / 0.713 / 0.710"],
    ["unregistered, 1 / 2 / 3 m", "0.863 / 0.853 / 0.851", "0.697 / 0.614 / 0.541"],
    ["ADASIS-like (2.5 m, 5 % roads missing)", "0.852", "0.574"],
    ["wrong map / no map", "0.851 / 0.856", "0.294 / 0.328"],
], header=["E78 pixel SD head (TEST)", "Lane F1", "Route success"], hl=(0,))
d.p("<b>Temporal (E79):</b> a training-free exponential average of the warped lane probability (coefficient 0.7 chosen on VAL) lifts failure-phase lane F1 from 0.748 to 0.790 (+0.042; consistency 0.863 → 0.956; clean 0.858 → 0.865) but only with good odometry "
    "(1 °/√s + 2 %: 0.753; 3 ° + 5 %: 0.657, harmful). Fusing the route attribute lowers route success (0.530 → 0.401): decide the route per frame. "
    "A learned memory (E80) adds +0.000 - 0.005 and was dropped. Map-free hypotheses at threshold 0.7 have precision 0.990 but recall 0.456.")
d.fig(F + "e79_fusion.png", "E79: failure-phase lane F1 per corruption with the averaging filter; map-free precision / recall against threshold.", 100)
d.callout("Verdict " + TIE("mixed"),
          "The SD map gives road and branch; the lane must come from perception (0.733 vs 0.46 geometric vs 0.328 no map). In our setting a training-free average gave more lane F1 than a learned memory (+0.04 vs +0.00 - 0.005; different base heads and splits), while the memory helped route success slightly. "
          "Corrections from about 85 review findings: the 'low-error hypotheses' table had been computed with the wrong model, frame spacing is 0.5 s / 1 s on TEST and 1.5 s on VAL, and the filter does see the map (it averages the map-conditioned output).")

# ------------------------------------------------------------------ loops 8-9
d.h1("9  Loops 8 - 9: perception first, assignment after (E81 - E90)")
d.fig(F + "p8_arch.png", "The loop 8 / 9 pipeline: map-free dense head → tracer → loose sparse graph cut into ~6 m edges → lane-template localisation of the SD map → relation-biased assignment network → extension with the SD route → graph tracks → planner.", 100)
d.h2("The unit of assignment (E83)")
d.fig(F + "p8_granularity.png", "Oracle route success by label unit on the perceived loose graph (VAL, 273 rows): whole traced piece 0.454, 10 m edges 0.956, 6 m 0.982, 3 m 0.993, vertex 0.993. The perceived graph covers 98.8 % of the route.", 100)
d.p("An assignment network with one token per ~6 m edge (60 features: geometry, dense-head evidence, per-edge SD features, scene token), three transformer layers of width 96 with a relation bias (successor, predecessor, left, right, same piece), "
    "<b>0.27 M parameters</b>, trained on 25,455 out-of-fold and perturbed samples in about 5 minutes, three seeds averaged, outputs keep (true lane) and route per edge.")
d.fig(F + "p8_route_vs_sigma.png", "Route success against map error (left, clean features) and per sensor failure at 2 m (right). Differences below about 0.03 are inside the noise.", 100)
d.table([
    ["assignment network, loop 8", "0.881", "0.879", "0.868", "0.839"],
    ["+ SD-route extension", "0.894", "0.896", "0.881", "0.849"],
    ["<b>loop-9 ensemble (bias + abstention training)</b>", "<b>0.876</b>", "<b>0.879</b>", "<b>0.876</b>", "<b>0.858</b>"],
    ["pixel SD head (E78), registered", "0.733", "0.714", "0.713", "0.710"],
    ["geometric rule on the perceived graph", "0.488", "0.482", "0.483", "0.449"],
    ["oracle labels on 6 m edges (not an upper bound)", "0.913", "-", "-", "-"],
], header=["Route success, TEST (809 rows)", "0 m", "1 m", "2 m", "3 m"], num=(1, 2, 3, 4), hl=(2,))
d.h2("Failure, extension and tracks (loop 8)")
d.table([
    ["night", "0.276", "0.450", "0.263"],
    ["three forward cameras off", "0.330", "0.633", "0.321"],
    ["front camera off", "0.501", "0.701", "0.493"],
    ["blur", "0.430", "0.540", "0.479"],
    ["extrinsics", "0.582", "0.601", "0.538"],
    ["fog", "0.850", "0.873", "0.879"],
    ["radar off", "0.829", "0.839", "0.902"],
], header=["Route success at 2 m (blur, extrinsics, radar off: 409 rows)", "network", "+ extension", "oracle labels"], num=(1, 2, 3))
d.ul([
    "<b>Extension (E84)</b> returns range where perception is short: the SD route shifted by the perceived lateral offset; extension points within 1.5 m of the route 0.731 with offset against 0.598 plain.",
    "<b>Lane-template localisation (E82)</b> lowers the centre-line ICP position error by 28 - 45 % (median position error 0.62 / 0.62 / 0.70 / 0.97 m against 1.11 / 1.13 / 1.16 / 1.34 m) because perceived lanes sit at (k − (n − 1)/2) × 3.5 m from the carriageway centre (±1.75 m for two lanes). It fails at night (2.45 m at 2 m) and the position gain does not carry over to route success (+0.014 / +0.016 / +0.007 / −0.005).",
    "<b>Graph tracks (E85):</b> failure-phase lane F1 0.645 → 0.755 (night 0.283 → 0.634, blur 0.465 → 0.728); at 3 °/√s + 5 % the failure-phase gain is gone (0.659 against 0.645) and the clean phase loses (0.760 against 0.843). Averaging the route attribute lowers route success 0.570 → 0.400.",
    "<b>Reference path (E86):</b> SD route + perceived offset 0.308 m at 20 m on 486 rows (SD centre line alone 0.949 m, HD lane 0.261 m; night 0.732 vs 1.445 m). A learned consumer: geometry only 0.88 m, with route probability 0.41 m, wrong map 1.09 m, no map 1.81 m. "
    "<b>Joint training (E86c) gave no clear gain</b> and lowered route success 0.868 → 0.769.",
    "<b>Lane F1 under failure of the graph pipeline is below the pixel head</b> (night 0.255 vs 0.507, blur 0.431 vs 0.581): the price of precision-first sparsification.",
])
d.callout("A protocol bug that changed numbers " + NEG("fixed"),
          "Rows with an <b>empty perceived graph were silently dropped</b> (night 140 of 809 rows, blur 9 of 409). All TEST samples were rebuilt and every dependent evaluation rerun. Night route success 0.333 → 0.276 (with extension 0.504 → 0.450), night lane F1 0.304 → 0.255, "
          "E85 failure phase 0.653 → 0.759 became 0.645 → 0.755. Clean cells are unaffected. (Separately, the E86 reference-path deviation 0.43 → 0.308 m and SD centre line 1.61 → 0.949 m changed because the all-rows values were inflated by rows whose driven trajectory is shorter than the evaluated distance; use the 486 comparable rows.) The loop-8 DEV sets still exclude empty graphs. <b>Do not quote any first-version loop 8 failure number.</b>", "neg")
d.h2("Loop 9: the biggest issues of loop 8")
d.fig(F + "p9_results.png", "Left: route success against per-road lateral map bias (clean features, map error 1 m): loop 8 ensemble, loop 9 ensemble (bias augmentation + abstention) and + per-road re-registration. Right: route-flagged length per frame with a wrong or missing map and under failure.", 100)
d.table([
    ["per-road bias σ 1 m", "0.848", "0.871"],
    ["per-road bias σ 2 m", "0.663", "0.847"],
    ["harder map + bias 1 m", "0.790", "0.847"],
    ["ADASIS-like map", "0.857", "0.868"],
    ["route-flagged length, wrong map (m)", "60.7", "4.2"],
    ["route-flagged length, no map (m)", "60.1", "0.0"],
    ["route-flagged length, right map (m)", "63.3", "63.8"],
], header=["TEST", "loop 8 ensemble", "loop 9 ensemble"], num=(1, 2), hl=(1, 4, 5))
d.p("<b>Retraining against map faults (E87)</b> was worth more than correcting them afterwards (+0.18 against +0.04 - 0.10): on top of the retrained network the per-road re-registration (E90) adds only +0.011 / +0.007 / −0.012 for the new networks (+0.04 - 0.10 for networks trained without bias). "
    "Abstention works on clean features but not under failure (a wrong map still gets 4.3 - 25.2 m of route; fog 7.0, night 23.3, blur 25.2 m; the label-free gate has AUROC 0.618 at night). "
    "<b>Held correction (E88)</b> helps only with trustworthy odometry: 0.679 vs 0.589 with exact poses, but 0.497 vs 0.590 at 3 °/√s + 5 %. Extension points do not close the lane-F1 price (night 0.252 → 0.282; loop 9 network at 1 m vertex spacing, the loop 8 value 0.255 is at 0.5 m). "
    "A consumer trained also on wrong and no-map samples reaches 0.95 / 0.96 m (loop 8: 1.01 / 1.12; geometry only 0.79 / 0.80).")
d.callout("Corrections of loop 9 " + TIE("honest"),
          "The 'true bias correction' is not an upper bound under drift, and the independent per-frame error test is meaningless (numbers identical to 0.001 by construction). The final E87 recipe was run on TEST twice (the first version had 0.815 at bias 2 m). 'Nine recipes' are nine networks.", "warn")

# ------------------------------------------------------------------ best system
d.h1("10  The best system at the end of week 1")
d.p("<b>Pipeline:</b> frozen GaussianCaR → map-free dense head (e37_final) → tracer → loose graph cut into ~6 m edges → lane-template localisation of the SD map → three-seed ensemble of the 0.27 M assignment network (loop-9 recipe, keep / route per edge) → SD-route extension → graph tracks → planner hand-over "
    "(output graph with attributes, reference path V(s) + o(s), map-agreement score). Cross-check branches: the geometric chain (registered map route, monitors, hold / restart / verify), the end-to-end lane-map head v2 (0.935 / 0.920), the pixel SD head (0.733) and the temporal average.")
d.table([
    ["Lane F1@1 m of the kept edges, no map", "0.857 (TEST, clean, 2 m); the grid chain of loops 3 - 4 reaches 0.876 (VAL 0.899)", "E83, E43"],
    ["Split / merge F1 @6 m (grid chain of loops 3 - 4)", "0.552 / 0.607 (key points); 0.714 / 0.637 (connectors); not measured for the final pipeline", "E44, E53"],
    ["Route success at map error 0 / 1 / 2 / 3 m", "0.876 / 0.879 / 0.876 / 0.858 (loop-9 ensemble); with extension 0.894 / 0.896 / 0.881 / 0.849 (loop 8)", "E83, E87"],
    ["Route success, per-road bias 2 m", "0.847", "E87"],
    ["Abstention (flagged route length)", "wrong map 4.2 m, no map 0 m, right map 63.8 m", "E87"],
    ["Robustness at 2 m (night / blur / 3 cameras / front camera off), with extension", "0.449 / 0.535 / 0.645 / 0.681; with the robust front end 0.575 / 0.689 / 0.634 / 0.717", "E87, E89"],
    ["Graph F1 under failure, rule system after E65f (night / blur / 3 cameras)", "0.259 → 0.454 / 0.433 → 0.558 / 0.648 → 0.685", "E57, E65"],
    ["Reference path at 20 m", "0.308 m (SD centre 0.949, HD lane 0.261); consumer 0.41 m", "E86"],
    ["Latency", "chain ≤ 41 ms without foundation (loop 3 / 4); loop 8 graph pipeline 0.27 s on one CPU thread + ~5 ms dense head; GaussianCaR ~0.4 s", "E43, loop 8"],
    ["Parameters", "dense head 2.7 M, refiner 0.76 M, key-point head 2.7 M, assignment network 0.27 M", ""],
], header=["Task", "Value (TEST unless noted)", "Source"], idcol=0, widths=["30%", None, "10%"])
d.fig(F + "p8_qualitative.png", "The loop 8 pipeline on TEST rows 10, 497 and 959 under clean, junction, fog, night and other failure conditions.", 82)

# ------------------------------------------------------------------ negatives
d.h1("11  What did not work")
d.table([
    ["1", "CNT-Net (transformer decoder), field-level temporal fusion, ensembles, flips, coordinate channels, distance-dependent threshold", "No gain over the U-Net / single field"],
    ["2", "Sparse DETR-style LaneFormer (E32)", "−0.058 F1, −0.178 route F against the grid; under-fitted"],
    ["3", "Learned link head (E42, E45), temporal B (E41), foundation read-out variants (E49 - E52)", "Ties rules / neutral / no gain"],
    ["3 - 4", "Regularisation, mirroring, ensembling against the seen-to-unseen shift", "Lowers VAL as well; the shift stays"],
    ["5", "Feature augmentation, extrinsics and radar repair, ID vote, Viterbi assignment, planner for a changed route (E66)", "No effect / not repairable / planner uses the route as a cue but does not follow a changed one"],
    ["5", "Hold with wrong verification", "10 better but 4 worse of 504 runs"],
    ["6", "WiSE-FT, BN only, more wrong-map training, v2 at night", "Worse than the baseline; v2 worse than E65d at night and more trusting"],
    ["7", "Learned memory (E80); fusing the route attribute", "+0.000 - 0.005; route success 0.530 → 0.401"],
    ["8", "Relation bias, network changes in the DEV study (23 runs); localisation gain at night; joint training E86c", "No gain; fails at night; no clear gain"],
    ["9", "Per-road re-registration (E90); extension points for lane F1 (E89); held correction at strong drift", "Noise; +0.01; hurts"],
], header=["Loop", "What", "Outcome"], idcol=0)

# ------------------------------------------------------------------ corrections
d.h1("12  Numbers and claims that were corrected - do not repeat them")
d.table([
    ["Loop 8 failure numbers of the first version", "Night route success 0.333 / 0.504, night lane F1 0.304, E85 0.653 → 0.759: empty-graph rows were dropped. Use 0.276 / 0.450, 0.255, 0.645 → 0.755."],
    ["E86 reference path 0.43 m, SD centre line 1.61 m", "The all-rows values were inflated by rows whose driven trajectory is shorter than the evaluated distance. Use 0.308 / 0.949 m on the 486 comparable rows."],
    ["'Strict graph −85 %, two thirds of length, 90 % availability'", "Learned keep ≥ 0.9: −95 %, 53 % of length, availability 0.675. The −85 % belongs to the probability filter."],
    ["'Every choice made on VAL' (loop 5)", "E64 regression is fitted on even TEST frames; E67 score gate, E69 validity rule and E70 - E72 thresholds are TEST quantiles: in-sample."],
    ["'TEST used once'", "Not true: loop 6 first candidate, loop 7 trust-head variant and loop 9 first recipe (0.815 vs 0.847) were run on TEST before being revised."],
    ["E75 alternative-route pair 0.805 / 0.887", "Belongs to the geometric transfer. The head's value is 0.828 / 0.887."],
    ["E79 'the filter never sees the map'", "False: it averages the map-conditioned output. E79 gain is +0.042 on TEST with the final head (+0.036 with the earlier head, +0.059 on VAL)."],
    ["E88 'true correction is an upper bound'", "Not under drift. The independent per-frame error test is meaningless."],
    ["E67 numbers", "Draw 0 of five (1.32 / 0.60 / 0.68 / 0.45 / 0.39); five-draw means 1.61 / 0.64 / 0.83 / 0.44 / 0.39."],
    ["E30 chain 0.932 / 0.899 / 0.851 / 0.820", "Fold heads, not E37 (0.885 / 0.856). Do not mix."],
    ["Scene count 323", "255 scenes (212 / 22 / 21). Mini '323 train / 81 val samples' is a different thing."],
    ["'Route attribute halves the consumer error' (0.88 → 0.41 m)", "The E86 consumers have no ego motion input; later loops show ego motion matters (week 2)."],
    ["Route-head seed noise", "Week 1 networks used three seeds or DEV SD 0.001 - 0.0014, but single TEST route heads (E60 / E65d / v2) are single runs; later work found a route-head seed spread of 0.05 - 0.09."],
], header=["Claim", "Correction"], idcol=0, widths=["32%", None])

# ------------------------------------------------------------------ open issues
d.h1("13  Open issues at the end of week 1", new_page=False)
d.ul([
    "<b>Everything adverse and the SD map are synthetic;</b> the SD route equals the driven route. A real navigation map and a measured per-road bias need an OpenStreetMap download (not done; needs the user's decision).",
    "<b>Perception under sensor loss:</b> the graph pipeline's lane F1 is below the pixel head at night and in blur; the foundation is frozen; no real night or rain data.",
    "<b>Abstention under failure is weak</b> (4 - 25 m of route with a wrong map; gate AUROC 0.618 at night).",
    "<b>Localisation under failure:</b> single-frame registration fails at night; a held correction helps only with trustworthy odometry.",
    "<b>Not differentiable:</b> tracer, grid-search registration and tracks. Joint training with a planner loss gave no clear gain, so 'end to end' means that the output graph is a differentiable interface. This led to week 2.",
    "<b>Evaluation:</b> 21 TEST scenes, mostly single seeds, intervals ±0.03; thresholds of loop 5 fitted on TEST; open loop only.",
    "Lane count and index are weak beyond one lane (SD lane count right for 94 / 64 / 15 % of edges in 1 / 2 / 3-lane stacks); latency 0.27 s is unoptimised.",
])

# ------------------------------------------------------------------ appendix
d.h1("Appendix  Experiment index, loops 1 - 9", new_page=True)
rows = [
    ["E15 - E26", "1", "CRN features: dense decoders, threshold, ensembles, tracking, tracer", "U-Net 0.613 → 0.670 F1@1 m (1,706-sample validation set); field is the limit", "tie"],
    ["E27 - E29", "2", "CRN vs GaussianCaR, feature level, loss weighting", "+0.243 VAL, +0.237 TEST; PCA-32 decoder level", "ok"],
    ["E30", "3", "Seen-to-unseen decomposition and training controls", "0.932 → 0.899 → 0.851 → 0.820", "info"],
    ["E31, E33", "2", "Tracer against skeleton; tracer step", "0.879 / 0.851 vs 0.854 / 0.821", "ok"],
    ["E32", "2", "Sparse-only LaneFormer", "0.821 / 0.791; route F 0.587 / 0.519", "neg"],
    ["E37 - E39", "3", "Final dense head, grid ceilings, cascade refiner", "0.885 / 0.856; coverage 87.5 / 94.1 / 98.2 %; refiner +0.001", "tie"],
    ["E40", "3", "Map prior (alignment, leakage, noise, outdated)", "exact pose 0.962 / 0.943; unaligned 1 m below no map", "tie"],
    ["E41, E42", "3", "Temporal A / B; learned link head", "A: graph 0.899 / 0.876; links tie", "tie"],
    ["E43", "3", "Final system", "TEST F1 0.876, route F 0.744, ≤ 41 ms", "ok"],
    ["E44, E53, E56", "4", "Key points, Bezier connectors, direction criterion", "split / merge 0.55 / 0.61; 0.71 / 0.64", "ok"],
    ["E45 - E52", "4", "Link scoring, path view, learned registration, read-out variants", "registration 0.54 → 0.43 m; others no gain", "tie"],
    ["E54, E55", "4", "Paired scene bootstrap; seed noise", "dense head SD 0.0025 / 0.0006", "info"],
    ["E57", "5", "Sensor perturbations", "night 0.259, blur 0.433, 3 cameras 0.648", "info"],
    ["E58", "5", "City transfer", "63 % / 90 % of the gap recovered with 400 / 1,600 rows", "info"],
    ["E59, E61, E62", "5", "Geometric route transfer; tracked registration; planner proxy", "0.923 at 1 m; lateral 0.25 m", "ok"],
    ["E60, E65", "5", "Route-conditioned head; failure remedies", "0.800 → 0.849; night F1 0.259 → 0.454", "tie"],
    ["E64, E66", "5", "Monitors; small planner module", "AUROC 0.953; planner does not follow a changed route", "tie"],
    ["E67 - E72", "5", "Hold, carry-forward, jumps, restart, verify", "hold 0.39 m; restart 0.58 → 0.25 m (in-sample)", "ok"],
    ["E73 - E76", "6", "Systematic fine-tuning, registration front end, alternative goal, trust", "v2 0.935 / 0.920; agreement AUROC 0.996", "ok"],
    ["E77 - E80", "7", "SD-map simulator, SD head, temporal average, learned memory", "0.733 (geometric 0.461); average +0.042", "tie"],
    ["E81 - E86", "8", "Sparsification, localisation, assignment, extension, tracks, reference path", "route 0.881 / 0.879 / 0.868 / 0.839; 0.308 m", "ok"],
    ["E87 - E90", "9", "Bias + abstention training, held correction, lane-F1 price, re-registration", "bias 2 m 0.663 → 0.847; wrong map 60.7 → 4.2 m", "ok"],
]
d.table([[a, b, c, e, T({"ok": "ok", "neg": "neg", "tie": "tie", "info": "info"}[t], {"ok": "positive", "neg": "negative", "tie": "mixed", "info": "reference"}[t])] for a, b, c, e, t in rows],
        header=["Id", "Loop", "What", "Result", ""], idcol=0, widths=["12%", "6%", "36%", "38%", "8%"])
d.p("<span class='small'>Sources: experiment journal (STATE.md), the loop 9 condensed report and the per-experiment files E27 - E72, and the reviewer findings of loops 1 - 9 (loops 1 - 3: 36 issues; loop 4: 27; loop 5: 30 + three further passes; loop 7: ~85; loop 8: 43 + 63 and 41 + 22 + 9; loop 9: 17). "
    "Numbers were taken from STATE.md, the loop 9 condensed report and the per-experiment files via a working digest (<code>reports/source/weekly/digest_week1.md</code>, not a primary source) and then reviewed against the primary sources (<code>review_week1.md</code>).</span>")

d.save(OUT)
