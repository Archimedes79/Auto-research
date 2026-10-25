"""Week 2 report (loops 10-17, 2026-10-18 .. 2026-10-24): HTML builder.  Run in gcar:  tools/env/_py14.sh build_week2_report.py
Numbers come from reports/source/weekly/digest_week2.md (every number there carries its source); keep both in step.
Output: reports/weekly/Week2_Loops10-17_Report_EN.html  (PDF: reports/build/print_weekly.ps1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

F = "reports/source/figures/"
OUT = W.ROOT + "/reports/weekly/Week2_Loops10-17_Report_EN.html"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))

d = W.Doc("From a learned lane-graph chain to a fully differentiable builder - and what happens when sensors fail",
          "Week 2 of the lane-centre-line graph project, loops 10 to 17: what was built, what worked, what did not, and which numbers can be trusted.",
          "Week 2 · Loops 10 - 17 · 17 - 24 October 2026",
          "Camera + radar BEV features (nuScenes) → sparse lane graph → map assignment → planner. Numbers are validation (VAL) unless marked TEST.")

# ------------------------------------------------------------------ contents + at a glance
d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")
d.h1("1  At a glance", new_page=False)
d.p("The task is to turn camera and radar features into a <b>sparse graph of lane centre lines</b>, attach a <b>navigation attribute</b> "
    "(which lanes belong to the route in a simulated SD map) and hand both to a planner. Week 1 ended with a modular pipeline (rules plus small networks) "
    "that was accurate but slow and not trainable end to end. In week 2 we replaced its parts by learned, then by <b>fully differentiable</b> modules, "
    "made the chain fast, and finally asked the question that had been open all week: <b>does it still work when a sensor fails?</b>")
d.stats([
    ("31 ms", "chain per frame after the cached foundation features", "builder, assembly, route head, consumer (foundation network, dense head and map rasterisation not timed); control chain needs 628 ms (idle RTX 3060, batch 1, cached foundation)"),
    ("0.746", "composite score S, VAL", "control 0.724. The edge comes from 38 % fewer vertices (13.5 vs 21.8 per 100 m), not from accuracy"),
    ("0.889", "route success, VAL", "head asked along the edges; vertex-based head 0.798, control 0.885 (two seeds)"),
    ("0.42 → 0.62", "S at night, failure half", "single frame → spatial hold (champion). Clean half of the same scenes: 0.750"),
    ("4.1 → 1.8 m", "planner path error at 20 m, night", "single frame → learned memory gate; with an exact map in the builder 0.28 m"),
    ("0.05 - 0.09", "route-head seed spread, failure cells", "up to 0.03 on clean frames; the failure-series cells of loop 17 showed 0.05 - 0.09. Gate, joint fine-tune and hold on B sit inside it"),
])
d.h2("What we learned this week")
d.ul([
    "<b>A learned trace network and a cross-attention route head work as a registration-free chain, but they do not beat the rules</b> "
    "(F1@1 −0.008 VAL / −0.017 TEST against the loose rule graph, −0.016 / −0.025 against the final rule graph; the gap is not seed noise).",
    "<b>The memory that works sits at the input.</b> A parameter-free spatial hold (ego-warped, decayed previous input, per pixel the stronger heat) "
    "lifts night F1 from 0.337 to 0.680 (TEST, loop 12) and recovers 60 % / 52 % / 13 % of the S loss at night / blur / three front cameras off (VAL, loop 17). "
    "The builder's own time recursion adds nothing.",
    "<b>The fully differentiable builder (loop 16) is sparse, fast and clearly worse where it counts for topology.</b> Junction score J 0.265 vs 0.405, "
    "lanes found 0.70 vs 0.80. The route-success gap turned out to be a representation error (one value per vertex of a long edge), closed on VAL by asking the head along the edges (no TEST number).",
    "<b>Under sensor failure the new builder loses as much as the old one.</b> Two things recover part of the loss: the hold with a small learned gate (+0.021 / +0.015 / +0.001 S at night / blur / three cameras off), "
    "and a builder trained on perturbed rows that also reads the SD map. The map benefit is real only if the registration is better than about 1 m; "
    "a wrong map is worse than no map.",
    "<b>The planner loss reaches the route head and the vertex positions and adds no measurable gain there.</b> Through the builder only the gradient reach was verified (no planner loss trained; the route loss of E202 was inconclusive). The label and position losses do the work.",
])
d.callout("How far is this from 'end to end'?",
          "The champion is the closest candidate, not a completely end-to-end system: the foundation network is frozen, the dense head is trained separately and cached, "
          "assembly, fork relocation and the validity threshold are rules at inference, route head and consumer are trained separately "
          "(the one joint run was inconclusive), and all results use exact ego poses and a simulated SD map.", "warn")

# ------------------------------------------------------------------ setup
d.h1("2  Task, data and how to read the numbers")
d.p("<b>Output.</b> Around the car (±50 m) a directed graph of lane centre lines with few vertices; per edge a probability of belonging to the route. "
    "From loop 15 on the goal is that every module on the training path is differentiable: no argmax, no matching, no rule step.")
d.p("<b>Data.</b> A nuScenes subset of 255 scenes (Boston and Singapore, day, dry), split by scene: TRAIN 212 scenes / 8,414 frames, VAL 22 / 873, TEST 21 / 833 "
    "(809 with a route label). The split is not location-disjoint. <b>Foundation:</b> a frozen, map-pretrained GaussianCaR; cached features of 200 × 200 cells of 0.5 m "
    "(32 PCA channels + 7 map logits + heat and direction of a dense head = 42 channels).")
d.fig(F + "dec_arch.png", "The two chains of the project. A: modular perception-first pipeline (most accurate). B: learned, differentiable chain (the end-to-end candidate). "
      "Both hand a graph with route attribute, trust signals and ego motion to the planner.", 100)
d.h3("Metrics")
d.table([
    ["F1@1 (F1@0.5)", "Point-wise F1 of the predicted lane points against the ground truth within 1 m (0.5 m)."],
    ["J", "Junction score: mean of split-F1 and merge-F1 at 6 m."],
    ["Vertices per 100 m", "Sparsity. A Douglas-Peucker oracle needs 14.3 at 0.3 m tolerance."],
    ["False length (FL), lanes found", "Share of predicted length that is not on a lane; share of ground-truth lane length that is matched."],
    ["S (composite, loop 16 on)", "0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 − FL), Sp = min(1, 14.3 / vertices per 100 m). "
     "<b>S rewards sparsity</b>; the noise band is 0.010; there is no confidence interval for S."],
    ["Route success", "Per frame, route points are covered and precisely covered (1.5 m) by the flagged edges, both ≥ 0.8. Map error cells 0 / 1 / 2 / 3 m, wrong map, no map."],
    ["Consumer path error", "Error at 20 m arc length of the path a small planner-side network predicts from graph, route probability and ego motion (93 k parameters)."],
], header=["Metric", "Meaning"], widths=["24%", None])
d.callout("Four rules for reading any number in this report",
          "<ol style='margin:0;padding-left:5mm'>"
          "<li><b>VAL or TEST.</b> TEST (21 scenes) was used once for the loop-16 champion and its control. In loops 10 - 15 TEST numbers were the main results and some variants were compared on TEST only; loops 16b and 17 are VAL only. Never compare a VAL with a TEST number.</li>"
          "<li><b>Noise.</b> Trace network seed ±0.002 F1; builder band 0.010 S; <b>route-head seed spread up to 0.03 on clean frames, 0.05 - 0.09 in the failure-series cells of loop 17</b>; with 21 scenes a success rate has an interval of about ±0.03.</li>"
          "<li><b>'Exact map' is an upper bound.</b> The simulated SD map is derived from the same lane map as the ground truth. The map alone scores S 0.693 (loop 17).</li>"
          "<li><b>Failures are synthetic.</b> Night, blur, lost cameras etc. are perturbations of the foundation inputs; no recorded failure was available.</li></ol>", "")

# ------------------------------------------------------------------ story
d.h1("3  The week as a story")
d.table([
    ["10 - 11", "17 - 18 Oct", "Can a learned network replace tracer and assembler, and a cross-attention head replace registration? Does a memory protect it under failure?",
     "Works, but F1@1 −0.008 VAL / −0.017 TEST (loose rule graph) and route success 0.007 - 0.026 below the single rule-based network (0.023 - 0.044 below the ensemble). A training-free memory brings it to about the robustness of the rule chain's own coasting.", TIE("mixed")],
    ["12", "18 Oct", "Are the planner-facing probabilities calibrated? Does a hold need a failure monitor?",
     "Route probability of the modular system calibrated (ECE ≤ 0.016; the learned chain is not: ECE 0.021, +28 % length); route confidence ranks frames (AUROC 0.88 - 0.91) except for lost front cameras. A per-pixel spatial hold needs no monitor: night F1 0.337 → 0.680.", OK("positive")],
    ["13", "19 Oct", "Are junction errors a missing-link problem? Can small verifiers or a learned consumer with trust inputs help?",
     "Junctions are a precision problem (1.8 × as many as real). Verifiers hurt under failure unless gated. Ego motion helps the consumer; trust inputs and a learned gate do not.", TIE("mixed")],
    ["14", "19 - 20 Oct", "Is the learned chain a usable planner interface? Does the planner loss train the navigation module?",
     "B equals A on the path-error metric (0.34 vs 0.36 m). Planner loss adds nothing; label loss trains the head.", OK("positive")],
    ["15", "22 - 23 Oct", "Can vertex positions be chosen end to end (MapTR-style refinement)?",
     "Yes: vertex error 0.54 → 0.43 m, F1@1 0.851 → 0.863. The planner loss reaches the positions and gains nothing. A seed check shows earlier route 'gains' were noise.", TIE("mixed")],
    ["16 / 16b", "23 - 24 Oct", "Which fully differentiable builder is best, and can it be fast?",
     "Champion D7: S 0.746 vs 0.724, 38 % fewer vertices, 31 ms vs 628 ms. Fork score and lane recall stay behind. Route gap closed on VAL by asking the head along the edges.", OK("positive")],
    ["17", "24 Oct", "How does the champion behave under sensor failure, with memory, a learned gate and a map?",
     "It loses as much as the control. Hold and gate recover part; a builder trained on perturbed rows plus a good map recovers more. Unseen failure types: the run crashed, no result.", TIE("mixed")],
], header=["Loop", "Dates", "Question", "Answer", ""], widths=["6%", "9%", "30%", "49%", "6%"], idcol=0)
d.p("<span class='small'>Verdict tags: green = answer is positive and supported; amber = mixed or inside the noise. The loop numbers 10 and 11 started on the evening of 17 October (end of week 1).</span>")

# ------------------------------------------------------------------ loops 10-11
d.h1("4  Loops 10 - 11: a learned trace network, a registration-free route head, a memory")
d.p("<b>Question.</b> The rule pipeline (tracer, assembler, key-point cut, Bezier connectors, registration) is accurate but made of hand-built blocks. "
    "Can one learned network replace it, and can the navigation attribute be produced inside the network without a registration step?")
d.h2("What was built")
d.fig(F + "p10_e94_arch.png", "E94, the learned trace network (1.67 M parameters). It grows the graph from the near field outwards: per active head a fan of 27 sampled points, "
      "the last six vertices and a crop of the already emitted graph feed two transformer blocks that predict heading bin, step class (2 / 4 / 6 m), offset, on-lane logit and an event (continue, end, fork, merge, border). "
      "Imitation training on ground-truth states, 10,000 steps.", 92)
d.ul([
    "<b>Trace network E94</b> with 192 seeds (heat peaks, non-maximum suppression 5 cells). The first version with 48 seeds lost lanes; with oracle seeds it beat the rules (0.921 vs 0.908), so the seed head, not the tracer, was the limit.",
    "<b>Route head E104b</b>: cross-attention from the 128-d vertex tokens to SD-map tokens (2,500 tokens at 2 m cells) with gated input fusion (0.69 M parameters). No registration step.",
    "<b>Memory (loop 11)</b>: the previous 42-channel input warped by the exact ego motion, mixed in with EMA weight 0.5 / 0.7, optionally replaced by the memory when a label-free monitor flags the frame.",
])
d.h2("Results")
d.table([
    ["Rule pipeline, loose graph", "0.891", "0.867", "0.286 / 0.390", "0.782", "reference"],
    ["Rule pipeline, final graph (key points + connectors)", "0.899", "0.875", "0.689 / 0.648", "-", "most accurate"],
    ["E94 learned trace network", "0.883", "0.851", "0.357 / 0.459", "0.744", "paired TEST −0.017 [−0.025, −0.009] vs loose"],
    ["E94 seed 1 (E111)", "0.884", "0.850", "-", "-", "seed noise ±0.002: the gap is real"],
], header=["System", "F1@1 VAL", "F1@1 TEST", "Split / merge F1 @6 m (VAL)", "Route F (VAL)", "Note"], num=(1, 2, 3, 4), hl=(2,))
d.table([
    ["Rule graph + assignment network E87 (single)", "0.860", "0.864", "0.858", "0.839"],
    ["Rule graph, ensemble of three", "0.876", "0.879", "0.876", "0.858"],
    ["Learned graph + E87, zero-shot (E102)", "0.815", "0.816", "0.811", "0.797"],
    ["E104b route head, threshold 0.6 (loop 10)", "0.813", "0.815", "0.795", "0.792"],
    ["E104b route head, threshold 0.5 chosen on VAL (loop 11)", "0.853", "0.843", "0.832", "0.820"],
], header=["Route success, TEST, by map error", "0 m", "1 m", "2 m", "3 m"], num=(1, 2, 3, 4), hl=(4,))
d.fig(F + "p10_summary.png", "Learned extractor against the rule pipeline: F1 by range (far band 40 - 50 m, TEST: 0.774 vs 0.800) and route success by map error.", 100)
d.fig(F + "p10_route_variants.png", "Route heads of loop 10: a head on a single step token learns the position prior (same flag for right and wrong map); a graph transformer and a pose / correlation-volume head give no route gain; cross-attention (E104b) is the best.", 100)
d.h2("Loop 11: a memory in front of the extractor")
d.table([
    ["night", "0.337", "0.428", "0.728", "0.293 → 0.69 coasting"],
    ["blur", "0.524", "0.640", "0.699", "0.474"],
    ["radar off", "0.813", "0.849", "0.838", "0.829"],
    ["three front cameras off", "0.653", "0.675", "0.666", "0.666"],
], header=["Failure (TEST, 10 s phase)", "Single frame", "EMA 0.7", "EMA 0.5 + hold", "Rule pipeline (single → own coasting)"], num=(1, 2, 3))
d.p("The memory lifts the learned chain at night and in blur, and the hold does nothing for radar loss or lost cameras (the plain EMA 0.7 still adds +0.036 / +0.022; the heat monitor does not fire there: 80 % of night frames flagged, 0.5 % with cameras off). "
    "Whole-chain route success at night rises from 0.093 to 0.346 (clean phase 0.847). The clean phase is not free: route success −0.024 [−0.050, −0.002] with EMA 0.7.")
d.callout("Verdict " + OK("positive") + TIE("mixed"),
          "The learned chain is the end-to-end candidate; the modular pipeline stays the accuracy reference. The memory makes the learned chain "
          "<b>about as robust</b> as the rule chain with its own coasting, not more robust (an earlier draft said 'more robust'; a reviewer corrected it).")
d.callout("Not adopted",
          "Wrong-direction training (E94b: direction 0.916 vs 0.856, but F1 −0.028 and lanes found 0.75 vs 0.84); covering-crop knock-out (E97) costs junction F1; thresholds 0.3 - 0.7 change nothing (E98); "
          "a memory of the route attribute (E106) does not help; the first route threshold (0.6) was replaced by the VAL-selected 0.5, which closed most of the 'price of removing the registration'.", "neg")

# ------------------------------------------------------------------ loops 12-14
d.h1("5  Loops 12 - 14: trust signals, spatial hold, verifiers, a planner-side consumer")
d.h2("Loop 12: calibration and the spatial hold")
d.ul([
    "<b>Calibration (E109).</b> The route probability of the modular chain is calibrated: ECE 0.006 [0.004, 0.008] clean and ≤ 0.016 under seven failure types at 2 m map error; with a wrong map it still flags 8.3 m of route per frame where the truth is 0. "
    "The learned chain's route probability (ECE 0.021) ranks but is not a probability (+28 % route length).",
    "<b>Route confidence (E110, modular system)</b> separates correct from wrong frames with AUROC 0.88 - 0.91 for clean, fog, radar off, blur and night, but only 0.69 - 0.71 for lost front cameras and extrinsics errors.",
    "<b>Better monitors (E108).</b> With an oracle failure flag the gain is +0.072 [+0.032, +0.119] in blur, +0.020 (not significant) at night; with a gradual onset the heat monitor flags only 35 % (night) / 17 % (blur) of failure frames.",
    "<b>Spatial hold (E112)</b> needs no monitor and no parameter: per pixel, all 42 channels come from the decayed ego-warped memory wherever its heat is higher than the current heat.",
])
d.fig(F + "p12_e112.png", "E112 spatial hold against single frame and whole-frame holds (failure-phase F1 and clean-phase cost). Night 0.337 → 0.680, blur 0.524 → 0.725; radar off flat; three front cameras off 0.653 → 0.600 (precision 0.60 → 0.47).", 78)
d.h2("Loop 13: junction topology and verifiers")
d.p("The learned graph predicts <b>1.8 times as many junctions as exist</b> (10,807 for 6,152); one-to-one precision is 0.30 / 0.34 for splits / merges against 0.85 / 0.72 for the final rule graph. "
    "A pairwise connector head did not help (it lowers junction F1 at every threshold). On TEST, a junction verifier lifts split / merge F1 from 0.385 / 0.428 to 0.437 / 0.504 (final rule graph 0.714 / 0.637) but lowers route F in nine of nine failure cells; "
    "a piece verifier raises clean F1 from 0.851 to 0.860 and hurts in five of six night and blur cells unless gated or trained on failure frames.")
d.fig(F + "p13_e114.png", "E114: the learned chain under failure with the spatial hold. Route success at night 0.093 → 0.444, blur 0.228 → 0.554, three front cameras off 0.290 → 0.242 (no memory helps there).", 100)
d.h2("Loop 14: the planner-side consumer")
d.p("A small consumer (input MLP + two-layer transformer, 93,314 parameters, three seeds) reads the 320 nearest 6 m edges of the traced graph, the route probability and the ego motion and predicts the path point at 20 m. "
    "On 546 TEST frames the learned chain reaches the level of the modular chain:")
d.table([
    ["geometry only", "0.90", "0.90", "0.90", "0.90"],
    ["+ route probability", "0.40", "0.40", "0.98", "0.98"],
    ["+ route + ego motion (B)", "0.34", "0.34", "0.76", "0.74"],
    ["modular chain A, same inputs", "0.36", "0.36", "0.70", "0.73"],
    ["+ soft verifier probabilities", "0.34", "0.35", "0.72", "0.70"],
    ["+ 128-d step token", "0.56", "0.55", "0.80", "0.78"],
], header=["Consumer inputs (TEST, error at 20 m, m)", "exact map", "2 m error", "wrong map", "no map"], num=(1, 2, 3, 4), hl=(2,), neg=(5,))
d.fig(F + "p14_e118.png", "E118, consumer under failure: single frame / monitored hold / spatial hold, clean-trained and failure-trained. Night 2.73 → 0.72 m (clean-trained, spatial hold) and 1.00 → 0.44 m (failure-trained). "
      "The hard cut of the piece verifier is 1.4 - 3.8 times worse in all nine cells.", 100)
d.callout("Verdict " + OK("positive"),
          "Hand the planner edges + route probability + ego motion, put the spatial hold at the input and train the consumer on failure frames. Do not hand over step tokens, and do not hard-cut the graph without a failure gate. "
          "<b>The planner loss reaches the route head and the consumer (AUROC 0.995 - 0.996) and adds nothing</b>; a route head trained from random weights on the planner loss alone does not learn the attribute (AUROC 0.53 - 0.55, one run per arm).")

# ------------------------------------------------------------------ loop 15
d.h1("6  Loop 15: are the vertex positions chosen end to end?")
d.p("A question after reading the condensed report: the trace network picks heading and step by argmax in a teacher-forced loop, so its positions are <b>not</b> end to end. "
    "A MapTR-style refinement decoder (1.37 M parameters, three layers, point-to-polyline loss, no matching) now makes the positions a differentiable output.")
d.fig(F + "p15_arch.png", "Traced graph → refinement decoder → planner. Each layer samples the feature pyramid at 21 points around every vertex, applies distance-biased self-attention and predicts a displacement (≤ 1.5 m along, ≤ 2.5 m across) and a keep logit.", 100)
d.table([
    ["Traced graph (E94)", "0.661", "0.851", "0.54 / 0.36", "63 %"],
    ["Refined (E120a, 3 layers)", "0.694", "0.863", "0.43 / 0.28", "73 %"],
    ["MapTR-style detach between layers (E120b)", "0.691", "0.862", "0.44", "-"],
    ["One layer (E120c)", "0.694", "0.861", "0.45", "-"],
    ["Lateral displacement only (E120d)", "0.692", "0.859", "0.49", "-"],
    ["Refined + keep gate 0.5", "-", "0.872 (P 0.845 → 0.877)", "-", "-"],
], header=["TEST, 833 frames", "F1@0.5", "F1@1", "Vertex error mean / median (m)", "within 0.5 m"], num=(1, 2, 3, 4), hl=(1,))
d.p("About half of the vertex-error gain is sliding along the lane, which the loss does not penalise. <b>The gradient arrives and the planner gains nothing:</b> the consumer error is 0.36 m on refined edges against 0.34 m on traced ones; "
    "with the planner loss alone the refinement is undone (F1@1 0.852), with the position loss kept it stays (0.863, 0.359 m).")
d.callout("A correction that mattered " + TIE("noise"),
          "Retraining the route head on refined tokens gave 0.875 / 0.868 / 0.854 / 0.845 against 0.853 / 0.843 / 0.832 / 0.820 - an apparent gain. A second training run of each head gave 0.868 / 0.862 / 0.853 / 0.834 (refined) and 0.864 / 0.860 / 0.862 / 0.850 (traced): "
          "the effect on map assignment is <b>not demonstrated</b>, and the 'price of B against A' of 0.01 - 0.04 quoted in loops 10 - 15 was a seed-0 value (0.008 - 0.019 with seed 1). From then on every route-head comparison uses two seeds.")
d.fig(F + "p15_results.png", "Loop 15 results: vertex error by range band (gain 13 - 22 % everywhere), keep gate, planner-loss variants.", 100)

# ------------------------------------------------------------------ loop 16
d.h1("7  Loop 16: eight fully differentiable builders, one champion")
d.p("<b>Hypothesis.</b> A builder made of <i>pieces and links</i> with survival read-out along rays can give much shorter graphs (target 14.3 vertices per 100 m) at the control's quality, "
    "with no argmax, no matching and no rule step on the training path. Success criterion fixed before the first run: S above the control and two of three (J ≥ 0.5, ≤ 17 vertices per 100 m, lanes found +0.03).")
d.h2("Search design")
d.p("Seven two-level factors (construction: grow a tree from tips / spray seeds, snap, link; pattern: fixed fan / teardrop; seeds; duplicate handling; context; initialisation; revisit) were covered by a Taguchi L8 array: "
    "<b>8 designs out of 128 combinations</b>. Successive halving 8 → 4 → 2 → 1 with a growing training budget (1 / 2 / 4 / 8 units of 1,200 iterations) cut the weakest at each rung. "
    "Before any training every design passed gates for geometry parity, operation scan, gradient reach of every parameter group and planner-loss reach. "
    "The control is the learned trace network E94 trained from the same stem (S 0.713 / 0.710 / 0.707 / 0.725 at 1 / 2 / 4 / 8 units; band 0.010).")
d.fig(F + "p16_ladder.png", "Successive halving. Crosses mark cut designs; the grey band is the control at the same training budget. The final point includes the assembly steps found after the ladder (anchoring, relocation, beam 288 / 144).", 92)
d.table([
    ["B  teardrop pattern instead of fixed fan", "+0.053", "confirmed by four flips (+0.078, +0.076, +0.028, +0.117)"],
    ["D  learned pair head instead of geometric duplicate removal", "−0.040", "confirmed (flip +0.046); BCE flat at about 0.44 - 0.55"],
    ["E  global context, G revisit", "+0.029 each", "not confirmed by flips (+0.005, −0.001)"],
    ["SPF2  hard-concrete keep gate", "−0.015", "lanes found 0.42 vs 0.69"],
    ["SPF3  ray reach 12 m instead of 24 m", "+0.020", "supported for D5 and D6, −0.009 for D7"],
    ["HF1 stage B / HF2 stage C / HF3 three children with MCL", "−0.004 / +0.001 / −0.000", "all inside the band; stage C lifts J by +0.04"],
    ["Eight more training units; time recursion", "no change", "S 0.724 → 0.725"],
], header=["Factor", "Effect on S (pooled)", "Evidence"], num=(1,), hl=(0,), neg=(1, 3))
d.h2("The champion")
d.fig(F + "p16_arch_final.png", "Champion D7 (1.2 M parameters): lattice seeds snapped onto lanes read the BEV features along a teardrop of rays; a verifier gives per-sample on-lane probability, its running product (survival, as in volume rendering) sets how far a ray is trusted and the vertex validity; three best endpoints become children; "
      "link and junction heads and a path-union assembly produce the graph that a route head and the consumer read.", 100)
d.table([
    ["VAL (873 frames)", "champion f1", "0.746", "0.870", "0.265", "13.5", "0.695", "0.042"],
    ["VAL", "control E94", "0.724", "0.883", "0.405", "21.8", "0.800", "0.062"],
    ["TEST, used once (833 frames)", "champion f1", "0.728", "0.839", "0.260", "13.8", "0.653", "0.060"],
    ["TEST", "control E94", "0.702", "0.851", "0.404", "22.0", "0.762", "0.089"],
], header=["Set", "System", "S", "F1@1", "J", "Vertices / 100 m", "Lanes found", "False length"], num=(2, 3, 4, 5, 6, 7), hl=(0, 2))
d.callout("Reading the table honestly " + TIE("mixed"),
          "The S gain (+0.021 VAL, +0.026 TEST) rests on sparsity and false length. Accuracy is lower: F1@1 −0.013 / −0.012, J −0.14, lanes found −0.10. The plan criterion was met in one of three points (vertices). "
          "Because S rewards sparsity up to 14.3 vertices, 'higher S' alone is circular evidence that the builder is better; Q (S without the sparsity term) is about 0.036 lower for the champion than for the control (loop 18 gap check, not re-verified here).")
d.h2("Where it is weak: forks")
d.fig(F + "p16_forks.png", "Fork diagnosis (649 true forks in 250 VAL frames): 41 % have stem or branch geometry not covered within 1 m, 35 % have both branches inside one predicted piece, 22 % are predicted within 6 m. True forks are shallow: median angle 5.5°, 38 % under 2° (two overlapping ground-truth lanes).", 88)
d.p("The junction head alone finds forks as well as the control (split / merge F1 0.42 / 0.52 against 0.35 / 0.46) but the assembled graph reaches only J 0.19 - 0.23. Two no-parameter assembly steps helped: snapping at 0.4 - 0.6 m radius and "
    "<b>fork relocation</b> (move the fork vertex up its stem to the foot of a strong heat peak; J 0.236 → 0.278, S 0.742 → 0.749). Three attempts to realise the head's forks inside the assembly, and a fork-coverage loss, were negative.")
d.h2("Loop 16b: the route gap was a representation error, and speed")
d.ul([
    "<b>Route success on TEST was 0.72 for the champion against 0.85 - 0.86 for the control (two seeds)</b> (vertex-based head). The first explanation (lane recall, untested vertex tokens) was <b>wrong and is retracted</b>: a recall-weighted fine-tune raised route coverage from 0.87 to 0.88 only, and zero tokens barely changed the result.",
    "Both graphs miss the same share of route edges (13.0 % vs 13.5 %), but a missed champion edge is 8.1 m long, not 4.6 m, and the head was asked once per vertex of a long edge. "
    "<b>Asking the head along the edges</b> (points ≤ 4 m, interpolated tokens): VAL route success 0.798 → 0.889 (two seeds 0.888 / 0.889), control 0.885; consumer error 0.31 → 0.26 m (control 0.28 m).",
])
d.fig(F + "p16_route_attr.png", "Route success of the ceiling, the vertex head and the head asked along the edges (VAL).", 92)
d.table([
    ["builder (CUDA graph)", "12.4", "5.4", "6.5"],
    ["memory stage (hold / gate)", "-", "-", "2.3"],
    ["assembly (CPU, vectorised)", "13.8", "13.2", "13.6"],
    ["route input + route head", "2.5 + 2.0", "2.5 + 2.3", "2.4 + 2.3"],
    ["consumer", "0.5", "0.7", "0.6"],
    ["<b>chain total (median)</b>", "<b>31</b> (p90 56)", "<b>25</b>", "<b>28.3</b>"],
    ["control: wavefront 619 + route 6.2 + consumer 3.0", "628", "-", "-"],
], header=["Stage, ms per frame (idle RTX 3060, batch 1, cached foundation features)", "loop 16b: champion f1, beam 288 / 144", "loop 16b: j1, beam 192 / 96", "loop 17: 192 / 96 with memory stage"], num=(1, 2, 3), hl=(5,))
d.p("Fast path = CUDA graphs for the builder and padded route head, vectorised assembly, with graphs identical in 120 of 120 frames and S 0.7455 vs 0.7458. The foundation network, dense head and map rasterisation are not timed. "
    "The CPU assembly (13.5 ms) is now the largest stage. Below beam 192 / 96 lane recall collapses (VAL250, f1 with time recursion: S 0.749 at 288 / 144, 0.746 / 0.736 / 0.703 for 192 / 96, 144 / 72, 96 / 48).")
d.fig(F + "p16_pareto.png", "Quality-time frontier: S against milliseconds per frame for beam sizes, with and without the time recursion.", 92)
d.callout("Not done / open",
          "No TEST number exists for the route head along the edges and the fast configuration (TEST is allowed once by the loop 18 prompt, not yet run). A replicate with the time recursion from step 0 failed (S 0.469: the prior built from the untrained network corrupts the seeds); "
          "a replicate without lineage reached S 0.741, so the result does not depend on how the champion was trained.", "warn")

# ------------------------------------------------------------------ loop 17
d.h1("8  Loop 17: the champion under sensor failure")
d.p("The champion had never been tested when a sensor fails. Every earlier chain had. The question decides whether it can replace the control.")
d.h2("Protocol")
d.p("22 VAL scenes, every third key frame (1.5 s apart), 291 frames per failure type: the first half of each scene is clean (137 frames), the second half (154 frames, about 10 s) uses the perturbed foundation features of one type. "
    "Main types: night, motion blur, three front cameras off. Milder types: fog, radar off, extrinsics error (3° class), one front camera off. The same failure-half frames <i>without</i> perturbation serve as matched reference "
    "(S 0.743, route success about 0.94; for route success the clean half (0.85) understates the loss (matched 0.94); for S the two references agree within 0.01). Choices such as decay 0.95 and EMA 0.5 were made with these 22 scenes in view; "
    "only the four milder types were never used for a choice.")
d.fig(F + "p17_arch.png", "The robust chain of loop 17: the memory stage (spatial hold or a learned gate) sits between the cached input and the builder; the map-conditioned variant adds five SD-map channels to the stem.", 100)
d.h2("What a failure costs, and what memory brings back")
d.table([
    ["night", "0.384", "0.424", "0.620", "0.644", "0.333 → 0.687 → 0.704", "0.10 → 0.43"],
    ["motion blur", "0.480", "0.516", "0.639", "0.652", "0.505 → 0.722 → 0.733", "0.22 → 0.48"],
    ["three front cameras off", "0.550", "0.580", "0.602", "0.598", "0.665 → 0.712 → 0.707", "0.38 → 0.50"],
    ["fog", "-", "0.696", "0.699", "-", "0.803 → 0.827", "0.59 → 0.67"],
    ["radar off", "-", "0.711", "0.711", "-", "0.814 → 0.834", "0.63 → 0.70"],
    ["extrinsics error", "-", "0.624", "0.626", "-", "0.693 → 0.754", "0.45 → 0.61"],
    ["one front camera off", "-", "0.667", "0.669", "-", "0.759 → 0.784", "0.52 → 0.59"],
], header=["Failure (VAL, failure half)", "Control single", "Champion single", "+ spatial hold", "+ learned gate", "F1@1: single → hold → gate", "Lanes found: single → hold"],
    num=(1, 2, 3, 4), hl=())
d.p("Clean half of the same scenes: S 0.750 (hold 0.752, gate 0.760). The failure costs the champion mainly <b>recall</b> (lanes found 0.71 → 0.10 at night, recall 0.856 → 0.218) but also precision (0.90 - 0.91 → 0.70 - 0.73; false length ×4). "
    "Paired F1@1 differences (scene bootstrap): hold − single +0.354 [+0.262, +0.447] night, +0.217 [+0.150, +0.282] blur, +0.047 [+0.021, +0.072] cameras; gate − hold +0.018 [+0.009, +0.029], +0.016 [+0.006, +0.023], +0.001 [−0.008, +0.010]. "
    "The hold recovers 60 % / 52 % / 13 % of the S loss. With the same hold the champion is <i>not</i> better than the control in F1 (−0.022 night, −0.008 blur, +0.066 cameras); it keeps its S edge through sparsity.")
d.fig(F + "p17_summary.png", "Loop 17 in four panels: (1) failure-half S by type and system, (2) route success under failure (two route-head seeds; dashed: the same frames unperturbed), "
      "(3) synthetic loss of perception, (4) the same builder trained on synthetic loss only, evaluated on the perturbed series. Panels 3 and 4 show that the gain is training, and the map adds little.", 100)
d.h2("What did not help")
d.table([
    ["Builder's own time recursion (E196)", "tie in S (0.419 / 0.510 / 0.571 off vs 0.424 / 0.516 / 0.580 on); F1@1 with hold −0.005 in blur and three cameras", "drop it: saves a second builder pass"],
    ["Failure-trained dense head (E197)", "single frame +0.101 F1@1 at night; with hold −0.021 [−0.046, +0.004]", "nothing on top of the hold"],
    ["Builder fine-tuned on ~77 failure frames per type (E198)", "single +0.026 night; with hold −0.030 [−0.041, −0.019] for lost cameras", "a sample-size result, not proof that weights cannot help"],
    ["Learned gate with three front cameras off (E199)", "+0.001 [−0.008, +0.010]", "memory cannot invent absent information"],
    ["Joint fine-tune: builder + route head (E202)", "control arm (route loss detached) as good or better; arms differ by up to 0.09", "gradient path works, no gain shown (one seed, loss 0.03)"],
], header=["Experiment", "Result", "Reading"], idcol=0)
d.h2("A map-conditioned builder")
d.p("Five SD-map channels are appended to the stem (zero-initialised convolution, so it starts as the champion). Arm B is trained on 842 perturbed TRAIN rows per type (seven types), clean rows and 20 % synthetic loss; arm C is the same without the map. "
    "A learned gate behind B (frozen) gives the best chain.")
d.table([
    ["champion + gate, no map input", "0.644", "0.652", "0.598", "0.420 / 0.588 / 0.427"],
    ["B + gate, exact map", "0.676", "0.677", "0.681", "0.818 / 0.883 / 0.814"],
    ["B + gate, map error 1 m", "0.660", "0.665", "0.666", "-"],
    ["B + gate, map error 2 m", "0.643", "0.648", "0.644", "-"],
    ["B + gate, no map at test time", "0.651", "0.653", "0.657", "-"],
    ["B + gate, wrong map", "0.605", "0.602", "0.584", "-"],
    ["map alone, exact (no perception, 137 frames)", "0.693", "-", "-", "0.628 (SD route)"],
    ["map alone, error 1 m / 2 m", "0.524 / 0.449", "-", "-", "0.401 / 0.307"],
], header=["VAL failure half", "S night", "S blur", "S 3 cameras off", "Route success night / blur / cameras"], num=(1, 2, 3), hl=(1,), neg=(6, 7))
d.callout("What the map really buys " + TIE("limited"),
          "Most of the night / blur gain over the champion comes from <b>training on perturbed rows</b> (lanes found 0.09 → 0.21 single frame), not from the map. The map input adds +0.022 / +0.020 / +0.025 S with hold; "
          "in the gated chain a 1 m error keeps about 35 - 50 % of the exact-map benefit and 2 m none. F1@1 exact − none is +0.055 [+0.030, +0.089] at night; <b>wrong − none is −0.045 [−0.068, −0.026]</b>: the chain has no abstain output. "
          "The exact map alone (S 0.693, F1@1 0.873) matches the chain in F1@1 and is above the failure-half chain B + gate (0.676); the clean chain scores S 0.75. So 'exact map' is an oracle-like upper bound. The earlier headline 'the SD map fills most of the gap' was dropped after this baseline.")
d.fig(F + "p17_best.png", "Graph quality and route success of the best chains under failure, including C + hold (route 0.66 / 0.70 / 0.38, a number not in a table of this report).", 100)
d.h2("The planner's view under failure")
d.table([
    ["champion chain, route head with exact map", "4.11", "1.81", "1.78", "0.21 (unperturbed)"],
    ["B, exact map in the builder", "0.31", "0.29", "0.28", "0.23"],
    ["B, no map in the builder (route head exact map)", "0.54", "0.33", "0.34", "-"],
    ["champion chain, route head without map", "4.00", "2.07", "2.06", "0.36"],
    ["B, no map anywhere", "0.55", "0.42", "0.46", "-"],
], header=["Consumer path error at 20 m, night (m), 82 frames per cell, three seeds", "single frame", "spatial hold", "learned gate", "same frames unperturbed"], num=(1, 2, 3, 4), hl=(1, 4))
d.p("A failure costs the consumer about 20 times at night without memory (4.11 vs 0.21 m) and about 9 times with memory (ratios derived from the table). A builder trained on perturbed rows brings the night error to 0.42 - 0.46 m <i>without any map</i>, close to the unperturbed 0.36 m; "
    "the exact map adds about 0.1 m, and its route lies close to the driven path (upper bound).")
d.fig(F + "p17_frames_dark.png", "Night frames: ground truth, single frame, hold, gate and control. Qualitative only; in two of four columns the gate is below the hold.", 62)
d.callout("Unseen failure types (E209): no result " + NEG("crashed"),
          "The design (train on fog, radar off, extrinsics, one and three cameras off; evaluate on night and blur) is written down, but the run crashed with a CUDA 'unknown error' and no checkpoint; <code>e142.done</code> is a false flag (renamed <code>.FALSE</code>). "
          "<b>No number about unseen failure types may be quoted.</b> The only evidence is loop 14's consumer (not the builder) with one type held out: 66 % (night, single frame) to 100 % (blur, three cameras) of the failure-training gain carried over.", "neg")

# ------------------------------------------------------------------ best systems
d.h1("9  The best systems at the end of the week")
d.table([
    ["<b>Clean frames, E2E candidate</b>: champion f1 + head along the edges + consumer", "S 0.746, F1@1 0.870, J 0.265, 13.5 vertices / 100 m, lanes found 0.695, route success 0.889 (two seeds), consumer 0.26 m; TEST (vertex head): route 0.72, the head along the edges was not run on TEST", "31 ms (j1 192 / 96: 25 ms)"],
    ["Control: E94 + E104b + wavefront", "S 0.724, F1@1 0.883, J 0.405, 21.8 vertices / 100 m, lanes found 0.80, route success 0.885, consumer 0.28 m; TEST (E94 + E104b): route 0.853 / 0.843 / 0.832 / 0.820", "628 ms"],
    ["<b>Failure, no map in the builder</b>: champion + learned gate (recursion off); the route head reads the exact SD map", "S 0.644 / 0.652 / 0.598 (night / blur / 3 cameras off), route 0.42 / 0.59 / 0.43, consumer at night 1.78 m", "≈ 29.6 ms with memory stage"],
    ["<b>Failure, map better than ~1 m</b>: map-conditioned builder B + gate", "S 0.676 / 0.677 / 0.681 (exact), 0.660 / 0.665 / 0.666 (1 m); route 0.82 / 0.88 / 0.81 (exact); consumer at night 0.28 m (exact) / 0.42 - 0.46 m (no map)", "≈ 30 ms"],
    ["Modular system A (accuracy reference, loops 8 - 9; all numbers TEST)", "Final graph F1@1 0.875, split / merge F1 0.714 / 0.637, route success ensemble 0.876 / 0.879 / 0.876 / 0.858", "0.27 s (CPU, unoptimised; the 41 ms of E43 is the loop 3 chain)"],
], header=["System", "Key numbers (VAL unless noted)", "Time per frame"], idcol=0)
d.p("<b>Honest status.</b> Both failure chains stay well below the unperturbed level (S 0.75, route success 0.94). TEST was not used after the loop-16 freeze. "
    "A suggested go criterion for a TEST run: failure-half S within 0.08 of the unperturbed reference and route success above 0.7 with a map of 1 - 2 m error (without any map nothing is flagged). The champion with gate meets neither; chain B with an exact map meets the S criterion against the matched reference (0.750) but not against its own clean half (0.761).")
d.fig(F + "p16_frames.png", "TEST frames at fixed percentiles: ground truth, control and champion (loop 16).", 80)

# ------------------------------------------------------------------ what did not work
d.h1("10  What did not work", new_page=False)
d.p("Negative results are results; each of these cost between minutes and hours and should not be repeated without a new idea.")
d.table([
    ["10", "Wrong-direction training, covered-crop removal, threshold sweeps, order of growth", "Direction 0.916 but F1 and lanes found lower; no effect otherwise"],
    ["10", "E87 retrained on in-sample learned graphs; heads on one step token; pose head and correlation volume", "0.556 route success; position prior only; no route gain"],
    ["11", "Memory of the route attribute (E106)", "No weight ≤ 0.5 changes route success by > 0.01"],
    ["12", "Hold with three front cameras off; radar off", "No gain, precision falls 0.60 → 0.47"],
    ["13", "Connector head; verifiers under failure; trust inputs; learned gate in the consumer (constant 0.45)", "Lower junction F1; verifier loses in 5 of 6 cells; no gain; gain was the parallel ego path"],
    ["14 - 15", "Planner loss through route head and positions (builder: gradient reach only; route loss E202)", "Gradient arrives, no gain; planner loss alone undoes the refinement"],
    ["16", "Growth family D1 - D4, learned pair head, keep gate, stage B / C, b = 3 MCL, hybrid, more training, fork-coverage loss, assembly-side fork recovery, ensemble, hysteresis", "All cut or inside the band; ensemble lifts F1@1 +0.014 but J −0.10"],
    ["17", "Builder recursion, failure-trained dense head, builder fine-tune on ~77 frames, joint fine-tune, synthetic loss of perception as stand-in for real failure", "Tie / nothing on top of hold / sample size / no gain shown / does not transfer"],
], header=["Loop", "What", "Outcome"], idcol=0)

# ------------------------------------------------------------------ caveats
d.h1("11  Numbers and claims that were corrected - do not repeat them", new_page=False)
d.table([
    ["Route-head seed spread 'up to 0.03' for every cell", "Up to 0.03 on clean frames (E121c refinement effect untested beyond that), but 0.05 - 0.09 in the failure-series cells. There, differences below about 0.1 are not interpretable without intervals (gate vs hold on route success, hold effect on B with lost cameras, E202 arms)."],
    ["'TEST once' for the whole week", "True for loop 16 only (one log entry). Loops 10 - 15 reported TEST as the main result; failure and memory variants of loops 11 - 14 were compared on TEST only (about 60 uncorrected comparisons in E105)."],
    ["'The SD map fills most of the gap'", "Map input adds about +0.02 S (about 20 % of the gap); the exact map is an upper bound; wrong map worse than none."],
    ["'Price of B against A 0.01 - 0.04 route success'", "A seed-0 value; 0.008 - 0.019 with seed 1. The 0.05 gap of loop 10 was mostly threshold 0.6 vs 0.5."],
    ["'The champion is more accurate'", "No: S edge is sparsity. F1@1 −0.013, J −0.14, lanes found −0.10."],
    ["Route gap explained by lane recall (loop 16, section 7)", "Retracted in 16b: representation of the attribute along long edges."],
    ["'Train tokens are out-of-fold' (loop 13 verifiers and consumers)", "Wrong: the trace network saw those rows. Only the dense-head inputs were out-of-fold."],
    ["'More robust than the rule pipeline' (loop 11)", "About as robust as the rule chain with its own coasting."],
    ["'A memory on the input brings most of it back at no cost'", "60 % / 52 % / 13 % of the S loss; clean J 0.247 → 0.222 and clean route success 0.846 → 0.838 (same setting; the builder recursion accounts for the rest of the 0.019)."],
    ["'Real failure frames' (loop 17)", "Synthetic perturbations of the foundation features; E206 trains and tests on the same seven types."],
    ["Experiment ids of loop 17", "Internal E130 - E144 are catalogued as E193 - E212 (E130 / E130b = E193 / E194, E131 = E195, E132 = E196, E133 = E197 / E198, E134 = E199, E135 = E200 / E201, E136 = E202, E137 = E203, E138 = E204 / E205, E139 = E206, E140 = E207, E141 = E208, E142 = E209, E143 = E210, E144 = E212). The catalogue ids E130 - E144 are different experiments."],
    ["Scene count 323", "255 scenes (212 / 22 / 21)."],
], header=["Claim", "Correction"], idcol=0, widths=["32%", None])
d.p("<span class='small'>Unresolved source disagreements: control timing 619 / 628 / 586 ms (628 ms follows the idle, batch-1 protocol); champion training lineage 24 vs 28 units; control consumer on VAL 0.27 vs 0.28 m; "
    "the TEST log labels the control 'E94 + E120' while the numbers match the unrefined E94. The loop-17 ledger was restored by replay after a rollback; the ledger rows, not the deleted code, are the evidence.</span>")

# ------------------------------------------------------------------ open issues
d.h1("12  Open issues and next steps", new_page=False)
d.ol = None
d.ul([
    "<b>Finish loop 17:</b> rerun E209 (unseen failure types) with checkpoints, add its result sentences, one fresh numbers review.",
    "<b>TEST once</b> for the head along the edges, the fast configurations, the gate and the map builder: allowed once by the loop 18 prompt (PROGRAM.md), not yet run; criterion above.",
    "<b>Map trust / abstain output</b> (dustbin column as in SuperGlue / LightGlue) before the map chain may be used; a differentiable rough registration of the sparse graph to the SD map, with the target of a benefit at 1 - 2 m error.",
    "<b>Joint end-to-end fine-tuning</b> after modular training, longer (4 - 8 units), builder + gate + route head + consumer with failure frames and a planner loss (E202 was one seed).",
    "<b>Grid head + sparsification trained end to end</b> (one loss from graph and route back into the dense head) against the modular version; a second-successor head for forks with a fork metric that follows geometry.",
    "<b>Speed:</b> compiled assembly and CPU / GPU overlap (assembly 13.5 ms is the largest stage); latency including the foundation network.",
    "<b>Evaluation gaps:</b> recorded (not synthetic) failures, failures at the start of a drive, long and combined failures, a real navigation map with measured error, a closed-loop planner test, a location-disjoint split check, confidence intervals for S and route success, more than one training seed for gate and builder B.",
], ordered=True)

# ------------------------------------------------------------------ appendix
d.h1("Appendix  Experiment index, loops 10 - 17", new_page=True)
rows = [
    ["E92", "10", "Ground-truth expert replay of the trace network", "F1@1 0.992 VAL / 0.991 TEST; split / merge 0.82 / 0.85 (VAL)", "tool"],
    ["E94", "10", "Learned trace network, 192 seeds", "VAL / TEST F1@1 0.883 / 0.851; rule 0.891 / 0.867", "ok"],
    ["E94b, E96 - E98", "10", "Wrong-direction states, order, covered crop, thresholds", "not adopted / no effect", "neg"],
    ["E102 - E104b", "10 - 11", "E87 on learned graph; token heads; cross-attention route head", "TEST route 0.853 / 0.843 / 0.832 / 0.820 at threshold 0.5", "ok"],
    ["E105 - E107", "11", "Warp + EMA + hold in front of the extractor; route memory; whole chain", "night F1 0.337 → 0.728; route memory no gain", "tie"],
    ["E108 - E112", "12", "Oracle flag, calibration, route confidence, seed noise, spatial hold", "night 0.337 → 0.680; ECE ≤ 0.016; AUROC 0.88 - 0.91", "ok"],
    ["E113, E116", "13", "Connector head; junction and piece verifiers", "split / merge F1 0.385 / 0.428 → 0.437 / 0.504; harmful under failure", "tie"],
    ["E114, E115", "13", "Learned chain under failure; consumer with trust and ego motion", "night route 0.093 → 0.444; ego motion helps, trust does not", "tie"],
    ["E117 - E119", "14", "Planner consumer; failure-aware consumer; planner loss through route head", "0.34 vs 0.36 m; night 1.00 → 0.44 m; no planner-loss gain", "ok"],
    ["E120, E121", "15", "Refinement decoder; planner loss through positions", "F1@1 0.851 → 0.863; no consumer gain", "tie"],
    ["E122 - E192", "16", "Control and eight builders D1 - D8, successive halving", "champion D7: S 0.746 vs 0.724 (VAL)", "ok"],
    ["K185 - K187", "16b", "Route head along the edges; fast runtime; compute frontier", "route 0.798 → 0.889; 31 vs 628 ms", "ok"],
    ["E193 - E194", "17", "Champion single frame under failure (three + four types)", "S 0.424 / 0.516 / 0.580 vs clean 0.750", "info"],
    ["E195, E196", "17", "Spatial hold; builder recursion off", "0.620 / 0.639 / 0.602; recursion a tie", "ok"],
    ["E197, E198", "17", "Failure-trained dense head; builder fine-tune", "nothing on top of hold", "neg"],
    ["E199", "17", "Learned gated memory (12.5 k parameters)", "+0.021 / +0.015 / +0.001 S (night / blur / cameras)", "ok"],
    ["E200, E201", "17", "Route head along edges under failure; hold vs gate", "hold +0.28 / +0.22; gate vs hold tie", "tie"],
    ["E202", "17", "Joint fine-tune builder + route head", "no gain shown", "neg"],
    ["E203", "17", "Latency of the robust chain", "28 - 33 ms", "ok"],
    ["E204, E205", "17", "Map-conditioned builder, synthetic loss; diagnosis", "+0.008 - 0.027 S on the perturbed series; map +0.005 - 0.012", "tie"],
    ["E206, E207", "17", "Map builder B trained on perturbed rows; gate behind B", "S 0.676 / 0.677 / 0.681 (exact map)", "ok"],
    ["E208", "17", "Matched unperturbed reference", "route 0.94, S 0.743", "info"],
    ["E209", "17", "Unseen failure types", "crashed, no result", "neg"],
    ["E210", "17", "SD map alone as lane graph", "S 0.693 exact; 0.524 at 1 m", "info"],
    ["E212", "17", "Consumer under failure", "night 4.11 → 1.78 m (champion); 0.28 m (B, exact map)", "ok"],
]
d.table([[a, b, c, e, T({"ok": "ok", "neg": "neg", "tie": "tie", "info": "info", "tool": "info"}[t], {"ok": "positive", "neg": "negative", "tie": "mixed", "info": "reference", "tool": "tool"}[t])] for a, b, c, e, t in rows],
        header=["Id", "Loop", "What", "Result", ""], idcol=0, widths=["12%", "6%", "34%", "40%", "8%"])
d.p("<span class='small'>Sources: experiment journal (STATE.md), loop indexes and ledgers, the loop 15 - 17 reports and the reviewer findings of each loop (review11 - review17). Numbers were taken from STATE.md, the loop 15 - 17 reports, the loop indexes and the loop 17 ledger via a working digest (<code>reports/source/weekly/digest_week2.md</code>, not a primary source) and then reviewed against the primary sources (<code>review_week2.md</code>).</span>")

d.save(OUT)
