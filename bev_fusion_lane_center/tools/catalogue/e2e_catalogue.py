"""Catalogue of all experiments (E15 - E121) in thematic order, with the tag for the use in a completely differentiable end-to-end system.
Tags: E = learned, differentiable, usable in an E2E system; P = usable, but only together with a classical step (the stage next to it is a rule or a geometric algorithm);
C = classical method, simulator, analysis or protocol (reference, not a part of an E2E system); D = tried, no gain or worse.
All numbers are quoted from the long report (TEST unless marked; F1 = point F1 at 1 m)."""

TAGS = {
    "E": ("E2E-usable", "learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside)"),
    "P": ("with a classical step", "learned, but needs a rule or geometric stage next to it"),
    "C": ("classical / reference", "classical method, simulator, analysis, protocol or literature review"),
    "D": ("dropped", "tried, no gain or worse"),
}

THEMES = [
    ("T1", "Foundation and dense lane field", "loops 1 - 4"),
    ("T2", "From the lane field to a graph", "loops 1 - 4"),
    ("T3", "Map prior and navigation route", "loops 1 - 7"),
    ("T4", "Perception-first graph with a learned assignment (reference system A)", "loops 8 - 9"),
    ("T5", "Learned graph extractor and registration-free navigation head (system B)", "loops 10 - 13"),
    ("T6", "Time and sensor failure", "loops 5 - 13"),
    ("T7", "Planner interface, trust and the planner loss", "loops 5 - 14"),
    ("T8", "Differentiable vertex positions (refinement of the traced graph)", "loop 15"),
]

# (theme, id, what, result, tag)
EXPS = [
    # ---- T1 foundation and dense lane field
    ("T1", "E15 - E16", "Frozen CRN camera-radar features on nuScenes Mini: layout, feature source, channel count, decoder changes", "heat-map F1 0.58; mirrored layout 0.37 - 0.43; channel count and level irrelevant (0.55 - 0.58)", "E"),
    ("T1", "E17", "Scale the CRN heads to 8,414 training frames: FCN (0.1 M) against U-Net", "FCN 0.264 to 0.493; U-Net +0.12 (0.613)", "E"),
    ("T1", "E18", "CNT-Net: ConvNeXt-V2 encoder with a transformer bottleneck as dense decoder (4.8 M)", "0.507, 0.11 below the U-Net", "D"),
    ("T1", "E23 / E25", "Ensemble of decoder families; coordinate channels and a distance-dependent threshold", "ensemble 0.650 against 0.670; coordinate channels not confirmed with full training", "D"),
    ("T1", "E24", "Error analysis of the CRN heads by range and place", "recall 0.81 (0 - 10 m) falls to 0.50 (40 - 48 m); the loss sits in the frozen features", "C"),
    ("T1", "E27", "Which frozen foundation carries the lane structure: detection-pretrained CRN against map-pretrained GaussianCaR (same 2.7 M U-Net head)", "F1 0.624 to 0.867 (VAL), 0.598 to 0.835 (TEST): +0.24, the largest single effect", "E"),
    ("T1", "E28", "Which feature level of the foundation to read", "decoder feature 0.874 / 0.840 (VAL / TEST); fused BEV 0.772 / 0.743; map logits 0.779 / 0.738", "E"),
    ("T1", "E29", "Loss weighting: false alarms three times costlier than misses", "moves the threshold only, F1 change at most 0.003", "D"),
    ("T1", "E30", "Train-to-test gap of the dense head: data amount, scenes, regularisation, mirroring", "no under-fitting; 0.033 head over-fitting plus a seen-to-unseen scene shift of 0.048 (VAL) / 0.079 (TEST)", "C"),
    ("T1", "E37", "Dense head at full maturity: all 8,414 rows, stabilised recipe, cross-fitted heads", "F1 0.959 / 0.885 / 0.856 (train-eval / VAL / TEST)", "E"),
    ("T1", "E49 / E50 / E52", "Better read-out: wider PCA (96), instance normalisation, both foundations fused", "no gain (at most 0.008)", "D"),
    # ---- T2 from the lane field to a graph
    ("T2", "E19 / E26", "Line forming: tuned threshold, smoothing, minimum length; point-by-point line tracer against skeleton", "0.613 to 0.670 by threshold alone; tracer +0.023 (0.693); a perfect field gives 0.997", "C"),
    ("T2", "E20", "Causal temporal fusion of the distance fields (CRN)", "0.675 against 0.670", "D"),
    ("T2", "E31 / E33", "Heat map + direction field to a directed graph with the tracer (step size, point spacing)", "F1 0.879 / 0.851 (VAL / TEST), route F 0.765 / 0.702; even a perfect field gives route F 0.838", "C"),
    ("T2", "E32", "Sparse DETR-style LaneFormer (4.3 M, 96 queries, explicit successor matrix) instead of the grid", "0.058 F1 and 0.178 route F below the grid approach", "D"),
    ("T2", "E38 / E39", "Ceiling of the grid proposals; cascade refiner (3 blocks, pooling + re-estimation)", "98.2 % of the ground truth within 2.5 m; refiner +0.001 F1, +0.014 route F", "P"),
    ("T2", "E41", "Temporal feature fusion (warped previous frame) against query propagation", "fusion +0.008 F1 and one third less flicker; query propagation 0", "E"),
    ("T2", "E42 / E45", "Learned link head; TopoLogic-style link scoring for splits and merges", "within 0.01 of the rule links", "D"),
    ("T2", "E43", "Loop-3 system: grid + refiner + links (+ map prior, + temporal)", "no map F1 0.876, route F 0.744; split / merge F1 at 6 m only 0.34 / 0.38", "C"),
    ("T2", "E44 / E53", "Learned split / merge key points (2.7 M) with node-guided assembly; Bezier connectors between key points", "split / merge F1 at 6 m 0.34 / 0.38 to 0.55 / 0.61 (key points) to 0.71 / 0.64 (connectors); route F -0.025 against the previous system", "P"),
    ("T2", "E46 / E47", "LaneGAP path view and repair; SeqGrowGraph (autoregressive growth) as a literature case", "only 0.428 of the ground-truth paths recovered; repair +0.001; led to the trace network", "C"),
    ("T2", "E51 / E56", "Node-conditioned dense head; direction consistency as acceptance criterion for connectors", "+0.003 F1; connectors lose 0.026 split F1", "D"),
    ("T2", "E54 / E55", "Statistics: paired scene bootstrap of the graph results; seed noise of the dense head", "seed SD 0.0025 (VAL) / 0.0006 (TEST) F1; absolute F1 about +-0.035, paired differences about 0.005", "C"),
    # ---- T3 map prior and navigation route
    ("T3", "E21 / E22", "Map under drift: lane assignment after alignment; Kalman tracking of the drift", "assignment 79.2 % to 91.9 %; node error 1.19 to 0.83 m", "C"),
    ("T3", "E40", "Map prior as three raster channels in the dense head (no-map control, pose noise, outdated map)", "exact pose +0.075 F1; 1 m error turns it into -0.057 unless the map is aligned first", "P"),
    ("T3", "E48", "Learned registration of the map prior (RAVE-inspired fine matching after a correlation search)", "median error 0.54 to 0.43 m at 1 m", "P"),
    ("T3", "E59", "Graph-ICP registration and geometric route transfer (no parameters), lane-level map", "route success 0.923 / 0.883 / 0.782 at 1 / 2 / 3 m against 0.640 / 0.305 / 0.153 without registration", "C"),
    ("T3", "E60 / E73", "Route-conditioned head: map raster and route in, lane and route heat maps out; systematic fine-tuning on perturbed features", "+0.050 lane F1 with the exact map; corrupted features carry the gain (+0.059 DEV)", "E"),
    ("T3", "E61", "Temporal tracking of the map alignment and of the lane-ID assignment", "route success stays 0.92 where per-frame registration falls 0.845 to 0.608", "C"),
    ("T3", "E74", "Graph-ICP registration in front of the end-to-end head", "route success nearly flat in the map error (+0.06 over the geometric transfer at 3 m)", "P"),
    ("T3", "E75", "Does the route head follow a changed navigation goal (counterfactual routes)?", "alternative route 0.828 against 0.887 for the driven one (geometric transfer 0.805 / 0.887; E60 0.535, E65d 0.699)", "E"),
    ("T3", "E76", "Map-trust output: learned scalar head against a label-free geometric agreement score", "agreement score AUROC 0.996 where the learned scalar reaches 0.82", "C"),
    ("T3", "E77", "ADASIS-like navigation map simulator and a rule-based lane assignment baseline", "rule assignment 0.46 against 0.93 at exact pose", "C"),
    ("T3", "E78", "Navigation-map conditioned head (5-channel raster) with a fine-tuning study", "route success 0.733 (exact) / 0.713 (2 m, registered), no map 0.328", "E"),
    # ---- T4 perception-first graph (system A)
    ("T4", "E81", "Precision-first graph: tracer strictness, probability filter, learned keep output", "keep 0.9 removes 95 % of the false lane length and keeps 53 % of the lane length", "P"),
    ("T4", "E82", "Rough localisation with a lane-count-aware template (ICP), held through a failure", "pose error 1.11 to 0.62 m; route success at most +0.016", "C"),
    ("T4", "E83", "Assignment network on the 6 m edges of a rule-built graph: relation-biased transformer (0.27 M) with route and keep outputs", "route success 0.881 / 0.879 / 0.868 / 0.839: +0.15 to +0.30 over the pixel-level head", "P"),
    ("T4", "E84", "Extension of the perceived route with the navigation route, shifted by the perceived lane offset", "+0.30 (three front cameras off), +0.17 (night), +0.01 (clean)", "C"),
    ("T4", "E85", "Graph tracks with keep evidence and bridging (temporal fusion on the graph)", "+0.110 failure-phase lane F1 (night +0.35); needs odometry within 1 degree per sqrt(s)", "C"),
    ("T4", "E86", "Reference path from the output graph: the localised navigation route shifted by the perceived lane offset", "0.31 m from the driven path (map line alone 0.95 m)", "C"),
    ("T4", "E86b / E86c", "First learned consumer of the graph; planner loss into the assignment network", "consumer 0.88 to 0.41 m with the route attribute; planner loss no clear gain (route success 0.868 to 0.769)", "E"),
    ("T4", "E87", "Training against map faults: per-road bias, wrong and missing maps, abstention targets (same network as E83; the recipe carries over to system B)", "bias 2 m: 0.663 to 0.847; wrong-map route flag 60.7 to 4.2 m; no cost on right maps", "P"),
    ("T4", "E88", "Held world-frame correction of the registration through a sensor failure, with the new networks", "partial: helps where the held correction is right; needs odometry", "C"),
    ("T4", "E89 / E90", "Lane F1 with the extension points; per-road lateral re-registration", "+0.03 lane F1 in the night only; re-registration +0.007, inside the noise", "D"),
    # ---- T5 system B
    ("T5", "E91", "Range analysis of the rule graph (TEST): where the accuracy is lost", "F1 0.957 (0 - 10 m) falls to 0.800 (40 - 50 m); a far lane is found for 92 % of the lines when the near part is found, 69 % when it is missed", "C"),
    ("T5", "E92", "Expert replay: a ground-truth expert drives the wavefront of the learned extractor", "F1 0.99; 25 - 31 % of junctions lost even with perfect decisions (duplicate annotations)", "C"),
    ("T5", "E94", "Trace network (1.67 M): stem pyramid, seed head, fan sampling, covered crop, two transformer blocks, merge pointer", "F1 0.883 / 0.851 (VAL / TEST) against 0.891 / 0.867 for the rule graph; no hand-built tracer, assembler or registration; seeds are heat-map peaks, vertices are appended by bookkeeping", "E"),
    ("T5", "E94b / E96 / E98", "Wrong-direction states and four heads; wavefront order; range-aware thresholds", "no gain beyond the noise", "D"),
    ("T5", "E97", "Knock-outs: covered crop, range embedding, merge and fork events", "the crop of the emitted graph carries the junction topology (split / merge F1 0.35 / 0.46 to 0.19 / 0.21 without)", "E"),
    ("T5", "E99", "Orientation of traced pieces by the network's own direction head (sign vote)", "+0.05 direction, +0.02 route F", "P"),
    ("T5", "E102", "Assignment network of system A on the learned graph (zero-shot and retrained)", "0.042 - 0.048 route success below; oracle 0.913 on both", "P"),
    ("T5", "E103", "Navigation attribute inside the network without registration: step-token head, graph transformer, learned localisation", "0.78 - 0.71 against 0.86 - 0.84 (rule graph + registration); 40,000 instead of 3,000 steps: +0.14 to +0.39", "E"),
    ("T5", "E104", "Cross-attention from vertex tokens to navigation-map tokens with gated fusion (0.69 M)", "route success 0.853 / 0.843 / 0.832 / 0.820 at 0 / 1 / 2 / 3 m, 0.007 - 0.026 below the registration-based network", "E"),
    ("T5", "E111", "Seed noise of the trace network (second training run)", "F1 differs by 0.001 - 0.002, an order below the gap to the rule graph", "C"),
    ("T5", "E113", "Junction diagnosis; connector head; junction verifier", "1.8 times too many junctions; verifier split / merge F1 0.385 / 0.428 to 0.437 / 0.504 (rule graph 0.714 / 0.637); connector head lowers it", "P"),
    ("T5", "E116", "Piece verifier (short graphs instead of false lanes), also under failure", "+0.043 precision on clean frames; lower F1 in night and blur unless gated or failure-trained", "P"),
    # ---- T6 time and sensor failure
    ("T6", "E57 / E58", "Synthetic sensor failures (night, blur, lost cameras, fog, radar, extrinsics) and city transfer", "F1 0.867 falls 70 % (night), 50 % (blur), 25 % (three cameras), 6 % (radar, fog); a new city costs 0.04 - 0.10", "C"),
    ("T6", "E64", "Output-based runtime monitors and a monitor-gated fallback to the map route", "failing frames found with AUROC 0.953; lateral error 1.03 to 0.43 m", "C"),
    ("T6", "E65", "Robust training: feature augmentation, fine-tuning on real perturbed features, AdaBN", "night 0.259 to 0.451, an unseen blur 0.433 to 0.540", "E"),
    ("T6", "E67 - E72", "Held registration, carried-forward graph, map lanes as lane graph, jumps and failure bursts", "night: hold 0.53 m better than per frame; carry-forward 0.651 to 0.741 for about 5 s; map lanes 0.887", "C"),
    ("T6", "E79", "Ego-motion-warped average of the lane output; low-error map-free hypotheses", "+0.042 lane F1 in a failure phase (no training)", "E"),
    ("T6", "E80", "Learned temporal memory: the warped previous output as extra input channels", "+0.000 - 0.005", "D"),
    ("T6", "E105", "Input memory in front of the trace network: ego-motion warp, convex mix and a hold triggered by a label-free heat monitor", "night F1 0.337 to 0.728, blur 0.524 to 0.699", "P"),
    ("T6", "E106", "Memory of the route attribute on the vertices of the learned graph", "no gain", "D"),
    ("T6", "E107 / E108", "Whole learned chain under failure; an oracle flag and a gradual onset", "night route success 0.093 to 0.346; a perfect flag adds +0.07 (blur) and nothing for lost sensors", "P"),
    ("T6", "E112", "Spatial hold: per-pixel maximum of the frame and the decayed, warped memory (no monitor)", "night +0.34, blur +0.20 over the single frame; not for lost front cameras", "E"),
    ("T6", "E114", "Whole learned chain under failure with the spatial hold and a route confidence", "route success night 0.09 to 0.44, blur 0.23 to 0.55", "E"),
    # ---- T7 planner interface
    ("T7", "E62", "Which route to hand to a planner (open-loop planning check)", "registered map route 0.35 m better than the perceived route; perceived route no better than straight ahead", "C"),
    ("T7", "E66", "Small learned planner (0.3 M CNN) on the module output: does it use and follow the route?", "route channel -20 % lateral error on curves; with a wrong route no more compliance than without", "E"),
    ("T7", "E109 / E110", "Calibration of keep and route probabilities; route confidence as trust signal", "route probability of A calibrated (ECE 0.006, at most 0.016 under failure), that of B over-predicts the route length by 28 %; route confidence AUROC 0.88 - 0.91", "E"),
    ("T7", "E115", "Learned consumer of the graph with trust inputs and ego motion", "ego motion: pooled failure error 1.09 to 0.71 m; a failure flag adds nothing", "E"),
    ("T7", "E117", "Consumer (93 k) on the learned chain: route probability, ego motion, soft / hard verifier inputs, step token", "0.90 to 0.40 m with the route probability, 0.34 m with ego motion (system A: 0.36 m); step token 0.56 m", "E"),
    ("T7", "E118", "The consumer under sensor failure: input memory, failure-frame training, an unseen failure type", "night 2.73 to 0.72 m (spatial hold) to 0.44 m (also failure training)", "E"),
    ("T7", "E119", "Planner loss into the route head; oracle route channel; route head from random weights", "no gain (0.34 m either way, oracle 0.316 m); planner loss alone does not create the attribute (AUROC 0.53 - 0.55)", "D"),
    ("T7", "VLA", "Literature review (not an experiment): is a vision-language-action model needed?", "language adds about +0.4 driving score, inside run noise; needs a workstation GPU", "C"),
    # ---- T8 differentiable vertex positions
    ("T8", "E120a - E120c", "Refinement decoder on the traced graph (vertices, edges, step tokens of E94 and the 42 BEV channels): 3 layers (or 1, or reference points detached as in MapTR) sample the BEV at the CURRENT vertices, graph-biased self-attention, displacement; point-to-polyline loss on the finished polylines, no matching (1.37 M)", "TEST F1@0.5 0.661 to 0.694, F1@1 0.851 to 0.863, vertex error 0.54 to 0.43 m; 1 layer 0.861, detached 0.862 (noise 0.002): depth and the gradient between the layers do not matter; the vertices slide 1.8 m along the lane", "E"),
    ("T8", "E120d", "The same decoder with displacement across the lane only", "TEST F1@0.5 0.692, F1@1 0.860 (sliding variants 0.863), vertex error 0.49 m instead of 0.43 m: the F1 gain is a lateral correction, half of the vertex-error gain is sliding along the lane", "E"),
    ("T8", "E120e", "Learned keep probability of the decoder as sparsification (hard gate on the vertices to measure it)", "threshold 0.5: F1@1 0.872 (precision 0.845 to 0.877, recall 0.882 to 0.866), false length 0.082 to 0.053; AUROC 0.868", "E"),
    ("T8", "E121a / E121b", "Planner consumer (E117) on the refined graph and the planner loss through the live vertex positions: retrained consumer, plug, fine-tuned, joint with and without the position loss, keep probability as input", "path error at 20 m 0.36 m against 0.34 m (wrong / no map 0.76 / 0.75 against 0.76 / 0.74): no gain; the gradient reaches the positions; with the planner loss alone the refinement is undone (F1@1 0.852), with the position loss it is kept (0.863, planner 0.359 m)", "D"),
    ("T8", "E121c", "Route head E104b retrained on the tokens of the refined graph (same recipe, 40,000 iterations) and read on the refined graph; the unchanged head for comparison", "route success 0.875 / 0.868 / 0.854 / 0.845 at map error 0 / 1 / 2 / 3 m against 0.853 / 0.843 / 0.832 / 0.820 on the traced graph (+0.022 to +0.026; the paired scene interval includes 0 at 0 m); the unchanged head reads the refined graph worse (-0.011 to -0.015); gap to system A 0.001 - 0.022; the planner consumer with these probabilities stays at 0.34 / 0.33 / 0.77 / 0.73 m (E121d); a second training run of each head gives 0.864 / 0.860 / 0.862 / 0.850 (traced) and 0.868 / 0.862 / 0.853 / 0.834 (refined): the route head varies by up to 0.03 between seeds, no effect of the refinement shown", "D"),
]


def counts():
    from collections import Counter
    c = Counter(t for *_, t in EXPS)
    return c


if __name__ == "__main__":
    print(len(EXPS), counts())


# ---- loop 16: rows of the ladder runs (generated by p16_catalogue.py from the ledger; ids from E124)
import json as _json
import os as _os

_P16 = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123_ladder/catalogue16.json"
if _os.path.exists(_P16):
    _d16 = _json.load(open(_P16, encoding="utf-8"))
    THEMES.append(tuple(_d16["theme"]))
    EXPS += [tuple(_r) for _r in _d16["rows"]]
