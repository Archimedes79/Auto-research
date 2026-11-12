# the sections of the junction report, written by hand (executed by build_junction_report.py; uses its helpers tbl, fig, f3 ...)
LEAD = ("Mirko's brief of 2026-11-11 asked for a lane head that recognises, on a 1 m grid, the lane direction, the junction entry and exit and the junction area, orders the connections inside a junction (entry 1 to exit 1, entry 2 to exit 2), "
        "prunes without losing the special nodes, stays differentiable and ends in a probability for every entry-to-exit pair. This page is the status of that thread: a ground truth that holds everything the brief asks for (E285), junction maps on "
        "GaussianCaR's frozen features (E286), a pair head that carries the order rule (E287), and a set of cheap measurements that decided what is worth building (E288 - E303). Rules of the thread: everything on the training path soft (D55), "
        "a TIE is kept and refined but never worse (D82), VAL only, TEST never touched.")

STATUS = tbl(["id", "question", "result", "state"], [
    ["E285", "a ground truth with everything: zone, conflict core, arms, ranked ends, ordered pairs, paint, visibility", "gates pass on all 873 frames; order rule 1899 / 1899; l6 merged (D83)", "<span class='pill ok'>done</span>"],
    ["E286", "junction maps (entry / exit heat with heading, zone, core, stop line, crossing) on B's frozen pyramid", "see the maps section: kp_f1 against B's event heat 0.40", "<span class='pill ok'>trained</span>"],
    ["E287", "pair head v2: order features, evidence along the curve, crossing penalty", "see the pairs section", "<span class='pill info'>night run</span>"],
    ["E288", "the close-node step as a differentiable operator", "TIE: Matrix-NMS R 1.75 sigma 0.1 = the hard step", "<span class='pill ok'>done (CPU)</span>"],
    ["E289", "the points are unstable", "nodes are locked to the 2 m grid along the line; the lines are stable (line probe on the GPU)", "<span class='pill info'>partly</span>"],
    ["E290", "time: how often do pairs flip between keyframes", "20 % flip; averaging the warped previous logit cuts that by 44 % but loses true pairs", "<span class='pill info'>measured</span>"],
    ["E293", "how far into the junction can points be put", "the lane heat does not tell true from false paths for 10 m; the chord-tangent curve is good 5 m deep", "<span class='pill ok'>done (CPU)</span>"],
    ["E296", "protect the special types in pruning", "TIE at p 0.9; the flag precision is the limit", "<span class='pill ok'>done (CPU)</span>"],
    ["E300", "where does the pair head fail", "far range (> 35 m: recall 0.23 against 0.38 near) and Singapore (0.24 against 0.35); unpainted ends are found 8 - 12 points less often, visibility does not matter (strict matching, two seeds)", "<span class='pill ok'>done (CPU)</span>"],
    ["E301", "splits and merges outside junctions", "closed by data: 99.8 % of the events are inside junctions", "<span class='pill no'>no target</span>"],
    ["E298 / E303 / E304", "driven pairs weighted higher, two learned handle scales per pair (the spline), far-range weighted maps", "code written and tested, queued behind the night chain (day chain 1)", "<span class='pill info'>queued</span>"],
    ["E302 / E305 / E306", "capacitated assignment with degree heads (entry may feed several exits), maps + pair head trained together (the pair loss reaches the heat), exact polygon targets (the legacy fill is half a cell off)", "code written and tested on real frames (CPU), queued (day chains 2 and 3)", "<span class='pill info'>queued</span>"],
    ["E291 / E292 / E294 / E295 / E297 / E299", "road-level lane counts, Matrix-NMS layer, nodes from 1 m maps, sigma head, chord keep-gate, time prior", "cards written", "<span class='pill info'>planned</span>"],
])

APPROACH = ("<p>The brief is a ladder, and the ladder was followed in its own order: the ground truth first (looked at on real frames), then the 1 m pixel maps, then nodes, pruning, pairs, time and the map. Every rung has a card with a written rule before the run, "
            "two seeds for a trained arm, VAL 873 frames, the controls named in the card. Where a cheap CPU measurement could decide, it ran first.</p>"
            + tbl(["the brief", "experiment", "what it answers"], [
                ["1 m grid: direction, entry / exit, junction area, paint", "E285 GT layer, E286 maps", "do the maps see the ends better than B's event heat"],
                ["no 2 m grid if that works (a pyramid for context)", "E294 (planned)", "nodes read from the 1 m maps"],
                ["splits / merges without a junction", "E301", "closed: there are five in VAL"],
                ["nodes with a type and a position uncertainty", "E295 (planned), E289", "sigma along / across the line; the grid phase"],
                ["prune by a matrix operation, special types never", "E288, E296, E292, E297", "Matrix-NMS = the hard step; protection at p 0.9; the layer inside the graph"],
                ["connect: entry 1 to exit 1, entry 2 to exit 2", "E285 (rule holds), E287 (rank features, crossing penalty), rule control", "the order is a property of the data; the head carries it"],
                ["a probability per entry-exit pair, splines, driven weighted higher", "E287, E303, E298", "pair head, learned handle scales, driven weight"],
                ["cross-attention over time", "E290, E299", "flip rate and the benefit of one buffered frame"],
                ["the map: road-level graph with lane counts", "E291, E302", "counts as capacities of the assignment"],
                ["what is overlooked", "E300, E293, E289", "range, city, depth, grid phase"],
            ])
            + "<p>Principles: <b>differentiable</b> (soft steps, Sinkhorn / dual softmax instead of matching on the training path; the hard steps are named: the peak extraction, the group / rank bookkeeping, the delivered thresholds); "
              "<b>TIE = keep and refine</b> (a follow-up card takes the TIE variant as its control, guards of D79 stay); <b>no leak</b> (the driven flag and the visibility labels are labels, never inputs; GT ends are keypoint candidates in training only).</p>")

ARCH = ("<p>The new parts sit beside the lane head B (E273): they read the same frozen GaussianCaR features through the same pyramid, so B's nodes, links, sections and assignment stay valid and the delivered graph of B can be compared with and without the "
        "connectors of the pair head.</p>"
        + tbl(["block", "literature it builds on", "what we do", "what is different or open"], [
            ["Features", "GaussianCaR (ours), CRN, BEVFormer", "frozen, cached, 39 channels", "no published method trains lane heads on frozen cached BEV features"],
            ["Junction maps", "CenterNet (heat maps with a Gaussian target), HDMapNet (dense heads), MGMap (multi-scale)", "1 m: zone, conflict core, entry / exit heat with heading; 0.5 m: stop line, crossing; dilated residual blocks or one attention layer at 4 m", "the targets are made by location from the junction GT layer; the paint is a map-expansion label"],
            ["Keypoints", "CenterNet peak extraction", "3 x 3 peaks, soft-argmax position (differentiable in the heat), heading at the peak", "the index choice is the one hard step (D7 convention)"],
            ["Pair head", "TopoNet / TopoLogic / LaneSegNet (pair logits), SuperGlue (dustbin), MaGRoad / SAM-Road (path evidence)", "self-attention over the keypoints, 13 geometry features, rank features of the entry / exit groups (the order rule), evidence along the Hermite curve, BCE + dual softmax with a 'none' entry, soft crossing penalty", "no lane paper puts an order constraint on entry x exit; Sinkhorn with capacities is the next step (E302: 27 % of the entries have more than one exit)"],
            ["Pruning", "Soft-NMS, Matrix-NMS (SOLOv2)", "one-pass decay on the 2 m grid, R 1.75 m, sigma 0.1; special types protected at p 0.9", "no gradient to the stronger node's probability: the order is a comparison"],
            ["Time", "CenterTrack (prior heat), RVLD (warped previous output), StreamMapNet / MapTracker", "measured first (E290): the benefit at 0.5 s keyframes is unknown", "E299 only if the replay says go"],
            ["Map", "SMERF, P-MapNet, SMART, TopoSD, OrienterNet", "lane-level assignment (E273); a road-level map with lane counts as capacities (E291)", "no published count constraint on a BEV lane graph"],
        ]))

GT = ("<p>The layer is made on the lane side from the lane GT of GtGenerator (D80: no edit of the shared generator, derived data of this project). The nuScenes intersection polygons are not used as the junction; the zone is what the connectors "
      "actually sweep (the connector runs grown by 1.8 m, closed by 2 m); the <b>conflict core</b> is where paths of different pairs cross, merge or diverge. Entries and exits are the first and last points of the connector runs, clustered into "
      "<b>arms</b> (the two directions of a road) with a lane rank (0 = leftmost) and the number of lanes. A pair carries its arm pair, ranks, turn angle, and whether the ego drove it (tier 1, non-causal label). The stop lines and crossings "
      "of the nuScenes map expansion are added as painted evidence. All targets by location; nothing is predicted.</p>"
      + tbl(["measure (VAL)", "value"], [
          ["junction frames", f"{GATES['junction']} of {GATES['frames']}" if GATES else "-"],
          ["order rule: multi-pair maneuvers that are non-crossing", f"{GATES['mono']} of {GATES['multi']} ({GATES['ident']} are the exact identity, entry i to exit i)" if GATES else "-"],
          ["spread of the entries of an arm along its heading above 3 m", f"{GATES['spread_gt3']} of {GATES['arms']} arms" if GATES else "-"],
          ["maps decode back (peaks within 1 m of the GT end)", f"entries {f3(GATES['decode_recall_ent'])}, exits {f3(GATES['decode_recall_ext'])}" if GATES else "-"],
          ["conflict core / zone area (p10 / p50 / p90)", " / ".join(str(x) for x in JST2["core_over_zone_q"]) if JST2 else "-"],
          ["intersection share of the road area per frame (median)", "27 % (p90 41 %)"],
          ["entries / exits PAINTED (stop line other than a turn stop within 3 m, or a crossing within 4 m)", f"{f3(JPAINT['entries_painted_share'])} / {f3(JPAINT['exits_painted_share'])}" if JPAINT else "-"],
          ["out-degree of the entries (1 / 2 / 3 / 4) and share above 1", f"{JDEG['out_degree_hist'][1:]}, {f3(JDEG['out_degree_gt1'])}" if JDEG else "-"],
          ["in-degree of the exits and share above 1", f"{JDEG['in_degree_hist'][1:]}, {f3(JDEG['in_degree_gt1'])}" if JDEG else "-"],
          ["natural piece ends without an event within 1 m", "0 of 5755 (the GT flags are complete; the 25 - 33 % missing flags of E283 are the delivered head's)"],
          ["pairs the ego drives", "14 - 23 % depending on the sample"],
          ["l6 against l4 (440 VAL frames)", f"pairs equal {CMP['pairs_equal']} / {CMP['junction4']}, exits {CMP['ext_equal']} / {CMP['junction4']}, entries {CMP['ent_equal']} / {CMP['junction4']} (13 frames up to 0.46 m)" if CMP else "-"],
      ])
      + "<p>Reading: the order rule is not a hope but a property of the nuScenes lane graph (every multi-pair maneuver is non-crossing); a one-to-one transport would be wrong for a third of the ends; paint is missing for most entries "
        "(61 %), so a head has to infer them from context; the zone is about 2.6 times the conflict core. The pictures below are the GT layer on real frames.</p>")

MAPS = ("<p>The junction maps head reads the 1 m, 0.5 m and 2 m features of B's pyramid (frozen: the maps are read from B's own features) and predicts at 1 m the zone, the conflict core, an entry heat and an exit heat with a heading each, at 0.5 m the "
        "stop line and the crossing. Three context depths (none, four dilated residual blocks, dilated blocks plus one attention layer at 4 m) and one joint arm (the pyramid trained together with the head) were trained, two seeds each, 3000 steps. "
        "The control is B's own stop / exit event heat read the same way. The number of the card, <code>kp_f1</code>, is the mean of the entry and exit F1 at a tolerance of 3 m, threshold 0.3, distance only (B has no heading).</p>")
MAPS2 = ("<div class='note'>Reading: the heading is learned to about 3 degrees; the head does not depend on paint (painted and unpainted ends are found about equally often, B's event heat finds painted ends 17 - 20 points more often); "
         "the far range (beyond 35 m) and Singapore remain the weakest. Against the E284 end head (node tokens) the end recall is only a little higher: the plateau near 0.6 is the next target (context, joint pyramid, range-weighted loss).</div>")

PAIRS = ("<p>The pair head takes the keypoints of the maps (3 x 3 peaks, soft-argmax position), gives them tokens (the 1 m features of the maps head plus position, heading, probability), lets them attend to each other, and scores every geometrically possible "
         "(entry, exit) pair. Arm p0 is the E284 head on better pixels; p1 adds the order rule (rank features and a soft crossing penalty); p2 adds the evidence along the Hermite curve (lane-centre probability of B, zone, core, stop line, crossing); "
         "q2 is p2 on the maps of the joint arm. The rule control (groups and ranks, one connected exit group per turn class, lane matching by normalised rank) is scored on the same keypoints.</p>")
PAIRS2 = ("<div class='note'>Reading of the controls (strict one-to-one matching, the same keypoints for the rule and the head; the first version of the controls was too loose and was corrected after a critic pass): the rule delivers about three times as many pairs and is wrong for two thirds of them (precision 0.33 against the head's 0.74), but its recall (0.35) is higher than the head's (0.29) and the ceiling of the E284 keypoints is 0.42 - 0.43 (both ends of a pair found). "
          "The pair recall of a head cannot pass the end recall of its keypoints; the lever is the maps (E286) before the pair decision. The pair flip rate between keyframes is 20 % (E290).</div>")

PRUNE = ("<p>Three cheap measurements answered the questions about pruning and stability without a training run. E288: the one-pass Matrix-NMS decay (R 1.75, sigma 0.1) reproduces the hard close-node step of record "
         "on both seeds (a TIE, kept: it is differentiable and can sit inside the training graph, E292). E296: protecting the stop / exit / split / merge nodes from the suppression raises the junction recall by 1 - 4 points but not the junction F1: "
         "the delivered flags are the limit. E289: the nodes are locked to the 2 m grid along the line (after a 0.5 m shift the median node sits exactly where it was), a shift by half a cell makes one node in four lose its partner; the lines "
         "are stable (the line probe runs on the GPU).</p>")
PRUNE2 = ("<div class='note'>E293 (first 240 VAL frames, 3295 connector runs): the lane heat is 0.97 - 0.99 on the true connector and 0.97 - 0.98 on false candidate paths for the first 10 m: it does not say which path is the real one until about 10 m "
          "before the exit; the chord-tangent Hermite curve between the true ends is within 1 m for 99.5 % of the points up to 5 m depth but only 77 - 89 % deeper in (median error 7 cm, p90 1.1 - 2.3 m): points can be put into the junction as far as the "
          "pair is known and the curve shape is learned (handle scales, E303); there is no paint on the connectors.</div>")

LOOK = ("<p>Everything is rendered from real predictions on VAL frames (no ground truth in the forward path). Six panels per frame: the front camera with the GT lane graph projected; the GT layer (zone, core, stop lines, crossings); the GT ends, arms and "
        "ordered pairs; the head's entry / exit heat with its keypoints (filled = matched within 3 m); the head's zone / core / stop-line maps; the pairs the pair head delivers (green true, red false, blue dashed missed).</p>")

WEAK = ("<ul class='tight'>"
        "<li><b>End recall plateaus near 0.6</b>; beyond 35 m (53 % of the GT ends are there) it is 0.55 - 0.58, in Singapore 0.56 - 0.59. The frozen GaussianCaR features are weak at range; a joint pyramid, more context or a range-weighted loss are the levers.</li>"
        "<li><b>Pair recall is capped by the end recall</b>; the keypoints of E284 allow 0.43 (both ends found), the head delivers 0.29, the rule 0.36 at precision 0.34. 52 - 54 % of the true pairs were delivered in neither of two consecutive frames (E290).</li>"
        "<li><b>The chord-tangent curve</b> is good for 5 m only; learned handle scales (E303) are the fix, the delivered connectors use the chord tangent today.</li>"
        "<li><b>Protecting special nodes does not help</b> (E296) because the flags are only 48 - 49 % precise; the real lever is the flag precision of the maps.</li>"
        "<li><b>Nodes are grid-locked</b> along the line (E289): the vertex phase jumps while the line is stable; evaluate lines, not vertices, for stability.</li>"
        "<li><b>What might be overlooked:</b> occlusion does not matter for the ends (0.57 visible, 0.58 hidden), the range matters a lot and so does the paint (painted entries are found 8 - 12 points more often); left- and right-hand traffic differ (Boston 0.35, Singapore 0.24 pair recall for E284); the planner needs the pairs the ego drives (14 - 23 %) and the route, not all pairs; "
        "the keyframes are 0.5 s apart, so the literature's gain from the previous frame (10 - 30 Hz) is unproven here; the nuScenes lane graph has no mid-block splits / merges (E301) and no paint on the connectors, so some of the brief's cases have no ground truth.</li>"
        "<li><b>Corrected after an independent code review (2026-11-11):</b> (1) the rule control and the failure splits counted a GT end as found when ANY keypoint lay near it, which flattered the recall: with strict one-to-one matching the rule's precision is 0.34 (not 0.13), the "more lanes are easier" effect disappears, paint starts to matter; (2) exits were marked "painted" by the stop line of their own arm's entry side: now crossings only (18.5 % of the exits, not 28 %); (3) the rank features of the pair head used the injected ground-truth keypoints in training but only real peaks at evaluation: now real peaks in both; (4) the legacy polygon fill puts every zone / paint target half a cell off the road (36 % wrong cells): an exact mode exists (E306), the numbers of this page are on the legacy targets and say so; (5) both maps were compared at the same fixed keypoint threshold although they are calibrated differently: the scorer now also reports the best threshold per map.</li>
        <li><b>Honest negatives:</b> E301 has nothing to train; E296 and the first Matrix-NMS settings (sigma 0.5) were not better than the hard step; the averaging with the previous frame (E290) smooths by delivering fewer true pairs.</li></ul>")

NEXT = ("<ol class='tight'>"
        "<li>Read the night results by their written rules (E286 arms, E287 arms, the joint arm, the line probe) and refine every TIE (D82).</li>"
        "<li>E303: two learned handle scales per pair for the delivered connector curve (CPU oracle bound first). E298: weight the pairs the ego drives. E302: Sinkhorn with learned capacities.</li>"
        "<li>The range lever: range-weighted heat loss, the attention arm, the joint pyramid.</li>"
        "<li>E292: the Matrix-NMS layer inside the training graph (the torch layer is written and equals the numpy function). E297: a soft chord keep-gate as the differentiable output stage.</li>"
        "<li>E291: road-level map with lane counts as capacities; E299 only if the E290 replay on the new dump says go.</li>"
        "<li>Open for Mirko: D4 / D5, the soft step E279, the three literature digests to merge into the catalogue, TEST once for a finished design.</li></ol>")

VERSIONS = (tbl(["what", "version"], [
    ["lane ground truth", "GtGenerator lane l6 (store 9ad1d4ce30, 34,149 keyframes, merged 2026-11-11, D83; the experiments of the night started on l4 0702a065c6: the junction targets are the same on 440 VAL frames)"],
    ["code", "training/grid/{jgt, qjunction, jmaps, jends, qmnms}.py, training/data/prep/junction_paint.py, training/train/{jmaps_train, jends_train}.py, evaluation/{jmaps_eval, jends_eval, jrule_control, jsplits, readout_replay}.py, tools/viz/junction_view.py; git tags lcr/E286.*, lcr/E287.*"],
    ["cards", "experiments/E285 - E303 (rule before the run, results in the cards); plan experiment_execution/plans/junction_thread_2026-11-11.md; decisions D82, D83"],
    ["dataset", "nuScenes v1.0-trainval; 8414 training frames, 873 VAL frames; TEST not used"],
    ["checkpoints", "runs/e286_*, runs/e287_*; not in git"],
]) + "<pre>wsl bash tools/env/_py14.sh jmaps_eval.py score --run e286_j1_s0\nwsl bash tools/env/_py14.sh jends_eval.py dump --run e287_p2_s0   &&   ... score --run e287_p2_s0 --seed 0\nwsl bash tools/env/_py14.sh junction_view.py frame scene-0096:10 --out DIR --pred e286_j1_s0 --pairs e287_p2_s0 --cam\nwsl bash tools/env/_py14.sh build_junction_report.py --media</pre>")

FOOTER = "Every number on this page is read from the result files of the runs and of the CPU analyses named in the cards; nothing is typed by hand except the verdict words, which follow the written rules of the cards. Written 2026-11-11 / 12."
