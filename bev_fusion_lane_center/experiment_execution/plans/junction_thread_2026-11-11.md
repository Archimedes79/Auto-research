# Plan of the junction thread (D82 / D83, 2026-11-11): from Mirko's design brief to experiments

Written by the lane chat. The brief (Mirko, 2026-11-11): recognise on a 1 m grid the lane direction, the junction entry / exit and the junction area, splits / merges outside junctions; then 2 m nodes with a type and a position uncertainty; prune nodes systematically, special types never; connect lane nodes, entries to exits (entry 1 -> exit 1, entry 2 -> exit 2), splits, merges; a probability for every entry-exit pair with splines, driven connections weighted higher; cross-attention over time; then the map. Plus: differentiable throughout; GT; literature; a road-level map with lane counts; what is overlooked. **Rules: TIE = keep and refine, never worse (guards of D79); everything on the training path soft (D55); VAL only, TEST never; GPU in the windows only.**

## 1. The ladder, one row per bullet of the brief (status 2026-11-11 23:00)
| # | brief | experiment | state / result |
|---|---|---|---|
| 1 | GT with everything (zone too large) | **E285** junction GT layer (zone, conflict core, arms, ranks, pairs, paint, painted, vis); **D83** merge of l6 and the l6 lane store | done; gates 873 frames; l6 store generating, merge after the night chains |
| 2 | 1 m maps: direction, entry / exit, junction area, paint | **E286** j0 / j1 / j2 (+ joint jj) on B's frozen pyramid | training (22:00 -> 00:30), scoring after |
| 2b | split / merge outside junctions on the 1 m maps | **E301** split / merge heat + heading added to the junction maps (planned, needs the maps code free) | card |
| 2c | "then no 2 m grid is needed" | **E294** nodes read from the 1 m maps only (E271 sparsification + E286 maps) against the 2 m node head | card (GPU, day) |
| 3 | 2 m nodes with type + direction | E273 node head (flags, heading) = B | exists |
| 3b | position uncertainty per node | **E295** sigma head (beta-NLL) + calibration + variance-vote read-out; after E289 step 1b says what is unstable | card |
| 4 | prune (matrix operation), special types never | **E288** Matrix-NMS TIE (= hard step, differentiable); **E296** the same with the special types protected (replay); **E292** the layer inside the training graph; **E297** keep-gate by chord distance (soft DP) | E288 done, E296 running, E292 code ready, E297 card |
| 5 | connect: lane nodes, entry -> exit with the ORDER, splits, merges | **E287** pair head v2 (rank features, crossing penalty, evidence), rule control, **E302** capacity / one-to-many assignment (Sinkhorn with learned row masses: 27 % of entries have > 1 exit) | E287 training queued; rule control done; E302 card |
| 5b | probability per entry-exit pair, splines, driven weighted higher | E287 output; **E298** loss weight for the pairs the ego drove (GT `driven`) and the route prior at inference | card |
| 6 | cross-attention over time | **E290** pair flips (20 %, the average cuts them 44 % but loses true pairs); **E299** one buffered frame: ego-warped prior heat / cross-attention on the keypoints (code needed) | measured; E299 card |
| 7 | link to the map pieces | E273 assignment (lane-level map, +0.071 with a map error); **E291** road-level map with lane counts (SD level: arms + counts, no connectors) | card |
| 8 | where does it fail (overlooked) | **E300** splits by driven / city / lanes / turn / painted / visible / range | E284 running |
| 9 | how far into the junction | **E293** depth probe | preliminary: evidence true = false paths for 15 m; Hermite from true ends within 1 m for 86 - 100 % |

## 2. What might be overlooked (the brief concentrates on the pair; these are the other directions)
1. **End recall is the ceiling**: the rule control on the E284 keypoints reaches pair recall 0.37 against the head's 0.29: two ends are found for about 0.34 - 0.38 of the true pairs. The pixel / end stage (E286) decides before the pair head can.
2. **Occlusion**: a car standing on the stop line hides the entry (the l4 `vis` / `occ` labels exist): E300 splits by visibility; a hidden end can only be inferred (map, time, the other lanes of the arm).
3. **What the planner needs**: not all pairs, the ones on the route and the ones the ego drives (14 % of the pairs): E298 / E300 report driven-pair recall; a closed-loop metric that responds is still open.
4. **Left- and right-hand traffic**: Boston vs Singapore: the turn rules mirror; the order rule is universal (non-crossing); E300 splits by city.
5. **Partial junctions at the window border**: connectors cut by the 100 m window give ends without partners; the l6 events (`partial`) and the evaluation by range.
6. **Grid aliasing**: the nodes are locked to the 2 m grid along the line (E289); the lines are stable: the delivered curve, not the node, should be what is evaluated for stability; E289 step 1b.
7. **The GT is not the truth**: the nuScenes map has no paint inside junctions (E293), stop lines are waiting points as well as entries (turn stops), connectors are arcs: a head that matches the map's idea is not a head that sees; the SD-level input (E291) tests how much of the structure the map must carry.
8. **Time at 2 Hz**: keyframes are 0.5 s apart (the literature uses 10 - 30 Hz): the benefit of the previous frame must be measured before it is built (E290 / E299).
9. **Differentiability is not gradient health**: every new module has a gradient report (selftests check that no parameter is dead); the named hard steps are the peak extraction (index choice), the group / rank bookkeeping and the delivered thresholds; the Matrix-NMS gives no gradient to the stronger node's probability (only the order).
10. **Budget**: the delivered graph must stay at <= 14.3 vertices per 100 m after the output stage; E288 / E296 keep the vertex guard.

## 3. Order of the work and the GPU
Night 2026-11-11 (22:00 - 07:00): E286 (6 runs) -> E287 (6 runs) -> joint arm + q2 (E286.jj / E287.q2) -> E289 line probe; CPU alongside: E288 / E293 / E296 / E300 replays, the l6 store. Thursday day window (08:30 - 17:30): E294, E295, E292 training (needs the merged code), E298, then E299. Merge of GtGenerator about 05:00 (after the chains), the loaders switch to l6 afterwards; runs on l4 stay valid records.
Rules for every card: written rule before the run, two seeds for a trained arm, VAL 873, the controls named in the card, a TIE is kept and refined in a follow-up card whose control is the TIE variant.
