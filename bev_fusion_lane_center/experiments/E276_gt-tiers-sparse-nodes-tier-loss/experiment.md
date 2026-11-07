# E276 Lane GT l4 (relevance tiers, camera visibility), a sparse node target and a tier-weighted loss for the lane head (baseline B, D58 / D59 / D60)

Card of 2026-11-07, written alone on Mirko's order (D60: "Plan all work for gt, do a test run, then generate a new gt version. Then refine the algo." and "Work without my presence"). It follows D59 (tiers in the GT and in the loss) and Mirko's notes on the result videos of E273 (HANDOVER, 2026-11-07): the own lane matters much more than the others, no demand where the camera cannot see, a wrong graph is worse than a missing one, the node density is too high, the endpoints and the intersections are weak. Priority D57. GPU only in Mirko's windows. No TEST. Nothing pushed.

**D55 status.** The head stays as E273 (soft up to the assignment, matching only by location in the targets). Nothing in this card adds a rule step to the forward path. The E275 read-out step is a separate card and not part of this one.

## 0. The card (what the tools run)
```json card
{
 "id": "E276",
 "card": "",
 "title": "tier-weighted loss (arm t) and tier-weighted loss on a sparse node target (arm s) for the E273 lane head, against the E273 baseline B on the l4 tier metrics",
 "kind": "block",
 "deterministic": false,
 "metric": "t12_F1_1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.04, "vs": "c"},
 "arms": {
  "c": {"recipe": "control B = E273 runs e273_p_s0 / e273_p_s1 (GT l3 training), scored on GT l4 with the tier metrics (no new training; the pyramid and node stage as trained)"},
  "t": {"recipe": "from the E273 phase-2 checkpoints (runs/e273_p_s<S>_p2/model.pt, the pixel stage is unchanged by this card) phases 3 4 5 6 7 as E273 with the tier switches: positives of the node losses x tier weight (3.0 / 1.5 / 1.0 for tier 1 / 2 / 3) x visibility weight (0.0 for a GT node that no camera sees), negatives of the existence loss x zone weight (2.0 / 1.5 / 1.0 / 1.0 for zone tier 1 / 2 / 3 / none), short-link loss x tier weight of the source node; targets of E273 (a node in every 2 m cell)"},
  "s": {"recipe": "as t, with the sparse node target: nodes of a piece only where the straight chord between kept nodes leaves a point of the piece by more than 0.3 m, at most 7.5 m between kept nodes, events and piece ends always kept (qtargets.node_table(sparse=(0.3, 7.5)))"},
  "g": {"recipe": "control of the assignment, as E273 / E274, scored inside every card"}
 },
 "control": "c",
 "variant": "t",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "t12_F1_1", "vs": "c", "min_effect": 0.02},
  {"metric": "t1_R_1", "vs": "c", "max_loss": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.05},
  {"metric": "lane_acc_err", "vs": "g", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01}
 ],
 "budget_gpu_h": 16,
 "kill": "a loss that is not finite, or a gate of phase 3 (node recall below 0.5 within 2 m) failing on a tier arm, or an arm that needs more than 1.5 x the E274 estimate (r: 2.4 h per seed for phases 2-7) for phases 3-7; or the tier labels of the l4 store fail a check of section 3 (stop before any training)",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (by hand, written now; `verdict276` compares the cards, the new tier numbers are not in PROBLEM.md, D59 point 3).**
- **t against c, KEEP** when the primary `t12_F1_1` (F1 at 1.0 m of the delivered graph against the GT lane points of tier 1 and 2 that a camera sees, section 2.4) gains more than +0.02 on BOTH seeds, and the guards hold: tier-1 recall `t1_R_1` not below c - 0.02, global topo F1 not below c - 0.05 (the price of not demanding unseen parts is reported, not forbidden), assignment `lane_acc_exact` t - g not below -0.01 and `lane_acc_err` t - g not below +0.03 (mean of the two seeds, as E274). **TIE** inside the band on both seeds, **CUT** when the gain is below -0.02 on a seed or a guard is broken, else INCONCLUSIVE (to Mirko).
- **s against t, KEEP** (the sparse target is adopted) when the DELIVERED NODES per 100 m of GT lane (`p.nodes.nodes_per_100m`; c: 31.5, seed 0; Mirko's "point density") fall by at least 25 % on both seeds (changed on 2026-11-07 evening, BEFORE any run, from `vert100_dp`: the oracle diagnostic P1 of section 7a shows that `vert100_dp`, the Douglas-Peucker vertices of the geometry, is the same for a perfect network under every node target, so it cannot judge a node target; it is reported) AND `t12_F1_1` stays within 0.02 below t on both seeds AND the guards hold: the assignment guards against g (as above) and `t1_R_1` not below t by more than 0.02 on either seed (no topo guard for this comparison). `t12_F1_1` more than 0.02 below t on a seed, or a broken guard, is CUT; no density gain with quality held is a TIE (t stays).
- **Reported, not judged:** every number split by tier and by visibility (node and point recall / precision, precision of delivered nodes in zone 1 / 2 / 3, false junction count in zone 1 / 2), the distribution of delivered nodes along the road, endpoint events P / R by tier, ms per frame, vertices per 100 m after Douglas-Peucker, links per node.
- **In-sample labels:** the weights (3.0 / 1.5 / 1.0, 0.0, 2.0 / 1.5 / 1.0) and the sparse parameters (0.3 m, 7.5 m) are first guesses written here before any run, not tuned on VAL; a second weight setting needs its own card.

**Predictions (the first two do not need training).** (1) Scored per tier, B is better in tier 1 than elsewhere: the F1 at 1.0 m of c in tier 1 + 2 (`t12_F1_1`) is at least 0.03 above its F1 over all GT points (the weakness is in the periphery). (2) Of the GT lane length in the window, 8 - 15 % is tier 1 + 2 and 35 - 60 % of the tier 3 length is not visible to a camera (l4 store, section 3: seen on 30 test keyframes: tier 1 + 2 about 14 %, visible 51 - 66 %). (3) t: `t12_F1_1` +0.04 over c on both seeds; global topo F1 -0.01 to -0.05. (4) s: `vert100_dp` 28.9 -> 20 or below with `t12_F1_1` within 0.02 of t. (5) The number of delivered nodes that lie in zone 1 and have no GT node within 2 m falls by at least 25 % in t (the weight of the false nodes works). (6) Endpoint events (end, start) in tier 1 improve in P and R by +0.05 or more in t (the weight at event cells inside tier 1; E274 weighted them everywhere and failed).
**Status after the CPU pre-experiments of section 7a (2026-11-07 evening, before any training run).** (1) hit at node level (B seed 0: node F1 tier 1 0.888 against tier 3 0.847, +0.04), the geometry-level number comes with the scoring pass; the gap is small, B is not much better in the own lane than elsewhere. (2) **missed**: tier 1 + 2 are 23.9 % of the GT lane length (16.3 + 7.6) and 32 % of the tier 3 length is not visible (30 % hidden by a box, 2 % outside every camera), measured on 1500 stored l4 frames. (4) **revised** (the original bet `vert100_dp` 28.9 -> 20 stands in the text above as written): the oracle shows `vert100_dp` is about 15 for a perfect network under any node target, so the arm s prediction is on the delivered nodes per 100 m: 31.5 -> 20 or below (-35 %), with `t12_F1_1` within 0.02 of t; whether `vert100_dp` also falls (fewer jittery nodes) is reported as a hypothesis, not a bet.

## 1. Why (what we know, VAL 873, E273 card seed 0 unless said)
- B: topo F1 0.596, junction F1 0.125 (P 0.07, R 0.65), end events P / R 0.36 / 0.35 (within 3 m), vertices per 100 m after DP 28.9 (budget 14.3, old champion 11.3), node F1 0.85, assignment `lane_acc` 0.98 per node (exact map), dustbin recall 0.49.
- E274 put MORE weight on events and intersections, i.e. on the parts that are often hidden; it was CUT (duplicates up, precision down). E275 (a hard read-out step) lifts the junction F1 in-sample: that is a rule on the output, this card is the trained alternative.
- The metrics of E273 count every lane of the 100 x 100 m window equally; nothing separates the own lane from a far side street and nothing excludes what no camera sees.

## 2. Design
### 2.1 GT l4 (autogt, `lane_tiers.py`; section 3 has the build)
Per point of every lane piece: `tier` 1 / 2 / 3, `vis`, `occ`, `tier_src`; per event `tier`, `vis`; per graph `ego_path`, `tier_info`. Tier 1 is the ego lane on the path the car really drives (the CAM_FRONT ego poses 25 m back and 55 m ahead of the keyframe, non-causal: labels only, never an input); where the car stands or the sequence ends the path goes on along the map (the straightest successor) to 55 m ahead. Tier 2: the left / right neighbours (road graph) of a tier-1 piece within 10 m of a tier-1 point. Tier 3: the rest (branches not taken, opposite lanes, far lanes). `vis`: at least one of the six cameras sees the ground point and no observed vehicle / pedestrian / two-wheeler box hides it (real height, the ground under a box is hidden). Parameters (tier_back 25, tier_ahead 55, path_tol 1.5, path_cos 0.7, neighbour_tol 10, min_ahead 40, min_back 20, extend, opposite_tier 3) are part of the lane signature. After the critic pass (2026-11-07, `ar-critic`, 14 findings, 4 blocking) the own lane BEHIND the car is extended along the map too where the car has driven less than `min_back` metres there (the start of a log, a car that has just started): without it the same own lane was tier 3 at the start of a log (L1 58 m against 84 m later in the same scene). The geometry helpers (camera projection, box hiding, category groups) are copies inside `lane_tiers.py`, so a change of the front GT files cannot change the labels without changing the lane signature. The l4 signature is **0702a065c6** (an earlier draft store `c96c18fb1f`, built before the critic fixes, is stale and is not used).
### 2.2 Targets (`training/grid/qtargets.py`, `training/data/lane_gt_data.py`)
`node_table` gets the tier and the DEMANDED flag of every node (the tier and flag of the node's own point in its piece; the event tier stored in the l4 events is not used here). Demanded = a camera sees the point, or it lies within `NEAR_BLIND` = 8 m of the ego and no box hides it (`occ` false): the cameras see no ground within about 5 m of the car, and with the camera flag alone the first 5 m of the own lane would not have been demanded and would have been invisible to the `t1_R_1` guard (critic finding 3). In the scorer and the loss `vis` always means this demanded flag; `sparse=None | (eps, max_chord)` (arm s) and `grid_targets` three more maps on the 50 x 50 grid: `tier` (node tier, 0 = no node), `vis`, `ztier` (the lowest tier of the GT lane points within the cell dilated by 1 m, 0 = no lane there). No other target changes.
### 2.3 Loss (`training/grid/qgraph.py`, `CFG`, defaults = the E273 loss)
`tier_w` (positives of the existence loss and of flags, offset, direction, section start, long-link endpoints at a GT node of that tier), `vis_w` (extra factor for GT nodes that no camera sees; 0 = not demanded), `neg_w` (negatives of the existence loss by zone tier: a false node costs more near the own lane), `link_tier_w` (short-link loss by the tier of the source node). Trainer flags `--tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1 --sparse 0.3 7.5`. Normalisation: the node terms and the short-link term are divided by the UNWEIGHTED count of GT nodes / positive pairs (as in E273), the long-link term by its own weight sum; so the tier weights also raise the scale of the node and link losses against the assignment and pixel terms in phase 7 by about 1.3 - 1.5 (the weights are the intended difference between t and c, this scale effect comes with them and is not separated by a control). Gates: with a tier or sparse arm the phase-3 gate drops its F1-against-E271 condition (it counts recall against all GT points: an arm that does not demand unseen nodes fails it by design) and keeps the floors (node precision >= 0.75, node recall >= 0.5 within 2 m, median position error <= 0.5 m); the trainer refuses an incomplete lane GT store for these arms.
### 2.4 Tier metrics (`evaluation/qgraph_eval.py`, the scorer; nothing in PROBLEM.md changes)
The delivered graph's polyline points are matched to the GT lane points of the frame (KD-tree, 1.0 m, as the F1@1.0 of the scorer) and every GT point carries its tier and `vis`. Recall `t<k>_R_1` = the share of the visible GT points of tier k that lie within 1.0 m of a delivered polyline point (`t<k>_R_1all` counts the invisible ones too). Precision `t<k>_P_1` per zone k: every delivered polyline point goes to the zone of its nearest GT point within 2.5 m (no GT point within 2.5 m: zone 0, off the lanes) and counts as right when that GT point is within 1.0 m; P = right points of the zone / all points of the zone. `t12_F1_1` = the F1 of recall pooled over tiers 1 + 2 (visible) and precision pooled over zones 1 + 2. These are geometry numbers, so a dense and a sparse node target are scored the same way. Node-level numbers (`node_P` / `node_R` by tier, against the dense GT nodes for c and t, against the sparse GT nodes for s) are reported next to them.

## 3. Work plan, gates, schedule (written 2026-11-07 ~19:30; the owner is the lane chat)
| step | what | gate | when |
|---|---|---|---|
| 1 | `autogt/lane_tiers.py` + the hook in `lane_gt.py`, developed in a copy (`temp/gt_l4`) so that the live lane signature (l3) stays readable while the E275 scoring runs | self-test (toy lane: path, extension, neighbours, opposite lane, crossing connector, standing car, boxes front and rear) | done 11-07 |
| 2 | test run on 30 real trainval keyframes (6 scenes): geometry equal to the stored l3 on every keyframe, tier lengths, visibility shares, top views | geometry identical (30 / 30), tier-1 length 72 - 117 m (mean 94 m), no empty tier 1, visible share of tier 1 0.60 | done 11-07, rerun on the final code after the critic pass (`temp/gt_l4/out/topviews.png`) |
| 2b | critic pass (`ar-critic`, mode code): 14 findings, 4 blocking: the autogt selftest needs `lane_tiers.py` in `NOT_GENERATOR`; the own lane behind the car (backward extension); the blind zone at the car (demanded flag, section 2.2); the stale test-run evidence. All four fixed, the rest taken or written down (sections 2.3, 6) | the draft store `c96c18fb1f` was stopped at 19,038 files and is stale | done 11-07 19:30 |
| 3 | the full l4 store: 34,149 keyframes, one thread at nice 15, about 1 h | the index says complete, 34,149 held | running from 19:36 into `data/autogt/lane/0702a065c6` |
| 4 | after SCORED: merge the copy into `autogt/` (the files are byte-identical, so the signature is the one generated), `lane_versions.csv` row l4 (already in the copy), tag `autogt/lane-l4`, `releases.csv` v11, AGENTS.md / README rows; the lane tests of autogt | the live signature equals the store's, `lane_graph_selftest` ok, geometry check against l3 on 500 random keyframes of the store | after the E275 scoring (~23:30) |
| 5 | head code: `qtargets` (tier, vis, ztier, sparse), `lane_gt_data`, `qgraph` loss switches, trainer flags, `qgraph_eval` tier metrics and the sparse GT nodes, `verdict276`; tests for each (known answers on the toy lane, the defaults reproduce the E273 loss bit for bit, a switch changes the loss in the intended direction, the tier metrics on the oracle = 1) and a critic pass | tests green, defaults reproduce E273 numbers on 3 stored batches | 11-07 night |
| 6 | baseline per tier: the dumps of the E275 scoring (`temp/e275/pred_*.pkl`, nodes of B s0 / s1 with e >= 0.2) scored on l4 (node level, CPU) and a card scoring pass of B with the tier metrics (GPU, 25 min per seed) | prediction 1 and 2 answered | after step 4 / 5 |
| 7 | GPU arms through `run.py` with a window launcher: t_s0, t_s1, s_s0, s_s1 (phases 3 - 7 from the E273 phase-2 checkpoints, about 2.5 h each) | gates of the trainer, kill criterion | from the next window after step 5 |
| 8 | verdicts by `verdict276`, ledger rows, section 7 of this card, `results/E276/README.md`, report section (E273 report v5 lacks E274 and E275 too) | numbers from the cards | after step 7 |

## 4. Choices made alone (D60), each a default Mirko can change
- Tier 2 = same-direction neighbours only; the opposite lane is tier 3 (`opposite_tier` 3). Tier 1 includes the lane behind the car (25 m). The alternative branches at a fork that are not driven are tier 3.
- Visibility is the camera only (six cameras, nominal rig calibration), boxes with lidar / radar points hide, static structures do not (the nuScenes map has no buildings); radar and range limits are not part of `vis`.
- The weights and the sparse parameters of section 0. `vis_w` = 0.0 means invisible GT nodes are not demanded, but a false node near them still costs (negatives are weighted by zone, not by visibility).
- The primary metric is geometric (F1 at 1.0 m on tier 1 + 2, visible), because it is the same for dense and sparse targets and does not depend on how a head places nodes.
- Both arms start from the E273 phase-2 checkpoints (the pixel stage is not touched by this card, it saves about 1 h per run and keeps the arms comparable); the E274 arm r trained all phases, so the cost of phases 3 - 7 is estimated from its log, not measured for this card.

## 5. Controls and checks
- c = B on the same GT l4 and the same scorer; the tier metrics of B s0 / s1 are the baseline per tier. g, the assignment scorer and the pairing are those of E273 / E274.
- Layer 0: tests of section 3 step 5; the oracle (GT through the chain) scores 1 on every tier metric; a shuffled tier map must change the loss. Layer 1: with all switches at their defaults the loss of 3 stored batches equals the E273 loss to 1e-6 (checks that nothing else moved). Layer 2: a null prediction (no nodes) scores 0 and the weights change nothing on it.
- The geometry of l4 equals l3 (so B's stored numbers on l3 are valid on l4), checked on the test keyframes and on 500 random keyframes of the store.

## 6. Risks and what would change the plan
- **Arm s learns an ambiguous target** (critic finding 10): no local rule says which cell on a straight lane carries the kept node (the choice follows the Douglas-Peucker split and the piece, which the network cannot see). With the false-node weight of 2 near the own lane the existence logit may sink along straights, and delivered nodes there vanish. The card's kill criterion catches it only through the phase-3 / 4 floors; the verdict then reads CUT through `t12_F1_1` (recall in the own lane), which is the honest outcome of this first guess. A softer sparse target (the dropped dense cells as "do not care" negatives) is the next variant if s fails only for this reason.
- **Tier weights and loss scale are confounded** (section 2.3). The comparison t against c is "the written weights, with the scale effect they bring"; a control with the same weights on all tiers (1 / 1 / 1) would separate them and is not in this card.
- The sparse target needs the link head to read nodes up to 7.5 m apart: still inside the 8 m short-link neighbourhood, so no long links are added for straight stretches; if the short-link precision falls below 0.7 in the phase-4 gate, arm s is dropped.
- `vis_w` = 0 lowers the global recall of unseen nodes by construction (topo F1 falls); that is why the topo guard is -0.05 and not -0.01. If the fall is larger the card reports it and Mirko decides whether global metrics matter.
- The visibility flag is a geometric estimate (boxes with real heights, nominal calibration); a wrong flag changes which nodes are demanded, not the ground truth geometry.
- Training cost: 4 runs of about 2.5 h each are two nights of the window. The first arms start at the next window after the code is ready; if the code is not ready tonight the launcher waits for the following window.

## 7. Summary
(open: filled when the arms are scored)

### 7a. CPU pre-experiments of 2026-11-07 (no model, no GPU; diagnostics, not verdicts; scripts are scratch in `temp/e276/`, kept in the git tag `lcr/e276-scratch`, promoted when a second card uses them)
**P1 - what the node targets themselves allow** (`exp_density_oracle.py`: a perfect network decoded from each node target, read out and scored by `qgraph_eval`, VAL 873 frames; an oracle is a sanity check, not an upper bound).

| node target | nodes per 100 m GT lane | vert100_dp (budget 14.3) | topo F1 | F1@1.0 | junction F1 | lanes found | Q |
|---|---|---|---|---|---|---|---|
| dense, a node in every 2 m cell (E273) | 33.0 | 11.4 | 0.9669 | 0.9957 | 0.9535 | 0.9853 | 0.9842 |
| sparse 0.3 m / 7.5 m (arm s) | 14.4 | 11.5 | 0.9669 | 0.9957 | 0.9535 | 0.9853 | 0.9842 |
| sparse 0.3 m / 15 m | 9.2 | 11.4 | 0.9668 | 0.9957 | 0.9535 | 0.9853 | 0.9842 |
| sparse 0.6 m / 7.5 m | 14.1 | 11.4 | 0.9669 | 0.9957 | 0.9535 | 0.9853 | 0.9842 |
(1) The sparse target keeps the geometry: every graph number of the oracle is the same to the fourth decimal, with 56 % fewer nodes (72 % at 15 m chords). (2) `vert100_dp` is 11.4 for a perfect network whatever the node target: it measures the complexity of the GEOMETRY (the old champion has 11.3), not the nodes. B's 28.9 is 2.5 times that: the excess 17 vertices per 100 m are jitter and duplicates in the delivered geometry, not the node grid. B delivers 31.5 nodes per 100 m against 33.0 of its dense target. So "the point density is very high" has two parts: the node count (the target is dense by construction; arm s removes 56 % of it) and the geometry noise (only a cleaner output lowers `vert100_dp`; the tier loss and the node precision are the levers). The rule of arm s was changed accordingly (section 0).

**P2 - what the l4 labels look like** (`exp_tier_stats.py`, 1500 random frames of the first 14,000 stored l4 frames; the full store repeats it): tier 1 is 16.3 % of the GT lane length (103 m per frame), tier 2 7.6 % (48 m), tier 3 76.1 % (479 m). Visible to a camera: tier 1 69 %, tier 2 68 %, tier 3 68 %; hidden by a box 20 / 25 / 30 %; outside every camera 11 / 8 / 2 %. Tier 1 by range from the car: 0 - 3 m 0 % camera-visible (the blind zone, 6.9 m per frame, demanded by the 8 m rule), 3 - 5 m 45 %, 5 - 8 m 72 %, 8 - 12 m 88 %, 20 - 35 m 74 %, 35 - 50 m 71 %: beyond the lead vehicle the own lane is hidden by a box in 26 - 32 % of its length, which the loss does not demand (the cost of "no demand without visibility"; a memory over frames is the lever for these parts). Mean driven path 18.5 m behind and 34.6 m ahead (the poses of the log are shorter than the 25 / 55 m asked); the map extension was used ahead in 47 % of the frames and behind in 35 %.

**P3 - baseline B per tier, node level** (`analyse_tiers.py` on the E275 node dump of E273 seed 0, e >= 0.5, node matching within 2 m, one-to-one, demanded GT nodes; seed 1 follows with the scoring pass):

| tier of the GT node / zone of the delivered node | GT nodes | share | demanded | recall (demanded) | delivered nodes in the zone | precision |
|---|---|---|---|---|---|---|
| 1 | 30,278 | 19.2 % | 82 % | 0.879 | 29,496 | 0.898 |
| 2 | 15,959 | 10.1 % | 82 % | 0.918 | 15,964 | 0.902 |
| 3 | 111,312 | 70.7 % | 70 % | 0.815 | 102,472 | 0.882 |
| off the lanes (no GT lane within 2.5 m) | | | | | 2,373 | 0.000 |
B is only a little better in its own lane than elsewhere (node F1 0.888 in tier 1 against 0.847 in tier 3): about 12 % of its own-lane nodes are missed and about 10 % of the delivered nodes in the own-lane zone have no GT node within 2 m (3.4 false nodes per frame). The weakness in topology and junctions (topo F1 0.596, junction precision 0.07) is therefore not a matter of the own lane versus the periphery at node level; the tier loss can move the own-lane numbers, the junction numbers need the links.

**P4 - how good is the map-extension rule of the tier-1 label** (`exp_extension_accuracy.py`: stored frames where the car drove at least 54 m ahead and 24 m behind (256 frames), the driven path cut and the label redone by the extension rule, compared with the driven label): extension ahead from 10 / 20 / 30 m of driven path: tier-1 recall 0.935 / 0.921 / 0.950, precision 0.947 / 0.946 / 0.965; 8 - 14 % of the frames with recall below 0.8 (a wrong branch at a junction). Behind (5 m driven): recall 0.926, precision 0.834 (the backward extension adds whole pieces: more than the 25 m asked); both cut: recall 0.859, precision 0.784, 26 % of the frames below 0.8. Reading: the extension labels the own lane right in about nine of ten frames; it over-labels behind the car; the clip of the extension to the asked length is a candidate for a later GT version (l5), not worth a regeneration now.

**P5 - temporal consistency of the delivered nodes** (video note 3; `exp_temporal.py`, the E275 node dump of B seed 0, the poses of the VAL file, the l4 store; 851 pairs of consecutive keyframes 0.5 s apart; the nodes of the previous frame are moved by the ego motion (the convention of the pose file is found by the GT nodes: the pose acts on the lane frame, median alignment 0.49 m) and count as persistent when a node lies within 2 m in the keyframe):

| group | GT nodes persistent | delivered nodes persistent | excess flicker (GT - delivered) |
|---|---|---|---|
| all | 0.999 | 0.975 | +0.024 |
| own lane (zone 1) | 0.999 | 0.987 | +0.012 |
| far lanes (zone 3) | 0.999 | 0.974 | +0.026 |
| near an intersection (within 3 m of an in-intersection connector) | 0.998 | 0.947 | +0.051 |
| elsewhere | 1.000 | 0.983 | +0.017 |
| own lane near an intersection | 0.997 | 0.957 | +0.040 |
| own lane elsewhere | 1.000 | 0.994 | +0.006 |
Mirko's observation is real and local: near intersections the delivered nodes flicker about three times as much as elsewhere (5.3 % against 1.7 % do not persist), and in the own lane near an intersection seven times (4.3 % against 0.6 %). At node level the flicker is small; the full dump of the scoring pass allows the same count for links and sections (the graph structure, where the videos looked worst).

**P6 - endpoints** (video note 4; `exp_endpoints.py`, B seed 0, the demanded GT events of the 873 VAL frames against the delivered nodes with the flag of that event type):

| GT event | n | flagged node within 3 m | within 6 m | any node within 3 m | median error of a flagged node within 6 m |
|---|---|---|---|---|---|
| end (a lane ends in the window) | 321 | 0.296 | 0.417 | 0.667 | 1.42 m |
| start | 339 | 0.469 | 0.599 | 0.664 | 1.09 m |
| split | 1520 | 0.584 | 0.670 | 0.918 | 0.89 m |
| merge | 1777 | 0.638 | 0.754 | 0.926 | 0.96 m |
| exit | 3914 | 0.688 | 0.816 | 0.913 | 0.97 m |
| stop | 3798 | 0.741 | 0.862 | 0.924 | 0.88 m |
| all events, tier 1 / 2 / 3 | 2153 / 1291 / 8225 | 0.713 / 0.695 / 0.650 | 0.819 / 0.851 / 0.766 | 0.939 / 0.887 / 0.900 | 0.84 / 0.82 / 0.98 m |
The position is not the problem where a node exists (median error 0.9 - 1.4 m): the weak points are the FLAG (a node is there, 91 - 93 % of the events, but only 58 - 74 % carry the right event flag within 3 m) and, for the lane ends and starts, the node itself (one third have no delivered node within 3 m). All demanded end events are in tier 3: no own-lane or neighbour-lane end occurs in the VAL frames, so the E276 weights (tier 1 / 2) cannot help the ends; they can help the flags of stops, exits, splits and merges in the own lane (0.71 against 0.65).

**P7 - the intersection label** (video note 1; `exp_intersection_label.py`, the l4 store, 873 VAL frames): only 77.3 % of the connector points that the lane graph marks as inside an intersection lie inside the intersection polygons (84.3 % within 1 m, 88.6 % within 2 m, the dilation of the head's target); 6.2 % of the polygon area has no lane within 6 m; 113 of 2600 polygons have no node id (no connector in them); the stop / exit points lie a median 1.35 m and p90 6.4 m before the polygon (6.7 % inside). The label and the lane graph disagree for 11 - 23 % of the connector points: part of the "intersection label did not always look good" is the label, not the head. Sent to the GT chat as a wish for a later lane version (an `intersection_lane` area built from the connectors). The pixel head itself scores F1 0.892 on the grown intersection mask (E274 scratch `pixel_eval.py`, 300 VAL frames, 1-cell tolerance).

**What the pre-experiments change in this card:** prediction 2 missed (section 0, status), prediction 4 revised and the s rule moved to the delivered nodes per 100 m (section 0), the label-noise estimate of P4 and the dense-vs-sparse equality of P1 are the evidence for the GT and the target choices of sections 2.1 and 2.2.
