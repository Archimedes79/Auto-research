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
**Status after the CPU pre-experiments of section 7a (2026-11-07 evening, before any training run).** (1) hit at node level (B seed 0: node F1 tier 1 0.888 against tier 3 0.847, +0.04), the geometry-level number comes with the scoring pass; the gap is small, B is not much better in the own lane than elsewhere. (2) **missed**: tier 1 + 2 are 23.2 % of the GT lane length (16.1 + 7.1) and 34 % of the tier 3 length is not visible (32 % hidden by a box, 2 % outside every camera), measured on 2995 frames of the complete l4 store (the visible share is one point below the bet range). (4) **revised** (the original bet `vert100_dp` 28.9 -> 20 stands in the text above as written): the oracle shows `vert100_dp` is about 15 for a perfect network under any node target, so the arm s prediction is on the delivered nodes per 100 m: 31.5 -> 20 or below (-35 %), with `t12_F1_1` within 0.02 of t; whether `vert100_dp` also falls (fewer jittery nodes) is reported as a hypothesis, not a bet.

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
`tier_w` (positives of the existence loss and of flags, offset, direction, section start, long-link endpoints at a GT node of that tier), `vis_w` (extra factor for GT nodes that no camera sees; 0 = not demanded), `neg_w` (negatives of the existence loss by zone tier: a false node costs more near the own lane), `link_tier_w` (short-link loss by the tier of the source node; `vis_w` also applies to the POSITIVE candidate links out of an unseen node: a missing real link there is not demanded, a false link out of it keeps its tier weight, as for the nodes). Trainer flags `--tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1 --sparse 0.3 7.5`. Normalisation: the node terms and the short-link term are divided by the UNWEIGHTED count of GT nodes / positive pairs (as in E273), the long-link term by its own weight sum; so the tier weights also raise the scale of the node and link losses against the assignment and pixel terms in phase 7 by about 1.3 - 1.5 (the weights are the intended difference between t and c, this scale effect comes with them and is not separated by a control). Gates: with a tier or sparse arm the phase-3 gate drops its F1-against-E271 condition (it counts recall against all GT points: an arm that does not demand unseen nodes fails it by design) and keeps the floors (node precision >= 0.75, node recall >= 0.5 within 2 m, median position error <= 0.5 m); the trainer refuses an incomplete lane GT store for these arms.
### 2.4 Tier metrics (`evaluation/qgraph_eval.py`, the scorer; nothing in PROBLEM.md changes)
The delivered graph's polyline points are matched to the GT lane points of the frame (KD-tree, 1.0 m, as the F1@1.0 of the scorer) and every GT point carries its tier and `vis`. Recall `t<k>_R_1` = the share of the visible GT points of tier k that lie within 1.0 m of a delivered polyline point (`t<k>_R_1all` counts the invisible ones too). Precision `t<k>_P_1` per zone k: every delivered polyline point goes to the zone of its nearest GT point within 2.5 m (no GT point within 2.5 m: zone 0, off the lanes) and counts as right when that GT point is within 1.0 m; P = right points of the zone / all points of the zone. `t12_F1_1` = the F1 of recall pooled over tiers 1 + 2 (visible) and precision pooled over zones 1 + 2. These are geometry numbers, so a dense and a sparse node target are scored the same way. Node-level numbers (`node_P` / `node_R` by tier, against the dense GT nodes for c and t, against the sparse GT nodes for s) are reported next to them.

## 3. Work plan, gates, schedule (written 2026-11-07 ~19:30; the owner is the lane chat)
| step | what | gate | when |
|---|---|---|---|
| 1 | `autogt/lane_tiers.py` + the hook in `lane_gt.py`, developed in a copy (`temp/gt_l4`) so that the live lane signature (l3) stays readable while the E275 scoring runs | self-test (toy lane: path, extension, neighbours, opposite lane, crossing connector, standing car, boxes front and rear) | done 11-07 |
| 2 | test run on 30 real trainval keyframes (6 scenes): geometry equal to the stored l3 on every keyframe, tier lengths, visibility shares, top views | geometry identical (30 / 30), tier-1 length 72 - 117 m (mean 94 m), no empty tier 1, visible share of tier 1 0.60 | done 11-07, rerun on the final code after the critic pass (`temp/gt_l4/out/topviews.png`) |
| 2b | critic pass (`ar-critic`, mode code): 14 findings, 4 blocking: the autogt selftest needs `lane_tiers.py` in `NOT_GENERATOR`; the own lane behind the car (backward extension); the blind zone at the car (demanded flag, section 2.2); the stale test-run evidence. All four fixed, the rest taken or written down (sections 2.3, 6) | the draft store `c96c18fb1f` was stopped at 19,038 files and is stale | done 11-07 19:30 |
| 3 | the full l4 store: 34,149 keyframes, one thread at nice 15, about 1 h | the index says complete, 34,149 held | done 11-07 20:51 into `data/autogt/lane/0702a065c6`; audited by the GT chat (34,149 of 34,149 frames load, per-point arrays consistent, no NaN, geometry equal to l3 on EVERY frame, 443 frames = 1.3 % without a tier-1 lane, 47 of them without any lane) |
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
- Layer 0: tests of section 3 step 5; the oracle (GT through the chain) scores 1 on every tier metric; a shuffled tier map must change the loss. Layer 1: with all switches at their defaults the node loss equals the committed E273 loss bit for bit on a fixed toy batch (`test_tier_loss.py` holds the frozen value 60.220611572265625, checked against the committed `qgraph.py`, and the review chat re-checked node loss, link loss and targets independently on random batches); "3 stored batches" in the first version of this card was not built. Layer 2: a null prediction (no nodes) scores 0 and the weights change nothing on it.
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

**P2 - what the l4 labels look like** (`exp_tier_stats.py`, 2995 random frames of the COMPLETE store, 34,149 frames, geometry equal to l3 in all 1360 frames checked): tier 1 is 16.1 % of the GT lane length (102 m per frame), tier 2 7.1 % (45 m), tier 3 76.8 % (485 m). Visible to a camera: tier 1 66 %, tier 2 63 %, tier 3 67 %; hidden by a box 23 / 29 / 32 %; outside every camera 11 / 8 / 2 %. Tier 1 by range from the car: 0 - 3 m 0 % camera-visible (the blind zone, 6.9 m per frame, demanded by the 8 m rule), 3 - 5 m 43 %, 5 - 8 m 70 %, 8 - 12 m 85 %, 20 - 35 m 69 %, 35 - 50 m 69 %: beyond the lead vehicle the own lane is hidden by a box in 31 - 33 % of its length, which the loss does not demand (the cost of "no demand without visibility"; a memory over frames is the lever for these parts). Mean driven path 18.1 m behind and 32.6 m ahead (the poses of the log are shorter than the 25 / 55 m asked); the map extension was used ahead in 50 % of the frames and behind in 35 %; 37 of 2995 frames have no tier-1 lane (no map lane near the path), 5 have no lane at all.

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

**P3, seed 1** (E275 close-out, `analyse_tiers.py` on the node dump of E273 seed 1, e >= 0.5): recall of the visible GT nodes tier 1 0.867, tier 2 0.929, tier 3 0.815 (seed 0: 0.879 / 0.918 / 0.815); node precision of the delivered nodes by zone 0.899 / 0.901 / 0.875 (seed 0: 0.898 / 0.902 / 0.882); 2,432 delivered nodes off the lanes (seed 0: 2,373). Same picture on both seeds: B is not much better in its own lane than elsewhere at node level.

**P5b - temporal consistency of the LINKS** (video note 3, the structure; `exp_temporal_links.py`, the full dump of B, 851 pairs of consecutive keyframes, both ends of a link of the previous keyframe moved by the ego motion; PERSISTENT when the keyframe has a link of the same graph with both ends within 2 m; delivered links p >= 0.5 with both ends e >= 0.5; GT links are the ceiling; seed 0 / seed 1):
| group | GT links | delivered links | excess flicker (GT - delivered) |
|---|---|---|---|
| all | 0.998 | 0.933 / 0.938 | +0.066 / +0.061 |
| short links | 0.999 | 0.958 / 0.960 | +0.041 / +0.039 |
| **long links** | 0.980 | **0.640 / 0.652** | **+0.340 / +0.328** |
| own lane (zone 1) | 0.998 | 0.957 / 0.955 | +0.041 / +0.043 |
| neighbours (zone 2) | 0.999 | 0.964 / 0.965 | +0.035 / +0.034 |
| rest (zone 3) | 0.998 | 0.927 / 0.936 | +0.071 / +0.062 |
| near an intersection | 0.996 | 0.845 / 0.857 | +0.151 / +0.138 |
| elsewhere | 0.999 | 0.960 / 0.963 | +0.039 / +0.036 |
| own lane near an intersection | 0.995 | 0.874 / 0.868 | +0.120 / +0.127 |
| own lane elsewhere | 0.999 | 0.981 / 0.979 | +0.019 / +0.020 |
Reading: Mirko's observation is the long links. A short link persists in 96 % of the cases, a long link (the connectors across a junction) in 64 - 65 % against 98 % in the GT: one in three of them is gone half a second later. Near an intersection the links flicker four times as much as elsewhere (14 - 16 % against 3.7 - 4.0 % do not persist), in the own lane near an intersection six times (13 % against 2 %). At node level the flicker was much smaller (P5: 5.3 % against 1.7 %), so the temporal inconsistency seen in the videos is mainly the choice of the long links between frames, together with their precision of 17 - 19 % (P9) and their importance for the topology (P10: a graph without them loses 0.12 topo F1). A memory over frames (hold a long link while it is not contradicted, or give the link head the previous frame's links) is the lever; not built.

**P8 - the assignment to the map elements, is it 1:1?** (video note 2; `exp_assignment_uncertainty.py` on the FULL prediction dumps of B, `temp/e276/predfull_e273_p_s0.pkl` / `_s1.pkl` written by the E276.c_s0 / c_s1 scoring jobs, exact map, e >= 0.5 nodes, VAL 873 frames, log `temp/e276/p8p9_s<k>.log`; seed 0 / seed 1):
| | seed 0 | seed 1 |
|---|---|---|
| delivered nodes | 150,305 | 151,265 |
| node accuracy (the node's map piece belongs to the GT lane of the node) | 0.976 | 0.976 |
| nodes sent to the dustbin / of them on a GT lane | 2.0 % / 23 % | 2.1 % / 23 % |
| delivered sections (>= 3 nodes with one section id) | 6,974 | 7,080 |
| section correct (majority map piece of the section) | 93.5 % | 94.0 % |
| map pieces claimed by two or more sections (claims per claimed piece) | 8.5 % (1.09) | 7.8 % (1.09) |
| map pieces per section (mean) | 1.22 | 1.21 |
| GT map pieces (>= 10 m inside the window, 11,177) that are the MAJORITY piece of no section | 45.9 % | 44.9 % |
Reading: at node level the assignment is accurate (0.976, the guard number of E273), and it is mostly 1:1 (a map piece is claimed twice in 8 % of the cases; a section carries 1.2 map pieces), so a loss that forces a strict 1:1 assignment has little to gain at this level. The last row is NOT a coverage number: a section claims only its majority piece, and overlapping pieces (connectors, parallel pieces) share nodes, so the row overstates the gap. I first read it as "45 % of the map pieces are uncovered" and wrote that; P8b below replaces that reading.

**P8b - which map pieces are really not claimed** (`exp_coverage.py`, the same dumps; per GT piece >= 10 m, from the delivered nodes that lie on a lane: a node belongs to its nearest piece and to the pieces tied with it within 0.3 m): NO NODE = no delivered node lies on the piece; CLAIMED = a node there has this very piece as its most probable map piece; SHARED = no node claims it but one is assigned to a piece that ties with it (an overlapping piece took the node, the scorer counts that node right); MISASSIGNED = nodes lie on it, all go to other pieces or the dustbin. Seed 0 (seed 1 in brackets):
| group | pieces | no node | claimed | shared | misassigned |
|---|---|---|---|---|---|
| all | 11,206 | 7.6 % (8.1) | 71.0 % (70.2) | 16.7 % (17.3) | 4.6 % (4.4) |
| tier 1 (own lane) | 2,450 | 5.4 % (6.1) | 71.7 % (70.9) | 17.8 % (18.0) | 5.1 % (4.9) |
| tier 1, seen by a camera | 1,892 | 4.8 % (5.4) | 73.2 % (71.6) | 16.9 % (18.0) | 5.2 % (5.0) |
| tier 2 (neighbours) | 759 | 0.8 % (1.1) | 98.4 % (98.0) | 0.8 % (0.7) | 0.0 % (0.3) |
| tier 3 | 7,997 | 9.0 % (9.4) | 68.2 % (67.3) | 17.9 % (18.6) | 4.8 % (4.7) |
Reading: the assignment does not leave half of the map uncovered: about 88 % of the pieces are claimed or shared and 4 - 5 % are misassigned; 5 - 8 % have no delivered node at all (in the own lane seen by a camera 4.8 - 5.4 %: that is the node recall gap of P3, not an assignment error). The neighbour lanes (tier 2) are claimed in 98 %. So the weakness shown in the videos is not the node-to-piece assignment or the coverage of the map; the places to look are the sections and links across the junction (P9) and the sections that mix two pieces (a section carries 1.2 pieces). Whether "horrible" refers to the picture of the assignment in the video (colours per piece) or to a metric cannot be told from these counts; the picture of the video is the next thing to check with Mirko's frames.

**P9 - link calibration and what abstaining costs** (video note 8; same dumps; a link is a delivered link between two delivered nodes with p >= 0.05; only links whose two ends are both matched one-to-one to GT nodes within 2 m are counted (a link with an unmatched end, i.e. at a false node, is left out: neither right nor wrong here), and such a link is RIGHT when its two GT nodes are joined by a GT link; the share of GT links delivered is taken over the GT links between matched nodes; seed 0 / seed 1):
- links 229,521 / 256,878, of which 50.3 % / 45.1 % are right; expected calibration error 0.121 / 0.117.
- reliability (mean probability -> share right): p 0.07 -> 0.007, 0.25 -> 0.036, 0.45 -> 0.152, 0.55 -> 0.268, 0.65 -> 0.396, 0.75 -> 0.536, 0.85 -> 0.694 (seed 1: 0.733), 0.97 -> 0.936 (0.940): the link probability is over-confident in every bin, most in the middle (a link at p 0.55 is right in one case of four).
- risk and coverage (precision of the links kept at probability >= t, and the share of the GT links delivered):
| threshold | links kept s0 / s1 | precision s0 / s1 | share of GT links delivered s0 / s1 |
|---|---|---|---|
| 0.5 (the read-out of B) | 141,821 / 140,226 | 0.792 / 0.800 | 95.1 % / 94.8 % |
| 0.7 | 123,377 / 122,634 | 0.860 / 0.868 | 89.8 % / 89.9 % |
| 0.9 | 92,505 / 91,535 | 0.936 / 0.940 | 73.4 % / 72.7 % |
- by length at p >= 0.5: short links 130,408 / 130,835 with precision 0.846 / 0.844; long links (the connectors across a junction) 11,413 / 9,391 with precision 0.169 / 0.194, that is 7 - 8 % of the delivered links but 27 - 32 % of the wrong ones. By tier of the source node (seed 0): 0.788 / 0.859 / 0.782, no gap between the own lane and the rest.
Caveat on the definition: "right" is a rule on node pairs and stricter than the graph scorer (which matches geometry); the long-link precision agrees with the scorer's own `link_long_P` of the cards (0.169 / 0.194 against 0.167 / 0.191), but an oracle that dropped every link the pair rule calls wrong lowered the topo F1 in a 30-frame smoke, so the pair rule is a lower bound of what is right at graph level (not run on the full set). Reading (links at unmatched nodes are not counted, so the share of wrong links among all delivered links may differ from these numbers): the probabilities can be used to abstain, and in link terms abstaining is cheap in the middle range (the graph metrics of P10 are less kind): raising the link threshold from 0.5 to 0.7 gains 0.07 precision for 5 points of coverage; the long links are the weak spot (four of five wrong), and the pictures (`results/E276/figures/links_uncertainty_e273_p_s<k>.png`, six frames each: width and opacity = probability, dashed below 0.5, green right, red wrong, orange = an end not matched) show them as the long red / orange chords across the junction mouths. Candidates, all read-out steps first (a hard selection on the output, D55 stop-gap): a separate threshold for long links, an abstain state "uncertain" between 0.5 and 0.8 that the planner can see, then a calibrated loss (the p-weighted focal term or a temperature) in training. The graph metrics for these read-outs are P10.

**P10 - link thresholds on the graph numbers of B** (read-out only; `exp_link_threshold.py`: the nodes at e >= 0.5 as in the card of B, the links of the full dump re-read at the thresholds (short links, long links) and scored by the E273 scorer on CPU; the variant (0.5, 0.5) reproduces the scoring card of B to 0.0000 on both seeds; VAL 873, a diagnostic by hand, in-sample, no verdict; seed 0 / seed 1):
| (short, long) threshold | topo F1 | topo P / R | junction F1 | F1@1.0 | lanes found | vertices per 100 m (DP) |
|---|---|---|---|---|---|---|
| (0.5, 0.5) = B | 0.597 / 0.595 | 0.586 / 0.609, 0.600 / 0.590 | 0.130 / 0.120 | 0.833 / 0.821 | 0.644 / 0.630 | 28.9 / 31.5 |
| (0.7, 0.7) | 0.592 / 0.580 | 0.705 / 0.511, 0.722 / 0.485 | 0.146 / 0.126 | 0.800 / 0.782 | 0.556 / 0.539 | 26.1 / 28.7 |
| (0.9, 0.9) | 0.445 / 0.429 | 0.857 / 0.301, 0.879 / 0.283 | 0.124 / 0.099 | 0.683 / 0.665 | 0.341 / 0.326 | 21.5 / 22.6 |
| (0.5, 0.7) | 0.594 / 0.583 | 0.638 / 0.555, 0.656 / 0.524 | 0.116 / 0.106 | 0.805 / 0.786 | 0.583 / 0.567 | 32.6 / 36.1 |
| (0.5, 0.9) | 0.545 / 0.534 | 0.702 / 0.445, 0.714 / 0.426 | 0.090 / 0.084 | 0.741 / 0.727 | 0.484 / 0.480 | 39.8 / 42.6 |
| (0.5, no long link) | 0.472 / 0.474 | 0.731 / 0.349, 0.734 / 0.350 | 0.068 / 0.070 | 0.676 / 0.680 | 0.401 / 0.406 | 45.6 / 46.8 |
(topo P / R: seed 0 first, then seed 1.) Reading: no link threshold lifts the topo F1 above the read-out at 0.5; (0.7, 0.7) is a tie in topo F1 (-0.005 / -0.015) and buys 0.12 topo precision for 0.10 topo recall, with F1@1.0 and lanes found falling by 0.03 / 0.09. The long links, 17 - 19 % precise, are indispensable for the topology: without them the topo recall falls from 0.61 to 0.35 and the topo F1 by 0.12 (they are the only links that join two sections across a junction), and a higher threshold for the long links alone lowers topo F1 and junction F1 at every value tried; the vertices per 100 m even rise without them (probably a graph in pieces carries more end vertices; not checked). So abstaining by a threshold is an operating-point choice (precision against recall; "better no link than a wrong one" is (0.7, 0.7)), not a quality gain: the gain has to come from making the long links right (fewer wrong candidates, a better label), not from cutting them. An abstain state for the planner (uncertain between 0.5 and 0.8) stays a candidate for the output format.

**What the pre-experiments change in this card:** prediction 2 missed (section 0, status), prediction 4 revised and the s rule moved to the delivered nodes per 100 m (section 0), the label-noise estimate of P4 and the dense-vs-sparse equality of P1 are the evidence for the GT and the target choices of sections 2.1 and 2.2.

### 7b. The night of 2026-11-08: training results
**Arm s (tier loss on the sparse node target), seed 0: failed the phase-3 gate (kill criterion of section 0), stopped 00:50, CUT by the written gate.** E276.s_s0 trained the 6000 steps of phase 3 (0.23 s per step, loss 2.15) and failed: node recall within 2 m 0.143 against the floor 0.5 (node precision within 2 m 0.876, flag recall 0.04 - 0.06 for stop, exit, split and merge against 0.28 - 0.36 for arm t seed 1 at the same gate); the run ended with exit 3, the scheduler took the s arms out (no s seed 1, no `sq`, no `verdict276` kind s). **Diagnostic by hand (CPU, 60 VAL frames, the phase-3 checkpoint `runs/e276_s_s0_p3`, read-outs at three thresholds, scored against the sparse GT nodes, `temp/e276/diag_s.sh`, cards in `runs/e276diag/`):** delivered nodes per 100 m 3.6 at e >= 0.5 (node recall 0.146, precision 0.693), 50.0 at e >= 0.3 (recall 0.910, precision 0.310), 71.8 at e >= 0.2 (recall 0.949, precision 0.225), against 14.4 per 100 m in the sparse target. So the head knows where the sparse nodes are (recall 0.91 at e >= 0.3) but cannot decide WHICH of the roughly three candidate cells along a straight lane carries the node: the position of a Douglas-Peucker / chord vertex on a straight stretch is not a function of the picture, the existence probability smears over the candidates and stays under 0.5. This is a hypothesis for the cause (the three-threshold counts are its evidence, not a test of it). The sparse target is therefore not learnable by this head with this loss; the P1 oracle showed that the geometry does not need the dense nodes, not that the head can place the sparse ones. Next: a set-based node loss (one-to-one matching of predicted nodes to the sparse GT nodes inside the loss, allowed by D55) so that a node may sit anywhere along a straight lane, or the trained thinning of E277 (the soft close-node step), which does not change the target.

**Arm t (tier-weighted loss, dense node target), both seeds: verdict276 kind t = TIE (written by the scheduler at 01:19, `temp/e276/verdicts.log`, rule of section 0; VAL 873, lane GT l4, in-sample).** The tier weights do not lift the own lane; B stays the baseline.
| seed 0 / seed 1 | B | arm t | gain |
|---|---|---|---|
| t12_F1_1 (F1 at 1 m, own lane + neighbours, visible parts; the primary) | 0.888 / 0.875 | 0.886 / 0.885 | -0.002 / +0.010 (band 0.02) |
| tier-1 (own lane) recall | 0.851 / 0.811 | 0.838 / 0.842 | -0.013 / +0.031 |
| tier-3 (far lanes) recall | 0.796 / 0.769 | 0.698 / 0.670 | -0.098 / -0.099 |
| topo F1 | 0.597 / 0.595 | 0.580 / 0.573 | -0.017 / -0.022 |
| lanes found | 0.644 / 0.630 | 0.557 / 0.529 | -0.087 / -0.101 |
| node precision / recall | 0.873 / 0.833, 0.868 / 0.833 | 0.892 / 0.804, 0.888 / 0.790 | |
| false nodes in the own-lane zone (share of the delivered nodes there) | 10.2 % / 10.1 % | 10.1 % / 10.8 % | no fall |
| delivered nodes per 100 m | 31.5 / 31.7 | 29.7 / 29.4 | -5.6 % / -7.4 % |
| assignment p - g exact / error | -0.0072 / +0.0722, -0.0061 / +0.0706 | -0.0044 / +0.0778, -0.0058 / +0.0731 | guards held |
Predictions of section 0: t12_F1_1 +0.04 **missed** (-0.002 / +0.010), the 25 % fall of false nodes in the own-lane zone **missed** (none), global topo F1 -0.01 to -0.05 **held** (-0.017 / -0.022), end / start events in tier 1 not testable (no own-lane end occurs in VAL). Reading: the loss moved capacity from the far lanes (tier-3 recall -0.10, lanes found -0.09 / -0.10, the weights make the head silent where it is not demanded) to nowhere: the own lane did not gain. The node head's limit in the own lane is not the weighting of the loss (the own-lane recall of B is 0.81 - 0.85 with 82 % of the own-lane nodes visible): it is what the features and the 2 m grid can resolve. What the tier labels are good for stays: the metric (t12_F1_1, the zone numbers) and, perhaps, the demand rule for the memory of a later experiment. Not adopted (TIE: the cheaper variant is B).

### 7c. Interim status, night check 2 (2026-11-08 03:25)
Ran: smokes (ok), arm t seeds 0 / 1 (trained, gate of phase 3 passed, scored, verdict TIE, section 7b), arm s seed 0 (failed the phase-3 gate, CUT; no seed 1), baseline B scored on l4 with full dumps, the CPU analyses P1 - P10 (section 7a). No NaN in any run; the only failed gate was arm s (by design: exit 3, the scheduler stopped, restarted without the s arms). Spun off from this card (own cards): E277 (the soft close-node step in the head: `E277.u_s0` / `u_s1` trained, scoring `uq_s0` running, `uq_s1` queued, `verdict277` automatic) and the E278 diagnostic (phase-6 models scored). Pending: the E277 / E278 scoring cards and verdicts, the comparison pictures, the report page and the artifact update; the card sections 7 of E277 and the results page `results/E277/README.md` are written when the verdict is there.
