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
- **s against t, KEEP** (the sparse target is adopted) when `vert100_dp` falls by at least 25 % (c: 28.9 per 100 m, seed 0) on both seeds AND `t12_F1_1` stays within 0.02 below t on both seeds AND the guards hold: the assignment guards against g (as above) and `t1_R_1` not below t by more than 0.02 on either seed (no topo guard for this comparison). `t12_F1_1` more than 0.02 below t on a seed, or a broken guard, is CUT; no density gain with quality held is a TIE (t stays).
- **Reported, not judged:** every number split by tier and by visibility (node and point recall / precision, precision of delivered nodes in zone 1 / 2 / 3, false junction count in zone 1 / 2), the distribution of delivered nodes along the road, endpoint events P / R by tier, ms per frame, vertices per 100 m after Douglas-Peucker, links per node.
- **In-sample labels:** the weights (3.0 / 1.5 / 1.0, 0.0, 2.0 / 1.5 / 1.0) and the sparse parameters (0.3 m, 7.5 m) are first guesses written here before any run, not tuned on VAL; a second weight setting needs its own card.

**Predictions (the first two do not need training).** (1) Scored per tier, B is better in tier 1 than elsewhere: the F1 at 1.0 m of c in tier 1 + 2 (`t12_F1_1`) is at least 0.03 above its F1 over all GT points (the weakness is in the periphery). (2) Of the GT lane length in the window, 8 - 15 % is tier 1 + 2 and 35 - 60 % of the tier 3 length is not visible to a camera (l4 store, section 3: seen on 30 test keyframes: tier 1 + 2 about 14 %, visible 51 - 66 %). (3) t: `t12_F1_1` +0.04 over c on both seeds; global topo F1 -0.01 to -0.05. (4) s: `vert100_dp` 28.9 -> 20 or below with `t12_F1_1` within 0.02 of t. (5) The number of delivered nodes that lie in zone 1 and have no GT node within 2 m falls by at least 25 % in t (the weight of the false nodes works). (6) Endpoint events (end, start) in tier 1 improve in P and R by +0.05 or more in t (the weight at event cells inside tier 1; E274 weighted them everywhere and failed).

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
