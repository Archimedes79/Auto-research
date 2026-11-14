# E344 - plan: from five named rule steps to none; the query head on the 1 m grid ("Q-Lane")

Plan of 2026-11-14 17:30, written by the lane experiments chat on Mirko's request of the same afternoon (after the report: "der Bericht sagt, dass E2E differenzierbarkeit fehlt. Bitte arbeite einen Plan aus, diese herzustellen. wir hatten auch mal ueber eine komplette Query architektur geredet, die auf dem 1m grid aufsetzt, positionsunsicherheit zulaesst und dann sowohl smoothness und verbindungen und alles optimiert"). **Status: plan only; nothing built, nothing run.** Needs Mirko's answers to section 7 and a fresh `ar-critic` (cards) before the first line of code (D84 point 6 parked the "unstructured transformer head"; this plan proposes a grid-anchored variant of it and asks whether that is wanted).

## 0. The card (what the tools run)
```json card
{
 "id": "E344",
 "card": "",
 "title": "plan: remove the named output selections of the delivered graph (snap / wiring, threshold, top-K, close-node step, Douglas-Peucker) one by one and, as the end state, replace the keypoint / pair / connector chain and B's node / link heads by one query head anchored on the 1 m grid; the plan card's metric is the number of named selections, the guard is the accuracy",
 "kind": "variant",
 "deterministic": true,
 "metric": "named_output_selections",
 "direction": "lower",
 "band": 0.0,
 "prediction": {"value": 0, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E324 s (D94): five named output selections (snap and wiring of the connector ends, probability threshold p >= 0.1, top-K = 32 keypoints, close-node step 1.5 m, Douglas-Peucker 0.3 m), plus the group / rank bookkeeping (named, no gradient) and the frozen sensor features (D50)"},
  "q": {"recipe": "Q-Lane at stage Q5 (section 4): the delivered graph is a probabilistic graph (nodes with existence, type, position mean and sigma, heading; edges with probabilities); the only selection left is the consumer's threshold"}
 },
 "control": "c",
 "variant": "q",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "named_output_selections", "vs": "c", "min_effect": 1},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 0.0,
 "kill": "a stage gate of section 4 fails twice (one refinement round each, D82: a TIE is refined, not dropped); the plan then stops at the last stage that passed and the report says which selection stays",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: plan"): this card is the plan; every stage of section 4 gets its own card with a rule written before the run. The equal-accuracy bar of the whole plan is the removal rule of D84 point 4 / D87: snapped topo F1 and the differentiable (unsnapped) read-out each within 0.01 of the control on both seeds, guards held; at the end the planner reads probabilities, so the threshold is no longer ours.

## 1. What "end-to-end" means here, and where we stand
The rule (D55, D88): everything on the training path is soft, a loss on the output reaches every module (checked with `evaluation/drivers/e2e_gradient_report.py`, "no silent tensor"), hard selection only on the delivered output and named in every report. Today the delivered graph has five named selections, one named bookkeeping step and the frozen sensors:
| # | step | what it carries (VAL) | soft replacement that exists | state |
|---|---|---|---|---|
| 1 | snap + wiring of the connector ends (E311) | about two thirds of the topology gain (69 % / 73 % of it is the choice of ONE nearest piece, the rest the end geometry) | E320 / E324 s: soft attachment replaces the geometry (+0.016 / +0.015 over E312); E325: a learned wiring head (545 parameters), runs tonight | geometry soft, the one-piece wiring still a rule until E325 |
| 2 | probability threshold p >= 0.1 | 0.5 -> 0.1 gives +0.046 / +0.052 lanes found | none needed: deliver the probability (and a position sigma, E318) | the metric is bound to a threshold; E318 lifted today (D96) |
| 3 | top-K = 32 keypoints | **not an accuracy cap: K binds in 0.2 - 2.6 % of the VAL frames (measured 2026-11-14 on the E312 dumps: 12.2 valid ends per kind on average, 90th percentile 21, maximum 32)**; it is an index choice | dense queries (this plan, track B) | deferred since D84 |
| 4 | close-node step 1.5 m (E275) | topo F1 0.597 -> 0.626, duplicates down | E279 soft step (KEEP, but it trains the pyramid jointly and so invalidates the junction heads), E288 Matrix-NMS (TIE), E292 layer (not adopted) | named |
| 5 | Douglas-Peucker 0.3 m | the vertex budget (E297: 19 - 20 vertices per 100 m in the delivered graph, budget 14.3 counted on the metric's own Douglas-Peucker basis) | E297 soft chord keep gate (plan card, never run) | named |
| 6 | group / rank bookkeeping of the keypoints (order rule) | features of the pair head, no gradient by design (D82) | `jgt._groups` vectorised (E342); a learned order in track B | named |
| 7 | frozen sensor features (D50) | no loss reaches camera or radar | none planned for the lane head | decision of Mirko |

## 2. Principle
A selection is replaced when something soft learns it at the equal-accuracy bar (D87: an E2E-required replacement without an E2E alternative is a TIE kept and refined, never dropped). Where nothing soft is needed (the threshold), the answer is to deliver the probability and measure without the threshold. Two tracks run in parallel: **A** closes the gaps in the system we have (small, cards exist, GPU already booked); **B** replaces the chain by one query head on the 1 m grid, which removes steps 1, 3, 4 and 5 by construction instead of imitating them.

## 3. Track A: close the gaps in the system we have
| id | what | selection it removes | state / next |
|---|---|---|---|
| A1 | E325 learned wiring head; then the verdict at the equal-accuracy bar (snapped topo F1 >= E324 s - 0.005 on both seeds) | 1 (the wiring) | tonight; result section follows |
| A2 | **threshold-free reporting** (new card, CPU): topo F1, lanes found, junction F1 as a function of the threshold, the area under the curve, calibration of the connector probability (ECE) and of sigma (ENCE, E295 / E318); E318 (position sigma, lifted) delivers sigma | 2 | card next, then E318 on the weekend queue |
| A3 | E317 (geometry loss) and E319 (heading-guided exchange), lifted | feed 1 and 3 (better ends) | critic running, then queue |
| A4 | E297 soft chord keep gate: run it as written (plan card since 2026-11-11), judged by the vertex budget and topo F1 | 5 | card to be updated to the E324 s control, then one GPU job |
| A5 | the close-node step: no new experiment until track B decides; the one-to-one loss of Q-Lane removes the need (section 4) | 4 | parked, with the reason |
| A6 | E341 / E342 / D99: compute, not selections | none | as planned |

## 4. Track B: Q-Lane, one query head on the 1 m grid
**Idea (what we talked about before: E294 "nodes from the 1 m maps", E295 "a position spread per node", E297 "soft pruning", D82 "stay differentiable"):** every cell of the 1 m grid (100 x 100 = 10 000, on the same frozen pyramid as today, so comparisons stay fair) is a query. No top-K, no Hungarian matching: the queries are anchored at their cell like the 2 m nodes of B and like the end heat of E307 (CenterNet style), which keeps the training easy on 8 414 frames; free DETR-style queries (TP30, TP05, TP06 in the corpus) need far more data to converge and are kept as the later option B2.
**What a query outputs:** an existence logit; a type (normal, split, merge, entry, exit, as B has them); a sub-cell position mean and a position uncertainty (sigma along and across the line: the Laplace NLL of E318, the heat target of sigma 1.0 m of E307); a heading with a concentration; and a **successor distribution**: a softmax over a fixed set of forward slots in the query's own frame (relative offsets 1 - 12 m inside a forward cone) plus a dustbin ("no successor here"), computed densely by a convolutional head (no pair MLP over 10 000 cells; the MACs stay in the range of the current maps head).
**Junction connectors** become chains of ordinary nodes through the junction (the lane GT already has the connector polylines): the entry -> exit pairing is the path of the soft successor chain, computed by B's soft absorbing walk (2.8 ms today), so there is no pair head over a top-K set and no wiring rule: the end IS a node of the graph and its link is a probability. Open point for the card of Q3: connectors of 10 - 30 m against a slot radius of 12 m (chain nodes every 1 - 2 m inside the junction, within the vertex budget through the keep gate of A4).
**Output:** a probabilistic graph: nodes with existence, type, mean, sigma, heading; edges with probability. The planner (and the time fusion of D90 / D91) reads probabilities and uncertainties; the only selection left is the consumer's threshold.
**One loss for everything** (the "optimises smoothness and connections and everything" part): existence (focal + dice, as E307), type (CE), position (Laplace NLL with sigma), heading (cosine), successor (CE with the dustbin, dual softmax), **duplicate suppression as a one-to-one loss** (a Sinkhorn / dual-softmax over each query's neighbours replaces the close-node step), **smoothness** (curvature of the soft path and C1 continuity of the tangent at every node: the incoming and the outgoing Hermite curve share the tangent), connector geometry (the Chamfer loss of E317), the order / non-crossing penalty (E287), a vertex budget (the soft keep gate of E297). All of them but the first four exist in some form in `training/grid/`.
**Stages and gates** (each stage a card with its rule before the run, two seeds, VAL, fresh critic; one refinement round on a TIE):
| stage | what | cost | gate |
|---|---|---|---|
| Q0 | representation ceiling, CPU, GT only: render the GT graph into the query code (existence, type, offset, heading, successor slots), decode with the soft read-out, score with the standard scorer | days of code, no GPU | topo F1 of the decoded GT >= 0.97 and the decode runs in <= B's node + link read-out; if not, the representation is changed before any training |
| Q1 | dense 1 m node head with sigma, soft keep gate and one-to-one loss, on the frozen pyramid (E294 revived on the current system) | 4 - 6 GPU h per seed | plain-graph nodes: node F1 within 0.01 of B's 2 m head (0.851 / 0.850), vertices within the budget, ms <= B's node head (2.9 ms) + 1 ms |
| Q2 | + successor links and the soft sections, replacing B's links / sections (5.5 + 2.8 ms) | 6 - 8 GPU h per seed | B's plain delivered graph: topo F1 within 0.01 (0.626 / 0.625), lanes found within 0.015 |
| Q3 | + junction: dense entry / exit types, connector chains; no snap, no top-K, no close-node step | 8 - 12 GPU h per seed | unsnapped read-out (D85: counts) within 0.01 of E324 s snapped on both seeds, guards held |
| Q4 | joint training of everything with the smoothness and geometry losses; calibration of the existence and of sigma | 8 - 12 GPU h per seed | equal-accuracy bar + ECE / ENCE reported; smoothness measured (curvature of the delivered lines against the GT's) |
| Q5 | the system against E324 s on VAL, two seeds, then ONE TEST with Mirko's own recorded OK | CPU + short GPU | named selections: 0; accuracy within the bar |
Total GPU: about 60 - 100 slot hours for two seeds, i.e. two weeks at the rate of the last days, with the code in front of each stage; the plan is cut after any stage that fails its gate twice, and the report says which selections remain.
**Alternatives considered:** (B1) keep the junction layer and replace only the top-K by a soft token set (all cells weighted by their heat, attention instead of a hard index choice): small, but it keeps the snap, the wiring and the close-node step; (B2) free DETR-style queries with Hungarian matching inside the loss: allowed by D55, the literature has it (TP30 and followers), but slow to train on 8 414 frames and without the 1 m anchoring; (B3) do nothing beyond track A: the five selections stay named, each with its measured price.

### 4b. Variant B0 (Mirko's sketch of 2026-11-14): points on the 2 m grid, information pooled by attention, then soft pruning
"First design points on a 2 m grid, let information pool, prune the irrelevant points; the points at junctions get a large position uncertainty; every point estimates position, position uncertainty, direction, type and connection after the self-attention and pooling steps; the position estimate is itself improved by the self-attention." Read as a cascade: (1) propose one point per 2 m cell with a first position, sigma, heading, type; (2) neighbourhood self-attention (a 30 m window, 15 x 15 cells, about 70 MMACs per layer; global attention over the 2 500 cells would be 0.8 GMACs per layer) plus coarse pooled tokens from the 4 / 8 m levels for the long range; (3) soft keep gate (existence x importance, the chord cost of E297); (4) every attention layer adds a zero-initialised residual to the position and to log sigma (iterative refinement, the look-forward-twice and deformable-reference-point ideas of VS19 / VS20 in the corpus); (5) the connection (successor) after the last layer. It is Q-Lane on the 2 m grid of B instead of the 1 m grid, with B's node head as the first stage, so it is the cheaper start. Measured on the VAL ground truth on 2026-11-14 (scratch `temp/_session_scratch_s292/gt_end_grid.py`, `gt_arm_reg.py`): (a) **the 2 m grid is not a resolution obstacle for the ends**: 0.2 % of the GT entry / exit ends share a 2 m cell with another end of the same kind (0.0 % on a 1 m grid); the nearest other end has a median distance of 4.8 m (entry) / 5.2 m (exit), 2.0 % / 1.3 % are closer than 2 m and 26 % / 17 % closer than 3.5 m (neighbouring lanes); the position has to come from the sub-cell offset, as in B (median 0.28 m on the nodes, but locked to the grid, E289); (b) **the ends of one arm are regular, which is what a self-attention can use to improve the positions**: in the 3 818 groups of two or more ends (mean size 2.2) the spread along the heading (the stop-line alignment) has a median of 0.47 m (90th percentile 2.1 m) and the lateral gap between neighbouring ends a median of 3.43 m (10th - 90th percentile 2.81 - 4.00 m). The keypoint ends of E312 are 1.44 m from the GT ends on VAL (E310; 0.6 m on the training frames, which contain injected GT cells), and the pair head's set blocks already attend over the keypoints but only feed the pair logit: the position is the soft-argmax of a 3 x 3 window of the heat and nothing else. The first test of the idea is therefore cheap and needs no new architecture: **a zero-initialised position / log-sigma residual after the set blocks of the pair head, supervised by the end loss of E317 / E318** (a card after E317 / E318 have run; prediction: end error 1.44 m -> at most 1.1 m, unsnapped topo F1 +0.005). Q0 gets the grid size as a parameter (2 m and 1 m) so that B0 and Q-Lane are decided by the same ceiling measurement. Risks named: a large sigma is an escape hatch of the NLL (beta-NLL and the calibration ENCE are the guards, the position loss keeps a term that does not depend on sigma); long junction connectors (10 - 30 m) as chains of uncertain 2 m points cost vertices, so the connector stays a Hermite link between two sharp ends and only its interior carries the large sigma; a hard drop of tiny points for speed is a named internal selection, the delivered weights stay soft.

## 5. Track C: the sensors (D50)
The frozen GaussianCaR features mean that no loss reaches camera and radar. Not planned for the lane head: after Q5 a probe with the BEV system chat (unfreeze the last stage of the BEV encoder, one seed, the same gradient audit). D50 stays Mirko's decision.

## 6. How the plan is measured
(a) the table of section 1 re-written at every stage with the number of named selections; (b) the gradient audit (`e2e_gradient_report.py`) per stage: no tensor on the path without a gradient; (c) threshold-free metrics and calibration (A2); (d) the equal-accuracy bar of section 0 and ms per frame on an idle GPU (E342 part B), because Q-Lane replaces B's heads (24.2 ms in total with the pyramid) and has to be judged on time as well; (e) every report names the selections that remain.

## 7. Questions for Mirko (nothing is started on track B before the answers)
1. Is the dense grid-anchored query head (section 4) what you meant, rather than free DETR-style queries (B2)? D84 point 6 parked the "unstructured transformer head": does it stay parked for B2 and is it lifted for Q-Lane?
2. Is the equal-accuracy bar (snapped and unsnapped topo F1 within 0.01 of E324 s) the right price for removing the selections, and may Q-Lane replace B's node / link heads at the end (it must also be as fast)?
3. Priority against D99 (smart pruning) and track A on the GPU: A first (it is booked anyway), Q0 on the CPU in parallel?
4. What does the planner need from the output: probabilities and sigma per node and edge (the plan assumes yes)?

## 8. Referee
Not yet: a fresh `ar-critic` (cards) on this plan and on the Q0 card before any code.
