# E355 - DESIGN: one set of tokens instead of grid nodes plus keypoints: connect, prune, connect again, decode the types (D104); with step 0, the ground truth through the variant that keeps a few nodes inside the junction (CPU)

Card of 2026-11-15 19:45, version 2 of 20:20 after the first critic pass (REVISE: the pruning target of version 1 was trained before and failed (E276 arm s) and was not cited; step 0 was mostly known; "minimum spacing by the code", "one pruning does three jobs", "everything soft" and "five selections to two" overclaimed; the junction interior dodged D104; the compute was low; the literature was missing). Written by the Spurmitte chat that Mirko addresses as "Fable" on his question of 2026-11-15 (D104, verbatim: "Don't set nodes closer than 2m Consider 2.5m. Consider node setting + pruning also on intersections when there is no information where to pull the node to. Why do we have nodes and keypoints? Why not only transformer nodes, which are connected, pruned, connected and then decoded into differnent types. Very simple design."). **A design card: nothing is built, and no trained stage starts without Mirko's recorded decision** (D84 point 6 deferred a transformer head "until more planning"; this card is that planning). Its one runnable part is step 0 (CPU, section 0).

**The short verdict of this card on itself (so that nobody reads more into it):** most of the design is what we already have under new names: B's node + link code, the sparse node target of E276, the keep gate of E297, the refinement of E344 variant B0 and today's pair head; and it is NOT yet simpler than the system of record when counted honestly (section 3). Three things are new and are worth testing, in this order: (1) ONE pair head for every edge of the graph on ONE node set (no second family, no snap); (2) the ends as token TYPES that read the 1 m features by sampling; (3) nodes set inside the junction and pruned to a few per connector (D104's sentence, read faithfully). It supersedes track B of the plan E344 (Q1 - Q5) if Mirko takes it.

## 1. Why there are nodes AND keypoints today (the honest answer)
History, not design. (a) The lane head B (E273) is already "one graph": nodes on the 2 m grid, short links, LONG links for the junction connectors, the ends as node flags; on the ground truth that code loses only 0.019 topo F1 (E346, 873 VAL frames; lanes exact, the loss sits in the connectors). (b) B could not LEARN the junction part: junction F1 0.12 - 0.13, half the end flags at no GT end (E283), entry recall 0.57 / 0.58 on its frozen node tokens (E284). So "the ends are nodes, nothing to snap" is true of the CODE and was true of B; it did not make the junction learnable. (c) A second family was therefore added on B's frozen pyramid: the junction maps head at 1 m, keypoints as 3 x 3 peaks (top-K = 32), a pair head, Hermite connectors, a snap that joins the connector ends to B's pieces. It works (snapped topo F1 0.626 -> 0.6981 / 0.7010 with E324 s), and the seam between the families is where rule steps sit (the snap and its one-piece wiring: two thirds of the topology gain; top-K; the peak rule), next to B's close-node step. (d) Attention is not what was missing as such: B's link head has self-attention over the active nodes and the pair head has two set blocks over the keypoints. What the two families never had is ONE set on which the same head decides every edge.

## 2. The design
**Minimal core (the part a first trained stage would build):**
- **Tokens:** one token per cell of ONE lattice over the 100 x 100 m window, spacing 2.5 m (40 x 40 = 1600; 2.0 and 3.125 m in step 0), also inside intersections. A token holds a feature vector (d about 96), a position (cell centre + offset), a heading.
- **Layers (L about 4):** (1) read the frozen features around the token's current position (a 3 x 3 patch of the 1 m map, pooled, and the 2 m map: one bilinear tap would undersample the 1 m map, E347); (2) self-attention among the tokens of a local window (about 30 m) plus a few pooled global tokens; (3) a zero-initialised residual on position and heading (E344 variant B0).
- **Connect 1:** existence per token; cheap edge logits to the tokens of a forward cone (a candidate set by geometry, named); one message layer along the soft edges.
- **Prune:** by EXISTENCE only in the core: the keep weight is the existence probability, a multiplier in training; the survivors that enter connect 2 are the tokens with the largest weights (an index choice, see "soft or not").
- **Connect 2 (about 100 - 250 survivors):** ONE pair head for all edges: lane edges between neighbouring nodes and junction edges entry -> exit, each a cubic between two nodes with two handle scales (the long-link form: median fit error 0.014 m in E346; `qtargets.decode` fits scales only for links longer than 8 m today, shorter ones use chord / 3). Inputs as today's pair head (two tokens, relative geometry, evidence sampled along the candidate curve). Edge probabilities are sigmoids with the dual softmax as an auxiliary term, as the pair head has it: a single row softmax with "none" was tried in B (E273 v6) and reverted (junction F1 0.122, topo F1 0.467), and forks need more than one outgoing edge.
- **Decode:** type (lane point, fork, merge, entry, exit, start, end), position with sigma, heading per survivor; probability and handle scales per edge; the delivered graph = survivors and edges above one threshold.
**Extensions, each its own later card:** (X1) pruning by importance for the vertex budget; (X2) a few kept nodes inside the junction per connector (step 0 decides whether the code gains from them); (X3) about 300 free queries with a Hungarian loss instead of the lattice; (X4) the recursive state of E354 on the token features; (X5) the tokens reading the cached features without B's pyramid (E352's subject).

**What version 1 claimed and what holds:**
- *"One pruning does three jobs."* Not as written. Only the close-node step is a delivered step today; the Douglas-Peucker stage is the metric's own basis (PROBLEM.md), not a step of the delivered path; and "interior tokens get no target and are pruned" is exactly B's intersection suppression of today, not a new answer.
- *A pruning target of Douglas-Peucker vertices by location* was TRAINED and failed: E276 arm s (`node_table(sparse=(0.3, 7.5))`): node recall within 2 m 0.143 against a floor of 0.5, CUT at the phase-3 gate; the network cannot decide which of about three candidate cells along a straight lane carries the node. So X1, if it is ever built, needs a loss without a per-cell target along the line (the reconstruction of the line through the kept nodes plus a budget, the form E297 proposes and never ran, or an ordered one-to-one assignment along the piece), with E276's gate as its kill. The literature points the same way: published set heads place a fixed number of equidistant points per element (10 - 30), not simplification vertices (section 8).
- *"Minimum spacing is a property of the code."* False as written: one node per cell guarantees nothing about nodes in neighbouring cells (a node sits at the piece point nearest its cell centre, an event at the event position; 32 % of B's GT nodes lie within 1.75 m of another, E292). Two readings of D104 are kept apart: (i) the lattice is not finer than 2 m / 2.5 m (holds by choosing the lattice; this excludes the 1 m lattice of E344's Q-Lane); (ii) no two delivered nodes closer than 2 m / 2.5 m: needs a merge radius in the TARGET (nodes of two pieces that run closer than the radius, e.g. the two branches right after a fork, are merged, so the fork moves downstream): step 0 measures its price on the GT; E356 measures the same radius as a read-out step on the stored dumps.
- *"Everything soft until the final threshold."* No. Named: (1) the survivor choice before connect 2 is an index choice in the MIDDLE of the path (the D7 convention of today's top-K; a deviation from D55's "selection only on the output"), with GT tokens injected in training, a train / evaluation difference like today's GT cells; (2) the forward cone of connect 1 and the geometric `allowed` gate of connect 2 are candidate selections; (3) the final threshold. Whether the count of rule steps falls is counted with the table of E344 section 1 in the stage cards, not claimed here. Only X3 (free queries) has no selection before the threshold.
- **Inside the junction (D104, read faithfully):** nodes are SET at the lattice positions inside the junction and pruned to a few per connector (not to none). E346 rejected only the unpruned chain (a node in every cell a connector crosses: 57 - 60 % of the connectors collide). Whether a pruned chain collides, and whether it buys anything over the direct entry -> exit edge, is the new measurement of step 0. If it does not, the core keeps the direct edge and the interior tokens stay context only.

## 3. Simpler or not, counted honestly
| | system of record (E324 s on B) | token design, minimal core |
|---|---|---|
| node families | 2 (B's nodes; keypoints) | 1 |
| heads | node, link, section walk, map assigner; maps head, pair head, attach | token layers, connect 1, pair head; the map assigner (8.3 ms, attaches the graph to the SD map) STAYS and is in every gate |
| edge deciders | 3 (B's short links, B's long links, the pair head) | 1 |
| glue | snap + one-piece wiring, attach | none |
| suppression | close-node step (rule), peak rule | learned through existence + attention (three earlier tries to learn it failed: E277, E292, E347) |
| GT codes | B's node table; junction maps + pairs | one node table (+ the junction pairs as edges) |
| loss terms | about 10 across two stages | about 8 in one stage |
| training stages | B, then maps, then pairs (frozen in between) | one |
The gain is in the rows "families", "edge deciders", "glue" and "stages". It is not in the amount of code or loss terms, and the learned suppression is the open risk.

## 4. Ground truth, matching, loss (core)
- **GT code:** `qtargets.node_table` on the chosen lattice (the lattice size is a module constant today and the file is under the E343 freeze: a runtime parameter as E346 patched it), events always kept, junction pairs as entry -> exit edges with fitted handle scales. No sparse target in the core (E276).
- **Matching:** by location (B's rule, the one that trained on this data), with one-to-many targets early and one-to-one late (the DDQ recipe: dense, then distinct, then one-to-one), because E347's one-hot cell target fired on both sides of a cell border (18 - 25 % surplus). A global Hungarian matching only for X3.
- **Loss:** existence (focal), type (CE), position (Laplace NLL, E318), heading (cosine), edges (BCE + dual softmax auxiliary), curve geometry (E317): all exist in `training/grid/`.

## 5. Compute (recomputed by the critic from the stated sizes; to be counted before any trained stage)
1600 tokens, d = 96, L = 4, a 12 x 12 window: attention 0.18 GMACs, projections and feed-forward 0.71; connect 1, the message layer, the sampling projections and the decode on top (not counted yet); connect 2: today's pair head costs 0.022 GMACs for 32 x 32 pairs, so 250 x 250 ungated pairs cost about 1.3 GMACs and 0.1 - 0.3 only if the geometric gate passes 8 - 22 % of the pairs (unmeasured). **Honest total: 1.2 - 1.5 GMACs**; global attention over all tokens would add about 2. Against it: the maps head without paint 1.78 GMACs plus B's heads (never counted in GMACs; 2.9 + 5.4 ms) and the pair head; the pyramid (4.68 of about 8 GMACs) is untouched. So the saving in MACs is modest; the claim of this card is fewer parts, not less compute. Time: the junction forward of today is 14.7 ms (E312) / 19.7 ms (E324 s) on a quiet host (E351); a head of 30 - 40 kernel launches is launch-bound at batch 1 (a guess of 4 - 8 ms), measured with E351's protocol in the first trained stage. Training memory of the window gather at batch 8 is unmeasured.

## 6. Risks and kills
1. **Ends on a coarse lattice** (E347: the ends on a 2 m grid lost 0.05 - 0.125 keypoint F1 and 0.17 - 0.19 crowded recall, both arms failed). Answer: 1 m patch sampling; kill in stage T2 below.
2. **Learned suppression** failed three times (E277, E292, E347) and a free set slides along the lane (literature); kill in T1.
3. **Our own negative:** E32 (96 free DETR-style queries, old stack) lost 0.058 F1 and 0.178 route F to the grid head. This time differs by anchored lattice tokens on frozen features and one-to-many labels early; X3 repeats E32's idea and is last.
4. **Junction F1 falling back to B's 0.12 - 0.13:** B had this code. The difference here is the pair head with evidence along the curve (worth 0.012 - 0.023 unsnapped topo F1, E331 / E332) on tokens that read 1 m features; the evidence maps (lane heat, zone, core) come from the maps head today, which the design removes: the core must predict them as auxiliary token outputs or sample B's lane heat only: open, decided in T0.
5. **Lattice phase** (E289: B's nodes are locked to the grid phase; a 1 m shift moves 25 % of them): matters for a recursive state (E354) on tokens; reported as jitter in T1.
6. **Few labels per token / slow schedules** (literature: one-to-one alone trains slowly).
7. **Compute** above the system of record fails D86.

## 0. The card of step 0 (CPU, no training; the only runnable part; built with the E346 / E356 tool by its owner, the lane experiments chat, after E356 has run)
```json card
{
 "id": "E355",
 "card": "",
 "title": "step 0: the ground truth through the D104 variant of B's node + link code: nodes ALSO inside the junction (intersection suppression off), pruned along every piece to the sparse set of E276 (kept where the chord leaves the piece by more than 0.3 m, at most 7.5 m between kept nodes, events always kept), on the 2.0, 2.5 and 3.125 m lattice, with and without a merge radius equal to the lattice in the target; decoded and scored with the standard scorer on the 873 VAL frames against the undecoded lane GT",
 "kind": "block",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.03, "vs": "g1"},
 "arms": {
  "g1": {"recipe": "control: the undecoded lane GT (arm g1 of E346, stored: topo F1 0.9858, lanes found 1.0000, F1@1.0 1.0000, junction F1 0.9995)"},
  "v20": {"recipe": "the variant on the 2.0 m lattice (suppression off, sparse 0.3 / 7.5)", "run_ids": ["E355.v20"]},
  "v25": {"recipe": "the variant on the 2.5 m lattice", "run_ids": ["E355.v25"]},
  "v31": {"recipe": "the variant on the 3.125 m lattice (the E346 tool refuses 3.0 m)", "run_ids": ["E355.v31"]},
  "m25": {"recipe": "v25 with a merge radius of 2.5 m in the target (reading (ii) of the spacing rule)", "run_ids": ["E355.m25"]}
 },
 "control": "g1",
 "variant": "v25",
 "pairing": "position",
 "seeds": [0],
 "rule": [
  {"metric": "topo_F1", "vs": "g1", "max_loss": 0.03},
  {"metric": "lanes_found", "vs": "g1", "max_loss": 0.03},
  {"metric": "F1@1.0", "vs": "g1", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the tool is deterministic: with the suppression ON and the sparse set OFF at 2.0 m it must reproduce E346's arm l2 to 1e-6 (topo F1 0.9669, lanes found 0.9853, F1@1.0 0.9957, junction F1 0.9535), and with the sparse set 0.3 / 7.5 the row of E276 P1 (the same four numbers); the red `test_qtargets.py` of E346 section 4 stays named as open",
 "evaluate": {"split": "val"}
}
```
**Rule in words (by hand, label "by hand: GT replay"; `decide.py` on the block alone is not the verdict):** the question is whether kept nodes inside the junction are a code worth predicting. A variant arm PASSES when it loses at most 0.03 against the lane GT on topo F1, lanes found and F1@1.0. **The interior nodes are WORTH IT** (X2 goes into the design) when the passing variant has a junction F1 at least 0.01 ABOVE E356's arm of B's code on the same lattice (direct edges only) AND at most 5 % of the connectors still have a collision cell after the pruning; otherwise the core keeps the direct entry -> exit edge. **Lattice:** the coarsest lattice whose variant is within 0.01 topo F1 of v20. **Merge radius:** m25 is reported against v25: the share of events merged or moved and the losses are the price of reading (ii).
**Reported for every arm:** kept nodes per frame (mean, 95th percentile, maximum: the size of the survivor set), kept nodes inside the junction per connector, the collision share after pruning, the share of connector length that comes back through no link (5.5 % at 2 m in E346), the distances between kept nodes (share closer than 2.0 and 2.5 m), junction F1, the lane / connector split as in E346.
**What is already known and is NOT re-measured here:** the sparse set with the suppression ON gives the numbers of l2 to four decimals at 2 m (E276 P1); B's code at 2.5 and 3.125 m and the close-node radius of 2.0 / 2.5 m are E356's arms and are the reference rows of this result.
**Prediction (guesses):** v25 loses 0.03 topo F1 (at the bar); its junction F1 is NOT 0.01 above E356's (the pruned interior nodes still collide for more than 5 % of the connectors, because crossing connectors cross whatever the pruning): X2 is not worth it and the direct edge stays; m25 merges 5 - 10 % of the events.

## 7. Trained stages (each its own card with a rule before the run, two seeds, VAL, a fresh critic; none before Mirko's recorded decision)
| stage | what | gate (numbers; anything else is reported) |
|---|---|---|
| T0 | ORACLE nodes: the GT nodes of the chosen lattice as tokens, only the ONE pair head trained (the scan's first step: is one head for all edges enough when the nodes are right?) | delivered-graph topo F1 at least 0.90 and junction F1 at least 0.60 on both seeds (the code's ceiling is 0.967 / 0.95; B's learned junction F1 is 0.13, the oracle with B's node recall 0.63, E273) |
| T1 | the core on B's frozen pyramid, lanes only (tokens, layers, existence, the pair head on lane edges) | the delivered graph against B + close-node step, no Douglas-Peucker stage in either: topo F1 not below 0.626 / 0.625 by more than 0.01, dup_frac not above by more than 0.01 WITHOUT a close-node step; kill: node recall within 2 m below 0.5 at the first gate (E276's gate) |
| T2 | + entry / exit types | entry / exit recall within 0.05 of the maps head without paint (E332 arm a) at matched precision; a miss sends the junction-zone tokens to a 1.25 m lattice |
| T3 | + junction edges: the whole graph, no snap, no top-K, no close-node step | topo F1 of the delivered graph not below E324 s snapped by more than 0.01 on both seeds (0.6981 / 0.7010), junction F1 not below E341 by more than 0.03, guards as E341, with the map assigner in the path |
**ADOPT (D86 weights):** T3 passes on both seeds AND parameters, GMACs and ms per frame (quiet host) are not above the system of record: then the token head is the simpler system at equal accuracy and is OFFERED for adoption; the adoption itself is Mirko's recorded decision (D94). A pass with more compute is "kept and refined" (D87).

## 8. Literature (`literature/scan_2026-11-15_set_based_lane_graph.md`, 15 table rows; the full-text numbers went through a summarising fetch, digest grade; deep reads of Relationformer and CGNet are ordered)
- **Closest designs:** Relationformer (one set of node tokens plus a relation token, Hungarian on nodes, connects once at the end, tested on 20 - 80 tokens, no BEV lane result); CGNet (adjacency per decoder layer: 1 / 3 / 6 rounds give TOPO 39.6 / 41.3 / 42.2, a junction heat map, a Bezier head for connected pairs; 38.8 M parameters, 241.9 GFLOPs, 11.2 FPS: a compute warning); TopoPoint (vertex and lane tokens in one decoder); UniTopo (a connection as its own query). **No lane-graph paper prunes tokens between connection rounds**; DDQ is the detection precedent (dense, then distinct, then one-to-one). "Connect, prune, connect" is a new combination with no published number.
- **Nodes and keypoints** are one object in the set designs (a junction is a class, a heat-map aid, or a degree); two objects exist only where a dense map is used, as in our chain.
- **Spacing:** no paper sets a minimum node distance in metres; practice is 10 - 30 equidistant points per element and merge radii of 0.15 - 1.5 m; denser points helped once (LaneGAP 10 -> 40 points: TOPO 0.450 -> 0.472) and saturated once (MapTR). The published vertex target on straight lines is equidistant points, not simplification vertices (consistent with E276's failure).
- **Junction interiors** are tied to their two ends in the published remedies (a Bezier of the connected pair, an end-to-start distance prior, the connection as a query): edges, not free interior nodes.
- **Training risks seen in the papers:** duplicates on a free set; few tokens with labels under one-to-one matching (RATopo 23.9 -> 32.2; MapTRv2: one-to-many reaches in 24 epochs what one-to-one needs 110 for); schedules of 110 - 400 epochs.

## 9. Referee
First pass 2026-11-15 (fresh `ar-critic`, mode cards, no code run): REVISE as a design card, and step 0 "do not build as written". All eleven findings are taken into this version: E276 arm s cited and the importance pruning moved out of the core; step 0 reduced to what is new (the kept interior nodes, the merge radius, the counts) and tied to the E356 tool with exact kill rows; the spacing, pruning, softness and selection-count claims corrected; the D104 reading of the junction as an arm; the history corrected (attention existed; B had ends as nodes); the honest count of simplicity; the compute recomputed (1.2 - 1.5 GMACs); the missing risks; gates with numbers and T0 (oracle nodes) first; the literature from the scan. It verified 0.019, 57 - 60 %, 0.014 m, 0.12 - 0.13, E283, E284, 0.626 -> 0.6981 / 0.7010, the E347 numbers and the E330 GMACs and ms. A second fresh pass precedes the build of step 0.

## 3b. Summary
(empty until step 0 exists)
