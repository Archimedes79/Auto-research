
### 0. Summary

**Problem.** In the current best method the way from the frozen features to the graph runs through hand-built blocks: a dense head (learned) gives a heat map and a direction field, a point-by-point **line tracer** (`line_tracer.py`) turns them into polylines, a rule-based **assembler** (`graph_from_dense.py`) orients, cuts and links them, a **key-point head** with a node-guided cut (`node_head.py`, `node_assemble.py`) and **Bezier connectors** (E53) repair the topology, a **cascade refiner** (`refiner.py`) verifies the points, and the **edge cutting** of loop 8 (`p8_seg.py`) makes the 6 m tokens. Only the dense head, the key-point head, the refiner and the assignment network have parameters. For an end-to-end network these hand-built blocks (the red ones in the figures) should not be in the main path.

**Proposal.** Replace the whole extraction by one differentiable network that returns a **sparse graph** (a padded set of vertex and edge tokens with masks), and build the graph **step by step from near to far**: seeds near the ego vehicle first, a learned point-by-point trace network that grows each lane outwards ring by ring (12.5 / 25 / 37.5 / 50 m), every far step conditioned on the already finished near graph. The trace network is the learned form of what worked best so far (the tracer with its fan of headings, centring, on-lane check and merge on contact); two alternative arms (a vertex-first graph transformer in the style of SAM-Road, and a ring-progressive anchored set decoder) are included as controls. Everything is trained by imitation of the ground-truth graph (RNGDet++ style: noisy states, corrective expert actions), first by **distilling the rule pipeline** to prove parity, then by ground-truth imitation with the features.

**What the measurements say (new, section 1).** The perceived graph loses accuracy with range (F1 at 1 m 0.957 in the first 10 m, 0.874 at 30 - 40 m, 0.800 at 40 - 50 m); most of the far loss is offset (recall at 2.5 m is still 0.935 at 40 - 50 m), not missing lanes; a far lane is found more often when its near part is (0.92 against 0.69), but a hard gate on verified near parts costs more recall than it gains precision, and the near geometry predicts the next 10 m, not the next 30 m. A near-to-far design should therefore condition softly, read the features at every step, and treat range as an input.

**Experiments.** Twelve experiments (E92 - E103) in five phases after the measurements (E91, done), with three decision gates (parity with the rule pipeline, gain in the far band, parity of the integrated route success), about 50 GPU hours in total. An honest expectation is a small gain in F1 (the dense field is the known bottleneck, E26 / E38) and a larger gain in the one thing the rule pipeline cannot do: it removes every red block from the main path and lets the planner gradient reach the extraction.

### 1. Where the current extraction loses accuracy

All numbers are new (2026-10-17, `scripts/p10_range.py`, `p10_extrap.py`): the perceived loose graph of the clean-trained dense head `e37_final` and the relaxed tracer of E57 (the proposals of the assignment network), against the ground-truth lane graph; TEST has 833 frames, VAL 873.

| range from the ego | precision @ 1 m | recall @ 1 m | F1 @ 1 m | recall @ 2.5 m | lost to offset (R2.5 - R1) | lost completely (1 - R2.5) |
|---|---|---|---|---|---|---|
| 0 - 10 m | 0.976 | 0.939 | 0.957 | 0.992 | 0.053 | 0.008 |
| 10 - 20 m | 0.962 | 0.925 | 0.943 | 0.990 | 0.065 | 0.010 |
| 20 - 30 m | 0.936 | 0.891 | 0.913 | 0.980 | 0.089 | 0.020 |
| 30 - 40 m | 0.898 | 0.852 | 0.874 | 0.968 | 0.116 | 0.032 |
| 40 - 50 m | 0.833 | 0.770 | 0.800 | 0.935 | 0.165 | 0.065 |

[[fig: report_final/figures/p10_range.png | Left: precision, recall and F1 at 1 m and recall at 2.5 m of the perceived graph by range from the ego (TEST). Middle: share of the far part (30 - 50 m) of a ground-truth lane that is found, when the near part (up to 20 m) of the same lane is found (>= 80 % within 1 m) or missed. Right: ground truth only, lateral error of a straight or constant-curvature continuation of the last 15 m of the near part.]]

Four findings shape the design.

1. **Far errors are mostly offset, then false lines, then misses.** From 30 m on, 12 - 17 % of the ground-truth points lie 1 - 2.5 m beside the graph, 3 - 6 % are missed completely, and precision falls to 0.83 at 40 - 50 m. A learned step that reads the features at the candidate position (the 'centring' of the tracer) addresses the first; a range-aware evidence threshold the second.
2. **The far part depends on the near part, softly.** The far part of a ground-truth path is found for 91.6 % of its points when its near part is found, against 69.3 % when the near part is missed (TEST; VAL 91.5 % against 82.2 %; 2,754 against 291 ground-truth polylines on TEST). Of the false far points (more than 1.5 m from the ground truth, beyond 30 m), 92 % (TEST; VAL 85 %) sit on predicted lines that have no near part or whose near part is not a verified true line; only 8 % extend a verified near line. A **hard** gate on verified near parts (an oracle check against the ground truth) raises the far precision from 0.834 to 0.869 but lowers the far recall from 0.780 to 0.673, because 15 % (VAL 11 %) of the far ground-truth points are in graph components without a near part (side roads that start beyond 20 m, annotation pieces). The conditioning must be soft: near-graph tokens as input, not a rule.
3. **The near geometry does not extrapolate far.** On ground-truth paths, a straight continuation of the last 15 m of the near part has a median lateral error of 0.34 m at 10 m beyond, 1.14 m at 20 m and 1.75 m at 30 m (only 47 % and 40 % of the paths within 1 m at 20 and 30 m); a constant-curvature model is worse (0.37 / 1.62 / 2.47 m). The history (6 vertices, about 12 m) is a good input for the next steps; beyond 10 m the features have to carry the lane.
4. **Topology, not points, is the largest gap.** The system finds 0.844 of the ground-truth points but recovers only 0.428 of the ground-truth paths (TEST, E46): the annotation has about twice as many pieces (31.5 against 15.7 per frame) and fewer interior dead ends; split / merge F1 at 6 m is 0.55 / 0.61 after the key-point cut and 0.71 / 0.64 with Bezier connectors (E44, E53), at a small cost in route F.

What earlier experiments established (all in the report, sections 4 and 5):

| result | number | consequence for the design |
|---|---|---|
| the field, not the sparsifier, is the bottleneck (E26, E38) | tracer on the oracle field F1 0.993, on the predicted field 0.88 - 0.90 (VAL); relaxed proposals cover 98.2 % of the ground-truth points within 2.5 m, 87.5 % within 1 m | gains must come from reading the features at the vertices and from graph context, not from a better tracer rule |
| a sparse-only decoder underfits (E32, 'LaneFormer') | F1 0.821 against 0.879 for the grid approach, route F 0.587 against 0.765, training F1 only 0.880 | do not start from scratch with Hungarian-matched queries: anchor to proposals and distil the working pipeline first |
| the cascade refiner mostly verifies (E39) | +0.001 F1, +0.014 route F, existence head lifts precision 0.898 -> 0.919; 1 block = 3 blocks | the refinement is information-limited at 1 - 2.5 m; its existence head is worth keeping inside the step network |
| learned link heads tie the rules (E42, E45); key points add the topology (E44, E53) | links within 0.01; split / merge F1 0.34 -> 0.55 (key points) -> 0.71 (connectors) | topology gains came from learned nodes and connector geometry: fork / merge events and pointers are the right place for learning |
| train-to-unseen shift of the frozen features (E30) | F1 0.932 own training scenes, 0.899 held-out train scenes, 0.851 VAL | training inputs must be out-of-fold or perturbed like detector output; scheduled sampling on own predictions of train frames would be over-optimistic |
| the exact map prior and temporal fusion are separate plug-ins (E43) | +0.066 F1 (map), 0.001 - 0.007 (refiner, temporal) | keep them as additional inputs, not as part of this design |

### 2. Requirements

R1 **No hand-built block in the main path**: no tracer, no cutter, no registration, no tracker between the features and the graph tokens; parameter-free differentiable operations (bilinear sampling, max-pool NMS, top-k, softmax) are allowed inside the network. R2 **Sparse output**: a padded set of at most N_v = 800 vertices and N_e = 900 edges per frame with masks; no raster as the final output. R3 **Near to far** in the training order, the inference order and the conditioning. R4 **Planner interface unchanged**: edges of about 6 m with keep / route attributes (E83 / E87), so that the integrated network can feed the assignment network directly. R5 **Frozen foundation**: the 39 BEV channels (`dec32`, `logit7`, 200 x 200 cells of 0.5 m) are the input; the sensors get no gradient (a documented limit). R6 **Budget**: one 12 GB GPU shared with other jobs, a training run of at most about 1 h, inference at most about 100 ms per frame (the rule chain needs 41 ms per sample without the foundation, E43).

### 3. From algorithm to network: element-level mapping

**3.1 The tracer and its neighbours, element by element.** The left column names the element in our code, the right column its learned counterpart. Every element has a training signal that can be generated from the ground-truth graph or from the rule pipeline's own decisions (for distillation).

| element in the rule pipeline (file: function) | what it does | learned counterpart | training signal |
|---|---|---|---|
| seeds (`line_tracer.py: seeds_from`) | valley minima of the smoothed field below `t_seed`, best first, blocked within `r_seed` of finished lines | **seed head**: 1-channel logit map on F0, max-pool NMS (7 x 7), top-K = 48, sorted by range; block rule becomes the 'covered' input | focal BCE on gaussian targets at ground-truth lane points sampled every ~10 m (plus all seeds of the rule tracer when distilling) |
| step direction (`_scan`: fan of 9 headings within +-45 deg, field read 1 m and 2 m ahead, turn penalty `lam`) | picks the heading with the lowest field | **heading head**: 9-way classification over the same fan plus a residual of +-6 deg; inputs are the features sampled at 9 headings x 3 look-aheads (27 points, three scales) | cross entropy to the corrective expert bin (distillation: the rule's choice) |
| step length (`step` = 1 m) | fixed | **step class** {2, 4, 6 m}: the expert takes the longest step whose chord deviates <= 0.3 m from the polyline (adaptive, sparse by construction) | cross entropy |
| centring (`_center`: weighted centre of the cross-section, +-1.5 px) | moves the new point to the valley floor | **offset head** after a second sampling of the cross-section (7 lateral offsets, +-2.5 m, as `refiner.pool`) at the predicted point | Huber to the signed distance to the ground-truth line |
| on-line check (`t_on`, `gap_max` = 2) | stops after more than 2 steps off the line | **on-lane head** (sigmoid) plus an event class 'end'; the miss counter is part of the head state | BCE with false-alarm weight |
| merge on contact (`cov` stamp, `r_cov`, `merge_deg` = 40 deg) | joins a foreign line at small angle, crosses at large angle | event class 'merge' with a **pointer** over the already emitted vertices (dot product plus an angle bias); crossing = no event | cross entropy over the vertex set + 'none' |
| duplicate rejection (`drop_duplicates`, `r_seed`) | removes short lines lying on longer ones | the **covered crop** (16 x 16 cells, 1 m) of emitted vertices is an input, heads attend to each other (head-to-head attention); duplicates are not rewarded | implicit; checked by the duplicate rate |
| orientation (`graph_from_dense.orient`: mean agreement with the direction field) | flips polylines to the travel direction | each head carries a direction flag (trace with / against the travel direction) and the step network predicts the heading in the travel direction | expert direction from the ground-truth polylines |
| assembly, junction lead (`assemble`: end -> start links by distance and heading, split 5 m upstream and merge 5 m downstream) | creates splits and merges | events 'fork' (spawn a child head with a predicted branch angle) and 'merge' (pointer); the lead is learned from the ground-truth junction geometry | cross entropy / L1 on the branch angle |
| key points (`node_head.py`: split / merge heat maps, NMS, `node_assemble.cut_nodes`) | learned junction nodes bias the cut | the two junction heat maps stay as **auxiliary heads** on the stem; their value at a head is an input of the event classifier | focal BCE as in E44 |
| Bezier connectors (`node_assemble.synth_connectors`: cubic Bezier from a split S to a merge M, accepted by the mean heat) | recovers missing branches | **connector head**: pairwise scores for (fork token, merge token) pairs within 4 - 40 m (geometry kernel x semantic similarity, as TopoLogic / E45); the curve is a cubic Bezier decoded from the two end headings (differentiable) | BCE to ground-truth reachability |
| cascade refiner (`refiner.py: Block`: pool at +-2.5 m, 1-D convolutions along the line, offset and existence) | verifies and smooths | the **second block** of the step network (cross-section pooling) and a final pass of 1-D dilated convolutions along each finished polyline (smoothing, existence) | Huber to the distance field, BCE existence |
| edge cutting, edge features (`p8_seg.py`) and edge classifier (`p8_assign.py`, E83 / E87) | 6 m tokens, 60 hand-built features, keep / route | the step class makes ~6 m edges natively; the token embedding of the step network replaces the hand-built features; the edge transformer stays and receives the SD map as tokens | as in E83 / E87 |

**3.2 The literature, mapped to the same elements.** What we use from each paper is stated in terms of an element of section 3.1 (numbers are author-reported on other datasets and are only guidance).

| method (as scanned in the literature groups 3.1 - 3.14) | idea | where it enters |
|---|---|---|
| RoadTracer (CVPR 2018) | iterative search guided by a CNN decision function; evaluate at a fixed error budget | the step network is its decision function; the evaluation protocol of section 9 (recall at a fixed false length per range band) |
| RNGDet / RNGDet++ (transformer, imitation learning) | graph grown vertex by vertex; noisy walks with corrective expert actions | training scheme of section 5.6; step outputs with an end / fork event |
| SAM-Road (CVPRW 2024) | vertex NMS on probability masks, edge existence by a light transformer over candidate edges | arm C (vertex-first), the seed head, and the final edge transformer (E83) |
| Sat2Graph (ECCV 2020) | graph-tensor encoding, single pass, decoding not differentiable | **not used** (decoding would be a red block) |
| LaneGAP (ECCV 2024) | lanes as root-to-leaf paths, path set prediction | the path-level evaluation (E46) and an optional path-consistency loss on the traced chains |
| SeqGrowGraph (ICCV 2025) | autoregressive vertex insertion, Bezier centre lines | step events and the cubic Bezier connector |
| TopoLogic / CGNet (2024) | learnable distance kernel x semantic similarity for links; junction-aware queries | connector head (pairwise score) |
| VectorMapNet / MapTR / LaneSegNet-style queries (E32) | polyline set prediction with Hungarian matching | arm B, with anchors instead of Hungarian matching because of the E32 result |
| GLD-Road (ISPRS 2025) | global nodes, then local search that repairs only broken connections | a repair stage for far dead ends: heads parked at dead ends are given one more step with a lowered threshold (experiment E99) |
| MapTracker, HRMapNet, StreamMapNet (memory) | history in BEV or queries | later: warped vertex tokens as seeds (E103), not in the first phases |

### 4. Near to far: six mechanisms

M1 **Ring wavefront.** Heads grow in stages: a head that crosses the ring boundary is parked; the next stage unparks it and adds the seeds of the next ring that are not covered by existing vertices. Near lanes are therefore complete before any far step is taken.
M2 **Conditioning on the finished near graph.** Three channels: the covered crop of emitted vertices (positions of all finished lines around the head), head-to-head attention (other heads of the frame as keys), and the pointer set (all emitted vertices as keys). This is the soft form of 'the far part is found when the near part is': the measured 0.92 against 0.69 is the signal the network can use, a hard gate loses recall (section 1).
M3 **Range as input.** Fourier features of the range and of the polar angle enter the head token; the evidence for 'on-lane' and for the events has a learned range bias, so that far steps need more evidence (the 'short but reliable' rule of the task) without a global threshold.
M4 **Curriculum and scheduled sampling over rings.** Training starts with states up to 20 m, then 35 m, then the full range; the near context of a far state is first the ground truth, then the ground truth corrupted like detector output (10 % dropped lines, 1 m shifts, false lines), and later (ablation) the network's own out-of-fold predictions, so that far decisions learn to correct near errors instead of trusting them.
M5 **Range-weighted losses and sampling.** Training states are drawn with a ratio near : mid : far of 1 : 1 : 2, the loss weight grows with the range (1 + range / 25), and the evaluation reports every metric per range band.
M6 **Multi-scale reading by range.** The pyramid F0 / F1 / F2 (0.5, 1, 2 m cells) is sampled at every candidate; with the range embedding the network can learn to rely on coarse features far away, where the BEV features are blurred.

[[fig: report_final/figures/p10_rings.png | Left: the ground-truth lane graph of one TEST frame split into the four stages. Right: what each stage does. The ring radii are a starting point (12.5 m steps), to be tuned on VAL.]]

### 5. Arm A: the learned trace network, at element level

**5.1 Inputs and data (files that exist).** Features: `~/exp_gcar_b123/feats/{part}_dec32.npy`, `{part}_logit7.npy` (memory-mapped, 39 channels after concatenation, 200 x 200, rows = frames; loader `dense_head.Data`), normalised with the channel statistics of the training set. Ground truth: `graph_gt_{part}.pkl` (per frame a dict with `pts` = polylines in metres in the ego frame, in the travel direction, `edges` = successor links, `kind`, `start_open`, `end_open`). Auxiliary dense targets: the exact distance field (`gt_dist_exact`) and the direction field (`dir_{part}.npy`) of the dense head. For distillation: the out-of-fold heat and direction maps of the three cross-fitted dense heads (`train_oof`, as used for the assignment network) and the decisions of the rule tracer logged by a copy of `line_tracer._trace` that also returns, per step, the heading index of the fan, the centring offset, the field value and the snap target. Out-of-fold inputs matter: the dense head's own training frames are over-fitted (E30).

**5.2 Modules.** (Sizes: batch B frames, N heads per frame, token width d = 128.)

| module | input -> output | details | parameters |
|---|---|---|---|
| `Stem` | x [B, 39, 200, 200] -> F0 [B, 64, 200, 200], F1 [B, 96, 100, 100], F2 [B, 128, 50, 50] | two 3 x 3 conv blocks (BatchNorm, ReLU), two max-pool + conv blocks; optional input channels: heat (1) and direction (2) of the dense head (frozen at first, jointly trained later) | 0.55 M |
| `AuxHeads` | F0 -> heat [B,1,200,200], direction [B,2,...], split / merge heat [B,2,...] | 1 x 1 convolutions; focal BCE as E27 / E37 / E44; keeps the stem honest and gives the events a prior | 0.02 M |
| `SeedHead` | F0 -> seed logits [B,1,200,200] -> top-K (K = 48) after 7 x 7 max-pool NMS, sorted by range | sub-pixel offset by a 2-channel regression; each seed spawns two heads (with / against the travel direction) | 0.02 M |
| `HeadState` | per head: position (2), heading (2: cos, sin), direction flag, history = last 6 vertices in the heading frame (12) and step classes (6), miss counter, range embedding (16 Fourier features of range and polar angle), covered crop [1, 16, 16] (emitted vertices of all heads, rotated into the heading frame, 1 m cells, width 1.5 m) | MLP -> head token [d] | 0.1 M |
| `Sampler` (parameter-free) | features at 27 fan points (9 headings of +-45 deg x look-aheads 2 / 4 / 6 m) and at 7 lateral offsets (+-2.5 m) of the candidate -> [N, 27, 291], [N, 7, 64] | `grid_sample` on F0, F1, F2 and on heat + direction (291 = 64 + 96 + 128 + 3); differentiable in the sampling positions, so the second block can move the point | 0 |
| `TraceBlock` x 2 | tokens: head (1) + fan (27) + history (6) + covered crop patches (16) + [block 2: cross-section (7)] -> updated head token | 3 transformer layers, 4 heads, feed-forward 512, pre-norm; after each layer a **head-to-head attention** over the N heads of the frame (keys: head tokens, bias: negative distance); block 1 predicts heading and step, block 2 re-samples at the predicted point and predicts offset, on-lane, events | 0.8 M |
| `StepHeads` | head token -> heading logits [9] + residual [2], step class [3], offset [1], on-lane [1], event [5] (continue, end, fork, merge, border), branch angle [1], pointer logits [M + 1] | pointer = scaled dot product of a projected head token with the embeddings of the M emitted vertices (plus the cosine of the heading difference as a bias) and a 'none' slot | 0.1 M |
| `PolyHead` (final pass) | each finished polyline [n, d] -> smoothed points, existence per vertex, orientation | dilated 1-D residual convolutions along the line (as `refiner.ResConv1d`), as in E39 | 0.2 M |
| `ConnectorHead` | pairs (fork token, merge token) within 4 - 40 m -> score, Bezier control scale | geometry kernel x semantic similarity (E45) | 0.1 M |

About 2 M parameters in total; one inference step of 48 heads costs well below 10 ms on the RTX 3060 (estimate), a frame needs about 25 steps of 2 - 6 m per lane, so about 100 ms with batching (to be measured).

**5.3 One step, exactly.** (1) Sample the fan and the covered crop; (2) block 1: heading bin k, step class c, residuals -> candidate point q = p + (c + d_l) u(theta + phi_k + d_phi); (3) block 2: sample the cross-section at q, offset o -> q' = q + o n; (4) on-lane probability, event distribution, pointer; (5) emit vertex q' with embedding v = MLP([q', theta', head token]); (6) apply the event: 'continue' (head moves to q'), 'end' (head stops), 'fork' (head continues, a child head starts at q' with heading theta' + psi), 'merge' (edge from q' to the pointed vertex, head stops), 'border' (head stops); (7) a head whose on-lane probability stays below the range-dependent threshold for more than g = 2 steps stops (the learned form of `gap_max`).

**5.4 Inference loop** (the rings of section 4):

```
heads = seeds(top-K, near first)                      # SeedHead, both directions
for stage s in 0..3:                                   # rings 12.5 / 25 / 37.5 / 50 m
    heads += new_seeds(ring s, not covered by emitted vertices)
    while any(head.active and head.range <= R[s]):
        out = TraceNet(features, heads, vertices)      # one batched step for all heads of all frames
        vertices += emit(out); heads = apply_events(out)
        park(heads with range > R[s])
    unpark(parked heads)                               # they continue in the next stage
graph = PolyHead(chains) + ConnectorHead(fork, merge pairs)   # sparse vertex / edge tokens, masks
```

**5.5 Losses.** L = L_head + L_step + L_off + L_on + L_event + L_ptr + L_seed + L_aux + L_graph. L_head, L_step, L_event, L_ptr: cross entropy (events class-balanced with focal weights); L_off: Huber on the signed distance of the emitted point to the ground-truth line (sampled from the distance field, differentiable, as `refiner.losses`); L_on: BCE with false-alarm weight lambda_fp = 3 as in E32 / E39; L_seed: focal BCE as E44; L_aux: the dense-head loss of E37; L_graph: on the final graph the point existence and a path-consistency term (the predicted chain from a root to a leaf must stay within 1.5 m of one ground-truth path, LaneGAP-style, optional). All range-weighted by w(r) = 1 + r / 25.

**5.6 Training (expert, noise, curriculum).** *Expert:* for a state (p, theta, history) the expert projects p on the ground-truth polyline (arc position s*), takes the longest step c in {2, 4, 6} m with chord deviation <= 0.3 m, the heading bin of the chord, the offset of the endpoint to the line, and the event from the ground-truth graph (end if the polyline ends within the step, fork if another successor starts within 3 m, merge if the endpoint lies within 1.5 m of another polyline with a heading difference < 40 deg, border beyond 48 m). *States:* 32 states per frame drawn from the ground truth with lateral noise N(0, 0.5 m) (clipped at 1.5 m) and heading noise N(0, 10 deg); the expert action corrects back to the line (RNGDet++-style imitation, a cheap form of DAgger). *Context:* the ground-truth polylines within the range of the state (plus 2 m) form the covered crop and the pointer set, corrupted with probability p_c (10 % lines dropped, 1 m shifts, 5 % false lines). *Curriculum:* epochs 1 - 4 only states up to 20 m, 5 - 8 up to 35 m, then the full range; the near : mid : far sampling ratio 1 : 1 : 2. *Scheduled sampling:* from epoch 9 the context is, with probability p_ss (rising to 0.5), the network's own graph of the lower stages (out-of-fold or perturbed inputs, section 5.1) instead of the corrupted ground truth. *Optimiser:* AdamW, one-cycle, learning rate 1e-3, weight decay 1e-2, batch 16 frames x 32 states, 12,000 iterations (about 30 - 40 min, input-bound: 16 frames x 39 x 200 x 200 half-precision values per step), mixed precision.

**5.7 From tokens to the planner.** The emitted vertices and edges are the sparse graph. The assignment network of E83 / E87 receives per edge the learned token embedding (128-d), the edge length and heading, and the SD-relative features computed by differentiable operations (distance and signed offset to the localised SD route, lane count); the localisation is replaced in phase 5 by SD tokens that the edge transformer attends to. The gradient of the route loss flows through the edge transformer into the step network's tokens (the frozen foundation is the only place where it stops).

### 6. The two alternative arms (controls)

**Arm C: vertex-first graph transformer (SAM-Road-style, parallel).** Vertex candidates: the K = 400 peaks of the lane heat map after max-pool NMS (radius 2.5 m) plus the junction heat maps; vertex token = features sampled at the peak (three scales) + position + range embedding + heat value. Candidate edges: k = 8 nearest neighbours within 12 m with heading compatibility (at most 3,200 edges); edge token = [h_i, h_j, relative position, heat integral along the segment (8 samples), direction cosine]. A 4-layer graph transformer (attention over the edge list, relative-position bias) outputs the edge existence and direction and a per-vertex offset; **near to far inside the network**: layer l may only attend to tokens within the radius R_l (12.5, 25, 37.5, 50 m), so that far tokens read the already refined near tokens; training with class-balanced BCE on edges labelled by consecutive ground-truth vertices. Decoding is a threshold on the edge probabilities (no tracer). Cheap, stable and fully parallel; its weakness is the vertex recall of the peak proposals and the lack of per-step feature reading along the lane.

**Arm B: ring-progressive anchored set decoder.** Three decoders (rings 0 - 20, 15 - 35, 30 - 50 m, overlapping by 5 m) with M = 48 queries each, every query anchored at a point where a lane crosses the ring boundary (ground truth in training, peaks of the heat map on the boundary in inference) and predicting 11 points over 20 m, an existence logit and a successor logit; deformable cross-attention to F0 / F1 / F2 and to the tokens of the previous ring. Assignment by the nearest ground-truth polyline at the anchor (no Hungarian matching, the reason E32 trained slowly), deep supervision on every layer, auxiliary dense loss, warm start of the stem from the dense head. It is the cleanest answer to 'what if E32 is repaired'.

### 7. Code plan

| file (new, `scripts/`) | content | main elements |
|---|---|---|
| `p10_expert.py` | expert state generator from `graph_gt_*.pkl`; replay | `ExpertGraph(gt)`: arc-length parametrisation, `project(p)`, `step_targets(p, theta)` (bin, class, offset, event, pointer target), `replay(seeds)` -> graph (used by E92) |
| `p10_rulelog.py` | copy of `line_tracer` with logging | `_trace_log` returns per step (heading index, centring offset, field value, miss counter, snap target); `distill_dataset(split)` writes `distill_<split>.npz` |
| `p10_data.py` | datasets | `StateData` (features by memory map, 32 noisy states per frame, covered crop raster, context corruption), `FrameData` (full frames for inference), collate with padding and masks |
| `p10_net.py` | network | `Stem`, `AuxHeads`, `SeedHead`, `sample_fan(...)`, `sample_cross(...)`, `TraceBlock`, `StepHeads`, `PolyHead`, `ConnectorHead`, `TraceNet` (forward for training states and for a step of the wavefront) |
| `p10_train.py` | training | stages, curriculum, scheduled sampling, mixed precision, checkpoints, queue-runner compatible arguments (`--arm`, `--distill`, `--curriculum`, `--sched`, `--rings`) |
| `p10_infer.py` | wavefront inference | `Wavefront` (rings, parking, seeds, events, pointers, border), graph assembly, dedupe-free output (`graphs_<name>_<split>.pkl` in the format of `graph_eval`) |
| `p10_eval.py` | metrics | reuses `graph_eval.evaluate` (point P / R / F1, route F, split / merge F1, direction, lanes found, false length), `path_eval` (path F1), new: per-range-band metrics (from `p10_range.py`), recall at a fixed false-length budget per band, paired scene bootstrap (`boot_eval.py`) |
| `p10_arm_c.py`, `p10_arm_b.py` | the alternatives | `VertexGNN`, `RingDecoder` with the same data and evaluation |
| `p10_tests.py` | tests | see below |

Tests that gate every phase: **T1** shapes and masks (padding must not change the output of unpadded frames); **T2** expert replay: the expert's own decisions reproduce the ground-truth graph with point F1 at 1 m >= 0.99 and split / merge F1 at 6 m >= 0.9 (this also validates the graph assembly and the metrics); **T3** rule replay: the logged decisions of the rule tracer, replayed through the same decoder, reproduce the rule graph exactly (the oracle field gives 0.993, as in E31); **T4** overfit 32 frames to point F1 >= 0.98 within 2,000 iterations; **T5** permutation test: shuffling the order of heads or seeds changes the output graph by less than 0.002 F1; **T6** no future leakage: stage s output is independent of the features beyond ring s (checked by zeroing them); **T7** range-band metrics of the ground truth against itself equal 1.

### 8. The set of experiments

[[fig: report_final/figures/p10_roadmap.png | Phases and decision gates. G1 stops the learned tracer if it cannot reproduce the rule pipeline; G2 asks for a measurable gain; G3 compares the integrated network with the modular system.]]

The baseline for every row is the rule pipeline in the same protocol: point F1 at 1 m VAL 0.899 / TEST 0.876, route F 0.799 / 0.744, split / merge F1 at 6 m 0.31 / 0.42 (VAL) before and 0.71 / 0.64 (TEST) after the key-point cut and connectors, path F1 0.43 - 0.49 (TEST), per-range F1 of section 1 (the final graph's range bands are measured in E91b).

| id | question | arms and set-up | metrics, success criterion | cost / risk |
|---|---|---|---|---|
| E91 | Where are the errors by range? (done) | section 1; E91b: the same bands for the final rule graph (E43 + E44 + E53), by error type (offset, miss, false, topology) | table of section 1; band F1 of the baseline | 1 CPU-hour; done |
| E92 | Is the code path correct? (tests T1 - T7) | expert replay on the ground truth; rule-decision replay on the oracle field and on the predicted field | replay F1 >= 0.99 (GT) and the rule graph reproduced exactly; metrics of the ground truth = 1 | 2 CPU-hours; a failure means a bug, not a result |
| E93 | **Parity (G1):** can the network reproduce the rule pipeline? | distil the logged decisions of `line_tracer` on out-of-fold fields (8,414 frames): inputs (a) heat + direction only, (b) + 39 channels, (c) + covered crop, (d) + history; 2 seeds | VAL F1 at 1 m >= 0.889 (baseline - 0.01), route F >= 0.78, band F1 within -0.02 of the rule graph in every band | 3 CPU-h for the logs + 4 x 1 GPU-h; if (a) fails the network or the loss is wrong |
| E94 | Does imitation of the ground truth with the features beat distillation? | same four input sets, expert = ground truth (noisy states, corrective actions) | **G2:** VAL F1 >= 0.909 or far-band (30 - 50 m) F1 +0.02 over the rule graph on VAL; otherwise the network is a red-block-free equivalent | 4 GPU-h; the dense field limits it (E26 / E38) |
| E95 | Do curriculum and scheduled sampling reduce the propagation of near errors? | uniform ranges against curriculum; context: clean GT / corrupted GT / own out-of-fold predictions; metric: far-band F1 as a function of the near-band error of the same lane | far F1 +0.01; slope of 'far found given near missed' improves (0.69 -> 0.8) | 6 GPU-h |
| E96 | Does the order matter? | ring wavefront against one stage (all heads together) against far-to-near (negative control) | near-to-far at least +0.005 F1 over one stage; far-to-near worse | 3 GPU-h |
| E97 | Which near-to-far conditioning carries the gain? | ablate covered crop, head-to-head attention, pointer set, range embedding one at a time | band F1 and duplicate rate per ablation | 5 GPU-h |
| E98 | Range-aware evidence | learned range bias against a global threshold; recall at a fixed false-length budget per band (RoadTracer-style) | recall at 2 m / frame false length per band, far band +0.02 | 2 GPU-h |
| E99 | Topology by events | fork / merge pointer + connector head against rule assembly + E44 cut + E53 connectors; optional repair step for parked dead ends (GLD-Road-style) | split / merge F1 at 6 m >= 0.65 / 0.60 (TEST, within 0.05 of the rule graph), path F1 >= 0.45, route F within 0.01 | 4 GPU-h; topology is limited by annotation conventions (E31 / E46) |
| E100 | Arm C | vertex-first GNN with layer-wise ring gating, same data and protocol | F1, route F, speed against arm A | 3 GPU-h |
| E101 | Arm B | ring-progressive anchored set decoder | as E100; compare with E32 (0.821) | 4 GPU-h |
| E102 | **Integration (G3):** tokens -> keep / route | the assignment network (E87 recipe) on the learned tokens instead of the hand-built features; frozen against joint training; with and without the SD tokens instead of the registration | route success within 0.01 of the modular system, lane F1 within 0.01, planner consumer error within 0.05 m | 6 GPU-h |
| E103 | Temporal and SD tokens | warped vertex tokens of the previous frame as extra seeds; SD-map tokens with a learned correction instead of the template registration | failure-phase lane F1 against the graph tracks (E85: 0.645 -> 0.755); route success under a per-road bias | optional; 6 GPU-h |

Order and parallelism: E92 and E93's logging run on the CPU while the code of E94 is written; E93 / E94 share the data builder; E95 - E98 are ablations of the best E94 arm and run on three GPU slots; E100 / E101 start after G1 and use the same loaders. Total about 50 GPU-hours and 10 CPU-hours.

### 9. Evaluation protocol

Selection on VAL (22 scenes) only; TEST (21 scenes) once per finalist; at least two seeds per configuration (seed noise of the dense head is 0.0025 F1 on VAL, E55; scene-level 95 % intervals of absolute TEST scores are about +-0.035 F1, paired differences between graphs of the same dense output resolve about 0.005, E54). Every metric of `graph_eval` plus: **per-range-band** point precision / recall / F1 at 1 m and recall at 2.5 m (as in section 1); **recall at a fixed false-length budget** per band (RoadTracer: the operating point is chosen by the error budget, not by the best F1); **path F1** (E46); **duplicate rate** (share of predicted length within 1 m of a longer predicted line); **topology** (split / merge F1 at 3 / 6 m); **runtime** per frame; the **near-to-far dependence** statistic of section 1 (far found given near found / missed) and the false-far-point split (attached / isolated), which is the direct test of whether the network uses the near graph.

### 10. Risks and what the earlier results say

* **The field limits everything** (E26 / E38 / E39): the learned tracer reads the same frozen features as the dense head. A gain is only plausible where graph context helps (offset along a lane, continuation, topology), so the expectation is +0.005 to +0.02 F1 in the far bands and a better topology, not a jump; the main reward is removing the red blocks and opening the gradient path.
* **Sparse-only decoders underfit** (E32): hence anchoring, distillation first, auxiliary dense losses, and arm B only as a control.
* **Autoregressive error accumulation**: scheduled sampling, corrupted contexts, parking at ring boundaries; the near-to-far dependence statistic is monitored.
* **Over-optimistic training inputs** (E30): contexts and distillation inputs are out-of-fold or perturbed.
* **Annotation conventions cap the topology** (E31 / E46: oracle route F 0.84 for the rule assembler): the events are trained on the ground-truth graph itself, so conventions can be learned, but not what the images do not show.
* **Throughput**: 16 frames of 39 channels per step are input-bound; the plan uses half precision memory maps and a cache of the stem outputs only if the stem is frozen.
* **Evaluation noise**: absolute TEST scores carry +-0.035; decisions use paired differences and VAL.

### 11. Decisions I need from you

1. Start with **arm A** (the trace network, E92 - E99) and run arms B and C as controls after the parity gate, as proposed? (The alternative is to start with arm C because it is cheaper and gives a quick red-block-free baseline.)
2. The **ring radii** (12.5 m steps) and the step classes {2, 4, 6 m} are starting values; are there planner needs (e.g. a minimum range for the ego lane, junction look-ahead) that should fix them?
3. Is a **result that only matches the rule pipeline** (gate G2 not reached) acceptable for the end-to-end goal, or should the accuracy gain be a hard requirement?

### Appendix A: code skeletons (to be implemented; shapes are those of section 5)

Conventions as in `refiner.py`: points in metres in the ego frame, first coordinate = row (x), second = column (y); `grid_sample` takes (y / 50, x / 50). The fan samples all heads of all frames in one call.

```python
# p10_net.py - sampling of the fan (parameter-free) and one trace block
def sample_fan(feats, pos, theta, looks=(2.0, 4.0, 6.0), n_dir=9, max_turn_deg=45.0):
    # feats: [F0 [B,64,200,200], F1 [B,96,100,100], F2 [B,128,50,50]] (+ heat/direction as extra channels of F0)
    # pos [B,N,2], theta [B,N] (rad) -> [B,N,27,C]   with C = 64 + 96 + 128 + 3 = 291
    k = (torch.arange(n_dir, device=pos.device) - (n_dir - 1) / 2) * (2 * math.radians(max_turn_deg) / (n_dir - 1))
    ang = theta[:, :, None, None] + k[None, None, :, None]                       # [B,N,9,1]
    d = torch.tensor(looks, device=pos.device)[None, None, None, :]               # [1,1,1,3]
    u = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                         # [B,N,9,1,2]
    q = pos[:, :, None, None, :] + d[..., None] * u                               # [B,N,9,3,2]
    q = q.reshape(pos.shape[0], pos.shape[1], 27, 2)
    return torch.cat([sample_map(F, q) for F in feats], -1)                       # refiner.sample_map: bilinear, [B,N,27,C_l]


class TraceBlock(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, c_fan=291):
        super().__init__()
        self.fan = nn.Sequential(nn.Linear(c_fan, d), nn.GELU(), nn.Linear(d, d))
        self.fan_pos = nn.Parameter(torch.zeros(27, d))                           # (heading bin, look-ahead) embedding
        self.hist = nn.Linear(3, d)                                               # (u, v, step class) of a history vertex in the heading frame
        self.cov = CoveredCrop(d)                                                 # [N,1,16,16] -> 16 patch tokens
        self.layers = nn.ModuleList([TLayer(d, heads) for _ in range(layers)])    # attention inside one head's token set
        self.h2h = HeadToHead(d, heads)                                           # attention between the heads of one frame, key bias = -distance

    def forward(self, head_tok, fan, hist, cov, pos, head_mask):
        B, N, d = head_tok.shape
        tok = torch.cat([head_tok[:, :, None], self.fan(fan) + self.fan_pos, self.hist(hist), self.cov(cov)], 2)   # [B,N,1+27+6+16,d]
        for lyr in self.layers:
            tok = lyr(tok.reshape(B * N, -1, d)).reshape(B, N, -1, d)
            h = self.h2h(tok[:, :, 0], pos, head_mask)                            # [B,N,d]
            tok = torch.cat([h[:, :, None], tok[:, :, 1:]], 2)
        return tok[:, :, 0]


def decode_step(out, st, steps=(2.0, 4.0, 6.0), bin_deg=11.25):
    k, c = out.heading.argmax(-1), out.step.argmax(-1)                            # [B,N]
    phi = (k - 4) * math.radians(bin_deg) + torch.tanh(out.res_phi) * math.radians(6.0)
    ln = torch.tensor(steps, device=k.device)[c] + 0.5 * torch.tanh(out.res_len)
    th = st.theta + phi
    q = st.pos + ln[..., None] * torch.stack([torch.cos(th), torch.sin(th)], -1)  # candidate point; block 2 re-samples here
    return q, th
```

```python
# p10_expert.py - corrective expert for imitation (ground truth only, no network)
class ExpertGraph:
    def __init__(self, gt, ds=0.5):
        self.poly = [resample(p, ds) for p in gt["pts"]]                           # graph_eval.resample
        self.arc = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in self.poly]
        self.succ = successors(gt["edges"], len(self.poly))
        self.tree = cKDTree(np.concatenate(self.poly)); self.owner, self.idx = owner_index(self.poly)

    def project(self, p):                                                          # -> (poly id, arc position s*, distance)
        d, j = self.tree.query(p); i, a = self.owner[j], self.idx[j]
        return i, self.arc[i][a], d

    def ahead(self, i, s):                                                         # point at arc position s, continuing into the successor piece (first successor)
        while s > self.arc[i][-1] and self.succ[i]:
            s, i = s - self.arc[i][-1], self.succ[i][0]
        return interp(self.poly[i], self.arc[i], min(s, self.arc[i][-1])), i

    def step_targets(self, p, theta):
        i, s, dist = self.project(p)
        for c in (6.0, 4.0, 2.0):                                                  # longest step with chord deviation <= 0.3 m
            tgt, j = self.ahead(i, s + c)
            if chord_deviation(self.between(i, s, s + c)) <= 0.3:
                break
        phi = wrap(math.atan2(tgt[1] - p[1], tgt[0] - p[0]) - theta)
        k = int(np.clip(round(math.degrees(phi) / 11.25) + 4, 0, 8))
        off = signed_distance(p + c * unit(theta + (k - 4) * math.radians(11.25)), self.poly[j])
        ev = self.event(i, s, c)                                                   # end / fork (+ child angle) / merge (+ pointer target) / border / continue
        return dict(k=k, c=c, off=off, on=float(dist <= 1.0), ev=ev)
```

```python
# p10_train.py - one training iteration (states, context, losses)
def train_step(net, batch, ep, cfg):
    feats = net.stem(batch.x)                                                      # F0, F1, F2 (+ aux heads)
    states = batch.states                                                          # [B,32]: position, heading, history, range; noisy ground truth
    ctx = batch.context(ep, p_corrupt=cfg.p_c, p_self=cfg.p_ss(ep))                # covered crop + pointer set: corrupted GT or own lower-stage graph
    out = net.step(feats, states, ctx)                                             # blocks 1 and 2, StepHeads
    w = 1.0 + states.range / 25.0                                                  # range weight
    loss = (w * (ce(out.heading, batch.k) + 0.3 * ce(out.step, batch.c) + huber(out.off, batch.off)
                 + bce(out.on, batch.on, fp=3.0) + ce(out.event, batch.ev, focal=True) + ce(out.ptr, batch.ptr))).mean()
    loss = loss + cfg.w_seed * focal_bce(net.seed_logits(feats), batch.seed_target) + cfg.w_aux * dense_loss(net.aux(feats), batch.dense_target)
    return loss
```

```python
# p10_infer.py - wavefront (rings), batched over frames
class Wavefront:
    RINGS = (12.5, 25.0, 37.5, 50.0)

    def run(self, net, x):
        feats = net.stem(x); heads = net.seed_heads(feats); verts = VertexSet(x.shape[0])
        for R in self.RINGS:
            heads.add(net.new_seeds(feats, verts, ring=R))                         # uncovered seeds of this ring
            while heads.any_active(range_le=R):
                out = net.step(feats, heads.state(), verts.context())
                verts.emit(out); heads.apply(out)                                  # events: continue / end / fork / merge / border
                heads.park(range_gt=R)
            heads.unpark()
        return net.assemble(verts)                                                 # PolyHead + ConnectorHead -> padded vertex / edge tokens
```
