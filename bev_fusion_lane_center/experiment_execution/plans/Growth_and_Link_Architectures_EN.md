# Differentiable lane-graph growth and spray-and-link: architecture proposals

Camera + radar BEV, lane centre-line graph with navigation route (nuScenes) · design note, 2026-10-23 · follows loop 15 (E120 / E121)

> **Status: design only.** Nothing in this note has been implemented, trained or measured. Numbers marked *(measured)* come from existing runs (E94, E120, loop 15 and the stored graphs); the geometry numbers in section 4.2 are an illustrative derivation with assumed error sizes, not measurements. Every component is meant to be differentiable (bilinear sampling, attention, expectation, soft masks); what stays discrete is listed per architecture.

## 1. The questions and the short answers

| question | short answer |
|---|---|
| Can a network generate the sampling points itself, at several distances along short and long lines, with wide angles for short lines and narrow angles for long lines? | Yes. A *pattern head* outputs polar offsets (range, bearing) from the state of the tip, exactly as deformable attention predicts offsets from a query; the bearing range is tied to the range by an error model (section 4.2), so the pattern is a pear: short rays wide, long rays narrow. Gradient reaches the pattern through the bilinear reads. |
| Can it pull information from the points, refine its estimate over time, and keep the resulting point in the set to do it again? | Yes: an *inner loop* (two passes: near and wide, then far and narrow along the arc estimated from the near samples), a *commit* of the result into a point set, an *outer loop* in which the children become the new tips, and a *revisit* pass over the whole set (E120). "Over time" can also mean frames: the set of frame t-1, warped by the ego motion, starts frame t. |
| Must it estimate its own validity, and start everywhere? | Yes. Seeds are a lattice (about 1.6 k points) that snap onto the lane and output a validity; most die in the first step. Validity is a soft mask on the losses and on the route head, not a threshold. |
| Alternative: start from points and connect them? | Architecture **L** (spray, snap, link): free query points become oriented lane pieces, then a link head scores candidate pairs inside the same fan, using image evidence along a connector curve. Fork and merge are rows or columns with two links. |
| Both together? | Architecture **H**: one softmax over image rays *and* nearby pieces; choosing a ray grows a new vertex, choosing a piece is a link. |

## 2. Where the current system stands (all *measured*, TEST = 833 frames of 21 scenes)

| fact | value | why it matters here |
|---|---|---|
| E94 samples a fixed fan | 9 bearings (-45 ... +45 deg) x 3 ranges (2, 4, 6 m) = 27 points, +7 cross-section points at the chosen candidate | same angular width at every range; nothing beyond 6 m |
| E94 step | class {2, 4, 6 m} + residual, argmax, trained teacher-forced | the choice is discrete, no loss sees the finished graph |
| E94 steps on the test graphs | mean edge 4.65 m, median 5.8 m, 90th percentile 6.0 m; two thirds of all edges lie between 4.5 and 8 m, i.e. in the 6 m step class (6 m plus residual) | straight lanes get a vertex every 6 m although longer steps would do |
| vertex density | 22 vertices per 100 m of edge (175 per frame) against 14.3 per 100 m (93 per frame) for a Douglas-Peucker oracle on the ground-truth lanes at 0.3 m tolerance (12.7 at 0.5 m; mean chord 7.0 m at 0.3 m, 7.9 m at 0.5 m) | room for a sparser, longer-step representation; E94 graphs also contain false pieces, so the ratio overstates the gain somewhat |
| seeds | oracle seeds 0.921 against 0.908 F1@1 for the rule pipeline; E94 was limited by the number of peaks (48 -> 192) | "start everywhere" attacks the seed limit |
| junction topology | split / merge F1 0.36 / 0.46 (E94) against 0.69 / 0.65 (system A, final graph) | the link head of L attacks the weakest part of B |
| quality of the traced graph | F1@1 0.851, F1@0.5 0.661 (E94); 0.863 / 0.694 after the E120 refinement; 0.872 with the keep gate; system A 0.867 (loose) / 0.875 (final graph) | baselines for the experiments in section 8 |
| E120 finding | vertex error 0.54 -> 0.43 m; half of it comes from sliding along the lane, lateral-only refinement gives 0.49 m at the same F1 (0.860); depth and detaching between layers do not matter | the along-lane position is poorly constrained by a point-to-polyline loss; variable steps mean the loss must not care about it either |
| route head | seed noise up to 0.03 route success between two training runs | every route-head comparison below needs two seeds |
| planner loss | reaches the positions (E121) but gives no gain; too weak to shape topology (E119 / E121) | geometry and link losses stay |

Discrete parts left in B: seeds (peak picking), which vertices exist, edges and topology (event classes, merge pointer argmax), the argmax step, teacher forcing.

## 3. Building blocks used

| block | role in G / L / H | source | status of the reference |
|---|---|---|---|
| bilinear sampling at predicted positions (deformable attention, offsets predicted from the query, reference points refined layer by layer, gradient blocked at the previous reference) | pattern sampling, snap, connector sampling | [Deformable DETR](https://arxiv.org/pdf/2010.04159) | checked by search (details not against the PDF) |
| anchor queries whose extent modulates the attention | oriented-box sampling of a piece (length / width) | [DAB-DETR](https://arxiv.org/pdf/2201.12329v4) | checked by search |
| sparse instances with fixed + learnable keypoints, recurrent propagation of instances to the next frame with ego-pose compensation | pattern with fixed and learned points; temporal start of frame t from frame t-1 | [Sparse4D](https://arxiv.org/pdf/2211.10581), [v2](https://ar5iv.labs.arxiv.org/html/2305.14018), [v3](https://arxiv.org/pdf/2311.11722) | checked by search |
| look-up at the current estimate, update, repeat | inner loop (re-centre, re-sample) | [RAFT](https://arxiv.org/pdf/2003.12039) | checked by search (the recurrent unit being a GRU is from memory) |
| expectation over a softmax (soft-argmax, integral regression) | next point as expectation over rays | [Integral Human Pose Regression](https://alphaxiv.org/abs/1711.08229); bias of the expectation: [Gu et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_Removing_the_Bias_of_Integral_Pose_Regression_ICCV_2021_paper.pdf) | checked by search |
| learned duplicate removal as a relation module | duplicate head for snapped pieces | [Relation Networks for Object Detection](https://arxiv.org/pdf/1711.11575) | checked by search |
| next-N-vertices transformer with validity inside a local window | ancestor of G (and of E94) | [CenterLineDet](https://arxiv.org/pdf/2209.07734) | checked by search |
| lane segments + topology head | pieces + link head | [LaneSegNet](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06d86daa7001b7ea755ac4c80dc93497-Abstract-Conference.html) | checked by search |
| DAG of lane nodes with node states (continue / fork / merge or terminate) | event-free alternative in G and L | [DAGMapper](https://arxiv.org/pdf/2012.12377) | checked by search |
| vertex and edge maps, GNN adjacency; local candidate edges instead of all pairs | candidate lists in L | [InstaGraM](https://arxiv.org/html/2301.04470v3); SAM-Road (read in loop 8, see `loop2/STATE.md`) | InstaGraM checked by search |
| path-level supervision | path term on the soft adjacency | [LaneGAP](https://arxiv.org/pdf/2303.08815v3) | checked by search |
| point-to-polyline loss without matching, MapTR-style refinement | position loss | our E120; MapTR (see `report_final/text/lit_extra15.md`) | read in loop 15 |
| scheduled sampling / DAgger-style replacement of expert states by own states; slot-style competition across child slots; Hermite curve; sparsemax; volume-rendering transmittance (survival along a ray); soft NMS | training recipe, child slots, connector, soft selection of endpoints, ray verification, near-duplicate suppression | Bengio et al. 2015; Ross et al. 2011; Locatello et al. 2020; Martins and Astudillo 2016; Mildenhall et al. 2020; Bodla et al. 2017 | from memory, not re-checked |

## 4. Architecture G: differentiable growth with an adaptive fan

![G architecture](assets/p16_g_arch.png)

### 4.1 Idea

A *tip* is a pose (p, theta), a state token and a validity. A step reads the BEV features at a learned pattern of points in front of the tip, decides where the lane continues by an expectation over the rays (not an argmax), corrects the tip itself, and commits the result into a point set. The committed children are the tips of the next round. Nothing is placed by a grid, a peak or a Hungarian match.

### 4.2 The sampling pattern: short rays wide, long rays narrow

![fan geometry](assets/p16_fan.png)

*Error model (my derivation, assumed sigmas, not measured).* The chord from the tip to the next vertex at range r has a bearing relative to the tip heading. Three things make this bearing uncertain: the lateral error of the tip itself (sigma_y / r, large for short chords), the heading error of the tip (sigma_theta, constant), and the curvature of the lane (sigma_kappa * r / 2, large for long chords). With a coverage factor k = 2.5:

`half_fan(r) = k * sqrt( (sigma_y / r)^2 + sigma_theta^2 + (sigma_kappa * r / 2)^2 )`

With sigma_y = 0.5 m (the order of the measured vertex error of the traced graph, 0.54 m), sigma_theta = 5 deg and sigma_kappa = 1/40 per m before and 1/150 per m after the curvature has been estimated from the near samples:

| range r (m) | E94 fan today (deg) | half-fan, one stage (deg) | half-fan, two stages (deg) |
|---|---|---|---|
| 1.5 | 45 | 49.4 | 49.4 |
| 3 | 45 | 27.5 | 27.0 |
| 6 | 45 | 20.4 | 17.5 |
| 12 | - | 25.6 | 15.0 |
| 24 | - | 44.9 | 17.2 |

The tip-position term and the curvature term are equal at r* = sqrt(2 sigma_y / sigma_kappa): 6 m before and 12 m after the curvature is known. So long rays can only be narrow if the pattern is built in two stages: a near, wide stage that estimates the local direction and curvature, then far rays centred on the extrapolated arc (panel b). That is the two-pass inner loop of G.

*Pattern head.* From the tip state it outputs (a_m, b_m) for M rays (M about 23, ladder 1.5 / 3 / 6 / 12 / 24 m with 7 / 5 / 5 / 3 / 3 rays, initialised on the table above):
`r_m = r_min (r_max / r_min)^sigmoid(a_m)`, `phi_m = c(r_m) + half_fan(r_m) * g * tanh(b_m)`,
with c(r) = kappa_hat * r / 2 (zero in the near stage) and a gain g in [0.5, 2] from the tip state (wider after a low-validity step). Three points per ray (at 1/3, 2/3 and the end) are read, from a pyramid level that grows with r. This is 69 reads per tip against 34 in E94.

### 4.2a The pattern as sketched: teardrop, shared rings, verified long rays, n hypotheses (adopted)

![teardrop pattern](assets/p16_teardrop.png)

![n-best hypotheses at a fork](assets/p16_beam.png)

*The sketch (redrawn to scale in panel a).* From the tip, 12 rays: 0 deg 24 m, +-9 deg 20 m, +-20 deg 12 m, +-50 deg, +-90 deg, +-125 deg and 180 deg 4 m. As an envelope: `r(phi) = max(4 m, 24 m / (1 + (phi / 20 deg)^2))`. Samples sit on shared rings (4, 8, 12, 18, 24 m), clipped to the envelope: 12 ends + 16 intermediate points = 28 points, about the budget of E94 (27 + 7), but reaching 24 m instead of 6 m and covering the full circle at 4 m.

*Relation to the pear of 4.2.* The sketch is a monotone teardrop: the wider the angle, the shorter the ray. A bend is not covered by a widening far fan (the U-shape of my error model) but by iteration: short steps in the bend, long steps on straights (panel c, steps 24 m / 5.8 m / 24 m in the illustration). So the far fan stays narrow; the re-aiming of far rays along the arc estimated from the near samples becomes an optional refinement. The full circle at short range serves seeds (no heading), tip errors, lateral context (neighbouring lanes) and the look back at committed points. For a tip with a known heading, wide and backward rays are not candidates for the next vertex (coverage mask on endpoints that lie on committed points, direction prior); they stay as context tokens. For seeds the pattern is mirrored (two lobes), and the direction head resolves the sign.

*Long lines are verified by their intermediate samples.* Per ray, a verification score v_k at every sample (lane present at sample k given that it was present up to k-1; a small causal 1-D convolution or recurrence along the ray), the survival `T_k = v_1 ... v_k`, and the probability that the last verified sample is k, `P(k) = T_k (1 - v_(k+1))` (panel d). The endpoint of the ray is the expected (or most probable) last verified sample. This is the discrete-time survival model, and the same algebra as the transmittance of volume rendering (from memory). Consequences:
- The step length is the longest chord that stays on the lane: a chord that cuts a corner leaves the lane at some sample and fails there. The Douglas-Peucker rule of 4.3 item 5 follows from the verification, no separate step-length target is needed (the beam figure: 24 m on the straight branch, 3.8 m on the curved left branch).
- A long ray cannot hop onto the endpoint of a neighbouring lane, because its intermediate samples between are off the lane.
- The strict product is conservative ("better short than false lane graphs"); a tolerant variant allows one missing sample with a learned penalty (occlusion).
- Sample spacing on long rays must stay at 4 - 6 m (the sag between samples on a 15 m radius is about 0.2 m).
- Labels come from the geometry alone: per ray, the furthest sample such that all samples up to it lie within 0.5 m of a ground-truth lane (no expert, no matching).

*The n endpoints with the highest likelihood become n hypotheses.* Likelihood p = P(k) of the ray times a learned direction prior; endpoints within 3 m of a higher-ranked one are suppressed (soft NMS), endpoints on committed points are masked. The n best (n = 3 by default) become children, each with its p as validity; children with p below 0.3 are pruned (beam figure, step 1), the living ones are new tips with their own teardrop (step 2). This replaces the single expectation of 4.3 item 4 as default: two modes at a fork are two hypotheses, not their average. Hypotheses on the same lane merge through the coverage pointer.

*Differentiability.* Endpoint positions (expectation over the samples of a ray), verification scores, survival and likelihood are differentiable. The choice of the n best and the NMS are selection only: gradient flows through the selected endpoints and their likelihoods, not through the choice. In training all endpoints are kept with their likelihood as weight up to a budget N_max (global top-N_max on validity, selection only); a strictly soft variant uses sparsemax weights over the endpoints (support from the forward pass, gradient through the weights; from memory) and is an ablation. Both are covered by the "selection only" rule of the gate in the triage plan.

*Refined in `Hypothesis_Tree_Spec_and_Tests_EN.md`:* validity = survival up to the endpoint x direction prior (P only chooses the endpoint along a ray), redundancy by segment dominance, bottleneck path validity, tree -> DAG by merge and edge split, b >= 2 children in a fixed-shape tree with a beam; a numpy reference passes 18 spec tests.

*Illustration only:* panels c and d and the beam figure are drawn by scripts from hand-set numbers and a toy scene, not model outputs.

### 4.3 Step module (per tip, all tips in parallel, padded to a budget)

Items 4 and 5 describe the expectation read-out (variant R1). With 4.2a the default read-out is R2: verified rays and the n best endpoints as hypotheses; R1 stays as ablation.

1. Pattern head -> polar offsets -> positions (differentiable in the tip pose and the head outputs).
2. Bilinear reads of the frozen BEV pyramid at the positions -> ray tokens (feature, relative range and bearing; a small 1-D attention along the three points of a ray).
3. Cross-attention: the tip token pulls from the ray tokens, from nearby *committed points* (this replaces the covered-crop raster of E94 and prevents re-tracing) and from the history tokens.
4. Heads: ray weights w (softmax over rays); child position `p + sum_m w_m (r_m u(theta + phi_m) + delta_m)` with a small per-ray residual delta; up to 3 child slots with their own logits and a competition across slots, so that two children choose different rays; an existence logit per slot (a "stop" is all slots off); correction (dp, dtheta) and curvature estimate of the tip; validity; a 128-d step token per child (the token that goes to the route head, as now).
5. Step length is a consequence of where the weight sits: weight on the 12 m ray is a long step on a straight lane, weight on the 2 m ray a short step in a bend or at a junction. This is the sparsification: the imitation target for the weights is the largest range whose chord stays within 0.3 m of the ground-truth lane (Douglas-Peucker rule; mean chord 7.0 m on the test lanes, so the ladder must reach 12 m).
6. Output density: the committed chain is resampled to a fixed arc-length spacing (2 m) by linear interpolation, which is differentiable in the vertex positions; the planner and the route head see a fixed density although the vertex density is variable.

### 4.4 Loops

- *Inner loop, x2:* pass 1 near and wide, pass 2 far and narrow on the arc from pass 1; the corrected tip re-centres the pattern; weights are shared (RAFT-like). Whether a third pass pays off is an experiment (X3).
- *Commit:* children with existence above a soft gate are appended to the set P (positions, tokens, validity, parent index). During training all slots up to the budget are kept and weighted; at inference a threshold and top-k keep the budget.
- *Outer loop:* the children are the next tips (about 16 rounds, ring wavefront from near to far as in E94). Training uses free roll-out; backprop through the last 3-4 rounds (detaching between rounds did not matter in E120).
- *Revisit:* after the last round all committed points are refined together by the E120 decoder (graph-biased self-attention, bilinear reads), which fixes early points with the evidence found later.
- *Frames:* the set of frame t-1, ego-warped, supplies initial tips of frame t (as the instance propagation of Sparse4D v2 / v3); the input memory of the chain stays as it is.

### 4.5 Start everywhere

About 1.6 k lattice points over the +-50 m BEV (2.5 m spacing, jittered in training). Each seed gets a *ring* pattern (short rays at all bearings, no heading prior) and a snap step: move laterally onto the lane, estimate the heading axis (the direction comes from the dense direction head and is resolved later by the route), output validity v0. Seeds with low validity stop here; the rest grow in both directions along the axis. Several seeds snap to the same lane; they are merged by the soft pointer to existing points (the soft form of the E94 merge pointer) and by the density term. Estimated cost: about 0.11 M bilinear reads in the first round, about 20 k in a later round with 300 tips (assumptions, not measured; E94 runs at most 192 heads).

### 4.6 Losses (no matching, no peak picking)

- Position: point-to-polyline distance, prediction to ground truth weighted by validity, and ground truth to prediction as a soft minimum over the validity-weighted points (the coverage term). This is the E120 loss; it does not care where along the lane the vertices sit, which suits variable steps.
- Validity: BCE / focal on all tips and seeds (target: within 0.5 m of a ground-truth lane).
- Child slots: left-to-right ordering of the ground-truth children (no matching), or the best slot only.
- Density: soft occupancy (splat the points on a 0.5 m raster), penalise cells above 1 so that duplicates are not free.
- Warm start: imitation with the E94 expert (heading, step) for the first epochs, then free roll-out with a growing share of own states.
- Chain: route head and consumer on the output set, planner loss on, as in E121; the route head is compared with two seeds.

### 4.7 Differentiability audit

| element | gradient path | what is not differentiable |
|---|---|---|
| pattern positions | bilinear weights of the reads (local: 4 cells per level; large moves need the coarse level and a good initialisation) | - |
| next point | softmax weights, expectation, residual | the expectation is biased toward the centre when the softmax is flat (known for soft-argmax): temperature and residual |
| existence, validity | sigmoids; `log v` is added to the attention logits of the route head | threshold at export |
| commit | concatenation of tensors with masks | - |
| seeds | lattice positions are inputs; the snap moves them | no peak picking |
| topology | parent links by construction; merges by a soft pointer | argmax of the pointer only at export |
| budget | all slots kept in training | top-k at inference, number of rounds |

### 4.8 Variants for the ablations

G0: E94 fan unchanged, soft step and free roll-out only. G1: adaptive pattern, one stage. G2: two stages (near / far on the arc). G2b: teardrop of the sketch with verified rays and n = 3 hypotheses (default after 4.2a). G3: G2 / G2b with commit and revisit. G4: G3 with start everywhere.

### 4.9 Risks

Free roll-out is unstable without the warm start. Expectation over rays averages two modes at a fork (hence the child slots). The ray ladder is a hyper-parameter (range 24 m is a guess). Dense seeds raise duplicates and the class imbalance of validity. The E120-type position loss is blind to along-lane position; the step length now comes from the verification (4.2a), a vertex-count penalty stays available. Short wide rays have a single sample and cannot be verified: they can land on a neighbouring lane (beam figure, hypothesis 3, p = 0.17), hence the direction prior and the masks. Hypotheses can multiply; the coverage merge, the NMS radius, n and the budget N_max keep the frontier bounded.

## 5. Architecture L: spray, snap, link

![spray, snap, link schematic](assets/p16_l_schematic.png)

![L architecture](assets/p16_l_arch.png)

### 5.1 Idea

Bottom-up: many free query points, each pulled onto the lane to become a short oriented lane piece (pose, length, token, validity); pieces exchange information by self-attention; then candidate pairs of piece ends are scored by a link head that looks at the image along a connector curve. The graph is the set of pieces with a soft adjacency matrix. Depth is constant; there is no sequential growth.

### 5.2 Stages

1. **Spray.** N_s queries (jittered lattice, or learned anchors initialised by k-means on training lanes, as Sparse4D does for its 3-D anchors, from memory), pose from the dense head, state token from the feature at the point. Positions are inputs, not outputs, so seeding itself does not need gradients.
2. **Snap, 3 layers.** Per query: oriented-box sampling along and across the piece (DAB-DETR-like length / width modulation of the deformable reads), self-attention between queries with a relative-pose bias (distance, heading difference) so that neighbours agree on the lane axis, updates dp (mostly lateral), dtheta, dl, validity v.
3. **Duplicates.** Either a learned pair head (relation-network style: for nearby pairs predict "duplicate", keep = v times the product of (1 - duplicate) over higher-ranked pieces) or *longitudinal repulsion*: the self-attention predicts a displacement along the axis away from neighbours, so that pieces relax to an even spacing of about one piece length (this also gives a near-fixed density without any argmax).
4. **Candidate pairs.** For each piece end the k <= 6 nearest piece starts inside the fan of the end (gap <= 9 m, bearing within the same cone as in G, heading difference bounded). About 110 pieces describe the ground-truth lanes of an average test frame (652 m of lane per frame, about 6 m per piece incl. gap), so a frame has a few hundred candidate pairs, not N x N.
5. **Link head.** A Hermite connector between end(i) (position, heading) and start(j); K = 8 image samples along it, verified one by one as on a long ray (survival product of the samples: a connector whose intermediate samples are off the lane is not a link); attention over the samples with a query made of the two piece tokens and the relative pose -> link logit. Sigmoid, not softmax: a split is one end with two links, a merge one start with two incoming links.
6. **Iterate, 2-3 times.** Graph-biased self-attention over the pieces with log p_ij as the attention bias updates the tokens; the link head runs again (tokens now carry the neighbours' evidence).
7. **Gap pieces.** Where a link spans more than one piece length, the connector is sampled at the piece spacing and one more snap pass places the new pieces: positions are a differentiable function of the end poses.
8. **Output.** Pieces and soft adjacency A, tokens and validity to the route head (validity added to its attention logits) and the planner consumer.

### 5.3 Losses

Piece position and validity as in G. Link BCE on the ground-truth adjacency (piece i is followed by piece j if the arc-length gap along the same ground-truth path is below a limit). Path-level term on the soft reachability `A + A^2 + ...` (sparse matrix powers) against ground-truth path co-membership, in the spirit of LaneGAP, because the planner loss alone was too weak to shape topology.

### 5.4 Differentiability audit

| element | gradient path | what is not differentiable |
|---|---|---|
| piece pose | deformable reads, attention, displacement heads | - |
| duplicates | soft keep factor or repulsion displacement | - |
| candidate list | built from piece poses by geometry (k nearest in the cone) | the selection of the k candidates (no gradient to *which* pairs are candidates; the scores of the chosen ones are differentiable) |
| link probability | sigmoid on attended connector samples; connector is a smooth function of the end poses | threshold at export |
| gap pieces | connector sampling | their number (a function of the gap length) |

### 5.5 Variants

L0: pieces and a pair MLP without image evidence. L1: Hermite connector with image samples. L2: L1 with graph-biased iterations. L3: L2 with gap pieces. Orthogonal: duplicate head versus repulsion; k-means anchors versus lattice.

### 5.6 Risks

Pieces at junctions are short and hard to detect (the argument of LaneGAP against pieces); the link head has to tolerate fragmented pieces there. The soft adjacency can hide cycles; the path term should be checked for them. Candidate lists miss true pairs when a piece end is badly placed (recall of the candidate step must be measured). Duplicates are the main failure of spraying.

## 6. Hybrid H: ray-or-piece softmax

![hybrid](assets/p16_h_unify.png)

L produces pieces; G grows from the open ends that the link step left dangling. In the fan of an open tip the keys of the attention are image rays *and* the tokens of existing pieces inside the cone, plus a stop logit, in one softmax. Choosing a ray extends the graph with a new vertex (growth), choosing a piece is a link (join or merge), choosing stop ends the lane. The E94 merge pointer is the discrete precursor of the piece keys. The temporal memory feeds the pieces of frame t-1, ego-warped, as additional starting pieces.

## 7. Comparison

| | G (growth) | L (spray, snap, link) | H (hybrid) |
|---|---|---|---|
| start | lattice seeds snap, then grow in both directions | lattice or anchor queries snap into pieces | L first, G from dangling ends |
| depth | sequential, about 16 rounds | constant (3 snap + 2-3 link layers) | both |
| density | variable (long steps on straight lanes), resampled to fixed | fixed by relaxation (piece spacing) | both |
| fork / merge | child slots; soft pointer to existing points | one end with two links, one start with two links | both, in one softmax |
| discrete left | top-k at inference, rounds | candidate selection, export threshold | both |
| closest to what exists | E94 trace network (fan, steps, wavefront) + E120 | E120 positions + route-head attention; links are new | needs G and L |
| main risk | roll-out stability, mode averaging at forks | duplicates, fragments at junctions | complexity |
| attacks which weakness of B | argmax / teacher forcing, 6 m step cap, seeds | junction topology (split / merge F1 0.36 / 0.46), seeds | all three |

## 8. Experiment plan (ordered; each step builds on the previous one)

Baselines *(measured)*: E94 F1@1 0.851, F1@0.5 0.661; E120a 0.863 / 0.694, vertex error 0.43 m; keep gate 0.872; system A 0.867 / 0.875; split / merge F1 0.36 / 0.46 (E94) against 0.69 / 0.65 (A); 22 vertices per 100 m; planner path error 0.34 m at 20 m; route success of the route head 0.853 / 0.843 / 0.832 / 0.820 (seed 0) and 0.864 / 0.860 / 0.862 / 0.850 (seed 1) at 0 / 1 / 2 / 3 m map error.

| id | change | hypothesis | metrics | decision rule |
|---|---|---|---|---|
| X1 | G0: expectation over the existing 27 rays + free roll-out with warm start, no other change | argmax and teacher forcing cost accuracy through error accumulation along long chains | F1@0.5 / 1, vertex error along / across, error versus position in the chain (drift at 40 m) | keep if F1@0.5 gains >= 0.01 or drift falls at unchanged F1@1 |
| X2 | G1 / G2: adaptive pattern, ladder to 24 m, near / far stages | two thirds of the edges sit in the 6 m step class; longer steps give a sparser graph at equal quality | vertices per 100 m at equal F1@1; F1 at equal vertex budget; junction exits | keep if <= 17 vertices per 100 m with F1@1 >= E94 (oracle 14.3) |
| X3 | inner passes 1 / 2 / 3, with and without the arc re-aiming | the second pass is what makes long rays narrow; a third gives little | F1@0.5, vertex count, time per frame | choose the smallest number that keeps X2's gain |
| X4 | G4: start everywhere against 192 peak seeds | seeds are the limit (oracle seeds 0.921 against 0.908) | lane recall by lane length, false length share (duplicates), reads per frame | keep if recall rises without losing precision, and cost stays within 2 x E94 |
| X5 | L: spray + snap only (no links), ablate self-attention between queries and repulsion | information exchange between neighbours and repulsion make the pieces consistent and evenly spaced | piece-level F1, duplicate rate, spacing variance | pieces with F1@1 >= 0.85 and duplicate rate below 10 % |
| X6 | L: link head with / without image evidence along the connector; candidate recall | image evidence at the join helps most at forks and merges | link precision / recall, split / merge F1, path continuity, candidate recall | split / merge F1 >= 0.5 (E94 0.36 / 0.46, A 0.69 / 0.65) |
| X7 | H: ray-or-piece softmax against G and L side by side | one softmax avoids duplicate or conflicting decisions | all of the above | keep if it beats the better of G and L in junction metrics |
| X8 | chain: route head and consumer on the output of the best variant, planner loss on / off | variable vertex density does not hurt the map assignment; validity gate helps | route success (two seeds per head), planner path error at 5 ... 40 m | report with seed spread; no claim below 0.03 route success |

## 9. Constraints and limits

- Not implemented and not measured; the table in 4.2 is a model with assumed sigmas.
- Small training data: the earlier set decoder E32 lost 0.058 F1 against the raster, one run; any query-based variant may do the same without the warm start.
- Train-set graphs of E94 are in-sample for the networks that read them (E120 / E121 caveat); cross-fitted graphs exist (out-of-fold token files) and should be used for X8.
- Bilinear sampling gives local gradients only; large pattern moves rely on the coarse level.
- Route-head seed noise is up to 0.03 route success; single-seed gaps below that are not results.
- The position loss is blind to the position along the lane; sparsity needs its own term.
- Cost numbers in 4.5 and 5.2 are estimates from assumptions, not measurements.
- Exact poses and a simulated SD map are used throughout the project; the architectures say nothing about pose or map faults beyond what E87 / E104 showed.

## 10. Literature status

Checked by search in this session: Deformable DETR, DAB-DETR, Sparse4D (v1 - v3), RAFT, integral regression and its bias, Relation Networks for Object Detection, CenterLineDet, VectorMapNet, LaneGAP, InstaGraM, LaneSegNet, DAGMapper, RoadTracer (abstract), RNGDet++ (existence only), TopoGPT and BeyondFormer (abstract snippets only), the 2026 satellite road-graph paper on local candidate edges.

Correction to the earlier chat answer: of RoadTracer the search confirmed only "iterative search guided by a CNN decision function"; the fixed step length and angle bins I described are from memory. The list of lane-detection papers in that answer (LaneATT, CLRNet, Anchor3DLane, STSU, FOLOLane, RCLane, GANet, LaneAF, TopoNet, TopoMLP, TopoLogic, PivotNet, BeMapNet, iCurb, RNTR, LaneGraph2Seq details) was not re-checked; PolyWorld and Sat2Graph were not returned by the search. Treat these as pointers, not as verified descriptions.

Sources used for the checked items: [Deformable DETR](https://arxiv.org/pdf/2010.04159) · [DAB-DETR](https://arxiv.org/pdf/2201.12329v4) · [Sparse4D](https://arxiv.org/pdf/2211.10581) · [Sparse4D v3](https://arxiv.org/pdf/2311.11722) · [RAFT](https://arxiv.org/pdf/2003.12039) · [Integral Human Pose Regression](https://alphaxiv.org/abs/1711.08229) · [Relation Networks](https://arxiv.org/pdf/1711.11575) · [CenterLineDet](https://arxiv.org/pdf/2209.07734) · [VectorMapNet](https://arxiv.org/pdf/2206.08920) · [LaneGAP](https://arxiv.org/pdf/2303.08815v3) · [InstaGraM](https://arxiv.org/html/2301.04470v3) · [LaneSegNet](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06d86daa7001b7ea755ac4c80dc93497-Abstract-Conference.html) · [DAGMapper](https://arxiv.org/pdf/2012.12377) · [RoadTracer](https://arxiv.org/pdf/1802.03680) · [RNGDet++](https://arxiv.org/pdf/2209.10150) · [TopoGPT](https://www.emergentmind.com/papers/2606.31814) · [BeyondFormer](https://arxiv.org/pdf/2609.07511) · [satellite road graph, local candidate edges](https://arxiv.org/pdf/2602.23290)

Reproduction of the figures and the numbers in section 2: `scripts/fig_p16_design.py all` (figures, fan table); the vertex statistics were computed from `~/exp_gcar_b123/p10/tok_e94_a_test_clean.pkl` and the ground-truth lanes of the TEST rows (Douglas-Peucker at 0.2 / 0.3 / 0.5 / 1.0 m: 15.8 / 14.3 / 12.7 / 11.5 vertices per 100 m of 542 841 m of lane).
