# GR05 - DAGMapper: full-text extraction

| field | value |
|---|---|
| paper | DAGMapper: Learning to Map by Discovering Lane Topology. Namdar Homayounfar, Wei-Chiu Ma, Justin Liang, Xinyu Wu, Jack Fan, Raquel Urtasun. ICCV 2019 (Proceedings pp. 2911-2920 per a search-result summary; the arXiv comment field says "Published at ICCV 2019"); uploaded to arXiv only on 2020-12-22 as v1 (the only version), arXiv 2012.12377. **The task brief says ICCV 2021: that is wrong, the venue year is 2019.** Links actually opened: arxiv.org/abs/2012.12377, ar5iv.labs.arxiv.org/html/2012.12377 (full paper incl. appendices A - C), arxiv.org/html/2012.12377v1 (same text, used for a cross-check of Algorithm 1), patrick-llgc.github.io reading note (third-party, opinions marked as such) |
| axis, tier, K ids | A3 (sparse lane graph; growth agents), tier 1 core; K11 (DAG of lane nodes with states), K161 (short-unroll training with Chamfer loss; see section 5.6: K161 is in the compatibility study but not in `register.md`) |
| level-1 note(s) | [GR05](../papers/GR05_dagmapper.md) (corrections in section 8) |
| code extraction | no official repository. A third-party, strongly modified re-implementation exists inside the road-boundary benchmark Topo-boundary (GPL-3.0, read only, none of fork / merge / Chamfer inside): [topo-boundary](../code/topo-boundary.md) |
| sources read | ar5iv HTML of v1, nine separate queries: structure, Sections 3.1 / 3.2 / 3.3, Section 4 (tables), appendices A and B, precision queries on direction / position / fork, introduction + topology + generalisation + conclusion; arxiv.org/html v1 (Algorithm 1, fork / merge questions); arxiv.org/abs page; web search for code (nothing official); GitHub repository search "DAGMapper" (0 results); Uber research pages (404). Not readable: the ICCV supplement PDF (binary through the fetch tool; the tool kept a local copy of the 5.7 MB PDF in its tool-results folder, which I did **not** open), the CVF page (HTTP 403). Read 2026-10-25 |
| reading marks | every statement about the paper is **digest**: all pages came through the summarising fetch tool (no page was seen raw), so numbers and equation symbols may be mis-transcribed; the table values were returned cell by cell and agree with the level-1 note and with the paper's own text where I could cross-check. Statements marked "my arithmetic" or "my reading" are mine, not the paper's. Statements about our own code / plans are **checked** (files read) |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## 1. Problem and claim
- Predicted: the lane **boundaries** (paint lines, not centre-lines) of highway fork and merge areas as a directed acyclic graph of boundary nodes with a state per node, from a single image: the aggregated **lidar-intensity** BEV of one mapping-vehicle pass, 5 cm per pixel, strips up to 8000 x 1200 px (400 x 60 m). Purpose: cheaper HD-map production (annotators correct a draft).
- Gap closed: earlier recurrent boundary tracers (HRAN) assume a fixed lane order and cannot change the number of boundaries; DAGMapper adds an explicit topological state and grows the graph greedily with a queue of fork nodes.
- The claim that matters: with the state head, a convolutional-RNN memory and a symmetric Chamfer loss on densely sampled polylines, a walker trained end to end reaches F1 76.3 at 2 px (10 cm) against 66.9 for a distance-transform plus skeleton baseline, with 89 % of ground-truth boundaries having correct topology. At coarse tolerance (5 and 10 px) the baseline is slightly better (Table 1).
- Not claimed and not measured: any fork or merge detection score, any merge edge (a merge is a termination), inference speed.

## 2. Method as built

### Pipeline / architecture (digest)
```
lidar intensity BEV 1 x H x W (5 cm/px; H <= 1200, W <= 8000)
  -> global feature net: LinkNet-style encoder-decoder + FPN, instance norm   -> F  [C, H/4, W/4]   (C not stated; 20 cm per cell, my arithmetic)
  -> DT net: residual layers on F                                              -> D  [1, H/4, W/4]   thresholded inverse distance transform
  -> concat [F, D]   (D acts as attention on lane positions)
per node i with parent P(i) at x_P(i), state s_P(i) (one-hot):
  axis-aligned crop of [F, D] centred at x_P(i) + one-hot s_P(i)  = 11 channels, h x w not stated
     -> Direction header: conv-RNN -> global MAX pool -> Linear -> direction of the next RoI      (cosine loss)
  RoI rotated along the predicted direction, bilinear interpolation of [F, D, one-hot s_P(i)]
     -> Position header: conv-RNN -> light encoder-decoder -> softmax map over positions in the RoI, upsampled to the original image size
     -> State header: same RoI, conv-RNN as the direction header but global AVERAGE pool -> softmax over 3 states   (focal loss)
test: skeleton end points of the binarised D = initial vertices; graph growth by Algorithm 1; uncovered regions of D re-seed missed boundaries
```
- The conv-RNN is a vanilla recurrent cell with convolutions in place of matrix products and ReLU in place of tanh (appendix); kernel, filters and hidden-state shape not stated. Whether weights are shared over steps is not said explicitly (the same headers are applied at each node of Algorithm 1).
- Only the proposed model has these heads; building blocks (appendix A): Conv2D, Residual (two blocks of instance norm - ReLU - conv, stride 2 in the first conv to downsample), IRC (norm, ReLU, conv), IRUC (IRC with nearest upsampling). Layer sizes are in Fig. 6 - 10 only (images, not text).
- 11 input channels of the direction header: with D (1 channel) and three one-hot state channels this would imply C = 7 (my reading, the paper does not say).

### Inputs, outputs
- Input: one image per sample, intensity of the lowest-elevation return of past and future sweeps aggregated with IMU odometry. Output: polylines of boundary nodes v_i = (x_i, theta_i, s_i) with x_i in global image coordinates, theta_i "the turning angle at the parent position" (angle or 2-vector, absolute or relative: not stated), s_i in {normal, fork, terminate}. Children: normal 1, fork 2, terminate 0.
- A **merge is a termination without a target**: the main text ties "terminate" to a merge (and to a lane end; the appendix calls it a "terminating point") and the paper describes no edge from the ending boundary to a node of the other boundary. The output is therefore a set of polylines with fork branches (a forest), not a DAG with merge edges (checked on two renderings; my reading of Algorithm 1 and the topology metric).

### Equations (numbered as in the paper)
- E1 G* = argmax over G of p(G | D); D = the BEV intensity image, the space of all graphs is exponentially large, hence the greedy procedure.
- E2 p(G | D) = product over nodes of p(v_i | v_P(i), D).
- E3 p(v_i | v_P(i), D) = p(theta_i | theta_P(i), s_P(i), x_P(i), D) * p(x_i | theta_i, s_P(i), x_P(i), D) * p(s_i | theta_i, s_P(i), x_P(i), D). Note that the state of node i is conditioned on the **direction** theta_i and the parent, not on the chosen position x_i: the state header reads the same rotated RoI as the position header (my reading, consistent with the appendix).
- E4 symmetric Chamfer: L(P, Q) = sum over points p_i of P of min over q in Q of ||p_i - q||_2 + sum over points q_j of Q of min over p in P of ||p - q_j||_2, with P and Q the **densely sampled** predicted and ground-truth polylines (sampling density not stated). Sums, not means; plain L2 norm, not squared.
- E5 total loss L = 1 * l_chamfer + 100 * l_cosine + 10 * l_focal + 10 * l_dt (weights chosen on the validation set). Not given: the cosine formula, the focal formula (gamma, alpha), the exact l2 form of l_dt, the softmax / RoI-sampling equations, the conv-RNN update equations.
- Algorithm 1 (inference): queue Q = initial vertices; pop a vertex; loop while the parent's state is not Terminate: theta_i = argmax, then x_i = argmax, then s_i = argmax (in that order); if s_i = Fork insert the vertex into Q; move to the child; stop when Q is empty. Breadth-first or depth-first order is not stated. The step "i <- C(i)" is printed as is and not explained.

### Losses (digest)
- **Chamfer** (E4) on the polylines produced by the rollout against the ground-truth boundary polyline; it is the only term that constrains the position header (no cross-entropy on the position map is listed). Which part of the prediction enters (the 5 unrolled vertices or the whole produced boundary) and whether the ground-truth side is the whole boundary or a local piece: **not stated**. The paper only says each ground-truth polyline is matched to its prediction by the Chamfer distance; the assignment is implicit in the start: every rollout begins at a ground-truth initial vertex (section "Labels").
- **Cosine** on the direction (weight 100: it compensates the scale of a sum of pixel distances, my reading; the number is tied to pixel units of a 5 cm raster).
- **Focal** on the state: "modified multi-label focal loss": the individual losses are summed and divided by the **sum of the focal weights** instead of the batch size, so that wrong predictions are emphasised and the over-represented "normal" class does not dilute them (paraphrase). Softmax over three states in the header, yet "multi-label" in the loss: not reconciled in the text.
- **l2 on the inverse distance transform**: ground-truth value 8 on a lane boundary, decreasing with distance, the head output clipped to [0, 10].
- **No Hungarian matching or any other assignment** anywhere. The assignment problem is avoided by construction: the rollout identity is given by its ground-truth start vertex and the loss is a set distance.

### Labels / ground truth
- Annotation: one polyline instance per lane boundary; a new polyline is started at a fork. The raster is 5 cm per pixel.
- Node spacing / step length along the polylines: **not stated**.
- Direction target, computed on the fly: project the parent anchor x_P(i) onto the closest boundary and take the turning function there.
- State target, computed on the fly: **window label** - if the current RoI contains a fork the target is Fork, if it contains a terminating point it is Terminate, otherwise Normal. So the state is "an event lies inside the lookahead window", not "this node is the event" (my reading; the RoI size h x w is not stated, so the positional tolerance is unknown). How a window containing both a fork and a terminating point is labelled: not stated.
- The second child of a fork: no separate target is described. The only stated mechanism is that the direction header receives the parent's state as one-hot input, "which helps it pick the correct lane when two boundaries are close". Algorithm 1 inserts the fork vertex into the queue and processes it again later; how the second pass differs from the first (possibly through the recurrent memory, which has advanced; my guess, not stated) is not described.

### Training recipe (digest)
- Initial vertices: all ground-truth initial vertices, each with a small Gaussian perturbation (std not stated).
- Argmax replaced by softmax everywhere on the training path; test time is argmax. How a coordinate is extracted from the softmax map in training (expectation?) is **not stated**.
- Teacher forcing: ground-truth states are fed to the headers; the **predicted** direction (anchored at x_P(i)) defines the next RoI unless x_P(i) is more than 30 px (1.5 m, my arithmetic) away from the lane boundary, in which case the ground-truth direction is used (a drift guard).
- Truncated back-propagation through time, 5 steps, for the conv-RNNs of the three headers; rollouts stop when the vertex leaves the image, when the RoI contains a ground-truth terminate state, or after 30 steps. Whether gradients pass the RoI cropping / rotation into the direction head: **not stated** ("all components differentiable and trained end to end").
- AMSGrad, lr 1e-3, 8 GPUs with Horovod, 12 h; one 1200 x 8000 image fits per GTX 1080 Ti (effective batch 8 images: my reading, the batch size is not stated); loss weights and the final model chosen on validation (the text says "lowest precision and recall", probably a slip for highest). Augmentation, epochs, iterations, learning-rate schedule: not stated.

### Inference (digest)
- Initial vertices: threshold the DT output, binarise, skeletonise, take skeleton end points (the threshold of the model's D and the skeletonisation procedure are not stated; the baseline cites Suzuki and Abe and a 32 px threshold).
- Greedy argmax chain per vertex (theta, then x, then s); fork vertices are re-queued; the chain ends at Terminate. No test-time step cap, no NMS, no merge or snap rule, no confidence threshold is described. After the graph is finished, regions of the DT image not covered by the graph start new walkers ("recovery"; a rule step, inference only).

## 3. Numbers (digest; table numbers as in the paper)
Protocol: 3000 images from 68 fork / merge areas of one North American highway; the 68 areas are split randomly 70 : 15 : 15 and all images of an area stay in one split; repeated for three random splits (Table 2 is the average over the three test sets; image counts per split not reported). Precision = predicted polyline points within the threshold of the ground-truth polylines divided by the predicted length, recall the reverse, summed over all images; counting within a threshold, not a one-to-one assignment. Thresholds 2 / 3 / 5 / 10 px = 10 / 15 / 25 / 50 cm. The F1 formula is not printed.

**Table 1 (comparison to baselines)**, P / R / F1 at 2, 3, 5, 10 px:

| method | P 2 / 3 / 5 / 10 | R 2 / 3 / 5 / 10 | F1 2 / 3 / 5 / 10 |
|---|---|---|---|
| DT baseline (distance transform, thresholded 32 px per side, skeleton) | 68.1 / 87.9 / 96.3 / 98.0 | 65.8 / 85.0 / 93.5 / 96.2 | 66.9 / 86.4 / 94.9 / 97.0 |
| HRAN with ground-truth initialisation (lane-counting module removed) | 55.6 / 71.0 / 84.0 / 90.9 | 45.9 / 58.1 / 68.3 / 74.0 | 50.1 / 63.7 / 75.1 / 81.4 |
| DAGMapper | 76.4 / 89.0 / 94.6 / 96.6 | 76.2 / 88.7 / 94.2 / 96.1 | 76.3 / 88.8 / 94.4 / 96.3 |

**Table 2 (ablation; G global feature net, M conv-RNN memory, R DT recovery, S state head; the check-mark placement was inferred by the fetch tool from flattened text, and it is consistent with the level-1 note and the paper text):**

| G | M | R | S | P 2 / 10 | R 2 / 10 | F1 2 / 3 / 5 / 10 |
|---|---|---|---|---|---|---|
| - | - | - | yes | 75.4 / 93.5 | 62.1 / 77.4 | 68.1 / 79.0 / 82.8 / 84.7 |
| yes | - | - | yes | 74.1 / 95.9 | 65.6 / 84.5 | 69.5 / 81.2 / 87.2 / 89.8 |
| yes | - | yes | yes | 73.3 / 95.4 | 72.8 / 94.4 | 73.1 / 85.6 / 92.1 / 94.9 |
| yes | yes | yes | - | 69.8 / 93.4 | 70.2 / 94.2 | 70.0 / 83.3 / 90.9 / 93.8 |
| yes | yes | - | yes | 77.0 / 96.9 | 75.0 / 94.2 | 76.0 / 88.3 / 93.7 / 95.5 |
| yes | yes | yes | yes (full) | 76.4 / 96.6 | 76.2 / 96.1 | 76.3 / 88.8 / 94.4 / 96.3 |

Deltas that decide transfer (my arithmetic from the cells above, F1 at 2 px unless said):
- State head removed: 76.3 -> 70.0 (-6.3; at 10 px -2.5); precision and recall at 2 px both fall by about 6. What "no state" means for stopping and forking in that variant is not said.
- Memory: +6.5 without recovery (69.5 -> 76.0), +3.2 with recovery (73.1 -> 76.3); the paper calls it the largest gain.
- Recovery: +3.6 without memory (69.5 -> 73.1; recall at 10 px 84.5 -> 94.4) but only +0.3 with memory (76.0 -> 76.3; recall at 10 px 94.2 -> 96.1, precision at 10 px -0.3): once the walker has memory the re-seeding rule step is worth little.
- Global feature network: +1.4 (68.1 -> 69.5) at 2 px, +5.1 at 10 px.

**Table 3 (new highway, 336 images from 114 fork / merge areas, a different state about 1000 km away; the model was trained on all images of the first highway, train + val + test; no adaptation: "no domain adaptation is required")**: precision 57.4 / 72.1 / 83.4 / 88.6, recall 53.0 / 66.2 / 76.1 / 80.9, F1 55.1 / 69.0 / 79.5 / 84.6 at 2 / 3 / 5 / 10 px.

**Topology (text only):** each predicted polyline is assigned to the ground-truth boundary with the largest overlap within 20 px (1.0 m, my arithmetic); 89 % of ground-truth boundaries have the correct topology. Reported for the proposed model only (in the text I saw), so an advantage over the DT baseline in topology is not shown numerically.

Qualitative / failure (Fig. 4, appendix C): forks spawn two new boundaries, merges end one or both boundaries; failures: partially undrawn boundaries lower recall, in one example the direction header found the wrong RoI at a fork and at a merge the upper boundary ran on over a straight one without stopping.

Compute: training 12 h on 8 GPUs, one 1200 x 8000 image per 1080 Ti. **Inference time, parameters, FPS, memory: not stated** (an autoregressive walker; a third-party note remarks that run time grows with the number of nodes, which is that note's opinion).

Caveats: pixel thresholds at 5 cm on lidar intensity, one mapping company's highways, boundaries not centre-lines, no fork / merge F1, no numbers on the DT baseline's topology; the numbers are not comparable with ours (0.5 / 1 / 1.5 m point F1 on camera-radar BEV).

## 4. Paper versus code
- **No official code.** The arXiv abstract page lists none, a web search found none, the GitHub repository search for "DAGMapper" returns 0 repositories, the Uber research page is 404, the CVF page returned 403 (not readable). The README of the road-boundary benchmark Topo-boundary states that the original work published neither code nor data.
- Third-party: Topo-boundary (TonyXuQAQ, GPL-3.0) ships a DAGMapper **re-implementation** for aerial curb detection, and its own README says it removes the direction head and the RNN and predicts the next-vertex probability map directly (VecRoad-like), and that it was hard to make converge. Its code (as read) has no fork logic, no Chamfer loss, a binary stop flag instead of three states, and a position loss that is BCE on a Gaussian target map. It is therefore **not evidence for or against** the paper's recipe. Details and licence: [../code/topo-boundary.md](../code/topo-boundary.md).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | A feature map to sample RoIs from (our frozen 39-channel BEV pyramid or the dense lane head output replace F and D: the DT head corresponds to our heat map with sigma 0.75 m), start vertices, ground-truth boundary / centre-line polylines for the Chamfer side. No lidar, no ego poses, no SD map needed by the training recipe. What depends on lidar intensity: only the learned front end (FPN on intensity), the DT target and the pixel-unit constants (30 px, 2 - 10 px tolerances at 5 cm); the walker and the loss are feature-map agnostic. |
| what it gives us | Axis A3 (and A4 as training regime): (a) a **drift guard** for free-unroll training, (b) a window-labelled event state with a focal loss normalised by the focal weights, (c) evidence that a short unroll with a set-distance loss trains a walker (the same family as our E120 loss, which we already have), (d) ablation evidence that history (conv-RNN memory) is worth +3 to +6 F1 at 2 px in a stateless-per-step walker. It gives **no** merge target, no fork-resolution method for two simultaneous children, no fork / merge metric. |
| conflicts | Hungarian: none. Argmax: test time argmax of theta, x, s, softmax relaxation only in training (train / test mismatch). Rule steps on the training path: GT start vertices, GT state inputs, the 30 px GT-direction guard, GT-terminate stopping, and at inference the skeleton seeds and the uncovered-region re-seeding. Trainable encoder: the paper trains its FPN and DT net end to end (we are frozen; fine, F is replaced). Sequential single-child walker with a fork queue: opposite of our level-batched tree (31 ms). Lidar: front end only. Global pose: none needed (image-local). GPL: nothing to copy (no official code; the third-party code is GPL-3.0, read only). |
| reuse mode | Idea only / re-implement from description. |
| effort | S: drift-guard substitution inside the existing tree loss (point-to-polyline distance is already computed), focal-normalised event loss on the junction head. M: a BPTT-over-levels variant with detached / non-detached tips (E120 showed detaching does not matter for positions, so low value). Blocker: none technical; the test needs a stability measure of the free unroll, not only S. |
| expected gain and risk | Drift guard: aims at the stated main risk of the tree plan (free-unroll training from the first step, exposure bias); expected effect on clean VAL: a tie in S, a gain in training stability / early-level quality; inference cost 0 ms, so a tie is decided for the guard only if it is as good as the baseline. Event loss: low expected gain, see 5.3. Re-seeding: a possible lever for lanes found (0.695 against 0.800 at equal sparsity) but an inference-only rule step; recovery was worth only +0.3 F1 at 2 px once memory was present. |
| feeds which experiment / K id | K11 (register line wording must be corrected, see section 8); K161 "short-unroll training with Chamfer loss" is listed in `experiment_execution/plans/Literature_Compatibility_Space_EN.md` but **not in `register.md`**: register it before use. Proposed new ids (to be registered): drift-guard teacher substitution inside a free-unroll tree; focal-weight-normalised event loss; coverage re-seeding of roots. |

### 5.1 What DAGMapper is, side by side with our two designs

| aspect | DAGMapper (paper) | trace network E94 (ours) | verified-ray hypothesis tree (ours) |
|---|---|---|---|
| step | one child per step; position = softmax over a rotated RoI; step length implicit (not stated) | fan sampling, argmax steps, ring wavefront | b children per tip from a teardrop of J = 12 rays; the step length is chosen by the survival (hazard) of verified samples |
| events | normal / fork / terminate, one head, window label | continue / end / fork / merge + merge pointer | none as a head on the tree path: end = all hypotheses die, fork = two living children, merge = assembly (cluster 1.5 m / 30 deg, edge split, snap 1 m); junction head + fork relocation on top |
| merge | termination, no edge | merge pointer = explicit edge | explicit edge by assembly |
| fork second branch | fork vertex re-queued, mechanism not described | fork event | both branches are children of the same tip (spec test T07: two living children at 18, 12, 6 deg) |
| training regime | GT start, GT state input, GT-direction guard at 30 px, free positions, BPTT 5, up to 30 steps | teacher-forced | free unroll from the first step on its own selections (the plan's stated main risk), denoising roots M7 planned |
| position loss | symmetric Chamfer over dense samples, sums of L2 | imitation + refinement (E120) | point-to-polyline both ways, validity weighted, coverage as soft minimum (E120 loss) |
| assignment | implicit via the GT start vertex | none (teacher forcing gives it) | none; K38 rMCL / relaxed WTA for the b children |
| memory | conv-RNN hidden state per header | transformer blocks over covered crop | stage C attention over siblings, ancestors up to the root and committed points |
| speed | sequential per node, no timing reported | - | level-batched, 31 ms per frame in total |

### 5.2 Is it really "matching-free"?
Assignment-free, yes: a set distance, no one-to-one matching, as our loss. Not free of rule steps on the training path: the rollout is anchored at ground-truth start vertices (this is the implicit assignment), the parent state fed to the headers is the ground-truth state, the stopping rule uses the ground-truth terminate state, and the direction is replaced by the ground-truth one whenever the anchor is 1.5 m off. Our rule "no rule step on the training path" would not allow that for the model path; as a training-time curriculum it is the same kind of thing as our teacher forcing in E94 and the planned denoising roots M7, and has to be labelled as such in a card. Also, the softmax-in-training / argmax-at-test mismatch is an exposure of the same kind we avoid by using selections with gradient through the selected items.

### 5.3 Which terms and tricks can be ported to a camera-radar BEV
1. **Symmetric Chamfer on dense samples** - we already have it in better form (E120: point-to-polyline, validity-weighted, plus a soft-minimum coverage term). Differences to check, none new: DAGMapper sums (no mean) and uses unsquared L2 (a gradient of constant magnitude, robust to outliers); the weight 100 on the cosine term is a unit effect of pixel sums and cannot be taken over. Nothing to port.
2. **Cosine direction loss** - on heading of the tip / child; our dense head already carries a 2-channel travel direction and the tips carry a heading, so this is probably already present in some form; no new information.
3. **Drift guard (30 px = 1.5 m)** - substitute the ground-truth heading (and optionally position) of a tip when it lies farther than a threshold from the ground-truth polyline, otherwise use the prediction. The distance is the same point-to-polyline quantity the tree loss computes. It sits between teacher forcing (E94, exposure bias) and the pure free unroll (tree plan). A sensible threshold in our units is a card parameter (1.0 - 1.5 m; 1.5 m equals our merge radius, which is a coincidence). Stop the gradient through the substituted quantity. This is the one item I would put on a card.
4. **Window-labelled event state with a focal loss normalised by the focal weights** - label for a tip: positive if a ground-truth fork / merge / end lies within its lookahead reach along the ground-truth path. In DAGMapper the head is indispensable because the walker has a single child and no other way to stop or branch (removing it costs 6.3 F1 at 2 px); in our tree, ends are hazards and forks are the b >= 2 slots, so that ablation does **not** transfer. The remaining use is an auxiliary label for the junction head (J 0.265 against 0.408 for the control at equal sparsity, the weak spot of the champion) where the class imbalance is the issue and the normalisation by the focal weights is a cheap stabiliser. Expected gain: low; E44 / E53 (key-point heads, split / merge F1 0.34 / 0.38 to 0.55 / 0.61 and 0.71 / 0.64) already showed that dedicated detectors help, so an auxiliary loss would have to beat those, and it would not be the same experiment as DAGMapper's ablation.
5. **Distance-transform l2 loss** - equivalent to our heat-map loss (Gaussian sigma 0.75 m versus DT peak 8 clipped to [0, 10]); the DT was concatenated as attention to the features: we have the heat map as the dense stage already. Nothing to port.
6. **Conv-RNN memory** - +3.2 to +6.5 F1 at 2 px. Our nearest equivalent is stage C attention over ancestors and committed points; DAGMapper supports having a path-history channel, it does not say how much our stage C is worth. A cheap check inside the tree plan: switch the ancestor keys off and measure; no new module needed.
7. **Seeds from DT skeleton ends and coverage re-seeding** - seeds are not a lidar topic; the recovery step brought +3.6 F1@2 without memory and +0.3 with it. For our lanes-found gap a coverage re-seed (heat-map mass farther than r from the built graph starts a new root) is the transferable idea; it is an inference-only rule step and needs a decision about the standing rule before use.

### 5.4 Fork and merge handling: what we can and cannot learn
- DAGMapper handles a fork by one state and a re-queue; the paper does not say how the second branch differs, and its own failure example (Fig. 4, row 4) is a fork where the direction header took the wrong RoI. Two near-parallel outgoing lanes are exactly the case where a single-child direction head struggles; that is the argument for b children per tip (K38).
- It handles a merge by terminating; a merge edge is created by nothing. Our assembly (cluster, split, snap) and the E94 merge pointer are ahead of it. If a merge target is wanted as a learned quantity, DAGMapper contains nothing to adopt.
- No fork or merge score is reported; our J metric (F1 at 6 m) cannot be compared with anything in this paper. The paper's topology figure (89 %) is per boundary, with a 1.0 m assignment tolerance.
- The level-1 claim "closest precedent for fork and merge handling" holds for the **training recipe** (set distance on a short free unroll with a ground-truth guard), not for fork / merge **decisions**.

### 5.5 Compute and our 31 ms
Sequential node-by-node growth with a conv-RNN on a rotated RoI per header per step has no stated inference time and a cost linear in the number of nodes; as published it does not fit a level-batched 31 ms budget, which is another reason to take only the training ideas.

### 5.6 Registers
K11 in `register.md` reads "DAG of lane nodes with continue / fork / merge states": the paper's states are normal / fork / terminate (a merge is a terminate), see section 8. K161 is missing from the register.

## 6. Not found / open
- RoI size h x w, step length, number of channels C, conv-RNN type and hidden shape, kernels and widths (only in Figs. 6 - 10), skeletonisation procedure and the DT threshold of the model, focal gamma / alpha, Gaussian-noise std, batch size, schedule, augmentation, sampling density of the Chamfer, order of the queue (BFS or DFS), test-time step cap, inference time.
- How a coordinate is read from the softmax map in training (the gradient path of the Chamfer into the position header) and whether gradients pass the RoI rotation into the direction header.
- Which part of the prediction enters the Chamfer in a 5-step window; how the second child of a fork gets its state target and its direction; how a window with both a fork and a terminate is labelled; what the "no state" ablation does about stopping.
- Whether the 89 % topology figure exists for the DT baseline; any fork / merge score.
- The ICCV supplement PDF (binary) and the figures 6 - 10 were not read; the supplement may contain the layer tables. The fetch tool kept a local copy of the supplement PDF (5.7 MB) in its tool-results folder; I did not open it because the rules allow page reading only. If the coordinator allows it, that file can be read for the layer tables.

## 7. Our experiments that touch it
- E94 trace network (events continue / end / fork / merge + merge pointer, teacher-forced; F1 0.883 / 0.851 VAL / TEST against 0.891 / 0.867 for the rule graph, split / merge F1 0.36 / 0.46 against 0.69 / 0.65 for the rule system per the concept plan): the same family, with an explicit merge edge that DAGMapper lacks.
- E44 / E53 key-point heads and Bezier connectors: split / merge F1 0.34 / 0.38 to 0.55 / 0.61 to 0.71 / 0.64. The level-1 note compares this with DAGMapper's state-head ablation ("in line with"); the two are not like-for-like (section 8).
- E120 matching-free point-to-polyline refinement (vertex error 0.54 -> 0.43 m; detaching between layers does not matter): our counterpart of the Chamfer term.
- E121 planner loss through positions: no gain (not related to the unroll recipe).
- Not tried: BPTT over several steps with non-detached tips as a separate arm, a drift guard, a focal-normalised event loss on tips, coverage re-seeding; no experiment uses K161.

## 8. Corrections to the level-1 note (GR05) and to the catalogue wording
1. The venue year: the paper is ICCV 2019 (arXiv upload 2020-12-22); the task brief's "ICCV 2021" is wrong. The level-1 note has it right.
2. **States:** the paper's states are normal (continue) / fork / **terminate**; there is no merge state. `OVERVIEW.md` (core table, GR05 row) and `register.md` K11 say "continue / fork / merge states" and the level-1 "Fit" line says the walker has fork / merge states per node: a merge is a termination without a target edge, the output is a forest of polylines with fork branches, not a DAG with merge edges. Suggested `why`: "Lane-boundary walker with normal / fork / terminate states (merge = terminate, no merge edge), trained on a 5-step free rollout with a symmetric Chamfer loss from ground-truth start vertices with a 1.5 m ground-truth-direction guard: nearest published recipe for a short-unroll set-distance loss; no fork / merge score, no code (K11, K161)."
3. "Closest precedent for fork and merge handling in a matching-free walker": true for the training recipe only (section 5.4). The fork mechanism of the second branch is not described in the paper; the merge is not handled.
4. "Matching-free": assignment-free, but the training path contains ground-truth start vertices, ground-truth state inputs, a ground-truth stopping rule and a ground-truth direction guard (section 5.2).
5. "step size comes from the position softmax inside the RoI": not stated in the paper; it is a plausible reading. The level-1 note also says the test-time step cap is not stated: confirmed, Algorithm 1 has none.
6. "Code: none found": confirmed for an official repository; a third-party GPL-3.0 re-implementation (strongly modified) exists in Topo-boundary: [../code/topo-boundary.md](../code/topo-boundary.md).
7. Dataset: add that the 70 : 15 : 15 split is by area and repeated three times (Table 2 averages three test sets), that the generalisation set has 114 areas on a highway about 1000 km away, and that the generalisation model was trained on the whole first data set (train + val + test).
8. The paragraph "Tried in our experiments" says the E44 / E53 gains are "in line with DAGMapper's state-head ablation (about -6 F1@2px without it)": the numbers are right (76.3 vs 70.0), but the ablation removes the only way of terminating or forking from a single-child walker, whereas E44 / E53 add detectors to an already complete tracer; treat it as loose support.
9. The DT baseline is better at 5 and 10 px (the level-1 note says so) and the recovery step is worth only +0.3 F1@2 once the memory is present (new; Table 2).
10. The level-1 item "whether gradients pass the RoI rotation / bilinear sampling into the direction head is not stated" is confirmed; the coordinate-from-softmax step (expectation or argmax in training) is equally not stated.
