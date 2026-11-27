# PL04 - PLUTO: full-text extraction

| field | value |
|---|---|
| paper | PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving. Jie Cheng, Yingbing Chen, Qifeng Chen. arXiv 2404.14327 [cs.RO], v1 only (22 Apr 2024); no venue on the abstract page; arXiv listing licence CC BY-NC-SA 4.0. Links opened: arxiv.org/abs/2404.14327, arxiv.org/html/2404.14327 (about ten queries), jchengai.github.io/pluto, github.com/jchengai/pluto (repository front page only) |
| axis, tier, K ids | A7 (navigation context to the planner), tier 1 core (queue priority P3); no K id yet (a proposal is in section 5) |
| level-1 note(s) | [PL04](../papers/PL04_pluto.md) (home note, loop 16, method + code page), [C14](../papers/C14_pluto-reference-lines-from-the.md) (loop 17, page digest) |
| code extraction | none written: on the coordinator's instruction (2026-10-25) the code extraction is skipped for this paper; section 4 lists where a later one should start. Repository https://github.com/jchengai/pluto |
| sources read | arXiv abs page (v1, metadata + abstract); arXiv HTML v1, about ten separate queries (whole paper: abstract to conclusion, Tables I - VII, Eqs. 2 - 9 and 12 - 15; the last 2,234 characters are arXiv page footer, no appendix exists); project page (videos only, no numbers); GitHub front page (file listing, README notes, no licence); raw `LICENSE` URL (HTTP 404); local files: the two level-1 notes, OVERVIEW.md, `reports/source/weekly/digest_week2.md` lines 158 - 168 (our E117 results). All read 2026-10-25 |
| reading marks | every statement about the paper, project page and repository front page is **digest** (a summarising fetch tool; no page was seen raw, numbers may be mis-transcribed; the ablation steps in the Table III summary below were recomputed by me from the returned values). Statements about the code are **memory / level-1 note** (carried over from PL04, not re-read here, see section 4). Statements about our own experiments are **checked** (digest_week2.md read). Section 5 is my analysis, not the paper's content |
| extracted by, date | extraction agent (Claude, sub-agent of the literature session), 2026-10-25 |

## 1. Problem and claim
- Learned closed-loop planner for nuPlan: from the vectorised scene (ego current state, 20-step agent histories, static obstacles, map polylines, traffic-light status) and a set of **reference lines** it predicts N_R x N_L candidate ego trajectories of 8 s (80 steps) with a confidence each, plus one trajectory per surrounding agent. A rule-based post-processor then picks among the top K.
- Gap closed: learned planners were good at longitudinal behaviour but weak at lateral decisions (lane change, bypass); the reference-line condition of the authors' earlier work (MPNP, three proposals) is generalised to a query grid (lateral queries from reference lines x learnable longitudinal queries). Imitation alone also suffers from shortcut learning; the paper adds a differentiable drivable-area loss and a contrastive loss with causality-changing negative augmentations.
- The claim that matters: on Val14 the learned planner with post-processing (93.21, R-score 92.06) passes the rule-based PDM-Closed (93.08, 93.20) in the non-reactive score; without post-processing it is 89.04 / 80.01.
- Honest limits for our reading: **the paper contains no ablation of the reference lines** (with against without), none of the number of lines, no variance over seeds, no runtime, no parameter count; the route restriction is hardly described in the paper (one clause) and only exists in the code (section 4).

## 2. Method as built (digest)

### Pipeline
```
 inputs (within 120 m of the AV; frame not stated)
   AV current state (pos, heading, vel, acc, steer; NO history)  --state-dropout encoder--> E_AV [1 x D]
   agents, T_H = 20 steps, 8 ch per step (diffs)  --neighbour-attention FPN--> E_A [N_A x D]
   static obstacles, 5 ch                          --2-layer MLP-->            E_O [N_S x D]
   map polylines, n_p pts x 8 ch                   --PointNet-like-->          E_P [N_P x D]
   E_0 = concat(E_AV, E_A, E_O, E_P) + Fourier PE(global pos, heading) + attribute embeddings   (Eq. 2)
   -> L_enc = 4 pre-LN transformer layers (D = 128)                                              (Eq. 3)
 lateral queries : reference lines -> same polyline encoder -> Q_lat [N_R x D]
 longitudinal q. : N_L = 12 learned, anchor-free         -> Q_lon [N_L x D]
 Q_0 = Projection(concat(Q_lat, Q_lon)) in R^{N_R x N_L x D}                                     (Eq. 4)
 L_dec = 4 layers; each: SelfAttn over dim 0 (lines), SelfAttn over dim 1 (long. queries),
                          CrossAttn(queries, scene tokens)                                       (Eq. 5)
 heads: T_0 [N_R*N_L x 80 x 6] = (px, py, cos, sin, vx, vy),  pi_0 [N_R*N_L]  (MLPs)            (Eq. 6)
        reference-free trajectory tau_free = MLP(E_AV')                                          (Eq. 7)
        agent prediction: 2-layer MLP, one trajectory per agent, P [N_A x 80 x 2]
 post-processing: top K = 20 by pi_0 -> LQR + kinematic bicycle roll-out -> rule score pi_rule
        pi = pi_rule + 0.3 * pi_0, output the argmax                                             (Eq. 15)
```
- Complexity argument for the factorised attention: full attention over the grid O(N_R^2 N_L^2), factorised O(N_R^2 N_L + N_R N_L^2).
- Output frame of the 6 channels (global, ego-centred, or relative to the reference line) is **not stated**; whether the reference-free trajectory enters the candidate set at inference is **not stated**.
- Fourier embedding: of the global position and heading of the latest agent / obstacle positions and of the first points of the polylines (frequencies and width not stated). Attribute embeddings: agent type, lane speed limit, traffic-light status ("such as": list may be longer).

### Reference lines as lateral queries (what the paper says)
1. Take lane segments within a radius R_ref of the AV (value not stated).
2. From each, a depth-first search follows the topological lane connections and chains the lane centre-lines.
3. Truncate to a uniform length, resample to a fixed number of points (neither value stated).
4. Encode with the **same polyline encoder as the map** (8 channels per point: offsets to the start point, to the previous point and to the left and right boundary points; how boundary points are obtained for a centre-line is not stated). The result is Q_lat; N_R is not stated (fixed, padded or variable unknown).
5. The only route statement: reference lines are "derived from the autonomous vehicle's surrounding lanes on its route" (one clause). There is no route input to the model, no roadblock, no de-duplication, no minimum length, no fallback rule in the paper text. Fig. 5 shows the global routing plan as light-blue lanes and the reference lines as purple dashed arrows (qualitative only).

### Teacher forcing (imitation loss)
- The end point of the ground-truth trajectory is projected onto the reference lines; the line with the smallest lateral distance is the **target line**. The target line is cut into N_L - 1 equal-length segments by distance; each segment belongs to one longitudinal query, the last query stands for "beyond the end of the line". The query whose segment holds the projected end point is the **target query**. Only that (line, query) mode gets the regression target; this is meant to prevent mode collapse. No equation for the assigned target trajectory; the projection method itself is not given.
- Eq. 8: L_reg = smoothL1(tau_hat, tau_gt) + smoothL1(tau_free, tau_gt); L_cls = CrossEntropy(pi_0, pi_0*) with pi_0* the one-hot target mode. Eq. 9: L_i = L_reg + L_cls.
- Prediction loss L_p: smooth-L1 on the one predicted trajectory per agent. (The equation number is not among those returned.)

### Auxiliary loss (differentiable interpolation)
- Non-drivable area is rasterised into a binary mask (500 x 500, 0.2 m, so 100 m x 100 m), a distance transform makes an ESDF. The ego footprint is N_c = 3 covering circles of radius R_c; their centres follow differentiably from each trajectory point and heading; the signed distance d_i of each centre is read by bilinear `grid_sample` (batched).
- Eq. 12: L_aux = (1 / T_f) sum_t sum_i max(0, R_c + eps - d_i^t). R_c and eps: not stated. The pseudo-code (Algorithm 1) normalises by the number of active (violating) entries + 1e-6 instead of T_f; the paper does not reconcile the two. With a modified cost map the same loss can penalise collisions.

### Contrastive imitation learning (CIL)
- For each sample x one positive augmentation x+ ~ T+ and one negative augmentation x- ~ T- are drawn; the mini-batch grows to 3 N_bs. The shared scene encoder gives latents h, h+, h-; a 2-layer MLP projection head gives z, z+, z- (which tokens are pooled into h: not stated; hidden width: not stated).
- Eq. 13: L_c = - log [ exp(sim(z, z+) / s) / ( exp(sim(z, z+) / s) + exp(sim(z, z-) / s) ) ], sim = cosine similarity, s = 0.1. One positive and one negative in the denominator (a softmax triplet), summed over all triplets of the batch; there is no in-batch negative pool mentioned.
- Targets: original and positive samples keep the original ground-truth trajectory; negative samples have **no trajectory target** (their ground truth may be invalid after the change) and enter only L_c.
- Eq. 14: L = w1 L_i + w2 L_p + w3 L_aux + w4 L_c, all four weights 1.0.
- Six augmentations. Positive (T+, ground truth stays valid): (1) state perturbation: small random disturbance of ego position, velocity, acceleration, steering angle; (2) non-interactive agent dropout: remove agents whose future boxes do not intersect the ego trajectory. Negative (T-, causal structure changed): (3) leading-agent dropout: remove all agents ahead; (4) leading-agent insertion: put a leading vehicle into the ego path where the ego trajectory becomes invalid (e.g. a collision), its motion copied from a random agent of the mini-batch; (5) interactive-agent dropout: remove agents that interact directly or indirectly (same criterion as (2)); (6) traffic-light inversion: when approaching a signalised junction without a leading vehicle, reverse the signal. Magnitudes, probabilities, insertion position: not stated.

### Training and inference
- nuPlan, 1 M training frames over all scenario types, 25 epochs, batch 128, AdamW (weight decay 1e-4), learning rate linear warm-up to 1e-3 over 3 epochs then cosine, 4 x RTX 3090, about 45 h with CIL and 22 h without. Planning horizon 8 s, history 2 s, features within 120 m.
- Inference = forward pass, top K = 20 of the N_R x N_L scored trajectories, closed-loop roll-out with LQR on a kinematic bicycle, PDM-style rule evaluator (progress, comfort, rules, TTC with the predicted agent trajectories, at-fault collisions excluded; weights not stated), final pi = pi_rule + alpha pi_0, alpha = 0.3; the post-processor only selects, it never edits a trajectory.

## 3. Numbers (digest; table numbers as printed)

Table I (parameters): T_H 20, T_F 80, D 128, L_enc 4, L_dec 4, N_L 12, N_c 3, cost map 500 x 500 at 0.2 m, alpha 0.3, temperature 0.1.

Table II, closed-loop Val14 (1,090 scenarios after filtering; 15 s at 10 Hz; "Score" = non-reactive; R-score = reactive):

| planner | type | Score | Coll. | TTC | Drivable | Comfort | Progress | Speed | R-score |
|---|---|---|---|---|---|---|---|---|---|
| Log-replay | expert | 93.68 | 98.76 | 94.40 | 98.07 | 99.27 | 98.99 | 96.47 | 81.24 |
| IDM | rule | 79.31 | 90.92 | 83.49 | 94.04 | 94.40 | 86.16 | 97.33 | 79.31 |
| PDM-Closed | rule | 93.08 | 98.07 | 93.30 | 99.82 | 95.52 | 92.13 | 99.83 | 93.20 |
| PDM-Open | learned | 50.24 | 74.54 | 69.08 | 87.89 | 99.54 | 69.86 | 97.72 | 54.86 |
| PlanTF | learned | 85.30 | 94.13 | 90.73 | 96.79 | 93.67 | 89.83 | 97.78 | 77.07 |
| Pluto without post-processing | learned | 89.04 | 96.18 | 93.28 | 98.53 | 96.41 | 89.56 | 98.13 | 80.01 |
| GameFormer | hybrid | 82.95 | 94.32 | 86.77 | 94.87 | 93.39 | 89.04 | 98.67 | 83.88 |
| PlanTF-H | hybrid | 89.96 | 97.06 | 93.38 | 97.79 | 91.08 | 92.90 | 98.01 | 88.08 |
| Pluto | hybrid | 93.21 | 98.30 | 94.04 | 99.72 | 91.93 | 93.65 | 98.20 | 92.06 |

(GC-PGP 61.09 / 54.91, RasterModel 66.92 / 64.66, UrbanDriver 67.72 / 64.87 are further rows.) Reading: without the rule post-processor the learned planner is far behind in the reactive simulation (80.01 against 92.06).

Table III, component ablation on a **subset of nuPlan (20 scenarios per scenario type, not Val14)**, single runs, no seeds:

| model | description | Score | step |
|---|---|---|---|
| PlanTF | reference | 87.55 | |
| M0 | base (query-based architecture with reference lines) | 87.04 | -0.51 against PlanTF |
| M1 | M0 + state dropout encoder (SDE) | 89.64 | +2.60 |
| M2 | M1 + auxiliary loss | 90.03 | +0.39 |
| M3 | M2 + reference-free head | 90.69 | +0.66 |
| M4 | M3 + CIL | 91.66 | +0.97 |
| M5 | M4 + post-processing | 93.57 | +1.91 |
| expert | log-replay | 94.24 | |

(steps computed by me from the table values.) Side effects the paper shows: M3 raises Drivable (98.38 to 99.19) but lowers Comfort (96.76 to 94.33) and Collisions (97.98 to 97.57); CIL lowers TTC (94.74 to 94.38); post-processing lowers Comfort (96.39 to 93.17), attributed by the authors to more emergency stops.
- **What the reference lines contribute: not isolated.** There is no run without reference lines, none with a different number of lines, none with a hard route filter against all lanes. The only indirect evidence: M0 is 0.51 below PlanTF, which uses polygon tokens with an on-route flag, so the line-conditioned query design alone gives no measurable gain in this ablation (a single run on a subset; the difference is probably inside the noise). The gains come from SDE, the auxiliary loss, the reference-free head, CIL and the rule post-processor.
- Table IV (N_L, on M4): 6 -> 88.89, 12 -> 91.66, 18 -> 90.18, 24 -> 87.90.
- Table V (top-K, on M5): K = 10 / 20 / 30 / 40 -> 93.18 / 93.57 / 93.58 / 93.02.
- Table VI (alpha, on M5): 0 (rule only) 90.64, 0.1 92.27, 0.3 93.57, 0.5 93.50, 0.7 93.12, 0.9 93.26; M4 without post-processing 91.66. The learned score is needed: the rule score alone is below the network alone.
- Table VII: constant-velocity agent prediction 92.82 against learned prediction 93.57.
- Compute: 4 x RTX 3090, about 45 h (CIL) or 22 h (no CIL), 25 epochs of 1 M frames, batch 128. **Parameter count, inference time, FPS, number of lines N_R: not stated anywhere I could read.** (C14 mentions 250 ms from a secondary table, still unverified; the paper has none.) My own order-of-magnitude estimate, not from the paper: D = 128 with 4 + 4 layers gives a few million parameters.
- Caveats: only Val14 is used, a rule-based post-processor is part of the headline number, the ablation set differs from the headline set, no seeds, the nuPlan lane graph is HD-map clean.

## 4. Paper versus code
Per the coordinator's instruction (2026-10-25) the code was **not** read in this extraction and no code file was written. What follows is the repository front page (digest) plus what the level-1 notes carried over from an earlier code-page reading (memory / level-1 note: not re-checked here).
- **Repository:** https://github.com/jchengai/pluto (the URL in the note is correct). Front page: `config/`, `script/`, `src/`, `run_training.py`, `run_simulation.py`, `requirements.txt`, `README.md`; branch `main`, 6 commits, 619 stars, 101 forks, 24 open issues. README: Python 3.9 conda env, nuplan-devkit installed from the Motional repository, then `requirements.txt`; checkpoint `Pluto-1M-aux-cil` on OneDrive; the README itself says the training part is not fully tested.
- **Licence: no LICENSE file** on the front page and the raw `main/LICENSE` URL returns 404, so the code is read only (describe, never copy). The arXiv text carries CC BY-NC-SA 4.0 (listing licence of the paper).
- **Where a later code extraction should start** (names from the level-1 note unless marked): (1) `RouteManager` (reference-line extraction); (2) the feature builder that packs the 100-point reference lines into the feature tensors (file not named in the notes; from memory the builder lives under `src/feature_builders/` and the scenario/route manager under `src/scenario_manager/`: unverified); (3) `pluto_model.py` (decoder, lateral x longitudinal query grid, reference-free head, state dropout, key padding); (4) the loss module with the ESDF hinge (from memory under `src/models/pluto/loss/`: unverified) and the contrastive augmentation code; (5) the `post_processing` / trajectory-selection code (LQR roll-out, rule scorer) only if the closed-loop part is of interest.
- **Facts the level-1 note attributes to the code and that the paper does not give** (still to be verified): start lanes within 3 m of the ego on a route roadblock with heading error below 90 deg; DFS over `outgoing_edges` restricted to route roadblocks; stop at 15 lanes or 100 m (unclear whether 15 is a depth or a cap on the number of lines); trim from the ego projection; drop paths shorter than 0.8 x length; de-duplicate when the maximum coordinate difference is below 0.5 m; 100 points per line (every 4th sample) plus a future projection of the ego; `use_ego_history=False`, state dropout 0.75, drop-path 0.2, dropout 0.1, 8 heads, default 6 modes (how this relates to N_L = 12 is unreconciled), `ref_free_traj=False` by default (the evaluation fallback for scenes without a reference line needs it switched on), the reference-line validity only masks output probabilities at evaluation.
- Within the paper: Eq. 12 (division by T_f) and Algorithm 1 (division by the active count) disagree.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Not the BEV raster. It needs a lane graph with successor edges, the ego pose in that graph (we are ego-centred: seeds are the vertices near the origin), a route score, and polylines resampled to a fixed point count. All exist after the route head: perceived DAG (vertices in ego metres, directed edges, validity), soft route probability per vertex, ego motion. No agents, traffic lights, boundaries or speed limits exist in our perception; the PLUTO encoder part for those inputs is not needed. A drivable-area raster for the auxiliary loss is not at hand (nuScenes map expansion could supply one for labels) |
| what it gives us | Stage A7: (a) a way to turn the DAG into **explicit candidate paths** (reference lines), each with scalar scores, instead of a bag of 6 m edge tokens; (b) a decoder layout (lines x anchors, factorised attention, one score per mode) and a teacher-forcing recipe; (c) a reference-free head as the abstention path; (d) causal negatives for contrastive training |
| conflicts | DFS and the 0.5 m de-duplication are rule steps with no gradient to the map; the route filter in the code is hard and roadblock-level (nuPlan only); teacher forcing picks the target line by a nearest-line rule (an assignment); the nuPlan lane graph is clean and long (100 m), ours is short, sparse and partly false; the model is designed for 80-step trajectories with speed and agents, ours is a geometric path at 5 ... 40 m; the code has no licence (read only); the nuPlan devkit is an unneeded dependency |
| reuse mode | Idea only / re-implement from the description (pure PyTorch, a few hundred lines); code read only |
| effort | S: path enumeration as a beam over the DAG plus per-path features fed to the existing E117 consumer, no new decoder. M: the full lines x anchors decoder with scored modes and soft teacher forcing. Blocker: none technical; the evaluation set for route errors (see 5.4) has to be defined first |
| expected gain and risk | On an exact map the route channel is already close to its ceiling (E117: oracle route channel 0.316 m, our consumer 0.34 m, so at most 0.025 m is left from a better route head); a gain, if any, would come from the path structure helping curved and junction cases, and it is an inductive bias, not new capacity (E117 already decodes 8 learned path queries). The wrong-map and no-map cells (0.72 - 0.76 m) are a choice problem at forks that no path enumeration solves without a route; PLUTO's own ablation does not show a gain from reference lines. Risk: a path enumerator on a graph with false forks multiplies lines; short graphs give lines shorter than the horizon |
| feeds which experiment / K id | Proposed techniques (to be registered before use): "scored path enumeration over the perceived DAG (beam, soft route score per path)", "soft teacher forcing over (line, anchor) modes", "route-flip negative augmentation". No K id assigned here |

### 5.1 Route-filtered DFS on a DAG with a soft route probability
PLUTO's rule is: start at lanes near the ego, follow successors, keep only route lanes, stop at a length, trim, de-duplicate. With a soft probability r_v in [0, 1] per vertex (and q_e the edge validity), three ways exist; the first two are what I would test, the third is PLUTO literal.
1. **Do not filter; score.** Enumerate all paths from the seed vertices up to the horizon (a batched beam over the DAG: frontier tensor of at most N_R paths, each step extends every path by its children and keeps the top N_R by path score; the DAG has no cycles, so the depth is bounded by the horizon divided by the vertex spacing, about 6 - 8 steps for 40 m at 7 m spacing, 13.5 vertices per 100 m). Give each path the scalars [log g_p = sum of log q_e (joint probability that all its edges are real), rho_p = length-weighted mean of r along the path, minimum r, valid length l_p, seed weight] as extra channels of every resampled point (r and q interpolated along the polyline) and as an additive log-prior on the attention logits of that line. The route then **enters as a feature, not as a gate**, so the gradient reaches the route head and a wrong route degrades gracefully: two lines with similar rho_p both stay in the candidate set and the scorer decides with ego motion (E117: ego motion is worth 0.40 -> 0.34 m).
2. **Threshold with fallback.** Keep paths with rho_p >= tau (tau around 0.5, to be tuned on VAL); if none passes, take all paths and let the reference-free head also answer. This is PLUTO's hard filter plus its reference-free head, expressed on soft inputs. The threshold is a rule step: acceptable only if it acts on detached scores and the kept paths keep their gradient (the rule "selection with gradient through the selected items" holds), and it should not be the default on the training path.
3. **Literal PLUTO** (hard route roadblocks): impossible, we have no roadblock id, and a wrong route removes the right path altogether.
- **Soft reachability without enumeration** is the alternative if enumeration is too slow: a forward pass over the DAG in topological order, a_v = 1 - prod_{u -> v} (1 - a_u * q_uv * [route weight]) (noisy-OR). It yields one number per vertex ("the ego can drive here, along the route"), which is what E117 already gets from the route probability; it cannot tell two paths from each other, so it does not add the path structure.
- **Forks.** DFS gives one line per leaf; lines share their prefix. At a fork with children c_1 ... c_k, route probabilities r_c need not sum to one; normalise across siblings with a dustbin, rho~_i = r_i / (eps + sum_j r_j), only as an extra feature. A merge needs no special rule in the DFS (several paths end in the same vertex), but then several lines share the suffix: that is correct for planning (they are different ways to reach it).
- **Partial or unknown route.** Beyond the end of the ADASIS-like horizon r is *unknown*, not zero. Feed a known-mask channel next to r, otherwise the planner reads "known off-route". With no usable route at all (map missing, route head abstains), r is uniform and the candidate set is the whole graph; this is the "no reference line" situation of PLUTO, handled by scoring instead of a separate head. A global SD shift of 2 - 3 m smears r over the parallel lanes of one road; the lines of all those lanes stay candidates.

### 5.2 Lines from a perceived graph with short and false lanes
- **Short graphs.** PLUTO drops lines shorter than 0.8 x length; on our graph this would drop everything where the graph stops early (we prefer short to false). Replace the rule by a **length mask and a soft length weight** (sigmoid in l_p); the decoder predicts path points at 5 ... 40 m also beyond the end of a short line (the line ends, the learned part extrapolates), with l_p given as an input channel.
- **False forks.** A false child doubles the number of lines near the ego. Their joint probability g_p is low when q_e is low, so the top N_R cut by score removes them; N_R should be small (4 - 8, my guess) and fixed for batching. The 0.5 m de-duplication is still useful for the second failure of our builder, the same lane found twice (two parallel chains within 0.5 m): do it on detached resampled points and merge the weights (noisy-OR) instead of discarding the duplicate's evidence.
- **Geometry.** Resampling a polyline by arc length (cumulative segment lengths, linear interpolation) is differentiable in the vertex positions, so a planner loss can reach the vertices (E121 tried a gradient to positions through the consumer: no gain, mean |gradient| per coordinate 3e-4 to 1.3e-3). Lane boundaries do not exist for us: use [p_i - p_0, p_i - p_(i-1), heading, r_i, q_i] as the point channels instead of PLUTO's left and right offsets.
- **Seeds.** PLUTO takes lanes within 3 m of the ego with heading error below 90 deg (level-1 note, from the code); ours are soft seed weights from distance and heading to the ego pose, which also covers an ego between two lanes.

### 5.3 Lateral and longitudinal decomposition, and what to keep
- **Lateral**: the discrete choice (which line) is a score over N_R lines; the continuous part is the offset to the line. I suggest decoding **Frenet residuals** (arc position s = 5 ... 40 m, lateral offset d(s) from the line) instead of absolute coordinates; the paper does not say which frame PLUTO uses for its 6 output channels, so this is a design choice, not a copy. The planner then only has to learn small offsets, and a perception error of the line is visible as a systematic offset.
- **Longitudinal**: our output is a geometric path at fixed arc lengths, so the 12 longitudinal anchors (their role is where on the line the trajectory ends, hence stopping and speed) are not needed; N_L = 1 would do. Table IV (6 -> 12 -> 24 gives 88.89, 91.66, 87.90) shows that the number matters for PLUTO's trajectories, not for a path. If the planner is later extended to a speed profile, the anchor design (equal segments of the line, last anchor beyond it) is the part to reuse.
- **Training the choice without argmax.** Teacher forcing by the nearest line is an assignment from the label side. A Hungarian-free replacement is a **soft target**: weights w_{p,k} = softmax over modes of (-distance^2 of the mode's end point to the GT end point / 2 s^2), cross-entropy against w and regression weighted by w (relaxed winner-takes-all; compatible with rMCL / relaxed WTA, MH01 / MH04 in the register). At inference the argmax of the scores is allowed.
- **Loss on the output.** The mixture should not be averaged into one path (a blur between left and right at a fork); keep the modes, report the top-1 path and the best-of-N error separately.

### 5.4 Contrastive learning with causal negatives
- PLUTO's idea: positives change nuisance factors (ego perturbation, agents that do not matter) and should keep the latent; negatives change the cause of the answer (a leading agent appears or disappears, the light flips) and should move it; negatives get no trajectory target.
- Map-side analogues for us: **positives** = ego-motion noise (already used: 0.15 m, 0.5 deg), dropping lanes that are neither on the route nor reachable from the ego, resampling jitter; **negatives** = **route flip** at the next fork (swap r between siblings), removal of the lane the GT path follows, insertion of a false fork on the GT path, route probability set to unknown. The planner's answer must change under a route flip, so a flipped sample is a legitimate negative and also the natural test of route dependence (E117d is a first look at it: the 30 m point moves to a forward-leading side road in 64 % of 51 frames, 3 seeds).
- Loss: Eq. 13 with the temperature 0.1 on a projected pooled latent of the consumer (our consumer pools nothing yet: a 64-d mean over its queries would do). The paper's gain from CIL is +0.97 on a single-run subset (90.69 -> 91.66) at twice the training time (45 h against 22 h) and with a tripled batch; the consumer is tiny, so the cost is small, but the expected gain is uncertain. Priority below the path-feature test.

### 5.5 Auxiliary losses
- L_p (agents) and L_aux (drivable-area ESDF with covering circles): we have no agents; the ESDF hinge could be replaced by a lane-support loss that samples the dense lane heat map (grid_sample) at the predicted path points (bilinear, differentiable, batched; the same mechanics as PLUTO's Algorithm 1 but with a heat map instead of a non-drivable ESDF). Untested; E121 (planner loss into positions) is a different thing. The reference-free head: M3 gained +0.66 over M2 with a side effect on comfort and collisions (-2.4 and -0.4); for us it is the fallback when the candidate set is empty.

### 5.6 Proposed experiment (not run; to be put on a card by the loop owner)
Arms on the same consumer data and protocol as E117 (E117 itself reported TEST, 546 frames with a driven path >= 20 m; the new arms are compared and selected on VAL, TEST only with the user's OK): (0) E117 as is; (1) E117 + path-level features from the scored enumeration, N_R in {4, 8}; (2) as (1) with the Frenet-residual output; (3) as (1) with hard threshold tau and fallback; (4) as (1) with the route-flip negative. Report path error at 20 m in the four map conditions (exact, 2 m, wrong, none), plus a **route-flip test** (flip the route at the next fork in 20 % of the frames, report error and the fraction of frames in which the top-1 path follows the flipped route), best-of-N error, the number of lines per frame, and ms per frame. Prediction I would write down: a tie on the exact map (the ceiling is 0.316 m), a small gain on curved rows, no gain in the wrong-map cells; the experiment is worth running only because path scoring is also what turns the perceived DAG into a planner-ready interface for other consumers (a rule-based PDM-style planner needs exactly such centre-line paths).

## 6. Not found / open
- R_ref, truncation length, point count per line, N_R, whether the DFS is route-restricted in the paper's own terms, de-duplication, minimum length: none stated in the paper (code values only via the level-1 note).
- Output frame of the trajectory channels; use of tau_free at inference; pooling for the contrastive latent; hidden width of the projection head; augmentation magnitudes and probabilities; R_c, eps; the loss equation numbers 1, 10, 11 (not returned by the reader); the state-dropout rate; polyline point counts and caps N_A, N_S, N_P; the coordinate frame of the inputs.
- No ablation of reference lines, of their number, of the route filter; no seeds; no parameter count or runtime; nothing on wrong or missing routes (the reference-free head is motivated by parking lots).
- The PDF and the supplementary were not read (the HTML has no appendix); the project page holds videos only (ablation videos for M1, M3, M4, no numbers).
- Code: the licence file is absent; internal file names are not verified (section 4).

## 7. Our experiments that touch it
- None used PLUTO's reference lines. Closest: **E117** planner consumer (input MLP + 2-layer pre-norm transformer encoder, d = 64, 8 learned path queries, 93,314 parameters, 25 epochs, 3 seeds, inputs = the 320 nearest 6 m edges of the traced graph with start / middle / end, route probability and ego motion). Error at 20 m on TEST (546 frames) for exact / 2 m / wrong / no map: geometry only 0.90 everywhere; + route probability 0.40 / 0.40 / 0.98 / 0.98; + ego 0.34 / 0.34 / 0.76 / 0.74; capacity control 0.34 / 0.34 / 0.76 / 0.74 [digest_week2.md l.162 - 165]. **E117d**: counterfactual route, 30 m point moves to a forward-leading side road in 64 % of 51 frames. **E119 / E121 / E202**: planner loss into the route head or into positions gave no gain (oracle route channel 0.316 m, so at most 0.025 m available).
- Not tried: path enumeration over the DAG, per-path scoring, Frenet-residual output, soft teacher forcing over modes, route-flip negatives, contrastive training, a lane-support auxiliary loss.

## Corrections to the level-1 note
1. PL04 / C14 describe the route-restricted DFS, the 3 m start rule, the 0.5 m de-duplication, the 100 m / 15 lanes stop and the 0.8 x length filter as properties of the method. In the **paper** none of these is stated: the paper has one clause ("lanes on its route"), no route input, and no numbers for radius, length, point count or N_R. They are code-only facts, still unverified by this extraction.
2. PL04 "Top K = 20 of N_R x N_L modes": K = 20 is the post-processing selection (Table V: K = 10 / 20 / 30 / 40 give 93.18 / 93.57 / 93.58 / 93.02), not an output size of the network; the "6 modes" default of the code is not reconciled with N_L = 12.
3. PL04 loss line: L_i also contains a smooth-L1 term for the reference-free trajectory (Eq. 8), so the reference-free head is trained whenever it exists.
4. C14 lists the negatives as "leading agent dropout / insertion, traffic-light inversion"; the fourth negative, interactive-agent dropout, is missing (six augmentations: two positive, four negative). Negatives carry no trajectory target.
5. PL04 "Fit" calls the route-filtered DFS the part that matches our design; the paper never shows what route-restricted reference lines contribute: no ablation exists, and the base query architecture (M0, 87.04) is 0.51 below PlanTF (87.55) in the authors' own table (subset, single run). The ablation table is on a different set (20 scenarios per type) from the headline Val14 numbers.
6. C14 "runtime ... 250 ms (secondary table, not verified)": the paper reports no runtime and no parameter count; the number stays unverified.
7. PL04 states the main-result numbers correctly (Table II checked); the reactive weakness without post-processing (R-score 80.01) is worth adding to the note.
