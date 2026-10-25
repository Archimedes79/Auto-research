# PL03 - PlanTF: full-text extraction

| field | value |
|---|---|
| paper | Rethinking Imitation-based Planner for Autonomous Driving. Jie Cheng, Yingbing Chen, Xiaodong Mei, Bowen Yang, Bo Li, Ming Liu (HKUST, HKUST Guangzhou, Lotus Technology per the project page). arXiv 2309.10443 (cs.RO, cross-list cs.AI), **v1 only**, submitted 2023-09-19. The venue "ICRA 2024" is stated by the GitHub README, not by the arXiv page (which lists none). Links opened: arxiv.org/abs/2309.10443, arxiv.org/html/2309.10443v1, arxiv.org/pdf/2309.10443 (binary, unreadable), jchengai.github.io/planTF, github.com/jchengai/planTF (landing page only) |
| axis, tier, K ids | A7 (navigation context to the planner), tier 1 core (OVERVIEW section 2, core table A7). No K id registered for PL03; neighbours: K17 (our route head + consumer), K95 (map features gated by ego state), K108 (ego-status caveat of open-loop metrics) |
| level-1 note(s) | [PL03](../papers/PL03_plantf.md); the same authors' follow-up is [PL04](../papers/PL04_pluto.md) |
| code extraction | none (decision of the user, 2026-10-25: code extraction only for the most meaningful papers); section 4 says where a later one should start. Repository https://github.com/jchengai/planTF |
| sources read | (1) arXiv abs page, (2) arXiv HTML v1, six separate queries: structure, method/inputs, Section III-A, all eight tables, III-B / III-C and figure captions, Section IV / V; (3) PDF fetch (binary, nothing readable); (4) project page; (5) GitHub landing page (README summary, file listing, no code file); (6) for context only: Forecast-MAE arXiv 2308.09882 (HTML), PLUTO arXiv 2404.14327 (HTML, first 100 000 characters), interPlan arXiv 2404.07569v2 (HTML), route-integration study arXiv 2512.03756, HYPE arXiv 2510.12733. Local files: level-1 note and the loop-16 card `sources/lit16/planner_nav.md` (card PL03, interface table 3a, adapters 3b). All read 2026-10-25 |
| reading marks | **Every statement about the paper is digest** (all pages came through a summarising fetch tool; no page was seen raw; numbers may be mis-transcribed, two internal inconsistencies are listed in section 3). Statements marked **[card]** are second-hand: they come from the loop-16 card, which was written from GitHub code pages (feature builder, map encoder, model) read in loop 16 through the same kind of tool; I did **not** re-read any code (instruction of 2026-10-25). **[memory]** = not seen on any page. **[mine]** = my analysis or estimate, not in any source |
| extracted by, date | Claude (Sonnet 5.5) extraction agent, 2026-10-25 |

## 1. Problem and claim
- Predicted: the ego trajectory for the next 8 s (used unaltered, no rule-based emergency stop, no post-optimisation), from 2 s of tracked-object history, the current and past ego kinematic state, traffic lights, the HD map, speed limits and the designated route. Trained by imitation on up to 1 M nuPlan frames (all 75 scenario types), tested in the nuPlan closed-loop simulation (15 s rollouts at 10 Hz, LQR tracker, kinematic bicycle model).
- Gap: pure imitation planners reach a high open-loop score (OLS) but a weak closed-loop score (NR-CLS / R-CLS). The paper asks which ego inputs and which augmentations matter, and names an "imitation gap": the tracker and the vehicle dynamics are unknown when the planner is trained.
- Claims that matter: (1) history motion is not needed and hurts closed loop (shortcut / copycat learning); single-frame kinematic states do the same; (2) a **state dropout encoder (SDE)** that randomly drops the kinematic tokens during training keeps the open-loop benefit and repairs the closed-loop score; (3) perturbing the ego state helps only together with **re-normalisation** of the coordinates; (4) the imitation gap is measurable (log-replay through the LQR tracker loses 5.65 NR-CLS on Test14-hard against perfect tracking) and an RL adapter closes it; (5) the resulting PlanTF competes with rule-based and hybrid planners without any rule.
- What the paper is **not**: it contains no description of the map or route representation and no ablation of lane or route inputs (digest of the full HTML). Everything about polygons, points and the on-route flag in this file comes from the code via the card. For us the paper is a published planner input contract (through its code), a lesson on ego-motion shortcuts and an augmentation recipe.

## 2. Method as built

### Pipeline / architecture
Paper text (digest): agent, map and ego features are encoded separately, concatenated and processed by a stack of transformer encoder layers; the model jointly predicts the other agents and the ego plan at scene level; the backbone is adapted from the authors' own motion-forecasting model Forecast-MAE. The paper gives **no** layer count, width, head count, decoder or loss. The shapes below are from the card and, where marked, from Forecast-MAE.

```
nuPlan: GT map, GT tracked objects, traffic lights, route (roadblock ids)
  |  feature builder: everything in the ego frame at t0 (ego at the origin), +-100 m square [card]
  v
 agents    [N <= 64, 2 s history]                -> agent encoder -> N x 128                   [card]
 map       [M polygons, 3 x 20 points, centre used] -> point encoder + attribute embeddings -> M x 128   [card]
 static obstacles                                 -> own small MLP encoder -> S x 128          [memory]
 ego state (state6 = x, y, yaw, velocity, acceleration, steering angle)
           -> SDE: one linear embedding + positional encoding per variable, one learnable query
              aggregates them by cross-attention; in training every token except position and
              heading is dropped with p = 0.75                                -> 1 x 128
 + positional embedding MLP([x, y, cos, sin]) of agent pose / polygon centre   [card]
  v
 concat tokens -> 4 transformer encoder layers (dim 128, 8 heads, drop-path 0.2)              [card]
  v
 ego token  -> MLP decoder -> 6 modes x 80 steps (8 s at 10 Hz) + 6 mode scores              [card]
 agent tokens -> MLP -> agent futures (auxiliary prediction)                                  [card]
```

### Inputs and outputs
- **Frame and normalisation:** all coordinates relative to the AV centre (paper, III-B); after a state perturbation the coordinates are re-normalised around the perturbed AV position, so the AV is again at the origin. No further scaling (z-score) is reported anywhere.
- **Map polygon tensor [card]:** one token per polygon; polygons are lanes, lane connectors and crosswalks inside a +-100 m square; no cap in the builder, padded to the longest sample of the batch. Per polygon three polylines (centre, left boundary, right boundary) of 20 points each (21 are sampled, the last is dropped). **The released encoder uses only the centre polyline.** Per-point features, 6 channels: offset to the polygon centre (2, translation only, no rotation), vector to the next point (2), cos and sin of the segment heading (2). A point encoder (`PointsEncoder`, internals not read) turns the points into one 128-d vector.
- **Polygon attributes added to the token [card]:** type (3 classes), **on_route (2 classes, an index lookup in all likelihood; verify)**, traffic-light status (4 classes, UNKNOWN allowed), speed limit (MLP of the value, or a learned "unknown" embedding when absent). Positional embedding from the polygon-centre pose (x, y, cos, sin).
- **on_route assignment [card]:** roadblock-id membership. nuPlan's mission route is a list of roadblock ids; a roadblock groups the parallel lanes of one carriageway between two junctions, so **all parallel lanes of a route roadblock are flagged 1**; the flag is therefore a road-level, not a lane-level attribute. How lane connectors get their flag is not recorded in the card (to check in the code).
- **Masks [card]:** a polygon is masked (key padding) if it has no valid point; a point is valid if it lies inside the square. No adjacency or successor input exists; topology is only implicit in coinciding end points.
- **Agents:** the paper says tracking data over a 2 s window (digest); the card says up to 64 agents. Per-agent features were not read [memory: position, heading, velocity, size, category, valid mask]. **Static obstacles:** not mentioned in the paper text; PLUTO (same authors, digest) encodes each as position, heading and box size (5 numbers) with a 2-layer MLP, and the PlanTF code very probably has an equivalent branch [memory].
- **Output:** the ego plan, 6 modes with scores, 80 steps of 0.1 s; at inference the best-scoring mode is executed [memory]. Other agents' futures are an auxiliary output [card].

### Equations
The paper has **no numbered equation for the planner**. What exists:
- SDE (words of the paper, my formalisation): e_i = Linear_i(s_i) + PE_i for each ego-state variable i; z = CrossAttention(q, {e_i : keep_i = 1}); in training keep_i ~ Bernoulli(0.25) for every i except position and heading, which are always kept (dropout rate 0.75). The rate used for the final model is 0.75 (Table VIII).
- RL adapter reward (Table V, digest): position tracking exp(-15 * ||p - p*||) with weight +1.0; action ||u||^2 with -0.01; action rate ||du||^2 with -0.1; longitudinal acceleration limit 1(a > 2.4) with -1; jerk limit 1(||da|| > 4.0) with -1; yaw-rate limit 1(||dtheta|| > 0.95) with -0.5. PPO, 80 K steps, learning rate 1e-3, actions = acceleration and steering rate. The adapter is evaluation-time machinery (Table IV) and is not part of PlanTF's reported numbers.

### Losses
- **Not stated in the paper** (digest of the whole text). Forecast-MAE's fine-tuning recipe (digest of arXiv 2308.09882): three-layer MLP multi-modal decoder with K modes (6 at evaluation), a separate 3-layer MLP for the confidences, Huber loss for regression and cross-entropy for the confidence, equal weights, winner-take-all over the modes.
- PlanTF code [memory, unverified]: ego regression loss (smooth L1 on the best mode) + ego mode-classification loss (cross-entropy) + agent prediction loss (smooth L1, masked). The best-mode choice is an argmin on the training path; **the later code extraction must verify the rule and the weights**. Auxiliary losses beyond the agent prediction: none known (PLUTO adds a drivable-area hinge loss and a contrastive loss; those are PLUTO's, not PlanTF's).
- Assignment: no Hungarian matching; the winner-take-all best mode plays that role for the 6 ego modes.

### Labels
Ego future from the log: 80 steps, in the (re-normalised) ego frame. After a perturbation the ground-truth future is **not** corrected by default: the paper tried a "future correction" by constrained nonlinear optimisation and found it harmful (Table III), so the log trajectory is used directly as supervision and the tracker absorbs small deviations (paper's explanation).

### Training recipe
Batch 128, weight decay 1e-4, 25 epochs, learning rate 1e-3 decayed to zero by a cosine schedule (README: 3 warm-up epochs), up to 1 M frames over all 75 scenario types, state perturbation with probability 0.5, SDE rate 0.75. Hardware and wall-clock time are not in the paper. README: building the 1 M-frame feature cache takes "20+ hours", training uses 4 - 6 GB of GPU memory per GPU at batch 32. The perturbation (random noise on the current ego state, smoothed history) is described in words only; noise magnitudes are not stated (digest).

### Inference
No rule step, no post-processing, no emergency brake; the plan goes to the LQR tracker. Test: Test14-random and Test14-hard (14 nuPlan scenario types x 20 scenarios; "hard" = the 20 lowest-scoring of 100 per type under PDM-Closed) and Val14 (1180 scenarios from 14 types). The online leaderboard was closed, so the public test set was used.

## 3. Numbers
All values digest, OLS / NR-CLS / R-CLS = nuPlan open-loop, non-reactive closed-loop, reactive closed-loop score (0 - 100).

**Table VI, comparison (Test14-random OLS / NR / R | Test14-hard OLS / NR / R | ms per frame incl. feature extraction)**

| planner | random | hard | ms |
|---|---|---|---|
| log-replay (expert) | 100.0 / 94.03 / 75.86 | 100.0 / 85.96 / 68.80 | - |
| IDM | 34.15 / 70.39 / 72.42 | 20.07 / 56.16 / 62.26 | 32 |
| PDM-Closed (rule) | 46.32 / 90.05 / 91.64 | 26.43 / 65.07 / 75.18 | 140 |
| GameFormer (hybrid) | 79.35 / 80.80 / 79.31 | 75.27 / 66.59 / 68.83 | 443 |
| PDM-Hybrid (hybrid) | 82.21 / 90.20 / 91.56 | 73.81 / 65.95 / 75.79 | 152 |
| RasterModel | 62.93 / 69.66 / 67.54 | 52.4 / 49.47 / 52.16 | 82 |
| UrbanDriver | 82.44 / 63.27 / 61.02 | 76.9 / 51.54 / 49.07 | 124 |
| GC-PGP | 77.33 / 55.99 / 51.39 | 73.78 / 43.22 / 39.63 | 160 |
| PDM-Open | 84.14 / 52.80 / 57.23 | 79.06 / 33.51 / 35.83 | 101 |
| **PlanTF** | **87.07 / 86.48 / 80.59** | **83.32 / 72.68 / 61.7** | 155 |

**Table VII, Val14** (other methods copied from the PDM paper): log-replay 100 / 94 / 80; IDM 38 / 77 / 76; GC-PGP 82 / 57 / 54; PlanCNN 64 / 73 / 72; PDM-Hybrid 84 / 93 / 92; PlanTF 89.18 / 84.83 / 76.78. PLUTO's Table II (digest) lists PlanTF on Val14 as score 85.30 and R-score 77.07, and PlanTF-H (PlanTF with PLUTO's post-processing, digest) as 89.96 / 88.08; the difference to 84.83 / 76.78 is not explained in either paper.

**interPlan:** the PlanTF paper has no interPlan result (it predates the benchmark), and the interPlan paper's Tables III / IV (digest) do not list PlanTF. **Not found.**

**Ablations that decide transfer** (Test14-random OLS / NR / R unless noted)

| Table | what changed | result |
|---|---|---|
| I | history, shared encoder / separate encoder | 90.20 / 56.50 / 56.28 and 90.28 / 61.02 / 59.85 (high OLS, closed loop collapses; hard: 48.60 and 51.98 NR) |
| I | no history; state3 (x, y, yaw) / state4 (+velocity) / state5 (+acceleration) / state6 (+steering) | 81.13 / 85.99 / 79.38; 86.42 / 81.32 / 75.75; 87.71 / 81.76 / 74.51; 88.45 / 83.32 / 77.52 (more kinematic state: OLS up 81 -> 88, NR-CLS down 86 -> 83) |
| II | SDE on state5 / state6 | state5: 87.71 / 81.76 / 74.51 -> 88.80 / 86.73 / 75.75; state6: 88.55 / 83.19 / 74.79 -> 87.07 / 86.48 / 80.59 (hard: NR 67.57 -> 72.68) |
| III | perturbation P, re-normalisation RN, future correction FC, state3 | none 78.92 / 71.86 / 70.87; P 80.85 / 74.28 / 71.69; P+RN 81.13 / 85.99 / 79.38; P+RN+FC 79.28 / 81.35 / 76.60 |
| III | same, state5 | none 88.44 / 80.67 / 74.50; P 89.20 / 79.85 / 72.43; P+RN 87.71 / 81.76 / 74.51; P+RN+FC 86.43 / 82.10 / 74.71 |
| III | same, state6 + SDE | none 88.33 / 77.28 / 74.10; P 87.71 / 77.70 / 75.18; P+RN 87.07 / 86.48 / 80.59; P+RN+FC 85.50 / 82.95 / 76.09 |
| III | history model, none / P+RN+FC | 88.99 / 65.84 / 65.58 and 89.94 / 65.14 / 66.03 (augmentation does not rescue history) |
| VIII | SDE dropout rate (state6 + SDE) | none 88.33 / 77.28 / 74.10; 0.25: 89.11 / 81.70 / 78.44; 0.50: 89.12 / 83.71 / 77.52; 0.75: 87.07 / 86.48 / 80.59 (rate above 0.75 not tested) |
| IV | log-replay planner through trackers (NR / R, random; NR / R, hard) | perfect 96.63 / 77.38; 91.61 / 71.34. LQR 94.03 / 75.86; 85.96 / 68.80. RL adapter 96.3 / 77.13; 91.65 / 71.62 |

- **What the lane and route inputs contribute: the paper has no ablation of them** (no run without the map, without the route flag, with a corrupted flag, or with fewer polygons). The only input ablations are on the ego state (Tables I - III, VIII). Outside evidence on the route input, not PlanTF and not comparable: HYPE (arXiv 2510.12733, digest, HiVT-based ego-proposal network) adds a route embedding and moves nuPlan NR score 0.8691 -> 0.8942 and R score 0.8736 -> 0.9036; a route-integration study on a different prediction model (arXiv 2512.03756, digest) adds a binary one-hot on-route flag per polyline (variant A1) and moves open-loop OLS from 82.87 to 83.59 on Val14, and neither paper tests a missing or wrong route.
- **Failure cases (project page, digest):** PlanTF struggles with dedicated operations such as leaving a parking zone, self-motivated lane changes, narrow passages, cyclists and cones, and nudging around parked vehicles; the authors attribute this mainly to the open-loop-training / closed-loop-testing mismatch.
- **Compute:** parameters not stated. Runtime 155 ms per frame including feature extraction and model inference in Python (Table VI; GPU/CPU not stated). Memory 4 - 6 GB per GPU at batch 32 (README). [mine] The 4-layer, 128-d encoder alone has about 0.79 M weights if the MLP ratio is 4 (4 x (4 x 128^2 + 2 x 128 x 512)); the full model is larger (point, agent encoders, decoders) but its size is not known.
- **Inconsistencies seen in the digest** (read the PDF before quoting): the "state6, no SDE" row of Table II (88.55 / 83.19 / 74.79) differs from the state6 row of Table I (88.45 / 83.32 / 77.52); the "-" row of Table VIII (88.33 / 77.28 / 74.10) equals the Table III row "state6 + SDE without P, RN, FC", not the plain state6 row of Table II. Either the digest mislabelled rows or the paper mixes runs.
- **Caveats:** one number per cell, **no seeds, no confidence intervals** (not found), so differences of about one point are not established; Test14-hard is selected by the failures of PDM-Closed and favours learned planners; scores are on the nuPlan protocol and not comparable with ours.

## 4. Paper versus code
- The paper does not describe the map encoder at all; the code is the only source for the polygon tensor, the on-route flag, the encoder sizes and the loss (section 2, marked [card] / [memory]). Level-1 and this file therefore rest on a code reading that was not repeated here.
- Repository: **https://github.com/jchengai/planTF** (owner jchengai; correct, no change needed). Landing page (digest): 379 stars, 32 forks, 11 commits on `main`, 17 open issues, last push 2024-07-11 (GitHub API value recorded in the loop-16 sources). Top level: `config/`, `docs/` (`other_baselines.md`), `script/` (`setup_env.sh`, sanity, Test14-random, Test14-hard, Val14 runs), `src/`, `requirements.txt`, `run_training.py`, `run_simulation.py`, `run_nuboard.py`.
- **Licence: no LICENSE file** in the top-level listing; the sidebar shows none and the GitHub API returned `null` in loop 16. Consequence: all rights reserved by default, **read only**, describe and re-implement, copy nothing. Checkpoints (PlanTF state6 + SDE, RasterModel, UrbanDriver) are offered through a download link; their licence is not stated, and they are trained on nuPlan (non-commercial research terms) and useless for nuScenes-based graphs anyway.
- Dependencies (README, digest): Python 3.9 in conda, `nuplan-devkit` installed in editable mode (this pulls the simulation, Hydra configs and the map API), the repository's `requirements.txt` (contents not read; PyTorch Lightning per the loop-16 notes; the PyTorch version is not stated), `script/setup_env.sh`, optional Weights & Biases or TensorBoard. None of this installs in our `gcar` environment (Python 3.12, PyTorch 2.8) and we install nothing without the user's OK.
- **Where a later code extraction should start** (file names are [memory] / [card], unverified; first list `src/`):
  1. the feature builder (`src/feature_builders/nuplan_feature_builder.py` [memory]): polygon extraction for lanes, connectors and crosswalks, the 21-point interpolation, the centre-polyline features, **the assignment of `on_route` by roadblock id**, traffic-light and speed-limit fields, the validity masks;
  2. the feature container (`src/features/nuplan_feature.py` [memory]): normalisation to the ego frame, collation with padding;
  3. the model (`src/models/planTF/planning_model.py` and `modules/map_encoder.py`, `layers/` [memory]): `PointsEncoder`, the attribute embeddings (is on_route an `nn.Embedding(2, 128)`?), the positional embedding, the SDE of the ego state, the decoder heads;
  4. the loss and trainer (a Lightning module, `_compute_objectives` [memory]): the weights, the best-mode rule, the agent prediction term; the state perturbation and re-normalisation augmenter (`src/custom_training` or a data-augmentor class [memory]).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | A list of lane pieces: P = 20 centre points per piece in ego metres, piece-centre pose (x, y, cos, sin), per-point valid mask, one route value per piece; type / traffic light / speed limit as "unknown"; optionally ego motion (speed, acceleration, steering or yaw rate from the vehicle bus) through an SDE. No images, no lidar, no BEV raster, no HD or SD map (the SD route enters through our route head). Agent and static-obstacle branches have no source in our pipeline and would be masked out or removed |
| what it gives us | Axis A7: a published input contract to which our graph + route probability can be handed over (adapters A1 - A3 of the compatibility study); the SDE and the evidence that ego kinematics raise the open-loop score while lowering the closed-loop score (our E117 is open loop); a perturb-and-re-normalise augmentation idea. Not a way to judge graph quality closed loop: that needs a simulator (OVERVIEW section 3, item 5) |
| conflicts | (1) No licence: re-implement from the description. (2) nuPlan devkit and Python 3.9 pins: unusable. (3) No connectivity input: forks / merges are implicit in coinciding end points. (4) Route flag is roadblock-level in training (all parallel lanes), ours is lane-level and soft. (5) Range +-100 m against our +-50 m. (6) Winner-take-all best mode on the training path [memory]: against our no-argmax rule if the loss were copied; a soft-min or a single-path loss avoids it. (7) nuPlan weights are useless; the model would be trained from scratch on 8,414 frames. (8) Ego state as input creates the shortcut |
| reuse mode | Re-implement from description (idea level); no code may be copied |
| effort | S for the graph-to-polygon adapter and tensors (a day, pure PyTorch, about 150 lines [mine]); M for the consumer, its ego-motion and route-corruption ablations and two seeds. No technical blocker; value uncertain (see next row) |
| expected gain and risk | Not a path-error gain over E117 (93 k parameters, 0.34 m at 20 m): the polygon transformer has at least 8 times the weights (0.79 M encoder estimate against 93 k) [mine] and would see 8.4 k frames. The gain is (a) proof that a published planner format takes our graph, (b) a controlled test of the ego-motion dependence of E117, (c) the soft-route embedding and validity bias as reusable pieces. Risks: over-fitting, forks without topology, calibration of the soft route value (OVERVIEW section 3, item 4) |
| feeds which experiment / K id | Proposed, to be registered before use: K-a soft route embedding (convex mixture of the two flag embeddings); K-b piece validity as an additive log-confidence bias on the attention keys; K-c state dropout on the ego motion of the consumer (E115 / E117); K-d graph pose perturbation with re-normalisation as consumer augmentation. Experiment ideas in 5.5 |

### 5.1 What the planner really consumes (the contract)
Per map polygon: 20 centre points as three 2-vectors each (offset to polygon centre, next-point vector, cos / sin), one token of 128 dimensions from a point encoder, plus four attribute embeddings (type, on_route, traffic light, speed limit) and a positional embedding of the polygon-centre pose; a polygon-level key-padding mask. No edges, no confidence, no boundaries in the centre-line-only configuration. The lane length is free (the number of points is not), so spacing varies with piece length [mine, from "20 points per polygon"; the interpolation rule must be checked in the code].

### 5.2 Adapter from our directed graph to polygon tokens
Our output: vertices in ego metres, directed edges (child-to-parent pointers) with forks and merges, validity per vertex, 128-d tokens, soft route probability per vertex.

1. **Cut the graph into pieces.** Mark junction vertices (out-degree >= 2 fork, in-degree >= 2 merge) and cut the graph into maximal non-branching chains; two neighbouring pieces share the junction vertex as end and start point. Split a chain longer than L_max (suggest 40 m, spacing 2 m at 20 points; nuPlan piece lengths are not reported) into equal parts. Clip a piece at the +-50 m square and resample the inside part. Chains are formed from thresholded edges, a discrete step; gradients reach positions, route values and validities through the resampling, which is the "gradient through selected items" case the project allows.
2. **Points per piece:** P = 20 centre points by arc-length resampling (linear between vertices; a smoothed connector curve is better where the builder has one). A chain between two vertices 7 m apart (champion: 13.5 vertices per 100 m) gives 0.35 m spacing, a 40 m piece 2 m. Ablate P = 10 versus 20 (the encoder cost halves). No left / right boundaries exist in our graph: use the centre-line-only configuration that the released encoder already uses.
3. **Features:** offset to the piece centre (arc-length midpoint), vector to the next point, cos / sin of the travel heading; the direction is carried by the directed edges, as in nuPlan.
4. **Forks and merges.** Default: **shared trunk** as in nuPlan, one piece per chain, branches start at the fork vertex; the transformer must find adjacency from coinciding end points (as PlanTF does, with the 86.48 NR-CLS above). Alternative: **root-to-leaf paths** (as in LaneGAP, MAT and PLUTO), the trunk duplicated per path, 2^b paths for b binary forks in view (usually few within 50 m; measure on TRAIN); each path then has a well-defined route value. Own addition beyond PlanTF: an N x N attention bias or a two-number feature (has successor, has predecessor) from the edge list, which costs nothing because the graph already has the edges.
5. **Token count [mine]:** 13.5 vertices per 100 m and 150 - 400 m of centre-line inside the square give 20 - 55 vertices, about 6 - 20 pieces; count on TRAIN before fixing the cap (suggest 48 with padding). With about 25 tokens the encoder is a few MFLOPs; time is launch overhead (unmeasured).
6. **Frame and range:** our graph is already in the ego frame at t0 (x forward, y left assumed; check the sign convention of the builder). PlanTF's range of +-100 m only matters through the learned positional MLP; train at +-50 m.
7. **Output side:** replace the 6-mode, 80-step decoder by our head (path points at 5 ... 40 m), trained by regression through the positions; no winner-take-all on the training path. Whether a single path suffices at forks is the open question of 5.4.

### 5.3 Soft route value instead of the binary flag
- PlanTF's flag is a 2-way embedding added to the token [card]. A soft value p in [0, 1] enters exactly as the convex mixture e(p) = (1 - p) E[0] + p E[1] = E[0] + p (E[1] - E[0]). This equals the binary embedding at p = 0 and p = 1, is differentiable in p (a gradient reaches the route head, although E119 found no gain from a planner loss there), and a 1-to-128 linear layer initialised from the two table rows implements it. Nothing is compared against an argmax.
- Piece value: the length-weighted mean of the vertex route probabilities along the chain. Because a route can leave a chain only at a fork, which is a chain end, p should be nearly constant inside a chain; a large spread inside a chain is itself a hint of a bad route head and could be passed as a second scalar.
- Distribution gap: in nuPlan training all parallel lanes of a route roadblock carry 1, so the model learns to choose the lateral lane among several flagged lanes by itself. A lane-level flag marks one lane; a road-level fallback (our open item 2) would mark all lanes of a road, which is *closer* to PlanTF's training distribution than our lane-level flag. Lane-level to road-level is a max over parallel pieces; the reverse is impossible.

### 5.4 What happens when the flag is wrong or missing
- **Evidence: none.** The PlanTF paper has no route ablation; the follow-ups I read (PLUTO, HYPE, the route-integration study) do not corrupt or remove the route either (PLUTO ablates the reference-line-free head, not the route; HYPE shows only a qualitative figure of proposals that "deviate" without a route). Nothing below is measured.
- **Missing (all pieces 0):** this never occurs in nuPlan training (a route always exists), so the input is out of distribution and the response unknown. [mine] expected: harmless on a road without branches (geometry and ego motion fix the lane), unreliable at forks. The code already uses a learned "unknown" embedding for the speed limit and an UNKNOWN traffic-light class; the same pattern, a third route class "unknown" with route dropout during training (suggest 20 %), is the cheap remedy and costs one embedding row.
- **Wrong (off-route branch high):** a route-conditioned imitation planner follows the flag. [mine] our E117 reaches 0.40 m with the route probability against 0.90 m without it and 0.316 m with an oracle route channel (E119), so the consumer depends strongly on the route; a wrong fork should cost more than a missing one. The model has no way to detect the error unless it gets a cue: pass the route head's confidence (entropy of p over sibling branches, or the Sinkhorn dustbin mass if the route head gets one) as a second scalar and train with fork flips at 10 - 30 %.
- **Soft hedge at a fork (p about 0.5 on both branches):** a single-path regression head will average the two branches into a path between them. PlanTF avoids this with 6 modes; for us either use a multi-hypothesis head (rMCL, K38) or accept the hedge as a measured baseline.
- **Validity (A3).** Three options, increasing in fidelity: (i) hard key-padding mask on pieces whose validity (min or mean along the chain) falls below tau, what PlanTF does with its geometric validity, no gradient to the validity; (ii) additive log-confidence bias on the keys, logits + log(clamp(v, eps, 1)), which multiplies the attention weight of a key by v and is differentiable (`scaled_dot_product_attention` accepts a float mask); (iii) validity as an input scalar added to the token. Recommended: (ii) plus (iii), with (i) only for padding. A low-validity piece with p high should still contribute little: v and p are separate inputs.

### 5.5 Proposed experiments (not run; to be put on a card by the loop owner)
- X1, format test: PlanTF-style consumer (4 x 128, 20-point pieces) against E117 on identical cached graphs (exact map, predicted graph), two seeds, path error at 20 m, ms per frame; control: E117.
- X2, ego motion: none / full / state dropout 0.75, on all frames and on a turning-only subset (K108); expectation [mine]: dropout costs open-loop error and shows how much of E117's 0.34 m is extrapolation.
- X3, route input: binary at 0.5 / soft / soft + confidence / unknown class; corruption sweep (route dropout 0, 20 %; fork flips 0, 10, 30 %); report path error overall and at fork frames.
- X4, validity: hard mask / log-bias / token feature, on exact and degraded graphs (loop-17 failure sets).

## 6. Not found / open
- The paper has no map or route section: points per polygon, features, the on-route rule, static obstacles, layer counts, decoder, loss and parameters are all absent from the text and rest on the card or memory.
- Whether the on_route embedding is a lookup table, how connectors get their flag, the exact interpolation rule for the 20 points, the `PointsEncoder` internals (max-pooled MLP as in Forecast-MAE, or not), the agent feature list, the static-object branch, loss weights and the best-mode rule.
- Any ablation of lane or route inputs, any corrupted or missing route, any map augmentation (the card found none).
- Seeds and confidence intervals; hardware and training time; the cause of the Val14 difference between PlanTF's own table (84.83 / 76.78) and PLUTO's (85.30 / 77.07).
- The PDF could not be read; Tables II and VIII need a check against it.
- interPlan numbers for PlanTF (not in either paper).

### Corrections to the level-1 note
1. "Fixes: position-only ego input": the released PlanTF uses **all six ego states (state6) with SDE at 0.75**; position-only (state3) is a study variant (best NR-CLS among the variants without SDE, 85.99, but OLS 81.13). With re-normalisation the ego x, y, yaw in the normalised frame are presumably constant zero [mine], so state3 is then effectively "no ego state".
2. "HTML extraction lacked the map section": the paper contains **no** map section (the whole HTML was queried); the map details in the note, and the line "Verified: checked", rest on the code page and should read "code-derived, not re-checked".
3. Venue: the arXiv page lists none; ICRA 2024 comes from the GitHub README (consistent, not wrong).
4. Val14: the PLUTO figures 85.30 / 77.07 are confirmed in PLUTO's Table II (digest); PlanTF-H there is PlanTF plus PLUTO's post-processing.
5. The dropout-rate numbers (81.70 / 83.71 / 86.48) are Table VIII; its baseline row equals the Table III row "state6 + SDE without augmentation", not a plain state6 (see section 3).
6. Compute: add the runtime of Table VI, 155 ms per frame including feature extraction (hardware not stated).
7. "6 modes" is not confirmed by the paper text (only the 8 s horizon is); it is from the code.
8. No interPlan result exists for PlanTF in the sources checked.
9. Licence: the note says nothing; verified in this pass: **no LICENSE file**, so read only.

## 7. Our experiments that touch it
No experiment uses a polygon-token consumer. Own consumer E117 (93 k parameters, edges + route probability + ego motion -> path): 0.90 -> 0.40 m with the route probability, 0.34 m with ego motion (system A 0.36 m), a step token 0.56 m. E115: ego motion lowers the pooled failure error 1.09 -> 0.71 m, a failure flag adds nothing. E118: consumer under sensor failure (night 2.73 -> 0.72 -> 0.44 m). E119: planner loss into the route head no gain (0.34 m either way, oracle route channel 0.316 m). E121a / E121b: planner loss through the positions, no gain (0.36 against 0.34 m). E187 (validation, 3 seeds, exact map): path error at 20 m 0.31 m with the vertex head, 0.26 m along the edges, 0.27 m control. None of these tests the ego-motion shortcut that PlanTF documents.
