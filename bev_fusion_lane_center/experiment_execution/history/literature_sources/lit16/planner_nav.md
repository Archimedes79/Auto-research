# Cluster PL - planner interface of a lane graph, navigation / route / SD-map attribution, evaluation through planning

Date: 2026-10-23. Reader's guide: cards PL01-PL49 (PL49 was added last) are ordered by relevance to our design (route attribution and graph-as-planner-input first, then nuPlan-style consumers, then end-to-end BEV planners with map instances, uncertainty, evaluation, further planners, prediction map encoders, SD-map priors). Section 3 holds the interface table, the adapter recipes and the answers to the brief's questions. Statements marked "(my synthesis)" are not from a paper.

## 1. Coverage statement

- **Searched** with WebSearch (20 queries, section 6) and **read** with WebFetch (arXiv HTML / ar5iv method sections, GitHub README pages and selected source files read as web pages): about 50 papers and 15 code repositories. Map input formats (counts, channels, masks, normalisation) mostly come from code pages, because the papers rarely state them.
- **Depth M (method section: tensors, losses, numbers):** PlanTF, PLUTO, Diffusion Planner, GameFormer, DTPP, PDM-Closed/Open/Hybrid, LaneGCN, VectorNet, QCNeXt (QCNet scene encoding), MTR, PGP, UniAD, VAD, VADv2, SparseDrive, DiffusionDrive, Hydra-MDP and Hydra-MDP++, NAVSIM v1 and v2 (pseudo-simulation), BEV-Planner (ego status), AD-MLP, UncAD, MAP, the two Gu et al. papers (map uncertainty; BEV-feature attention), SMART, P-MapNet, SMERF, OpenLane-V2-OSM, Neural Map Prior, OrienterNet, Online Navigation Refinement (MAT), GoalFlow, Perception-in-Plan, DriveTransformer, FUMP, RAP, DriveZero, SSR, SUPER-AD, UMPE, AMap, the topology-token paper (PL14), LFP, GET-DIPP, ST-P3, the BevAD study (PL49).
- **Depth C (code pages):** PlanTF (feature builder, map encoder, model, normalisation), PLUTO (feature builder, route manager), Diffusion Planner (map_process), GameFormer-Planner (data_utils), QCNet (dataset, map encoder), HiVT (dataset), VAD (plan_loss.py, config), tuplan_garage (route_utils, abstract planner), OMA-MAT README, READMEs of UniAD, LaneGCN, PGP, NAVSIM, trajdata.
- **Depth A only:** nuPlan benchmark paper (abstract), PKL (abstract), ControlMTR (abstract), PARA-Drive and ScenarioNet (search snippets), NavigScene and several 2026 preprints (SimWAM, DA-WAM, GuideFlow, DriveSafer; snippets only, not used for claims).
- **Not openable:** arXiv PDF URLs return unparsed binary through WebFetch (one such call left a binary copy in the session tool-results folder as a tool side effect; nothing else was stored). Therefore the QCNet CVPR paper, the PKL full text, the nuPlan full text and PARA-Drive were not read. Pages longer than 100,000 characters were read only to that limit (UniAD tail, PLUTO tail, parts of ONR, DriveZero, PL14, FUMP). The fetch tool returns a model-written extraction; items it reported as absent are written "not found".
- **Not found in the literature:** (i) a paper that feeds a perceived lane graph with topology into a nuPlan-style or rule-based planner and measures closed-loop effects; (ii) a planner-aware map-quality metric (three queries); (iii) a method that back-propagates a planning loss into graph edges or topology; (iv) an SD-map route used as a perception prior (SMART states it uses no route; SMERF and OpenLane-V2-OSM do not distinguish one).
- Several 2026 arXiv IDs (26xx) were opened; their numbers are self-reported.

## 2. Cards

### CARD PL01 - Online Navigation Refinement (MAT on OMA): SD roads to perceived centre-lines, route attribution after perception
- papers: Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps (authors not read; repo owner WallelWan; ICLR 2026 per repo README) https://arxiv.org/abs/2507.07487
- depth: M+C
- stage: S9
- idea: Input is an SD road graph (OSM; each road an ordered sequence of directed vectors with connectivity) and an online-perception (OP) map (lane centre-lines sampled uniformly with topology, plus boundary vectors) produced by MapTR / MapTRv2 / SeqGrowGraph at test time. A transformer assigns each perceived centre-line to exactly one SD road; an SD path (the route) then maps to a path of OP centre-lines by topological sorting. This is route attribution after perception, but hard and road-level, not a route probability per vertex.
- formulation: tokens = vectors [x1, y1, x2, y2, theta] -> 2-layer MLP. Block = Path-Aware Attention (enumerate valid root-to-leaf paths, reorder tokens by path index, grouped attention over adjacent tokens, restore, average over paths) + Spatial Attention (3-D grid x, y, orientation, Hilbert-curve serialisation, grouped attention) + FFN. Head: Prob_ij = softmax over roads j of (feature of centre-line i . mean-pooled road feature j / sqrt d), not Sinkhorn. Post-processing: start at the most confident centre-line, bidirectional beam search constrained by road connectivity. Channels [96, 192, 384, 768, 1536]; blocks MAT-T [2,2,2,2,2], MAT-L [4,4,4,12,4].
- input: SD road graph (vectors + connectivity) and OP centre-lines + boundaries in the ego frame; GPS pose is not an input (drift is treated as spatial noise).
- output: per-centre-line road id (unique; a road may own several centre-lines); the route is the SD path mapped to an OP path by topological sorting.
- training: loss 1.0 CE + 0.01 CTC on hand-annotated associations; dataset OMA built on nuScenes + OSM: over 30 K scenarios, 480 K road paths, 2.6 M lane vectors; train + val 31,724 (GT OP maps), test 5,573 (MapTRv2-generated OP maps), area-disjoint split; 50 epochs AdamW lr 1e-4, batch 128; regime: supervised.
- differentiable: partial - the softmax head is differentiable; path extraction (beam search) is not; trained apart from the OP generator, no gradient into it.
- sensors_data: nuScenes (Boston, Singapore) + OSM; OP maps come from camera BEV models; no radar.
- results: val NR-F1(50:95) MAT-T 78.2, MAT-L 78.7, best baseline 72.9 (EAM3); test with OP maps from MapTR / MapTRv2 / SeqGrowGraph: MAT-L 41.9 / 45.0 / 54.9 vs EAM3 36.3 / 39.1 / 50.8. Imperfect SD maps (appendix, val / test): global shift 10 / 20 / 50 %: 77.5 / 43.3, 76.0 / 42.9, 74.6 / 41.4; element absence up to 30 %: 75.5 / 43.1. Zero-shot nuScenes-C hard: rain 40.6, fog 42.1, dark 41.6. Ablation val: path-aware attention alone 74.1 vs 61.8 plain backbone.
- compute: latency 34 ms (T), 70 ms (L) on an A6000; parameters not stated.
- code: https://github.com/WallelWan/OMA-MAT (dataset OMA on Hugging Face `wanjiaxu/OMA`)
- fit_to_us: reuse_idea - effort M - their SD input is road vectors, ours is SD raster tokens, and the association is hard and road-level.
- compat_needs: OP lane graph as polylines with topology (our assembled graph); SD roads as vectors with connectivity; ego-frame alignment within GPS-level drift.
- compat_conflicts: hard unique assignment vs our soft route probability; beam-search post-processing not differentiable; trained on GT OP maps, tested on generated ones; separate from perception (no end-to-end).
- tried_before: E102-E104 (route attribution inside the learned chain; cross-attention kept, route success 0.853-0.820) is the end-to-end analogue; vector-SD association not tried.
- verified: checked

### CARD PL02 - PDM-Closed / PDM-Open / PDM-Hybrid (rule-based centre-line planner, closed-loop test bed)
- papers: Parting with Misconceptions about Learning-based Vehicle Motion Planning (Dauner, Hallgarten, Geiger, Chitta; CoRL 2023) https://arxiv.org/abs/2306.07962
- depth: M+C (partial)
- stage: S10, E
- idea: nuPlan planner that needs only the route's centre-line, forecasts of other agents and the drivable-area map for scoring: IDM proposals along the centre-line, rolled out by an LQR tracker on a kinematic bicycle model, best chosen by a nuPlan-like proxy score. PDM-Open (MLP on centre-line + ego history) reaches the best open-loop score without agent or map input beyond the centre-line; PDM-Hybrid adds a learned offset. Core message: open-loop and closed-loop scores are misaligned.
- formulation: centre-line from a graph search over the lanes of the route (algorithm name not confirmed on the pages read). Route = roadblock-id sequence, repaired in code by BFS (start off-route: backward depth 15, else forward depth 30; gaps between consecutive roadblocks bridged by forward BFS depth 30; loops cut). Proposals: 5 target speeds (20 ... 100 % of the speed limit) x 3 lateral offsets (-1, 0, +1 m) = 15, 4 s at 10 Hz; others forecast with constant velocity over 8 s; best proposal extended to 8 s by IDM; emergency brake if a collision is expected within 2 s; score weights in the supplement (not read). PDM-Open: centre-line at 1 m up to 120 m + 2 s ego history at 5 Hz, each projected to 512, MLP 2 x 512, 8 s output at 0.5 s.
- input: ego pose, lane graph (nuPlan map API), route roadblock ids, tracked objects, drivable-area map (scorer).
- output: 8 s trajectory.
- training: none for Closed; L1 imitation on about 177 k scenarios for Open / Hybrid.
- differentiable: no (Closed); the learned parts yes.
- sensors_data: nuPlan with privileged GT map and GT objects (paper: localisation errors, map imperfections, misdetections are not considered); Val14 = 1,118 scenarios.
- results: Val14 CLS-R / CLS-NR / OLS: PDM-Closed 92 / 93 / 42; PDM-Hybrid 92 / 93 / 84; PDM-Open 54 / 50 / 86; IDM 77 / 76 / 38; GC-PGP (lane-graph GNN planner) 55 / 59 / 83; Urban Driver 50 / 53 / 82; PlanCNN 72 / 73 / 64; log replay 80 / 94 / 100. 2023 nuPlan challenge: PDM-Hybrid 90, rank 1 of 25.
- compute: PDM-Closed 91 ms, PDM-Open 7 ms, PDM-Hybrid 96 ms (hardware not stated).
- code: https://github.com/autonomousvision/tuplan_garage
- fit_to_us: reimplement - effort M - the original needs nuPlan devkit and nuPlan maps; the principle (best route path through the graph by dynamic programming over route probability, IDM, proxy score) is small.
- compat_needs: lane graph with successors; route (roadblock ids or our per-vertex route probability); drivable-area raster; centre-line at 1 m up to about 120 m (our BEV reaches 50 m forward); speed limit.
- compat_conflicts: privileged-map assumption; the roadblock BFS repair has no counterpart for a predicted graph with gaps; centre-line quality is the single point of failure.
- tried_before: none (E117 is a learned consumer; no rule-based consumer in the project).
- verified: checked

### CARD PL03 - PlanTF (polygon tokens with on-route flag, state-dropout encoder)
- papers: Rethinking Imitation-based Planner for Autonomous Driving (Cheng et al., ICRA 2024) https://arxiv.org/abs/2309.10443 ; project page https://jchengai.github.io/planTF
- depth: M (partial: the HTML extraction lacked the map section) + C
- stage: S10, T
- idea: Transformer encoder (adapted from a masked-autoencoder forecasting model) over agent and map tokens; the ego plan is decoded from the ego token. Shows that kinematic ego state / history gives a shortcut: strong open-loop score, weak closed-loop score. Fixes: position-only ego input, state dropout encoder (SDE), perturbation augmentation with re-normalisation.
- formulation: map input (from code): polygons = lanes + lane connectors + crosswalks within a +-100 m square (no cap in the builder; padded to the longest sample in the batch). Per polygon three polylines (centre, left, right) x 20 points (21 sampled, last dropped). The released encoder uses ONLY the centre polyline: per point [p - polygon_center (2, translation only), vector to next point (2), cos / sin heading (2)] = 6 channels -> point encoder (`PointsEncoder`, internals not read) -> 128-d; added embeddings: polygon type (3), on_route (2, roadblock-id membership), traffic-light status (4, UNKNOWN allowed), speed limit by MLP or learned "unknown" embedding. Positional embedding from (x, y, cos, sin) of the polygon centre. Ego frame at the current pose. Key padding: polygon masked if no valid point; point validity = inside the square. No adjacency; no map augmentation found. Agents: 2 s history, up to 64. Depth 4, dim 128, 8 heads, drop-path 0.2, 6 modes, 80 future steps (8 s), SDE dropout 0.75.
- input: polygon feature dict (above) + agent histories.
- output: 6 trajectories with probabilities + predictions of other agents.
- training: imitation (loss terms not found on the page); ego-state perturbation with probability 0.5; 25 epochs, batch 128, lr 1e-3 cosine, weight decay 1e-4, at most 1 M frames; regime: teacher forcing / imitation.
- differentiable: yes for the network; the map is GT, no gradient to a map head.
- sensors_data: nuPlan, GT map + GT tracked objects; radius 100 m.
- results: Test14-random OLS 87.07 / NR-CLS 86.48 / R-CLS 80.59; Test14-hard 83.32 / 72.68 / 61.70; Val14 (own Table VII) 89.18 / 84.83 / 76.78 (PLUTO's table lists PlanTF at 85.30 / 77.07 on Val14: differs between papers). SDE dropout 0.25 / 0.5 / 0.75 on Test14-random: NR-CLS 81.70 / 83.71 / 86.48 while OLS falls 89.11 / 89.12 / 87.07.
- compute: cache of 1 M frames "20+ hours"; 4-6 GB GPU memory per GPU at batch 32 (README); parameters not stated.
- code: https://github.com/jchengai/planTF
- fit_to_us: adapt_code - effort M - requires nuPlan devkit and Python 3.9, but the feature dict is a clean contract to emit from our graph.
- compat_needs: polygons with 20 centre points, polygon centre, on_route flag, per-point valid mask, ego-frame coordinates.
- compat_conflicts: no connectivity input (topology only implicit); range 100 m vs our 50 m; route flag is roadblock-level (coarser than our lane-level route); ego motion as input causes the copycat shortcut (E117 uses ego motion).
- tried_before: none for polygon consumers; E117 is the own consumer.
- verified: checked

### CARD PL04 - PLUTO (route-restricted reference lines as lateral queries, contrastive negatives)
- papers: PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving (Cheng, Chen, Chen; 2024) https://arxiv.org/abs/2404.14327
- depth: M+C
- stage: S10, S4, T
- idea: One trajectory per (reference line x longitudinal query). Reference lines come from the lane graph and the route; the 12 longitudinal queries are learned embeddings. An auxiliary drivable-area loss and a contrastive imitation loss with positive / negative scene augmentations are added; a reference-free head covers scenes without a reference line.
- formulation: map polylines with 8 channels per point (p_i - p_0, p_i - p_{i-1}, p_i - p_left, p_i - p_right), PointNet encoder; speed limit and traffic-light status as attribute embeddings; Fourier embedding of global position. Reference lines (code, RouteManager): start lanes within 3 m of the ego on a route roadblock with heading error below 90 deg; depth-first search over `outgoing_edges` restricted to route roadblocks, stop at 15 lanes or 100 m; trim from the ego projection; drop paths shorter than 0.8 x length; de-duplicate when the maximum coordinate difference is below 0.5 m; feature = 100 points per line (every 4th sample) + future projection of the ego. Decoder: lateral queries (reference lines) x N_L = 12 longitudinal queries, D = 128, 4 + 4 layers; output 80 steps x 6 channels; top K = 20 of N_R x N_L modes. Loss = L_i (smooth-L1 on the target query selected by teacher forcing from the GT endpoint + CE on mode scores) + L_p + L_aux (hinge on an ESDF cost map 500 x 500 at 0.2 m, 3 covering circles) + L_c (softmax triplet, temperature 0.1), all weights 1. Positive augmentation: ego-state perturbation, dropout of non-interactive agents; negative: remove leading agent, insert a leading agent from another sample, remove interactive agent, invert traffic lights; batch tripled. Model code (pluto_model.py): defaults dim 128, 4 encoder + 4 decoder layers, 8 heads, 6 modes, drop-path 0.2, dropout 0.1, `use_ego_history=False`, state dropout 0.75, `ref_free_traj=False` (the reference-free head is optional; the evaluation fallback for scenes without reference line needs it enabled); the map encoder uses lane boundaries (`use_lane_boundary=True`); position embedding = Fourier embedding of (x, y, heading) of agents and polygon centres; key padding from the validity masks of agents, polygons and static objects; the reference-line validity only masks the output probabilities at evaluation.
- input: map polylines, agents (20 steps), static obstacles, reference lines.
- output: up to 20 scored trajectories.
- training: loss above; 1 M frames, 25 epochs, batch 128, AdamW; regime: imitation with teacher-forced reference-line choice + contrastive.
- differentiable: yes for the network; reference-line generation is a rule (DFS), no gradient to the map.
- sensors_data: nuPlan GT map.
- results: Val14 non-reactive / reactive score: PLUTO with post-processing 93.21 / 92.06, without 89.04 / 80.01; PDM-Closed 93.08 / 93.20; PlanTF 85.30 / 77.07; PlanTF-H 89.96 / 88.08; GameFormer 82.95 / 83.88; log replay 93.68 / 81.24.
- compute: 4 x RTX 3090, about 45 h with the contrastive loss (22 h without); parameters not stated.
- code: https://github.com/jchengai/pluto (README: training part not fully tested)
- fit_to_us: reuse_idea - effort M - DFS over route-filtered successors is exactly what our edge graph + route probability supports; the reference-free head is a model for the "no usable lane" case.
- compat_needs: successor edges, a route filter, polylines resampled to a fixed point count, drivable-area raster for the auxiliary loss.
- compat_conflicts: roadblock-level route; nuPlan devkit; DFS on a graph with false forks multiplies reference lines (only the 0.5 m de-duplication limits their number).
- tried_before: none.
- verified: checked

### CARD PL05 - Diffusion Planner (lane tensor 70 x 20 x 12, route lanes through adaLN)
- papers: Diffusion-based Planning for Autonomous Driving with Flexible Guidance (Zheng et al., ICLR 2025) https://arxiv.org/abs/2501.15564
- depth: M+C
- stage: S10, S5
- idea: A DiT-style transformer denoises the joint future of the ego and 10 neighbours conditioned on scene tokens; route lanes condition every block via adaptive layer norm; inference-time classifier guidance with energy functions (collision, speed range, comfort, drivable area).
- formulation: lanes (code map_process.py): 70 x 20 points x 12 channels [centre xy (2), vector to next centre point (2), left-boundary offset (2), right-boundary offset (2), traffic-light one-hot (4)]; speed limit and has-speed-limit stored per lane (70 x 1). Lanes = LANE + LANE_CONNECTOR within a radius, ranked by minimum distance of any centre point to the ego, nearest kept; successors NOT traversed; linear resampling to the fixed point count; empty slots zero-filled with an `avails` mask. Route lanes: 25 (Table 5) with the shared max_points (20) built by the same routine (paper: coordinates only), from route roadblock ids pruned by connectivity. Ego-centric, then z-score x (mu 10, sigma 20), y scaled by sigma 20. Neighbours 32 x 21 steps x 11 features (predicts 10). MLP-Mixer encoders; hidden 192, 3 encoder + 3 decoder blocks, 6 heads. Route embedding is added to the timestep embedding and applied via adaLN to all tokens. VP linear schedule (beta 0.1 ... 20), x0-prediction MSE, DPM-Solver++ 10 steps, temperature 0.5 (0.1 with refinement). Drivable-area energy = signed-distance cost map of the lanes; route not used in guidance. Encoder code (encoder.py): validity is read from all-zero rows, not from `avails` (a lane whose points all have zero geometry is masked through key_padding_mask; padding points inside a valid lane still enter the mean pooling); lane token = MLP-Mixer over the 20 points (token dim 64, channels 128) mean-pooled to 128-d, plus speed-limit Linear(1 -> 128) or a learned unknown embedding, plus traffic-light Linear(4 -> 128) taken from the first point of the lane; positional embedding Linear(7 -> hidden) of (x, y, cos, sin) + 3-way token-type one-hot; tokens concatenated [neighbours, static objects, lanes] and processed by plain self-attention blocks (no cross-attention); encoder.py has no separate route input (route handling sits in the decoder; code not read).
- input: ego state, neighbours, static objects, lanes, route lanes.
- output: joint ego + neighbour trajectories, 8 s at 10 Hz.
- training: x0-prediction MSE; about 11 M sampled nuPlan scenarios, 500 epochs, batch 2048, lr 5e-4; ego-state perturbation + quintic interpolation to the GT future; regime: denoising.
- differentiable: yes (network); guidance differentiates energies w.r.t. the sample, not the map.
- sensors_data: nuPlan GT map; delivery-vehicle data (about 200 h) as a second source.
- results: closed-loop non-reactive / reactive: Val14 89.87 / 82.80, Test14-hard 75.99 / 69.22, Test14 89.19 / 82.93; with refinement Val14 94.26 / 92.90, Test14-hard 78.87 / 82.00; delivery data 92.08.
- compute: about 20 Hz on one A6000 (appendix); training 8 x A100 80 GB; parameters not stated.
- code: https://github.com/ZhengYinan-AIR/Diffusion-Planner
- fit_to_us: adapt_code - effort M - the lane tensor contract is useful, but boundary offsets and traffic-light channels do not exist in our graph.
- compat_needs: fixed-size lane tensor in which invalid lanes are all-zero rows (the encoder derives its mask from zero geometry), ordered route lanes, ego-frame coordinates.
- compat_conflicts: no edges used; boundaries and traffic lights must be zero / unknown; 100 m vs 50 m range; trained only on the nuPlan distribution.
- tried_before: none.
- verified: checked

### CARD PL06 - GameFormer (level-k interactive prediction + planning) and GameFormer-Planner code
- papers: GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning (Huang et al., ICCV 2023) https://arxiv.org/abs/2303.05760
- depth: M+C
- stage: S10
- idea: Level-0 decoder predicts every agent from scene context; level k re-conditions each agent's decoder on the level k-1 futures of all other agents. The nuPlan pipeline adds reference-path selection from the route and a differentiable Gauss-Newton refinement.
- formulation: prediction model: agent history 11 steps (1 s), 22 agents; map up to 6 drivable lanes per target agent x 100 waypoints x 15 features (centre, left, right position and heading, speed limit, lane type, traffic light, stop-sign control), up to 4 crosswalks x 100 x 3, max-pooled to 60 lane + 20 crosswalk tokens; D = 256; K = 6 levels (K = 4 for open-loop planning). Route (paper): reference path from a cubic spline through traversed-lane waypoints, 100 m ahead, 1000 points at 0.1 m, 5 channels. Code (GameFormer-Planner): lanes (N, P, 3) = x, y, heading (+ 4 traffic-light one-hot = 7), route_lanes (N, P, 3) from route roadblock ids, crosswalks (N, P, 3); nearest N by minimum point distance; boundaries computed but not returned; `avails` (N, P) mask; ego frame at the present state; no other normalisation; N, P from config (not seen). Loss = L_IL (NLL of best mode per step + CE, every level) + 0.1 L_Inter (repulsive potential). Refinement: DIPP-style Gauss-Newton (max 30 iterations, step 0.3), cost weights learned through the optimiser.
- input: ego / agent histories, lanes, route, crosswalks.
- output: M joint trajectories (GMM) per level + refined ego plan.
- training: planning 20 epochs, AdamW lr 1e-4, batch 32, one RTX 3080; regime: imitation + interaction loss.
- differentiable: yes (optimiser differentiable, cost weights learned).
- sensors_data: nuPlan GT map; Waymo.
- results: nuPlan test benchmark overall 0.8288 (OL 0.8400, CL-NR 0.8087, CL-R 0.8376; second behind 0.8745); Waymo interactive (J, M = 6) minADE 0.9161, minFDE 1.9373, MR 0.4531, mAP 0.1376; WOMD closed loop success 73.16 % -> 94.50 % with refinement.
- compute: parameters not reported.
- code: https://github.com/MCZhi/GameFormer-Planner (project https://mczhi.github.io/GameFormer/)
- fit_to_us: reuse_idea - effort M - reference path from the route plus differentiable refinement could post-process E117's path; the level-k part needs agents, which we do not predict.
- compat_needs: route lanes, lane centre-lines with heading, mask.
- compat_conflicts: no connectivity; needs agents for the game part; nuPlan devkit.
- tried_before: none.
- verified: checked

### CARD PL07 - DTPP (tree policy planning with learned cost)
- papers: DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning (Huang et al., ICRA 2024) https://arxiv.org/abs/2310.05885
- depth: M
- stage: S4, S10
- idea: Ego plans a two-stage tree of candidate trajectories along reference paths; a query-centric transformer predicts all agents conditioned on every branch in one pass; a context-aware learned cost scores branches; softmax over costs + cross-entropy trains cost and prediction jointly.
- formulation: 3 reference paths (lane centre-lines from map and route, as in GameFormer); stage 1 (0-3 s) up to 30 target states over the paths, top 5 kept by learned cost; stage 2 (3-8 s) 6 targets per node, up to 30 branches; third-order polynomial speed profile; dynamic programming selects the first action. Encoder: 10 agents + ego, 20 past steps, 50 nearest map elements (lanes + crosswalks, MLP, dim 256, points per lane not stated), 3 self-attention layers; decoder with multi-axis (time / mode) causal attention. Cost = time-averaged weighted sum of ego features (jerk, acceleration, speed, lateral acceleration), a learned 5-d interaction feature and a Gaussian-RBF collision feature; weights from a 10-d MLP on the ego encoding. Loss = sum over stages beta^j (L_p smooth-L1 on the branch closest to the GT plan + 0.1 L_c cross-entropy), beta = 1.0, 0.2.
- input: lane graph (50 elements), route reference paths, agent histories.
- output: selected ego trajectory + predicted agent futures per branch.
- training: 100 k scenarios from the nuPlan validation split, 20 epochs, batch 16, AdamW lr 1e-4, one RTX 3080; regime: imitation / max-entropy IRL form.
- differentiable: yes (costs are differentiable functions of predicted states; softmax selection in training).
- sensors_data: nuPlan GT; 200 test scenarios (20 per type).
- results: nuPlan closed-loop non-reactive / reactive: DTPP 0.8964 / 0.8978; PDM 0.9061 / 0.9150; IDM 0.6396 / 0.6168; TPP 0.7388 / 0.7699; open-loop 0.7911.
- compute: 98.0 ms with pruning (199.6 ms without).
- code: https://github.com/MCZhi/DTPP
- fit_to_us: reuse_idea - effort M - a tree of route-conditioned candidates scored by a learned cost parallels our hypothesis-tree thinking on the planner side.
- compat_needs: reference paths from lane graph + route; map elements as tokens.
- compat_conflicts: needs agent futures; nuPlan devkit.
- tried_before: none.
- verified: checked

### CARD PL08 - VAD (map vectors as planner input and planning constraints; gradient into map points)
- papers: VAD: Vectorized Scene Representation for Efficient Autonomous Driving (Jiang et al., ICCV 2023) https://arxiv.org/abs/2303.12077
- depth: M+C
- stage: S10, L
- idea: Map queries predict 100 polylines x 20 points in 3 classes (lane divider, pedestrian crossing, road boundary). The ego query attends agent queries, then map queries (separate positional encoders); an MLP on [ego feature after agent attention, after map attention, ego status, command] outputs the plan. Training-only planning constraints use the predicted vectors.
- formulation: ego-boundary hinge L_bd = max(0, delta_bd - d), d = distance of each waypoint to the nearest predicted boundary point, delta_bd = 1.0 m; ego-lane direction L_dir = angular difference between the ego step vector and the nearest lane-divider vector within 2.0 m; collision hinge with delta_X 1.5 m, delta_Y 3.0 m. Only predicted instances with class score >= 0.5 enter. Config weights: bound 1.0, direction 0.5, collision 1.0, regression 1.0; map cls focal 2.0, points L1 1.0, direction cosine 0.005; BEV 200 x 200 over 30 x 60 m; 4-frame queue; 3 discrete commands. Code: min over all map points of the Euclidean distance (gradient reaches the selected point); atan2 of gathered points for the yaw term (gradient reaches map points); scores only act as boolean masks (no gradient); `perception_detach` exists in the bound loss, default False, not set in the config.
- input: multi-view BEV features, ego status (CAN bus), command.
- output: 6-step ego trajectory + map vectors + agent motion.
- training: map and agent losses of its own heads (matching as in MapTR, from memory) + planning losses; fully end-to-end.
- differentiable: partial - planning gradients reach predicted map point coordinates; class scores get none; matching and thresholds are discrete.
- sensors_data: nuScenes cameras, BEV 60 x 30 m.
- results: nuScenes val avg L2 / collision: VAD-Base 0.72 m / 0.22 %, VAD-Tiny 0.78 / 0.38, UniAD 1.03 / 0.31. Ablation (Table 2): full 0.72 / 0.22; no map interaction 0.86 / 0.29; no agent interaction 0.82 / 0.26; no constraints 0.76 / 0.28; only overstep 0.80 / 0.24; only direction 0.75 / 0.25. Rasterised vs vectorised map 0.74 / 0.39 vs 0.72 / 0.22.
- compute: VAD-Tiny 59.5 ms (16.8 FPS), VAD-Base 224.3 ms (4.5 FPS) on an RTX 3090.
- code: https://github.com/hustvl/VAD
- fit_to_us: reuse_idea - effort S - hinge constraints on the nearest vertex / edge (1.0 m, 2.0 m, confidence 0.5) are easy; the boundary term needs road boundaries, which a centre-line graph lacks.
- compat_needs: ordered map points with a class score; an ego trajectory head.
- compat_conflicts: Hungarian matching in the original map head; mmcv / mmdet3d stack; ego-status leakage on nuScenes open loop.
- tried_before: E121 (planner loss through the positions: no gain) is the same idea; VAD's own gain from the constraints is small (L2 0.76 -> 0.72).
- verified: checked

### CARD PL09 - UncAD (per-vertex Laplace uncertainty into planning and trajectory selection)
- papers: UncAD: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty (Yang et al., ICRA 2025) https://arxiv.org/abs/2504.12826
- depth: M
- stage: S10, L
- idea: Every predicted map point carries a Laplace scale; the scales are encoded into "uncertainty queries", concatenated with the map queries and used as keys / values for agent and ego attention; at inference candidate trajectories are scored against the Laplace ellipses of boundary points and risky ones are set to zero.
- formulation: map head output V_m of shape N_v x N_p x 4 (N_v = 100, N_p = 20; mu x, y and scale b x, y); loss = Laplace NLL, sum over points of log(2b) + |p - mu| / b; Q_unc = MLP(b); Q_m' = MLP([Q_m, Q_unc]) is the key / value of the agent and ego cross-attention; planner output 3 commands x N modes with confidences; UCAS: filter by command, score each trajectory point by NLL against the boundary ellipses, set the score to 0 if risk is high (threshold not stated), plus the base collision check and L2 distance to boundaries; the best remaining trajectory is output. Total loss = lambda_1 L_map (NLL) + lambda_2 L_mot + lambda_3 L_plan (weights not given).
- input: VAD- or SparseDrive-style map instances with scales.
- output: ego trajectory + selection.
- training: end-to-end, no extra uncertainty label; 8 x RTX 3090; epochs not given.
- differentiable: partial - uncertainty enters as features (differentiable); trajectory selection is a hard filter.
- sensors_data: nuScenes.
- results: nuScenes L2 / collision / drivable-area conflict rate DACR (a frame counts if any of the 4 ego corners lies outside the drivable area): VAD 0.72 / 0.23 % / 1.39 %; VAD + UncAD 0.62 / 0.17 % / 0.80 %; SparseDrive 0.61 / 0.10 / 1.06; SparseDrive + UncAD 0.60 / 0.07 / 0.88. Abstract: up to 26 % fewer collisions, up to 42 % fewer drivable-area conflicts, +1.9 % parameters.
- compute: +1.9 % parameters; latency not stated.
- code: https://github.com/pengxuanyang/UncAD
- fit_to_us: reuse_idea - effort S - our validity / keep probability is the analogue of 1 / b; there is no per-vertex position scale yet.
- compat_needs: per-vertex uncertainty in the same tensor as the positions; a candidate-trajectory scorer.
- compat_conflicts: thresholds unspecified; selection is non-differentiable.
- tried_before: none (E120 refinement and the keep gate exist, but not as planner input).
- verified: checked

### CARD PL10 - SparseDrive (sparse map instances as planner keys / values)
- papers: SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al., ICRA 2025) https://arxiv.org/abs/2405.19620
- depth: M
- stage: S10
- idea: Symmetric sparse perception (detection / tracking and online mapping share one decoder design) feeds a parallel motion planner. Map instances: N_m = 100 polylines x N_p = 20 points with anchors from K-means. The ego instance (initialised from the pooled smallest-scale front-camera feature) attends agents and map instances and plans in command-specific modes (K_p = 6, T_p = 6). Hierarchical selection: command filter, then collision-aware rescoring with the two most confident agent trajectories.
- formulation: ego anchor from kinematics, velocity from the previous prediction (against ego-status leakage); ego status decoded as an auxiliary output; agent-temporal cross-attention, agent-agent self-attention, agent-map cross-attention; memory queue of 3 frames; mapping range 60 x 30 m; tracking id once confidence exceeds 0.2.
- input: multi-view image features.
- output: ego trajectories per command + scores; map and agent instances.
- training: stage 1 perception from scratch (S: 100 epochs batch 8 lr 4e-4; B: 80 epochs), stage 2 joint 10 epochs with no frozen weights (no detach mentioned); regime: Hungarian matching for instances (from memory), imitation for planning.
- differentiable: partial - stage-2 gradients reach perception queries; instance matching, top-k and rescoring are discrete.
- sensors_data: nuScenes cameras.
- results: nuScenes val avg L2 / collision: SparseDrive-S 0.61 m / 0.08 %, -B 0.58 / 0.06 (VAD-style metrics); UniAD 0.73 / 0.61 and VAD 0.72 / 0.21 in the same table; collision-aware rescoring 0.12 -> 0.08 %; no ablation on map features or ego status.
- compute: 9.0 FPS (S), 7.3 FPS (B) on an RTX 4090; 85.9 M / 104.7 M parameters; training 18 + 2 h (S).
- code: https://github.com/swc-17/SparseDrive
- fit_to_us: reuse_idea - effort M - instance features + anchors as keys / values and per-command modes are portable, the mmcv-based code is not.
- compat_needs: map instance tensor (feature + 20 x 2 anchors) and detections for the rescoring.
- compat_conflicts: Hungarian matching; mmdet3d stack; K-means anchors over polylines assume a fixed point order.
- tried_before: E31 / E32 (matched sparse set decoder dropped) concern the perception side only.
- verified: checked

### CARD PL11 - UniAD (planner reads BEV + ego query; map queries reach it only through MotionFormer)
- papers: Planning-oriented Autonomous Driving (Hu et al., CVPR 2023) https://arxiv.org/abs/2212.10156
- depth: M (tail of the page not read)
- stage: S10
- idea: MapFormer produces map queries (300 thing queries for lanes / dividers / crossings with Hungarian matching + 1 stuff query for the drivable area) and a panoptic map; only the map queries go on to MotionFormer, where agent and map queries interact; the planner gets the ego query from MotionFormer, three learned command embeddings and the BEV feature, cross-attends the BEV in 3 decoder layers and regresses 6 steps (3 s). Inference-only occupancy optimisation picks the candidate minimising coordinate deviation + lambda_obs x Gaussian occupancy penalty (d = 5, sigma = 1, lambda_coord 1, lambda_obs 5).
- formulation: BEV segmentation map not passed to the planner; max-pool over command-embedded modalities builds the plan query.
- input: multi-view BEV features, command.
- output: 3 s plan; panoptic map; motion and occupancy.
- training: stage 1 tracking + mapping about 6 epochs; stage 2 end-to-end 20 epochs; image backbone gradient stopped in stage 1, image backbone and BEV encoder frozen in stage 2 (as read on the page); no other detaching described.
- differentiable: partial - gradients reach map queries only through MotionFormer attention; matching discrete.
- sensors_data: nuScenes cameras.
- results: Table 2 (avg L2 / collision): no intermediate tasks (row 10) 1.131 m / 0.773 %; full UniAD (row 12) 1.004 / 0.430; the planning table does not isolate the map module; Table 10: occupancy optimisation lowers collision but raises L2 (0.44 -> 0.54 at 1 s); minADE 0.751 -> 0.736 when mapping is added to tracking + motion.
- compute: 125.0 M parameters, 1.8 FPS on an A100 (numbers from the SparseDrive comparison).
- code: https://github.com/OpenDriveLab/UniAD (v2.0 branch: mmdet3d 1.x, torch 2.x per README)
- fit_to_us: baseline - effort L - the planner design (BEV cross-attention + command) is plain, the mmcv stack is the obstacle.
- compat_needs: map queries with positions; BEV feature.
- compat_conflicts: Hungarian matching; frozen BEV encoder in stage 2.
- tried_before: E31 / E32 (matched sparse decoder dropped).
- verified: partly (appendix cut)

### CARD PL12 - MAP (plan-enhancing online mapping + ego-status-guided planning)
- papers: MAP: End-to-End Autonomous Driving with Map-Assisted Planning (Yin et al., ICCVW 2025) https://arxiv.org/abs/2509.13926
- depth: M
- stage: S10
- idea: Planning query Q_map = cross-attention of the mapping module's memory features with the ego-status embedding; a parallel branch Q_plan = cross-attention of BEV features with ego status; a weight adapter alpha in [0,1] (sigmoid of a 64-d vector from ego status + command) fuses Q_fused = alpha Q_plan + (1 - alpha) Q_map. The mapping module receives gradients from the mapping loss and from the trajectory loss through Q_map.
- formulation: built on UniV2X (BEVFormer BEV, Panoptic SegFormer mapping), tracking / motion / occupancy removed; 5 s horizon (10 steps); loss = collision (BEV box overlap) + ADE + adaptive L2 / collision / off-road terms (weights not given); only the fused trajectory is supervised.
- input: BEV features, mapping memory, ego status (CAN bus), command.
- output: 5 s trajectory.
- training: initialised from UniV2X stage-1 checkpoint; regime: imitation.
- differentiable: yes (explicit statement of gradient into the mapping module).
- sensors_data: DAIR-V2X-seq-SPD (cooperative driving), not nuScenes.
- results: val L2 / collision / off-road: UniV2X (retrained) 3.46 / 0.34 / 0.74, MAP 2.48 / 0.94 / 0.89 (collision and off-road higher than the baseline on validation); test leaderboard MAP L2 2.56, collision 0.96, off-road 0.39 vs UniV2X 3.07 / 0.71 / 0.89. Ablation (data subsampled to 1/5): w/o plan-enhancing mapping 3.22 / 1.35 / 1.13; w/o ego-guided branch 3.52 / 0.45 / 1.35; full 2.88 / 2.48 / 0.45.
- compute: 2 x A800, about 50 h; transmission cost 1.69e5 BPS vs 8.09e5.
- code: https://gitee.com/kymkym/map.git
- fit_to_us: reuse_idea - effort S - mapping memory as planner query and an ego-status-dependent fusion weight are simple, but collision got worse.
- compat_needs: dense mapping memory features.
- compat_conflicts: cooperative-driving pipeline; different dataset.
- tried_before: E119 / E121 (planner loss into route head / positions: no gain).
- verified: checked

### CARD PL13 - UMPE (map prior encoder with confidence-biased attention, used with VAD)
- papers: Unified Map Prior Encoder for Mapping and Planning (arXiv 2605.02762; authors not read) https://arxiv.org/abs/2605.02762
- depth: M
- stage: S9, S10
- idea: One prior encoder for HD vectors, OSM vectors, satellite tiles and rasterised SD maps. Vector priors get SE(2) pre-alignment, multi-frequency sinusoidal point encoding and a 6-layer transformer producing tokens + confidences; BEV queries attend with dual cross-attention carrying a log-confidence bias, then presence-normalised channel gating. Raster priors: ResNet-18 with FiLM and grid_sample micro-alignment.
- formulation: merge X = LN(Y) + alpha W LN(Z), W zero-initialised, alpha ramps 0 to about 0.6; source dropout p = 0.3.
- input: priors + BEV query features.
- output: augmented BEV feature for MapTRv2 / MapQR mapping and for VAD's planning decoder.
- training: from scratch with a two-stage curriculum and separate learning rates for prior branches; whether planning gradients reach the prior encoder is not confirmed.
- differentiable: yes (the confidence bias is additive in the attention logits).
- sensors_data: nuScenes.
- results: planning with VAD: avg L2 0.72 -> 0.42 m, collision 0.22 -> 0.12 % (open loop only; no boundary metric, no NAVSIM); no planner-input noise test.
- compute: planning runs 60 epochs on 8 x A800.
- code: https://github.com/Ethan-Zheng136/UMPE
- fit_to_us: reuse_idea - effort S - the log-confidence additive attention bias is the soft form of our validity mask.
- compat_needs: per-token confidence.
- compat_conflicts: only VAD tested for planning; SparseDrive / UniAD / DiffusionDrive appear only as table baselines.
- tried_before: E40 / E48 (map prior at the grid, learned registration) are the closest.
- verified: checked

### CARD PL14 - Topology-conditioning token for end-to-end planners (GT vs predicted topology)
- papers: Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving (arXiv 2608.18035; authors not read) https://arxiv.org/abs/2608.18035
- depth: M (page cut at 100k characters)
- stage: S10, E
- idea: Traffic elements (4 light states, 9 sign classes lifted to 3-D) and the ego-centric topology (centre-line-to-centre-line and centre-line-to-element relations of OpenLane-V2) are rendered as a sentence, encoded by a frozen BERT-base (about 110 M parameters) into queries and concatenated with the planner's BEV queries.
- formulation: ego lane found from the ego position; LCLC gives ego-related centre-lines, LCTE the elements governing the ego lane; Q_plan = [Q_bev; Q_topo]; traffic-element 3-D centres from 2-D detection + UniDepthv2 + lidar.
- input: lane subgraph (as text) + BEV queries of the base planner.
- output: planner trajectory (base planners VAD, Orion on nuScenes; LTF, DiffusionDrive, DrivoR on NAVSIM; VAD, DriveTransformer-Large on Bench2Drive).
- training: planners fine-tuned with the extra tokens (details not read).
- differentiable: no through the text rendering and the frozen BERT; gradient reaches the planner only.
- sensors_data: nuScenes (OpenLane-V2 annotations), NAVSIM (traffic elements only), Bench2Drive (GT topology).
- results: nuScenes with VAD (ego scope, BERT encoder): GT topology avg L2 0.62 m / collision 0.16 %; topology predicted by TopoMLP 0.62 / 0.16 (identical); baseline VAD 0.72 / 0.22; full method 0.60 / 0.17; Orion 0.34 / 0.37 -> 0.26 / 0.23. Bench2Drive driving score VAD 42.3 -> 56.4, DriveTransformer-Large 63.46 -> 68.29 (traffic elements and topology combined).
- compute: BERT 110 M parameters (frozen); rest not read.
- code: none (project page https://zzongzheng0918.github.io/TE-Aware-E2E-AD/)
- fit_to_us: reuse_idea - effort S - a graph summary as one conditioning token is cheap; on nuScenes open loop the planner cannot tell GT from predicted topology, so the metric does not resolve graph quality.
- compat_needs: a text or vector summary of the ego-relevant subgraph.
- compat_conflicts: needs a text encoder; topology comes from OpenLane-V2 annotations, not from nuScenes map-expansion lanes.
- tried_before: none.
- verified: partly

### CARD PL15 - Producing and Leveraging Online Map Uncertainty in Trajectory Prediction
- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., CVPR 2024) https://arxiv.org/abs/2403.16439
- depth: M
- stage: S10, E
- idea: Online mapping heads (MapTR, MapTRv2 incl. a centre-line variant, StreamMapNet) output a Laplace location and scale per vertex; predictors (HiVT, DenseTNT) take (mu, b, class probabilities) as vertex features. It provides a ready nuScenes pipeline that scores a predicted centre-line map by downstream prediction.
- formulation: vertex (x, y) = two independent Laplace; loss sum of log(2b) + |v - mu| / b (regression weight 0.03 for MapTR / MapTRv2, 2 for StreamMapNet); std = sqrt 2 b; coordinates ego-relative, scene rotated to heading "up"; DenseTNT (VectorNet lane encoder) concatenates uncertainty to the vertex vector; HiVT changes from a vector set to a point set and doubles the node dimension; no resampling or ordering described.
- input: online vector map (vertices with scale and class probabilities), agent histories.
- output: agent trajectories (3 s from 2 s of history).
- training: staged; no end-to-end mapping + prediction (the "up to 50 %" claim concerns faster convergence of the predictor, e.g. DenseTNT 88 -> 44 epochs; the table label -22 % for the centre-line row does not match 99 -> 7 epochs).
- differentiable: yes inside each stage; no gradient from prediction to mapping.
- sensors_data: nuScenes (500 / 200 / 150 scenes), 2 Hz upsampled to 10 Hz; mAP at Chamfer 0.5 / 1.0 / 1.5 m.
- results: minADE / minFDE / MR without -> with uncertainty: HiVT + MapTR 0.4015 / 0.8418 / 0.0981 -> 0.3854 / 0.7909 / 0.0834; HiVT + MapTRv2-centre-line 0.3790 / 0.7822 / 0.0853 -> 0.3727 / 0.7492 / 0.0726; DenseTNT + MapTRv2-centre-line 0.8466 / 1.345 / 0.1520 -> 0.8135 / 1.311 / 0.1593 (MR slightly worse); DenseTNT with the GT HD map 0.8809 / 1.489 / 0.1903, worse than the online centre-line map without uncertainty. Largest MR gain -15 %; several DenseTNT cells get worse. Calibration not measured (future work).
- compute: not stated.
- code: https://github.com/alfredgu001324/MapUncertaintyPrediction
- fit_to_us: baseline - effort M - our graph pieces can be written as centre-line instances and scored by HiVT / DenseTNT on nuScenes, but the stack (trajdata, mmdet3d-based mapping code) is heavy.
- compat_needs: polylines with class probabilities and per-vertex scale, ego-relative frame.
- compat_conflicts: fixed point count per instance in the original models; HiVT / DenseTNT use no topology edges.
- tried_before: none.
- verified: checked

### CARD PL16 - Direct BEV-feature attention for behaviour prediction (cached mapping features into the predictor)
- papers: Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention (Gu et al., ECCV 2024) https://arxiv.org/abs/2407.06683
- depth: M
- stage: S10, B
- idea: Skip map decoding; the predictor reads the mapping model's BEV feature grid. HiVT: the BEV grid is split into flattened patches projected to N tokens, agent patches are queries, all map patches keys / values, vector lanes removed. DenseTNT: BEV features refined by a 1-D CNN and sampled at each lane vertex, concatenated with vertex positions (hidden size doubled), or agent subgraphs replaced by agent-BEV attention.
- formulation: BEV grid 200 x 100 for MapTR / MapTRv2 and 100 x 50 for StreamMapNet over 60 x 30 m; patch sizes (20, 10), (20, 20) for the centre-line model, (10, 5); attention MLP dims 512 (MapTR), 64 (MapTRv2, CL), 128 (StreamMapNet). Features are extracted once and stored (the pre-built datasets are about 500 GB each and contain the BEV features).
- input: stored BEV features, agent histories (and lanes for DenseTNT).
- output: agent trajectories.
- training: mapping models trained to convergence first, predictors trained on the stored features; joint training listed as future work.
- differentiable: yes inside each stage; no gradient from prediction to mapping.
- sensors_data: nuScenes, MapTR / MapTRv2 / StreamMapNet cameras.
- results: minADE / minFDE / MR vectorised -> BEV: HiVT + MapTR 0.4234 / 0.8900 / 0.0955 -> 0.3617 / 0.7401 / 0.0720; DenseTNT + MapTR 1.0462 / 2.0661 / 0.3494 -> 0.7608 / 1.4700 / 0.2593; HiVT + MapTRv2-CL 0.3657 / 0.7473 / 0.0710 -> 0.3652 / 0.7323 / 0.0710 (no change). No comparison to the GT map.
- compute: up to 73 % faster inference (MapTR + HiVT 22.4 -> 9.1 ms on an RTX 4090).
- code: https://github.com/alfredgu001324/MapBEVPrediction
- fit_to_us: reuse_idea - effort S - this is our situation (frozen cached BEV features, consumer reads them): a planner can read the 200 x 200 pyramid at the vertex positions in addition to the decoded graph.
- compat_needs: BEV features aligned to the ego frame; vertex positions for sampling.
- compat_conflicts: staged training means the consumer does not shape the BEV model (as with our frozen foundation).
- tried_before: E117 reads edges + route probability + ego motion; reading BEV features at vertices is not listed as tried.
- verified: checked

### CARD PL17 - SUPER-AD (BEV-segmentation uncertainty re-weights and filters diffusion proposals)
- papers: SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving (arXiv 2511.22865; authors not read) https://arxiv.org/abs/2511.22865
- depth: M
- stage: S10, L
- idea: Aleatoric uncertainty on the BEV segmentation head: per pixel and class a Gaussian logit (mean, std); the Monte-Carlo mean over T reparameterised samples gives probabilities, drivable classes are summed and a binary entropy is computed. A DiffusionDrive decoder proposes trajectories; each is scored by the minimum pixel safety along its path, which re-weights selection; trajectories crossing non-drivable pixels are discarded. A lane-following regulariser adds supervision.
- formulation: loss = -log of the average softmax over T samples of mu + sigma * eps; S_safe = p_pos (1 - H) + 0.5 H; lane loss = intent loss (L1 between a predicted per-point "follow lane" mask and an expert mask) + centre loss (distance of flagged points to the nearest lane-centre pixel of the BEV segmentation).
- input: BEV segmentation logits with std, DiffusionDrive proposals.
- output: selected trajectory.
- training: end-to-end with DiffusionDrive; weights not read.
- differentiable: partial - proposal weighting is soft, the exclusion of non-drivable crossings is hard.
- sensors_data: NAVSIM, camera only.
- results: navtest PDMS 87.7 (NC 98.1, DAC 97.0, EP 81.5, TTC 93.9); EPDMS navtest 84.26 vs 83.96 (DiffusionDrive camera + lidar); navhard + navsafe EPDMS 69.74 vs 66.24; ablation: BEV encoder 59.70, vector-map uncertainty 59.02, segmentation uncertainty 69.19, full 69.74; lane regularisation 66.55 -> 67.78 (text and table disagree on which uncertainty type is better; the table is used here).
- compute: not read.
- code: none given.
- fit_to_us: reuse_idea - effort S - a graph validity could weight path safety, and the drivable probability raster exists in our cached map-layer logits.
- compat_needs: raster of drivable probability along candidate paths.
- compat_conflicts: relies on BEV segmentation, not on a graph.
- tried_before: none.
- verified: checked

### CARD PL18 - AMap: forward-region errors matter more (ahead-aware map metric A-mAP)
- papers: AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction (arXiv 2512.19150; authors not read) https://arxiv.org/abs/2512.19150
- depth: M
- stage: E
- idea: Split the region of interest into ahead and rear halves relative to ego heading and report AP per half (A-mAP, R-mAP), because the road ahead matters more downstream. Motivation: MapTR output masked in front of or behind the ego and fed to HiVT (a predictor, not a planner). The method distils a teacher with future frames into a current-frame student.
- formulation: AP per half like global mAP; teacher with 5 future frames; multi-level BEV feature distillation with a GT-derived spatial mask; asymmetric query distillation (Hungarian matching of 100 student queries to 116 teacher queries, KL on matched logits).
- input: online map predictions (for the metric).
- output: A-mAP, R-mAP.
- training: teacher-student distillation (not needed for the metric).
- differentiable: n/a for the metric.
- sensors_data: nuScenes, Argoverse 2.
- results: forward mask 100 %: minADE +9.73 %, minFDE +13.95 %, MR +20.96 % (reference 0.3856 / 0.7919 / 0.0854); backward mask 100 %: +0.67 %, +0.66 %, -4.45 %. nuScenes (R50, no temporal): mAP 69.26, A-mAP 70.19, R-mAP 69.61. No planning experiment (only a qualitative figure).
- compute: not read.
- code: not found on the page.
- fit_to_us: reuse_idea - effort S - report graph metrics for the forward / route-relevant half.
- compat_needs: heading-aligned ROI split.
- compat_conflicts: the split is directional, not distance-weighted.
- tried_before: none.
- verified: checked

### CARD PL19 - Ego-status and open-loop critiques of nuScenes planning (BEV-Planner, AD-MLP) and the curb-collision metric
- papers: Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? (Li et al., CVPR 2024) https://arxiv.org/abs/2312.03031 ; Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes (Zhai et al., arXiv 2023) https://arxiv.org/abs/2305.10430
- depth: M
- stage: E
- idea: On nuScenes a model using only past ego motion and a command matches perception-based planners on L2; ego status in the planner dominates. New metric: curb collision rate (CCR) = share of trajectories that intersect annotated road boundaries rasterised at 0.1 m. Recommendations: stricter, more diverse metrics; report where ego status enters.
- formulation: AD-MLP input 21 values (4 past poses x, y, heading, velocity, acceleration, yaw rate) + one-hot command derived from whether the ego moves more than 2 m sideways in the next 3 s; Linear 21-512-512-18; 6 poses at 2 Hz; L1. BEV-Planner: R50, 704 x 256 images, BEV 128 x 128 (about 50 m), 4 past BEV frames concatenated without alignment, one learnable ego query, single L1 loss, no perception labels.
- input: ego history + command (AD-MLP); camera BEV (BEV-Planner).
- output: 3 s trajectory.
- training: L1 on trajectory points (AD-MLP down-weights samples whose predicted and GT points fall in the same 0.5 m occupancy cell).
- differentiable: yes.
- sensors_data: nuScenes.
- results: avg L2 m / collision % / CCR %: UniAD reproduced without ego status 1.03 / 0.77 / 1.93; official (ego status in BEV) 0.66 / 0.62 / 1.72; both 0.46 / 0.37 / 1.59; VAD-Base without 1.25 / 1.09 / 3.82, official 0.72 / 0.54 / 2.72, both 0.37 / 0.33 / 2.47; Ego-MLP 0.35 / 0.37 / 2.93; GoStraight 0.83 / 1.08 / 8.62; BEV-Planner 0.55 / 0.59 / 4.26, ++ 0.35 / 0.34 / 3.16; BEV-Planner + UniAD-style map task 0.96 / 0.89 / 2.60 (map helps turning, hurts straight driving: L2 and collision worse, CCR better). VAD with ego status: zeroing ego velocity gives L2 6.16 m; blank camera images leave planning almost unchanged (0.46 m / 0.54 %). AD-MLP: L2 0.29 m / collision 0.19 % vs UniAD 1.03 / 0.31 and VAD-Base 0.37 / 0.14 (quoted from the VAD table). 73.9 % straight driving in the main text, 87 % in Appendix E (not reconciled); about 70 % of headings within +-0.2 rad; the 0.5 m occupancy grid flags about 2 % of GT trajectories as colliding.
- compute: not stated.
- code: https://github.com/NVlabs/BEV-Planner (AD-MLP repository link not read)
- fit_to_us: reuse_idea - effort S - CCR-like boundary metric, ego-status ablation for E117, turning subset reported separately.
- compat_needs: GT road boundaries from the nuScenes map expansion.
- compat_conflicts: the command on nuScenes is derived from the future ego motion (leak); our route label (lane sequence driven in the next 6 s) is also future-derived.
- tried_before: none.
- verified: checked

### CARD PL20 - NAVSIM v1 (non-reactive simulation, PDM score)
- papers: NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking (Dauner et al., NeurIPS D&B 2024) https://arxiv.org/abs/2406.15349
- depth: M
- stage: E
- idea: 4 s planning from a recorded frame; the ego is rolled out by an LQR controller on a kinematic bicycle model at 10 Hz while other agents follow their log. PDMS = (NC x DAC) x (5 EP + 5 TTC + 2 C) / 12 with NC no at-fault collision (static-object contact 0.5), DAC drivable-area compliance (map-based), EP progress, TTC time-to-collision, C comfort.
- formulation: agent input = 8 cameras, merged lidar, ego status (pose, velocity, acceleration), one-hot driving command (left / straight / right, plus unknown); 2 s history at 2 Hz. The map is NOT an agent input (maps, tracks and occupancy are available for training targets only, not released for test frames; the scorer uses the map). Data OpenScene (a redistribution of nuPlan, 120 h at 2 Hz); navtrain about 103 k samples, navtest about 12 k; challenging-scene filter removes frames where constant velocity scores above 0.8 and frames where the human scores below 0.8 (constant velocity falls to 22 %, humans 95 %). Human-violation filter documented for v2.
- input: sensors + ego status + command.
- output: 4 s trajectory at 10 Hz (x, y, heading).
- training: n/a (benchmark).
- differentiable: n/a.
- sensors_data: cameras + lidar, nuPlan locations.
- results: navtest PDMS (NC / DAC / TTC / Comf / EP): constant velocity 20.6 (68.0 / 57.8 / 50.0 / 100 / 19.4); ego-status MLP 65.6 (93.0 / 77.3 / 83.6 / 100 / 62.8); TransFuser 84.0; LTF 83.8; UniAD 83.4; PARA-Drive 84.0; human 94.8. PDMS correlates better with the nuPlan closed-loop score than open-loop scores (qualitative on the page). Numbers differ across papers (Hydra-MDP lists TransFuser 78.0, PDM-Closed with GT perception 89.1).
- compute: cheap per scene (4 s rollout); no figures.
- code: https://github.com/autonomousvision/navsim
- fit_to_us: baseline - effort M - no radar, no nuScenes data, no map input for agents; DAC / DDC / LK are the map-sensitive terms and can be re-implemented.
- compat_needs: drivable-area polygons and lane geometry for scoring.
- compat_conflicts: nuPlan / OpenScene data only.
- tried_before: none.
- verified: checked

### CARD PL21 - NAVSIM v2 pseudo-simulation (EPDMS) and the map-sensitive sub-scores
- papers: Pseudo-Simulation for Autonomous Driving (Cao et al., CoRL 2025) https://arxiv.org/abs/2506.04218 ; NAVSIM docs (metrics) https://github.com/autonomousvision/navsim
- depth: M
- stage: E
- idea: Stage 1 rolls out the plan on a real frame; stage 2 scores the plan on pre-rendered synthetic observations (Gaussian-splat views) whose start points lie around the expert's endpoint (lateral steps of 0.5 m up to +-2.0 m, longitudinal steps of 5.0 m), weighted by a Gaussian kernel (sigma^2 = 0.1) of the distance to the planner's stage-1 endpoint; combined score = stage-1 score x weighted stage-2 average.
- formulation: EPDMS = (NC x DAC x DDC x TLC) x weighted average of TTC (5), EP (5), LK (2), HC (2), EC (2). DDC driving-direction compliance in {0, 1/2, 1}; LK lane keeping penalises staying far from the centre-line for long and is disabled at intersections; TLC traffic-light compliance; a penalty is ignored if the human commits the same violation.
- input: plan from any agent (4 s).
- output: EPDMS.
- training: n/a.
- differentiable: n/a.
- sensors_data: camera + lidar (nuPlan / OpenScene).
- results: correlation with nuPlan closed loop r = 0.89 (R^2 = 0.8, 83 planners; the text says 40 + 46 planners, the caption 37 + 46); single-stage open loop r = 0.83; ADE vs nuPlan open-loop score almost uncorrelated; navhard 450 stage-1 and 5,462 stage-2 observations. navhard EPDMS (snapshot 03/2026): PDM-Closed (privileged) 56.6; DrivoR 54.5; SimScale 53.2; GuideFlow 51.5; ZTRS 48.1; RAP 39.6; LTF 25.1; ego MLP 14.1; constant velocity 11.4.
- compute: about 6 x less than closed-loop simulation (README claim).
- code: https://github.com/autonomousvision/navsim
- fit_to_us: reuse_idea - effort M - DDC / LK / DAC as graph-sensitive sub-scores, and the idea of scoring a plan from perturbed start states.
- compat_needs: lane direction, centre-line distance, drivable area.
- compat_conflicts: nuPlan data and a rendering pipeline.
- tried_before: none.
- verified: checked

### CARD PL22 - nuPlan benchmark and its evaluation protocols
- papers: nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles (Caesar et al., arXiv 2106.11810) https://arxiv.org/abs/2106.11810
- depth: A
- stage: E, D
- idea: About 1,500 h of human driving in four cities, a lightweight closed-loop simulator (reactive and non-reactive agents) and planning-specific metrics. Route = roadblock-id sequence and the map = lane graph (both seen in planner code, PL03-PL06). Protocol splits used by the papers above: Val14 (1,118 scenarios), Test14-random, Test14-hard; scores OLS, CLS-NR, CLS-R.
- formulation: not read (abstract page only).
- input: not read.
- output: not read.
- training: not read.
- differentiable: n/a.
- sensors_data: nuPlan (Boston, Pittsburgh, Las Vegas, Singapore).
- results: see PL02-PL07.
- compute: not read.
- code: devkit https://github.com/motional/nuplan-devkit (named in the READMEs of PL03-PL06)
- fit_to_us: not_suitable - effort L - nuPlan logs and maps are not our nuScenes frames; its metric families are the template.
- compat_needs: nuPlan maps and logs.
- compat_conflicts: dataset.
- tried_before: none.
- verified: partly

### CARD PL23 - PKL: planner-centric metric for perception
- papers: Learning to Evaluate Perception Models Using Planner-Centric Metrics (Philion et al., CVPR 2020) https://arxiv.org/abs/2004.08745
- depth: A
- stage: E
- idea: Judge detections by how much they change a planner's output (a KL divergence between planner distributions on predicted vs GT inputs; the name and formula are from memory); abstract: downweights distant and slow objects; humans sided with it in 79 % of cases where it disagreed with standard metrics. Planner and dataset (nuScenes) not on the page.
- formulation: not read.
- input: not read.
- output: not read.
- training: not read.
- differentiable: not read.
- sensors_data: not read on the page.
- results: human study 79 % (abstract).
- compute: none seen.
- code: not checked.
- fit_to_us: reuse_idea - effort S - compare the consumer's output on predicted vs GT graph instead of comparing geometry.
- compat_needs: a deterministic or probabilistic consumer.
- compat_conflicts: none.
- tried_before: none (the oracle-vs-predicted swap of E119 / E117 is the same principle).
- verified: partly

### CARD PL24 - Input-swap evaluation protocols (GT vs predicted vs masked inputs into one fixed consumer)
- papers: composed from PL14 (Plug-and-Play TE), PL15 (Gu CVPR 2024), PL18 (AMap), PL19 (BEV-Planner), PL28 (GoalFlow oracle goal), PL23 (PKL)
- depth: M (composed from the cards)
- stage: E
- idea: Evaluate a map or graph by swapping the input of one fixed consumer and comparing outputs or metrics. Variants seen: GT vs predicted topology into VAD; GT HD map vs online centre-line map into DenseTNT; forward / backward masks of the predicted map into HiVT; oracle vs predicted goal in GoalFlow; zeroed ego velocity or blank images in VAD.
- formulation: (my synthesis) fixed consumer; input families {GT, predicted, masked by region, perturbed (shift, jitter, element drop), shuffled}; metrics sensitive to the swapped factor (boundary violation, direction, miss rate) and subsets (turning, forward half); report the swap gap; add an ego-status-removed variant so the consumer cannot ignore the graph.
- input: consumer + input variants.
- output: swap-gap metrics.
- training: none (frozen consumer) or staged.
- differentiable: n/a.
- sensors_data: nuScenes (PL14, PL15, PL18, PL19), NAVSIM (GoalFlow).
- results: VAD with GT vs TopoMLP topology: L2 0.62 / 0.62 m, collision 0.16 / 0.16 % (no gap); DenseTNT: online MapTRv2-centre-line 0.8466 / 1.345 / 0.1520 vs GT HD map 0.8809 / 1.489 / 0.1903 (online better); HiVT forward mask 100 %: minADE +9.73 %, backward +0.67 %; GoalFlow oracle goal 92.1 vs predicted 90.3 PDMS; VAD with zeroed ego velocity L2 6.16 m, blank images 0.46 m.
- compute: n/a.
- code: https://github.com/alfredgu001324/MapUncertaintyPrediction (prediction consumers)
- fit_to_us: reuse_idea - effort S - the project already did a partial swap (oracle route 0.316 m vs 0.34 m in E119; graph A 0.36 m vs B 0.34 m in E117), small gaps like the literature.
- compat_needs: a consumer that accepts GT and predicted graphs in one format.
- compat_conflicts: consumers that exploit ego status hide the swap gap.
- tried_before: partly - oracle-route and A-vs-B swaps with E117 (E119).
- verified: partly (synthesis)

### CARD PL25 - DiffusionDrive (truncated diffusion from anchors; map only as auxiliary segmentation on NAVSIM)
- papers: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025) https://arxiv.org/abs/2411.15139
- depth: M
- stage: S10, S4
- idea: Noise is added to K-means trajectory anchors (20 on NAVSIM, 18 on nuScenes), truncated at 50 of 1000 steps; 2 DDIM steps at inference; a 2-layer cascade decoder (shared across steps) with deformable BEV / PV cross-attention, agent cross-attention, timestep modulation; outputs confidence + offset. NAVSIM has no map query (agent cross-attention only; BEV semantic segmentation only as an auxiliary task); on nuScenes (built on SparseDrive) object and map queries are used.
- formulation: tau_k^i = sqrt(abar_i) a_k + sqrt(1 - abar_i) eps; loss = sum_k y_k L1(tau_hat_k, tau_gt) + lambda BCE(s_k, y_k), positive = anchor closest to the GT (winner-take-all-like nearest-anchor assignment); detection + BEV segmentation auxiliary losses.
- input: BEV / PV features, agent (and map) queries.
- output: top-1 scored trajectory.
- training: loss above; ImageNet-initialised ResNet-34 on NAVSIM; nuScenes planner trained 10 epochs from stage-1 perception weights; gradient to perception not stated.
- differentiable: partial - nearest-anchor assignment and top-1 selection are hard.
- sensors_data: NAVSIM camera + lidar; nuScenes cameras.
- results: NAVSIM navtest PDMS 88.1 (NC 98.2, DAC 96.2, TTC 94.7, Comf 100, EP 82.2; TransFuser 84.0); nuScenes avg L2 0.57 m (1 s 0.27, 2 s 0.54, 3 s 0.90), collision 0.08 %.
- compute: 45 FPS on NAVSIM, 8.2 FPS on nuScenes (RTX 4090); about 60 M parameters (NAVSIM model).
- code: https://github.com/hustvl/DiffusionDrive
- fit_to_us: reuse_idea - effort M - anchor-truncated denoising could refine E117's path, with anchors taken from route paths.
- compat_needs: BEV features, agent queries.
- compat_conflicts: nearest-anchor assignment is an argmin.
- tried_before: none.
- verified: checked

### CARD PL26 - Hydra-MDP and Hydra-MDP++ (map enters only through distilled rule-based sub-scores)
- papers: Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation (Li et al., 2024) https://arxiv.org/abs/2406.06978 ; Hydra-MDP++ https://arxiv.org/abs/2503.12820
- depth: M
- stage: S10, L
- idea: A planning vocabulary of 4096 / 8192 K-means trajectories (700 K nuPlan samples, 40 waypoints, 4 s) is scored by heads that imitate the sub-scores of the rule-based PDM simulation (NC, DAC, TTC, C, EP; ++ adds TLC, DDC, LK, EC) plus an imitation head. The map is used offline to compute the teacher scores, not as a network input; BEV segmentation / lane heads are not used by the student (++: optional lane and traffic-light supervision hurts planning).
- formulation: imitation = softmax CE against the distance to the log-replay; distillation = per-metric BCE against simulated sub-scores; inference cost = -(w1 log S_im + w2 log S_NC + w3 log S_DAC + w4 log(5 S_TTC + 2 S_C + 5 S_EP)) with grid-searched weights; ensembling of encoders and of sub-scores.
- input: image / lidar tokens + ego status (++: command as well).
- output: scored vocabulary entries; the lowest-cost trajectory.
- training: CE + BCE; vocabulary simulated offline.
- differentiable: yes for the heads; the teachers are rules.
- sensors_data: NAVSIM cameras + lidar.
- results: navtest PDMS: Hydra-MDP-V8192 83.0; weighted EP 86.5; scaled (V2-99 / ViT-L) 91.0; distilling only the overall PDM score gives 80.2; PDM-Closed with GT perception 89.1 (EPDMS 82.8 in ++). ++: ResNet-34 PDMS 86.6 / EPDMS 80.6, V2-99 91.0 / 84.1. The paper attributes the low DAC of neural planners to not knowing the drivable-area extent.
- compute: ~206 ms (ResNet-34) and ~271 ms (V2-99) latency (++).
- code: https://github.com/NVlabs/Hydra-MDP
- fit_to_us: reuse_idea - effort M - score candidate paths by graph-derived compliance sub-scores, to distil or to use directly.
- compat_needs: a simulator-style scorer from the lane graph and drivable area.
- compat_conflicts: needs a rule simulation per vocabulary entry.
- tried_before: none.
- verified: checked

### CARD PL27 - VADv2 (probabilistic planning over a vocabulary with map tokens)
- papers: VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning (Chen et al., 2024) https://arxiv.org/abs/2402.13243
- depth: M
- stage: S10, L
- idea: Planning vocabulary of N = 4096 trajectories (furthest-trajectory sampling over demonstrations; 6 waypoints / 3 s); planning tokens (embeddings of candidates) are queries in a cascaded transformer over scene tokens (vectorised map tokens predicting centre-lines, dividers, boundaries, crossings, supervised as in MapTRv2; agent tokens; traffic-element tokens; front-view image tokens); a navigation embedding (MLP, form not stated) and ego state are added; output = probability per action.
- formulation: loss = distribution loss (KL to the data distribution = CE to the nearest vocabulary entry) + conflict loss (vocabulary entries that conflict with other agents' GT futures or road boundaries are pushed down; map and agents enter only through these labels) + token losses.
- input: scene tokens, navigation embedding, ego state.
- output: probability over the vocabulary.
- training: loss above; regime: imitation with conflict labels.
- differentiable: planning gradient to the map head not stated.
- sensors_data: CARLA cameras; vocabulary statistics from nuScenes / NAVSIM.
- results: CARLA Town05 Long closed loop DS 85.1, RC 98.4, IS 0.87; NAVSIM navtest PDMS 89.3 and Bench2Drive DS 76.15 quoted for reference; no nuScenes planning numbers.
- compute: not read.
- code: https://github.com/hustvl/VAD (link given in the VADv2 abstract)
- fit_to_us: reuse_idea - effort S - conflict labels from the map: vocabulary entries leaving the drivable area get a negative label.
- compat_needs: boundary / drivable-area test per vocabulary entry.
- compat_conflicts: needs a vocabulary of trajectories.
- tried_before: none.
- verified: checked

### CARD PL28 - GoalFlow (goal point from a vocabulary, drivable-area-scored)
- papers: GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving (Xing et al., CVPR 2025; first author from memory) https://arxiv.org/abs/2503.05689
- depth: M
- stage: S10, S1
- idea: Goal points (x, y, theta) from a vocabulary of 4096 / 8192 trajectory endpoints are scored by a distance score (softmax over negative L2 to the GT endpoint) and a binary drivable-area score (shadow-vehicle box inside the drivable-area polygon); the best goal conditions a rectified-flow trajectory generator; a shadow trajectory generated without the goal detects unreliable goals.
- formulation: scorer = transformer decoder (query F_v + F_ego, keys F_bev) + two MLPs; final score w1 log d_dis + w2 log d_dac; rectified flow x_t = (1 - t) x0 + t tau, target v = tau - x0; goal encoded sinusoidally and concatenated; random masking of conditions; 128-256 samples; selector score = -lambda1 norm(f_dis) + lambda2 norm(f_pg).
- input: BEV features, ego features, goal.
- output: trajectory (and goal).
- training: flow matching + score losses; TransFuser-style scene encoder with HD-map cross-entropy, 3-D box classification and L1 as auxiliary losses (weights 10, 1, 10).
- differentiable: scoring / selection are hard.
- sensors_data: NAVSIM cameras + lidar.
- results: NAVSIM navtest PDMS 90.3 (NC 98.4, DAC 98.3, TTC 94.6, EP 85.0); with the GT endpoint as goal 92.1 (oracle; +1.8, mostly NC +1.4 and TTC +4.0, DAC -0.4); human 94.8; PARA-Drive 84.0 in the same table.
- compute: not read.
- code: https://github.com/YvanYin/GoalFlow
- fit_to_us: reuse_idea - effort M - a goal on the route graph: an endpoint vertex scored by distance and drivable-area compliance.
- compat_needs: drivable-area polygon.
- compat_conflicts: hard argmax of the goal.
- tried_before: none.
- verified: checked

### CARD PL29 - Perception in Plan / VeteranAD (anchor-guided perception, dense BEV map)
- papers: Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving (arXiv 2508.11488, AAAI 2026; authors not read) https://arxiv.org/abs/2508.11488
- depth: M
- stage: S10, S2
- idea: 20 K-means planning anchors initialise trajectory queries; per future step the guiding points of the anchors are projected into image and BEV coordinates and used as reference points for position-guided cross-attention to image / BEV features (planning-oriented sampling of perception features); autoregressive near-to-far refinement predicts offsets; classification score only at the last step. Map = dense BEV segmentation (cross-entropy), no vector or lane queries.
- formulation: P_ft = Delta P_ft + P_t per step; motion-aware layer norm moves queries across steps; agent cross-attention weighted by distance to the guiding points.
- input: three front images, LSS BEV.
- output: scored trajectories, BEV segmentation, boxes.
- training: one total loss over BEV, agent, regression and classification terms; AdamW lr 2e-4, batch 32, 16 epochs, 8 x RTX 3090; gradients appear to reach perception (inferred, not stated).
- differentiable: yes.
- sensors_data: NAVSIM, Bench2Drive; ResNet-34.
- results: NAVSIM navtest PDMS 90.2 (NC 99.1, DAC 98.3, TTC 96.1, EP 83.1); Bench2Drive open-loop L2 0.60, driving score 64.22, success 33.85 % (DriveTransformer has higher success 35.01).
- compute: see training.
- code: https://github.com/fudan-zvg/VeteranAD
- fit_to_us: reuse_idea - effort M - sampling BEV features along the planned path is the same principle as our rays / fans, applied to planning.
- compat_needs: BEV feature pyramid (we have it).
- compat_conflicts: the image branch needs image-space projection.
- tried_before: none.
- verified: checked

### CARD PL30 - DriveTransformer (task-parallel queries including map queries)
- papers: DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving (ICLR 2025; authors not read) https://arxiv.org/abs/2503.07656
- depth: M
- stage: S10
- idea: Agent queries (900), map queries (100) and one ego query interact in every block (sensor cross-attention, task self-attention, temporal cross-attention to a FIFO history of ego queries). Each map query is replicated per point with a position encoding in the sensor cross-attention and merged back by a PointNet (points per polyline and classes not given). Six planning mode embeddings; winner-take-all training.
- formulation: ego query from an MLP over CAN-bus info; ego position encoding updated after each block from the predicted trajectory; losses on task heads at all blocks; no detach described.
- input: sensor tokens.
- output: ego plan (6 modes), agent and map predictions.
- training: end-to-end; planning gradients appear to reach map queries (inferred, not confirmed).
- differentiable: yes; matching discrete.
- sensors_data: nuScenes, Bench2Drive.
- results: nuScenes open loop (Large, VAD metric) without ego status L2 0.40 m avg, collision 0.11 %; with ego status 0.33 / 0.07; Bench2Drive driving score 63.46, success 35.01 %.
- compute: latency 211.7 ms (Bench2Drive table).
- code: https://github.com/Thinklab-SJTU/DriveTransformer/
- fit_to_us: reuse_idea - effort M - map-instance queries as tokens with per-point position encoding and a PointNet merge.
- compat_needs: instance queries.
- compat_conflicts: Hungarian matching.
- tried_before: none.
- verified: checked

### CARD PL31 - FUMP (planning as motion prediction with map nodes)
- papers: Fully Unified Motion Planning for End-to-End Autonomous Driving (arXiv 2504.12667; authors not read) https://arxiv.org/abs/2504.12667
- depth: M (tail cut)
- stage: S10
- idea: Ego and other vehicles share a two-stage decoder (state-agnostic multimodal generation, then ego-state-aware refinement). Agent and map features from the SparseDrive perception become graph nodes, grouped into four zone subgraphs plus a global graph (PointNet, set attention); motion and plan queries cross-attend to these nodes.
- formulation: not read in detail.
- input: SparseDrive instance features.
- output: ego plan and agent motion.
- training: not read.
- differentiable: yes (end-to-end), details not read.
- sensors_data: nuScenes, NAVSIM, Bench2Drive.
- results: nuScenes open loop avg L2 0.39 m vs SparseDrive 0.61, collision 0.06 vs 0.08 %; NAVSIM PDMS 87.8 vs TransFuser 84.0 (85.1 reproduced; the abstract's +3.7 % does not match the table); Bench2Drive DS 45.67 vs VAD 42.35.
- compute: not read.
- code: not found on the page.
- fit_to_us: reuse_idea - effort M - map nodes in zone subgraphs.
- compat_needs: instance nodes.
- compat_conflicts: needs agents.
- tried_before: none.
- verified: partly

### CARD PL32 - Map-free leaders on NAVSIM 2025-26 (RAP, DriveZero) and the privileged-teacher pattern
- papers: RAP: 3D Rasterization Augmented End-to-End Planning (arXiv 2510.04333; authors not read) https://arxiv.org/abs/2510.04333 ; DriveZero: End-to-End Driving Beyond Human Demonstrations (arXiv 2609.06055, Xiaomi) https://arxiv.org/abs/2609.06055
- depth: M
- stage: S10, T
- idea: RAP renders annotated lane polylines and cuboids as lightweight 3-D rasters to create recovery-oriented training views and aligns raster and real features (no map input to the planner; maps only generate training data). DriveZero trains a privileged teacher (5.7 M parameters) on a local vector map of up to 256 tokens (traffic-light states attached to the governed lane elements), up to 96 agents and two route-based goal points (near and far anchor along the route, look-ahead scaling with speed), then distils into a camera-only student whose only route signal is a command embedded in one ego token; the student has no map head.
- formulation: student loss = winner-takes-all L1 over 64 proposals against the teacher rollout + BCE on six PDM-style sub-scores (weights not given); teacher: ego-to-agent cross-attention (2 layers), ego-to-map cross-attention (1 layer), 256-d tokens, Beta distributions over jerk and steering rate.
- input: RAP cameras + ego history (route for WOD-E2E); DriveZero student 4 cameras, ego kinematics, command.
- output: scored trajectories.
- training: RAP synthetic-to-real alignment; DriveZero teacher by reinforcement learning (96 GPUs, about 21 h) then distillation; regime: rl + distillation.
- differentiable: n/a (distillation).
- sensors_data: NAVSIM, Waymo, Bench2Drive (RAP); NAVSIM, HUGSIM (DriveZero).
- results: NAVSIM v1 navtest PDMS: RAP-DINO 93.8 (NC 99.1, DAC 98.9, TTC 96.7, EP 90.3), DriveZero 94.8 (95.3 with extra simulated data), teacher with GT inputs 95.8, human 94.8; navhard EPDMS RAP-DINO 36.9 (the pseudo-simulation paper lists 39.6), DriveZero 51.5 / 57.1, PDM-Closed 56.6.
- compute: RAP-DINO about 888 M parameters; DriveVFM pre-training 600 K + 200 K iterations at global batch 2,048.
- code: none (project pages https://alan-lanfeng.github.io/RAP/ , https://xiaomiautol3.github.io/DriveZero)
- fit_to_us: reuse_idea - effort M - a consumer trained on GT graph + route as teacher, usable to distil or to score.
- compat_needs: GT graph tokens for a teacher.
- compat_conflicts: a student without a map head makes graph quality invisible.
- tried_before: none.
- verified: checked

### CARD PL33 - SSR (16 navigation-guided scene tokens, no map supervision)
- papers: Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving (Li, Cui; ICLR 2025) https://arxiv.org/abs/2409.18341
- depth: M
- stage: S10, S9
- idea: Dense BEV (100 x 100) is squeezed into 16 tokens (dim 256) by a TokenLearner whose BEV features are modulated by the command through a squeeze-and-excitation layer; future-BEV prediction as self-supervision; waypoint queries attend the tokens; the output set is chosen by the command. No detection or map supervision in the main model.
- formulation: B^navi = SE(B, cmd); token = global-average-pool of B^navi times a learned spatial attention map; loss = L1 imitation + L2 future-BEV reconstruction (both weight 1.0).
- input: BEV features + command.
- output: trajectory.
- training: imitation + self-supervised BEV prediction; 11 h on 8 x RTX 3090.
- differentiable: yes.
- sensors_data: nuScenes, CARLA.
- results: nuScenes (MAX metric) avg L2 0.75 m / collision 0.15 % vs UniAD 1.03 / 0.31 and VAD-Base 1.22 / 0.53; AVG metric 0.39 / 0.06 %; tokens 8 / 16 / 32 / 64: L2 0.69 / 0.75 / 0.77 / 0.84, collision 0.20 / 0.15 / 0.16 / 0.41; removing navigation guidance: straight 0.72 / 0.29 % -> 0.71 / 0.10 %, turns 1.09 / 0.78 -> 1.04 / 0.66; CARLA Town05 Long DS 78.9.
- compute: 19.6 FPS (RTX 3090); parameters not reported.
- code: https://github.com/PeidongLi/SSR
- fit_to_us: reuse_idea - effort S - the command enters as a multiplicative channel gate on features; and a map-free planner reaching state-of-the-art nuScenes numbers shows the metric is no graph-quality test.
- compat_needs: BEV features.
- compat_conflicts: none.
- tried_before: none.
- verified: checked

### CARD PL34 - ST-P3 (dense BEV cost-volume planner on a 200 x 200, 0.5 m grid)
- papers: ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning (Hu et al., ECCV 2022) https://arxiv.org/abs/2207.07601
- depth: M
- stage: S10
- idea: Candidate trajectories sampled with a kinematic bicycle model are gated by the command and scored by f = f_o (hand-designed safety: overlap with occupied cells and road elements, lane-line distance) + f_v (learned cost volume from the prediction decoder, clipped) + f_r (comfort and progress), weights learned; a GRU refines the best candidate starting from the target point.
- formulation: planning loss = max-margin over samples + L1 to the expert; BEV 200 x 200 cells at 0.5 m (100 m x 100 m) on nuScenes, identical to our grid; heads include drivable area and lanes.
- input: BEV features, command, target point.
- output: refined trajectory (3 s).
- training: end-to-end; gradients appear to reach the shared network through the cost-volume head (inferred).
- differentiable: partial - candidate sampling and selection.
- sensors_data: nuScenes cameras; CARLA.
- results: nuScenes L2 1.33 / 2.11 / 2.90 m at 1 / 2 / 3 s (the table label "mm" is a unit error), collision 0.23 / 0.62 / 1.27 %; CARLA Town05 Short DS 55.14, Long 11.45.
- compute: not read.
- code: https://github.com/OpenPerceptionX/ST-P3
- fit_to_us: baseline - effort M - a dense-raster planner that can read the cached 39-channel features directly; a raster control for the graph consumer.
- compat_needs: BEV channels for drivable area and lane lines.
- compat_conflicts: sampling and argmin selection.
- tried_before: none (E117 is graph-based).
- verified: checked

### CARD PL35 - LaneGCN (lane-graph convolution with four edge types and dilations)
- papers: Learning Lane Graph Representations for Motion Forecasting (Liang et al., ECCV 2020) https://arxiv.org/abs/2007.13732
- depth: M+C (README)
- stage: S6, S10
- idea: Nodes are the straight segments between consecutive centre-line points (location = midpoint); node feature x_i = MLP_shape(end - start) + MLP_loc(midpoint). Four adjacency matrices: predecessor, successor (shifted identities), left and right neighbour (closest node on the adjacent lane by l2 distance). LaneConv: Y = X W0 + sum over {pre, suc, left, right} A_i X W_i, with dilated pre / suc terms A^k for k in {1, 2, 4, 8, 16, 32} (left / right undilated).
- formulation: MapNet = 4 LaneConv residual blocks, 128 channels; ActorNet 1-D CNN; FusionNet 4 modules A2L, L2L (another LaneGCN), L2A, A2A with distance thresholds 7 m, 6 m, 100 m; loss L_cls (max-margin to the best mode) + L_reg (smooth l1), alpha 1.0; actors and lanes within 100 m, frame centred on the agent with x along t-1 -> t0.
- input: lane nodes + 4 adjacencies, actor histories.
- output: K trajectories with confidences.
- training: 36 epochs, batch 128, Adam lr 1e-3 -> 1e-4; regime: winner-takes-all style max-margin.
- differentiable: yes.
- sensors_data: Argoverse (GT map); 100 m radius.
- results: Argoverse val (K = 6 minADE / minFDE): ActorNet only 0.91 / 1.66; + MapNet 0.79 / 1.29; full 0.71 / 1.08; K = 1 minFDE 2.97 (4.38 without map). LaneConv vs vanilla graph conv minFDE 1.10 vs 1.41.
- compute: about 11.5 h on 4 TITAN-X (paper), about 8 h on 4 RTX 5000 (README).
- code: https://github.com/uber-research/LaneGCN (archived January 2024)
- fit_to_us: reimplement - effort S - our edges map directly to pre / suc adjacency (edge nodes = LaneGCN nodes, the line graph of the vertex graph); left / right must be derived geometrically.
- compat_needs: segment nodes, a successor graph without gaps (dilation needs long chains), left / right neighbours.
- compat_conflicts: false forks and missing edges change the A^k powers; 6 m edges are coarser than typical LaneGCN segments (segment length not stated).
- tried_before: none (E42 / E45 / E46 learned link heads concern perception).
- verified: checked

### CARD PL36 - VectorNet (vectors, polyline subgraphs, global attention without lane topology)
- papers: VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation (Gao et al., CVPR 2020) https://arxiv.org/abs/2005.04259
- depth: M
- stage: S10
- idea: Every map element and trajectory is a polyline of vectors v_i = [start, end, attributes (type, speed limit, traffic-light state, timestamp), polyline id]; centre-line key points are sampled at equal spatial distance (spacing not stated). Polyline subgraph: 3 layers of shared FC (64 hidden) + layer norm + ReLU with concatenation of the max-pooled neighbourhood, final max-pool. Global graph: one self-attention layer over all polyline nodes with full connectivity (lane topology is not encoded).
- formulation: node-completion auxiliary loss (masked nodes reconstructed, Huber, alpha 1.0, features l2-normalised); coordinates centred on the target agent's last position; outputs per-step offsets rotated to the target heading; trajectory loss negative Gaussian log-likelihood.
- input: all polylines in the scene.
- output: future trajectory of the target agent.
- training: supervised; auxiliary masked-node reconstruction (self_supervised).
- differentiable: yes.
- sensors_data: Argoverse (GT map) and in-house data.
- results: Argoverse DE@3s: history only 5.24, map only 3.94, map + agents 3.84, + node completion 3.67; ConvNet (ResNet-18, 400 x 400) 4.49; test K = 1 DE@3s 4.01 vs 4.17 for the challenge winner.
- compute: 0.041 GFLOPs and 72 K parameters per target vs 10.56 GFLOPs and 246 K for the best ConvNet.
- code: no official release mentioned on the page.
- fit_to_us: reuse_idea - effort S - the simplest polyline encoder, structurally robust to missing edges because it uses none.
- compat_needs: polylines + attributes.
- compat_conflicts: no topology information.
- tried_before: none.
- verified: checked

### CARD PL37 - QCNet / QCNeXt (query-centric scene encoding with typed lane edges plus radius edges)
- papers: Query-Centric Trajectory Prediction (Zhou et al., CVPR 2023; paper PDF not readable here) ; QCNeXt: A Next-Generation Framework for Joint Multi-Agent Trajectory Prediction https://arxiv.org/abs/2306.10508 ; code https://github.com/ZikangZhou/QCNet
- depth: M (QCNeXt report) + C (dataset and map encoder)
- stage: S6, S10
- idea: Each scene element lives in its own local space-time frame; attention keys / values are concatenated with a relative pose embedding (distance, angle between the query orientation and the relative vector, relative orientation), so nothing depends on a global frame. Map: one polygon per lane segment (crosswalks two polygons); points on left boundary, right boundary and centre-line (raw map points, last point dropped; crosswalk centre-line resampled about every 2 m); polygon attributes position, orientation, type (4), is_intersection (3); point attributes orientation, magnitude, type (17), side (3).
- formulation: polygon-to-polygon edges = radius graph (max 300 neighbours, type 0) merged (max) with map-graph edges of types predecessor, successor, left, right (embedding of 5 types); point-to-polygon edges by ownership; layers: point-to-polygon then polygon-to-polygon attention; missing neighbour ids are skipped. Decoder: two stages (anchor-free recurrent proposals, 3 recurrent steps of 2 s, then anchor-based refinement with detached proposals); mixture of Laplace with scene-level winner-take-all; D = 128, 8 heads, 2 blocks (QCNeXt); AdamW 50 epochs, batch 32, lr 5e-4.
- input: polygons, points, typed edges, agent histories (50 steps).
- output: K joint futures (60 steps) with scores.
- training: mixture-of-Laplace NLL with winner-take-all; regime: wta.
- differentiable: yes (the recurrent stage-1 proposals are detached before refinement).
- sensors_data: Argoverse 2 GT map.
- results: Argoverse 2 validation (QCNet) minFDE6 1.25, minADE6 0.72, MR6 0.16, brier-minFDE6 1.87 (README); QCNeXt multi-agent avgMinFDE6 1.13 without ensemble, 1.02 with.
- compute: training needs about 160 GB GPU memory in total (README: 8 x RTX 3090 example); parameters not stated.
- code: https://github.com/ZikangZhou/QCNet
- fit_to_us: reuse_idea - effort M - relative geometry instead of a global frame, and radius edges next to topology edges so that a missing successor edge is not fatal; the av2 preprocessing is heavy.
- compat_needs: polygons with ordered points, typed edges.
- compat_conflicts: av2-style dataset class, PyG.
- tried_before: none.
- verified: checked

### CARD PL38 - MTR (768 polylines x 20 points, local k-NN attention, intention queries)
- papers: Motion Transformer with Global Intention Localization and Local Movement Refinement (Shi et al., NeurIPS 2022) https://arxiv.org/abs/2209.13508
- depth: M
- stage: S10
- idea: Map = up to 768 nearest polylines with up to 20 points (about 10 m in WOMD); per point position, direction and type; agent-centric coordinates; PointNet-like encoder (5-layer MLP width 64, max-pool, projection to 256). 6-layer transformer encoder with attention restricted to the 16 nearest polylines (4 to 64 tested). Decoder with 64 static intention queries (k-means of GT endpoints) and dynamic searching queries refined over 6 layers, cross-attending a dynamic collection of 128 polylines near the current predicted trajectory; NMS (2.5 m) picks 6 of 64.
- formulation: Gaussian regression per decoder layer (NLL on the hard-assigned positive component + CE on its probability) + auxiliary dense future prediction (L1); equal weights.
- input: polyline set + agent histories.
- output: 6 trajectories (GMM) per target.
- training: AdamW lr 1e-4, batch 80, 30 epochs, 8 GPUs, no augmentation; regime: wta (hard intention assignment).
- differentiable: yes except NMS.
- sensors_data: Waymo (GT map).
- results: WOMD test marginal mAP 0.4129, minADE 0.6050, minFDE 1.2207, MR 0.1351; joint mAP 0.2037.
- compute: parameter count not reported.
- code: https://github.com/sshaoshuai/MTR
- fit_to_us: reuse_idea - effort M - the dynamic map collection along the current predicted path is "sample map features around the plan"; k-NN attention.
- compat_needs: fixed-size polyline set with positions.
- compat_conflicts: no topology; NMS.
- tried_before: none.
- verified: checked

### CARD PL39 - PGP (policy over lane-graph edges, traversals aggregated by attention; nuScenes map expansion)
- papers: Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals (Deo, Wolff, Beijbom; CoRL 2021) https://arxiv.org/abs/2106.15004 ; code (re-implementation) https://github.com/nachiket92/PGP
- depth: M+C (README)
- stage: S6, S9, S10
- idea: Lane graph from the nuScenes map: centre-lines inside [-50, 50] m laterally and [-20, 80] m longitudinally around the target, split into snippets of at most 20 m discretised at 1 m; node features per pose (x, y, yaw, 2-bit stop-line / crosswalk flag); edges: successor (incl. branches and merges) and proximal (neighbouring lanes within distance and yaw thresholds; values not given). A policy header scores every outgoing edge (MLP 2 x 32, softmax over outgoing edges plus an end state); traversals are sampled as roll-outs (200 trajectories clustered to K = 10).
- formulation: GAT / GCN layers (dim 32), agent-node attention within a distance threshold; node embeddings along each path aggregated by 32-head attention into a 128-d context; 5-d latent z; decoder gives 24 values (x, y over 6 s at 2 Hz).
- input: lane graph with successor + proximal edges, agent histories.
- output: K = 10 trajectories.
- training: L = L_BC (NLL of edges visited by the GT future; a node counts as visited if its yaw matches) + L_reg (winner-take-all); 100 epochs with GT traversals, then 100 epochs with policy-sampled paths.
- differentiable: partial - roll-out sampling of traversals.
- sensors_data: nuScenes map expansion v1.3 (the same GT as our labels).
- results: nuScenes minADE5 1.30, minADE10 1.00, miss rate (5, 2 m) 0.61, off-road rate 0.03.
- compute: not stated.
- code: https://github.com/nachiket92/PGP
- fit_to_us: baseline - effort M - the edge policy is a route probability on edges conditioned on the agent's motion; our route head could fill that slot and supply the traversal.
- compat_needs: successor + proximal edges.
- compat_conflicts: sampling of paths; false forks.
- tried_before: E102-E104 (route attribution variants) concern the perception side.
- verified: checked

### CARD PL40 - HiVT (lane segments as independent vectors; common consumer of online maps on nuScenes)
- papers: Hierarchical Vector Transformer for Multi-Agent Motion Prediction (Zhou et al., CVPR 2022; paper not read) ; code https://github.com/ZikangZhou/HiVT
- depth: C (dataset code + README); paper not read
- stage: S10
- idea: In the dataset code each lane centre-line becomes N - 1 segments (start point and difference vector, no resampling); lane attributes (is_intersection, turn direction, traffic control) repeat for all segments of the lane; every lane segment connects to every agent whose position at the last observed step lies within local_radius = 50; one scene-level rotation (AV position and heading at the last step); no lane-to-lane edges, no successor information.
- formulation: not read beyond the dataset code.
- input: lane segment vectors + attributes, agent histories.
- output: multimodal agent trajectories.
- training: not read.
- differentiable: yes.
- sensors_data: Argoverse 1.1 GT map; Gu et al. (PL15) run it on nuScenes with online maps as a point set.
- results: Argoverse 1.1 val minADE / minFDE / MR: HiVT-64 0.69 / 1.03 / 0.10, HiVT-128 0.66 / 0.97 / 0.09 (README).
- compute: README: 35-40 min per epoch on an RTX 2080 Ti; preprocessing about 3.5 h; parameters not stated.
- code: https://github.com/ZikangZhou/HiVT
- fit_to_us: baseline - effort M - external consumer through the PL15 pipeline; PyTorch 1.8 / PyG 1.7 stack.
- compat_needs: segments with attributes.
- compat_conflicts: old framework pins.
- tried_before: none.
- verified: partly

### CARD PL41 - SMART (scalable map prior from SD + satellite maps, pre-trained on HD maps)
- papers: SMART: Advancing Scalable Map Priors for Driving Topology Reasoning (authors not read; ICRA 2025 per the page) https://arxiv.org/abs/2502.04329 ; project page https://jay-ye.github.io/smart
- depth: M
- stage: S9, B
- idea: A map-prior encoder is pre-trained on geo-referenced HD maps (Argoverse 2: 200 k scenes and 4.4 M HD maps in total; main model 200 k scenes / 810 k HD maps) from SD maps (OSM polylines) and satellite tiles only, then frozen and plugged into online topology models. No navigation route is used (the page says a route could be used to pre-compute priors but this is not evaluated).
- formulation: SD polylines resampled to 11 points, padded, sinusoidal embeddings, linear + 6-layer transformer with validity mask; satellite: Mapbox zoom 20 (about 0.11 m/px) resized to 500 x 250, ResNet-50 multi-scale features; prior = BEV grid 200 x 100 x 256 made by cross-attending SD then satellite features. TopoNet: the prior replaces the learnable BEV queries; TopoMLP: a cross-attention layer aligns the prior with perspective features.
- input: OSM polylines + satellite tiles at the GPS location; BEV or perspective features.
- output: BEV prior grid for the online model.
- training: SMART 8 epochs, 8 x A100, about 2 days; online models 24 epochs; weights frozen online; pose noise not discussed.
- differentiable: yes (frozen prior).
- sensors_data: Argoverse 2 HD maps for pre-training; OpenLane-V2 (tables below are the Argoverse 2-based subset).
- results: OpenLane-V2 val DET_l / TOP_ll / OLS: TopoNet 28.6 / 10.9 / 39.8 -> SMART-OL 46.1 / 27.5 / 51.1; TopoMLP 28.5 / 21.7 / 44.1 -> 46.6 / 37.0 / 53.1; SD-only prior (online) 34.2 / 16.5 / 44.1; satellite-only 41.2 / 22.2 / 48.1; offline SMART alone 37.9 / 31.9; geographically disjoint split OLS 26.9 -> 33.6; data scaling 1x / 18x / 40x OLS 45.3 / 49.3 / 51.1.
- compute: see training.
- code: none listed (project page only)
- fit_to_us: reuse_idea - effort M - validity-masked polyline tokenisation of the SD map (N = 11 points).
- compat_needs: geo-referenced SD vectors and the pose.
- compat_conflicts: needs GPS-level alignment; route unused.
- tried_before: E40 (map prior at the grid) and E48 (registration).
- verified: checked

### CARD PL42 - P-MapNet (OSM raster prior with cross-attention; HD-map prior as masked autoencoder)
- papers: P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors (Jiang et al., arXiv 2024) https://arxiv.org/abs/2403.10521 ; project https://jike5.github.io/P-MapNet/
- depth: M
- stage: S9
- idea: SD map (OSM road centre-line skeletons, weakly aligned because of GPS bias) rasterised into the ego frame; a conv encoder with sine position embedding gives SD tokens; BEV features (down-sampled by 4 to 50 x 25) cross-attend to them (2 layers best; 4 or more overfit). HD prior: ViT masked autoencoder (50 % mask, patch sizes 20 x 20 to 40 x 80) refines the initial segmentation; optional and heavy.
- formulation: fusion comparison at 120 x 60 m (camera + lidar mIoU): none 49.07, concat 54.73, CNN concat 56.27, cross-attention 60.20.
- input: BEV features, rasterised SD map, optional HD patches.
- output: segmentation, then vectors by post-processing.
- training: HD prior pre-trained 20 epochs, then fine-tuning 10 epochs.
- differentiable: yes (vectorisation is post-processing).
- sensors_data: nuScenes, Argoverse 2.
- results: nuScenes val camera + lidar mIoU: 60 x 30 m 44.40 -> 51.93 (SD) -> 53.07 (SD + HD); 120 x 60 m 49.50 -> 60.20 -> 61.77; 240 x 60 m 36.47 -> 48.87 -> 49.93; camera only 240 x 60 m 26.77 -> 42.20 -> 45.50; MapTR + SD at 240 x 60: mAP 8.03 -> 16.53. Including OSM service roads lowered mIoU 60.20 -> 58.53; no explicit GPS-noise experiment.
- compute: not read.
- code: project page (code announced there)
- fit_to_us: reuse_idea - effort S - cross-attention to a rasterised OSM prior is what our route head does with SD raster tokens, and the comparison supports it (+5.5 mIoU over CNN concat).
- compat_needs: rasterised SD map in the ego frame.
- compat_conflicts: weak alignment is tolerated, not corrected.
- tried_before: E40 / E48 / E104 (same family; cross-attention kept).
- verified: checked

### CARD PL43 - SMERF (SD navigation-map polylines fused by cross-attention into lane perception / topology)
- papers: Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (Luo et al., ICRA 2024, from memory) https://arxiv.org/abs/2311.04079
- depth: M
- stage: S9
- idea: OSM road polylines around the ego (GPS 3-DoF), each sampled at N = 11 points with sinusoidal embeddings (d = 32, T = 1000) + one-hot road type (6 + catch-all), linear layer, 6-layer transformer; cross-attention between BEV queries and SD features after each spatial cross-attention in the BEV encoder. All roads are used (road type labelled, no planned route distinguished).
- formulation: backbones BEVFormer-DeTR baseline and TopoNet.
- input: SD polylines + BEV features.
- output: lane detection and topology.
- training: not read.
- differentiable: yes.
- sensors_data: OpenLane-V2 subset A (Argoverse 2); the abstract claims gains up to 60 %.
- results: BEVFormer-DeTR baseline DET_l 17.0 -> 26.8, TOP_ll 2.3 -> 3.9, OLS 30.2 -> 34.8; TopoNet DET_l 28.2 -> 33.4, TOP_ll 4.1 -> 7.5, OLS 34.5 -> 39.4; gains largest at far range and at intersections. Noise robustness not evaluated.
- compute: not read.
- code: https://github.com/NVlabs/SMERF
- fit_to_us: reuse_idea - effort S - vector tokenisation of SD roads as an alternative to our SD raster.
- compat_needs: SD polylines and pose.
- compat_conflicts: no route channel.
- tried_before: none (our SD input is raster).
- verified: checked

### CARD PL44 - OpenLane-V2-OSM: SD maps for online road-network perception and reasoning
- papers: Enhancing Online Road Network Perception and Reasoning with Standard Definition Maps (arXiv 2408.01471; authors not read) project page https://henryzhangzhy.github.io/sdhdmap/
- depth: M
- stage: S9
- idea: Lane-centre-line detection and topology on OpenLane-V2 (Argoverse 2 subset A; the OLV2-OSM extension adds OSM for 1,000 maps). The SD map is fused as a raster (ResNet-18 features concatenated with BEV features + 2-layer ConvFuser for MapTR; stacked channel-wise for TopoNet) or as a graph (SD nodes augmented with BEV features at their positions, EdgeConv GNN, fused with centre-line queries by attention).
- formulation: not read beyond the fusion description.
- input: BEV features + SD raster or SD graph.
- output: centre-lines with topology.
- training: MapTR 110 epochs; TopoNet 24 epochs.
- differentiable: yes.
- sensors_data: OpenLane-V2 / OLV2-OSM.
- results: MapTR Chamfer mAP 40.0 -> 46.6 (OLV2 SD map) / 43.2 (OSM raster); the abstract's 30 % does not match the table (about 16.5 %). TopoNet OLS 34.8 -> 37.7 (+13 M parameters, raster) or 36.7 (+1.7 M, graph); TOP_ll 4.15 -> 7.71. Localisation-error robustness (performance loss, translation 0.25 / 0.5 / 1.0 / 2.0 m at 0 deg rotation): 0.45 / 0.90 / 3.07 / 10.82 %; at 10 deg rotation 13.64 ... 21.14 %. No route used.
- compute: +13 M or +1.7 M parameters.
- code: none (project page)
- fit_to_us: reuse_idea - effort S - graph-style SD fusion costs 1.7 M parameters vs 13 M for the raster fusion.
- compat_needs: an SD graph.
- compat_conflicts: perception range of OpenLane-V2.
- tried_before: E40 / E48 (misalignment sensitivity: +0.075 F1 with exact pose, -0.057 unaligned).
- verified: checked

### CARD PL45 - Neural Map Prior (global neural map memory updated by a GRU)
- papers: Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481
- depth: M
- stage: S8, S9
- idea: A global grid of features in world coordinates (sparse tiles, 0.3 m default; 0.6 m and 1.2 m reduce the gain) is queried with the ego pose; current BEV features attend the retrieved prior (patches 10 x 10, about 3 m; dimension 256); a 2-D conv GRU fuses and writes the result back; tiles are downloaded and uploaded on demand (about 70 % less memory than dense tiles).
- formulation: current-to-prior attention with separate position embeddings, so the model can lean on the prior at long range.
- input: current BEV features, retrieved prior features, ego pose.
- output: improved BEV features and updated prior tile.
- training: not read in detail.
- differentiable: yes.
- sensors_data: nuScenes.
- results: nuScenes val mIoU (all classes), without -> with prior: HDMapNet 32.73 -> 37.05; LSS 39.78 -> 44.80; BEVFormer 43.01 -> 48.54; VectorMapNet mAP 40.9 -> 44.8; night +7.73, rain +5.92; single traversal +2.30 vs multiple traversals +3.81; Boston re-split 26.35 -> 33.04. Pose-noise robustness not evaluated.
- compute: not read.
- code: not found.
- fit_to_us: not_suitable - effort L - a memory prior needs repeated traversals and accurate pose; it does not address the route question.
- compat_needs: repeated traversals, accurate pose.
- compat_conflicts: pose assumed known.
- tried_before: E41 (temporal feature fusion, +0.008 F1) is a short-range cousin.
- verified: checked

### CARD PL46 - OrienterNet (localisation in OSM by neural matching; the "no pose given" step)
- papers: OrienterNet: Visual Localization in 2D Public Maps with Neural Matching (Sarlin et al., CVPR 2023) https://arxiv.org/abs/2304.02009
- depth: M
- stage: S9
- idea: OSM rasterised (3 channels at 0.5 m, 128 x 128 m training maps; 50 classes: 7 area, 10 line, 33 point types, learned embeddings) and a neural BEV (32 x 32 m, 0.5 m, 8 channels + confidence) inferred from a single gravity-aligned image are correlated exhaustively over (x, y, theta) in the Fourier domain.
- formulation: P = softmax(M + prior) over the pose grid, K = 64 rotations in training, 512 at test; the pose is the argmax; sequences are fused by multiplying warped probability volumes.
- input: OSM raster, one image with known gravity.
- output: pose with a probability volume.
- training: loss = -log P at the GT pose (trilinear interpolation), supervised.
- differentiable: yes in training; argmax at inference.
- sensors_data: single camera + OSM (MGL dataset).
- results: MGL validation: position recall 15.78 % at 1 m, 47.75 % at 3 m, 58.98 % at 5 m; orientation 22.14 / 52.56 / 66.32 % at 1 / 3 / 5 deg.
- compute: 94 ms per localisation on an RTX 2080 (37 ms BEV, 51 ms matching); parameters not reported.
- code: announced, no URL on the page.
- fit_to_us: reuse_idea - effort M - a probability volume over pose for the registration front end gives explicit uncertainty instead of regression.
- compat_needs: SD raster and BEV features.
- compat_conflicts: the exhaustive pose grid costs memory.
- tried_before: E48 (learned registration, kept) and E74 (registration front end).
- verified: checked

### CARD PL47 - LFP (lane-level planning flags on double-edge lane queries; CARLA)
- papers: LFP: Efficient and Accurate End-to-End Lane-Level Planning via Camera-LiDAR Fusion (arXiv 2409.14170; authors not read) https://arxiv.org/abs/2409.14170
- depth: M
- stage: S10
- idea: N_d double-edge lane structures (left / right edges of N_p/2 3-D points, intersection and direction flags); heads predict boundary points, occupancy and a planning flag per point; the planned path is the midpoint of left / right points with flag 1. No explicit topology between lanes; the route command handling is not described.
- formulation: loss weights 3:2:1:3:4:5:1:0.1 over ROI, intersection, direction, occupancy, planning, edge, speed, signal terms; lane ROI selection by nearest pillars is a hard step.
- input: camera + lidar lane queries.
- output: lane flags, path, speed.
- training: joint loss; whether planning gradients reach lane perception is not stated.
- differentiable: partial.
- sensors_data: CARLA (about 126 K frames, 8 maps).
- results: Town05 Long DS 91.1, RC 99.6, IS 0.92; Town05 Short DS 96.7; 19.27 FPS (51.90 ms).
- compute: see results.
- code: none found.
- fit_to_us: reuse_idea - effort S - a per-point "on the plan" flag on perceived lane points is a per-vertex route flag, pairing with our route probability.
- compat_needs: lane queries with points.
- compat_conflicts: CARLA only.
- tried_before: E102-E104.
- verified: checked

### CARD PL48 - GET-DIPP (lane-graph encoder + differentiable integrated prediction and planning)
- papers: GET-DIPP: Graph-Embedded Transformer for Differentiable Integrated Prediction and Planning (arXiv 2211.06031; authors not read) https://arxiv.org/abs/2211.06031
- depth: M
- stage: S10
- idea: Per agent 6 nearest lanes and 4 nearest crosswalks; lane waypoints (boundaries, centre, headings, speed limit, types, traffic light, stop signs) become DGCNN vertices connected to their k = 20 nearest neighbours in feature space (not map connectivity); proxy waypoints (averaged subsets) cut memory from O(L x L) to O((L/l)^2); the DIPP differentiable optimiser plans on the prediction.
- formulation: loss = 0.5 L_predict + L_score + L_ADE + L_FDE.
- input: lane and crosswalk waypoints, agent histories.
- output: prediction and plan.
- training: end-to-end; the gradient path into the encoder is not described explicitly.
- differentiable: yes.
- sensors_data: Waymo (173 k / 25 k / 7 k frames).
- results: integrated planning position error at 3 / 5 / 10 s 1.710 / 3.420 / 8.201 m vs DIPP 1.726 / 3.913 / 9.365; collision 3.76 % vs 5 %; prediction ADE / FDE 0.712 / 1.808 vs 0.740 / 1.814.
- compute: not read.
- code: none given.
- fit_to_us: not_suitable - effort L - Waymo GT map; relevant only as an example of a feature-space k-NN graph (no explicit lane topology) feeding a differentiable planner.
- compat_needs: waypoints.
- compat_conflicts: GT map.
- tried_before: none.
- verified: checked

### CARD PL49 - BevAD study: what matters in end-to-end planners (BEV compression, path + speed outputs, commands as weak conditioning)
- papers: What Matters for Scalable and Robust Learning in End-to-End Driving Planners? (arXiv 2603.15185; authors not read) project page https://dmholtz.github.io/bevad/ (card added last, hence the out-of-order number)
- depth: M (page cut at 100k characters)
- stage: S10, T
- idea: Systematic study on Bench2Drive (CARLA, 220 short routes) and NAVSIM: (a) a high-resolution BEV (100 x 100) lets the planner latch onto spurious landmarks (causal confusion); masking 20 % of the left / right BEV cells and patchifying into scene tokens (patch size 4 best) removes this; (b) disentangled path + speed outputs beat entangled trajectories; (c) diffusion planners scale better; (d) navigation commands are weak conditioning (missed lane changes, route deviations), target points are suggested but not tested. Map-segmentation masks gave no significant difference from simple masking; online-map auxiliary supervision was pruned.
- formulation: tokenizer ablation (Bench2Drive driving score): no mask, p = 1: 66.86 (success rate 36.36); mask p = 1: 72.40; p = 2: 74.98; p = 4: 82.62 (success rate 57.43); p = 5: 66.44. Planning representation success rate: point estimator trajectory 51.7, point estimator path + speed 57.4, diffusion trajectory 56.2, diffusion path + speed 59.4.
- input: 6 cameras -> BEV (100 x 100) -> masked, patchified tokens; command, ego state.
- output: path + speed (disentangled) or trajectory.
- training: imitation; point-estimator and diffusion variants.
- differentiable: yes.
- sensors_data: Bench2Drive (CARLA), NAVSIM.
- results: BevAD-S 80.63 DS / 55.30 SR, BevAD-M 88.11 / 72.73 on Bench2Drive; NAVSIM PDMS 86.6 vs ParaDrive 84.0 and UniAD 83.4 (85.9 without masking, 85.6 without patchifying). No route-format or map-quality ablation.
- compute: not read.
- code: project page only (no repository named)
- fit_to_us: reuse_idea - effort S - warns against feeding a dense raster to the consumer; our graph tokens are already a compressed form, and the route as a channel rather than only a command is the stated weak point.
- compat_needs: BEV tokenisation (masking + patchify) if a raster is used.
- compat_conflicts: CARLA-based evidence only.
- tried_before: none.
- verified: partly

## 3. Cluster compatibility notes

### 3a. Interface contracts (what flows into the planner in the literature, and what our output must become)

| consumer | map unit | points per unit | per-point features | connectivity | route / goal | traffic light, speed | frame, normalisation | element count |
|---|---|---|---|---|---|---|---|---|
| PlanTF (code) | polygon: lane, lane connector, crosswalk | 20 (3 polylines; encoder uses the centre) | offset to polygon centre (2), next-point vector (2), cos / sin (2) | none | on_route flag per polygon (roadblock membership) | TL 4-way embedding, speed-limit MLP | ego at t0, rotated; +-100 m square | all in the square, padded to batch max |
| PLUTO (code) | polygon + reference lines | 20 / 100 | 8 ch (p-p0, p-p_prev, p-left, p-right) | reference lines by DFS over successors | route filters the DFS (roadblock ids) | attribute embeddings | polyline-local + Fourier global | N_R x 12 queries, top 20 |
| Diffusion Planner (code) | lane | 20 | 12 ch (centre, vector, left, right offsets, TL one-hot) | none (successors not traversed) | 25 route lanes via adaLN | TL per point, speed per lane | ego, z-score x (10, 20) | 70 lanes |
| GameFormer-Planner (code) | lane centre-line, route lanes, crosswalk | N, P from config (not seen) | x, y, heading (+ TL one-hot) | none | route lanes + reference path | TL one-hot | ego at present state | nearest N |
| DTPP | map element | not stated | MLP | none | 3 reference paths | not stated | ego | 50 |
| PDM-Closed / Open | centre-line | 1 m spacing up to 120 m (Open) | x, y, heading | graph search over route lanes | roadblock route, BFS repair | speed limit | ego | 1 path + offsets |
| LaneGCN | node = centre-line segment | segment | end - start, midpoint | pre / suc / left / right, dilations 1 ... 32 | none | none | actor-centred | within 100 m |
| QCNet (code) | polygon + boundary / centre points | raw points | relative geometry only | typed pred / succ / left / right edges + radius edges | none | type, is_intersection | relative (no global frame) | radius graph |
| PGP | node <= 20 m, poses at 1 m | <= 20 | x, y, yaw, stop / crosswalk flag | successor + proximal | edge policy | flags | agent-centred box | not stated |
| VectorNet / HiVT | vector / segment | polyline split into vectors | start, end, attributes | none | none | attributes | target- or scene-centred | all within radius |
| MTR | polyline | 20 | position, direction, type | k-NN (16) attention | none | none | agent-centred | 768 nearest |
| VAD / UncAD / SparseDrive | instance | 20 | xy (+ Laplace b) + 256-d query | none | 3 command embeddings / per-command modes | none | ego BEV 60 x 30 m | 100 |
| UniAD | map query | mask-based | query | none | 3 command embeddings | none | BEV | 300 + 1 |

Mapping of OUR output (vertex positions in ego metres, 128-d step token, validity / keep probability, successor edges, route probability per vertex, about 22 vertices per 100 m) onto these contracts gives three consumer families:
1. **Polyline set with attributes and padding mask** (PlanTF, Diffusion Planner, GameFormer, MTR, HiVT, VectorNet): adapters A1 + A2 + A3.
2. **Polyline set + successor edges (+ route filter)** (PLUTO, PDM-Closed, LaneGCN, PGP, QCNet): A1 + A4; our edge list is already the required structure, and edges-as-nodes give LaneGCN's segment nodes.
3. **Instance queries** (VAD, UncAD, SparseDrive, DriveTransformer, UniAD): A5.

### 3b. Adapters and conflicts (adapters are my synthesis from the formats above)

- **A1 graph -> lane pieces:** cut the vertex graph at fork / merge vertices into maximal chains; cut chains longer than L_max (PGP uses 20 m snippets; nuPlan lanes are map-defined); resample each chain by arc length to P points (P = 20 for PlanTF, Diffusion Planner, MTR, VAD, SparseDrive; 100 for PLUTO reference lines); per-point vector to the next point and heading by differences; polygon centre = point index 10 of 20 (PlanTF and PLUTO builders). With 6 m edges a 20-point piece spans up to about 114 m, so long lanes are nearly the identity and short connectors need interpolation.
- **A2 route:** (i) per-polygon on_route flag = route probability above a threshold, or a linear embedding of the probability in place of PlanTF's 2-way embedding; (ii) ordered route-lane list (Diffusion Planner 25 x 20; GameFormer route_lanes) from the chain of pieces with the highest mean route probability; (iii) PLUTO-style DFS where the roadblock filter is replaced by edge route probability above tau (their code: depth 15 lanes, 100 m, 0.5 m de-duplication); (iv) PDM centre-line = best path by dynamic programming over route probability from the vertex nearest to the ego, resampled at 1 m. nuPlan routes are roadblock-level (all parallel lanes of the block count as on route), ours is lane-level: lane-level to roadblock-level is easy, the reverse is not, so nuPlan planners saw lateral ambiguity in training.
- **A3 validity:** key-padding mask (PlanTF: polygon masked if no valid point; GameFormer `avails`; Diffusion Planner: mask derived from all-zero lane rows in its encoder, so invalid lanes must be zeroed); hard threshold 0.5 as in VAD; soft form as additive log-confidence bias in the attention logits (UMPE). Planners with a fixed N nearest lanes (Diffusion Planner 70, GameFormer nearest N) need top-N by distance after thresholding.
- **A4 edges:** LaneGCN pre / suc = our edges (or the line graph for segment nodes); left / right neighbours are not in our output (derive geometrically or drop). QCNet typed edges plus a radius graph tolerate missing topology.
- **A5 tokens -> instance queries:** pool the 128-d vertex tokens of a piece (mean or attention pool) to the query (VAD / SparseDrive keys), use 20 x 2 points as anchors / positional encoding (DriveTransformer: per-point position encoding + PointNet merge), class score from pooled validity.
- **A6 dense alternatives:** the cached 200 x 200 pyramid can be read at vertex positions (Gu et al. ECCV 2024), or graph channels rasterised for a cost-volume planner (ST-P3 grid is identical: 200 x 200 at 0.5 m).
- **Conflicts:** (C1) lane boundaries / road boundaries (Diffusion Planner channels 4-7, VAD's boundary hinge, UncAD boundary ellipses) do not exist in a centre-line graph and would have to come from the drivable-area logits. (C2) Traffic-light state and speed limit are absent (nuPlan planners accept UNKNOWN and "no speed limit", but behaviour at lights is then unobservable). (C3) Range: nuPlan planners use 100 m radii, our BEV is +-50 m; fine-tune or retrain. (C4) Frameworks: nuPlan devkit + Python 3.9 for PlanTF, PLUTO, Diffusion Planner, GameFormer, DTPP, PDM; mmcv / mmdet3d for UniAD, VAD, SparseDrive, DiffusionDrive, DriveTransformer; none installable in the gcar environment, so only head / loss ideas can be reused. (C5) Hungarian matching inside the UniAD / VAD / SparseDrive map heads conflicts with our matching-free rule if the perception side were reused. (C6) The command / route on nuScenes is derived from the future ego path (AD-MLP: more than 2 m lateral displacement within 3 s) and our route label (lane sequence driven in the next 6 s) is future-derived as well; a real mission route does not exist in nuScenes. (C7) Ego motion as planner input creates the shortcut documented by PlanTF (open-loop high, closed-loop low) and BEV-Planner (zeroing ego velocity: L2 6.16 m).

### 3c. Combinations the papers themselves demonstrate

- VAD / SparseDrive + per-vertex uncertainty (UncAD); VAD + prior encoder (UMPE, L2 0.72 -> 0.42); VAD + topology token with GT or predicted topology (PL14).
- HiVT / DenseTNT + MapTR / MapTRv2 / MapTRv2-centre-line / StreamMapNet, with vertex uncertainty (PL15) or with BEV features (PL16).
- TopoNet / TopoMLP + SD + satellite prior (SMART); TopoNet + SD graph (OLV2-OSM, SMERF); MapTR + SD raster (OLV2-OSM, P-MapNet); HDMapNet / LSS / BEVFormer / VectorMapNet + neural map prior (NMP).
- MapTR / MapTRv2 / SeqGrowGraph outputs + SD association (MAT, PL01): the only demonstrated route attribution on a perceived lane graph.
- Planner + learned or differentiable refinement: GameFormer + DIPP-style optimiser; Diffusion Planner + guidance and refinement; PLUTO + post-processing (89.04 -> 93.21 on Val14); Hydra-MDP + PDM sub-score distillation; GoalFlow + drivable-area scoring; SUPER-AD + DiffusionDrive.
- Privileged GT map + route teacher distilled into a map-free student (DriveZero).

### 3d. Answers to the brief's questions

1. **Representation our output must be converted to:** see 3a (three families) and adapters A1-A6. The simplest high-value contracts: PlanTF's polygon dict (20 centre points, polygon centre, on_route, valid mask) or PLUTO's reference lines (DFS on the route-filtered edge graph); the simplest rule-based consumer is the PDM centre-line (best route path, 1 m resampling).
2. **Route / goal encoding and precedent for attribution after perception:** planners encode the route as (a) 3 discrete command embeddings (UniAD, VAD, SparseDrive, DiffusionDrive, SSR gate, DriveTransformer modes), (b) a lane flag (PlanTF on_route), (c) route lanes as extra tokens (Diffusion Planner, GameFormer), (d) route-restricted reference lines (PLUTO, DTPP, PDM), (e) goal points along the route (GoalFlow, DriveZero teacher), (f) target points (TransFuser-style, from memory). Commands are reported as weak conditioning (missed lane changes, route deviations; PL49), and the route-input format has never been ablated in the papers read. Attribution after perception has one direct precedent, MAT / OMA (PL01), plus the lane flags that nuPlan planners compute on a given map (PL03). SMART, SMERF and OLV2-OSM use SD maps as geometry prior without a route.
3. **Missing, false, fragmented or low-confidence lanes:** nuPlan-style planners train on clean GT maps with only ego / agent augmentations (PlanTF: no map augmentation found; PLUTO: traffic-light inversion, agent dropout and insertion, and a reference-free head for scenes without a reference line). Models without connectivity (VectorNet, MTR, PlanTF, Diffusion Planner, HiVT) are structurally insensitive to broken edges; QCNet adds radius edges; LaneGCN, PGP, PLUTO and PDM depend on successors. Confidence use: hard threshold 0.5 (VAD), per-vertex Laplace scale as features and in trajectory selection (UncAD, Gu et al.), log-confidence bias (UMPE), BEV-segmentation uncertainty re-weighting proposals (SUPER-AD). Robustness tests exist for SD maps (OLV2-OSM translation 2 m: -10.8 %; MAT element absence 30 %: val F1 75.5 from 78.7) but not for planner-side map corruption (gap).
4. **Planning loss back-propagated into an explicit lane graph:** only into instance-level map polyline coordinates or queries: VAD (code-verified; class scores get no gradient; gain small: L2 0.76 -> 0.72), UncAD, MAP (explicit), SparseDrive / DriveTransformer (joint training, no detach), UniAD (through MotionFormer attention, BEV encoder frozen). No work reaches graph edges or topology. Our E121 / E119 null results are consistent with VAD's small ablation gain.
5. **Best test bed for graph quality (my synthesis from PL02, PL09, PL14, PL15, PL18, PL19, PL20, PL21, PL24):** nuScenes open-loop L2 / collision cannot resolve graph quality (identical L2 0.62 m and collision 0.16 % with GT and predicted topology in PL14; an ego-status MLP reaches L2 0.29-0.35 m; blank images change nothing). Ranked suggestion: (1) a deterministic centre-line planner (PDM-Closed principle: IDM along the best route path of the graph, proxy score) fed with GT vs predicted vs degraded graphs, measured with map-sensitive terms (drivable-area / boundary violation like DACR 1.39 % -> 0.80 % in PL09 or CCR in PL19, lane keeping, direction compliance, progress, path error at 5-40 m, restricted to the forward / route half as in PL18); no paper does this with a perceived graph. (2) E117 with an ego-motion ablation and a turning-only subset, graph swapped GT / predicted (E117 path error 0.34 m on B vs 0.36 m on A already shows a small gap). (3) External prediction consumers through the Gu et al. pipeline (minADE / minFDE / MR; forward-region sensitivity +9.7 % minADE). (4) NAVSIM-style sub-scores (DAC, DDC, LK) re-implemented on nuScenes; NAVSIM itself has no map input and no radar.
6. **Compute / code:** URLs are in the cards. Heavy items: nuPlan cache building (PlanTF 1 M frames 20+ h; PLUTO about 45 h on 4 x RTX 3090; Diffusion Planner 500 epochs on 8 x A100) and QCNet (about 160 GB GPU memory in total). Light and portable: VAD-style hinge losses, the PDM-style test bed, a PlanTF-style polygon encoder (4 layers, dim 128).

## 4. Reuse versus re-implement

- **Re-implement (small, from the method section):** PDM-style centre-line IDM test bed with oracle / degraded graph swap (PL02, with the route-repair idea of its BFS); PlanTF-like polygon consumer (depth 4, dim 128, 20-point centre polylines, on_route embedding, valid mask) with a state-dropout encoder for ego motion; VAD-style hinge constraints (boundary 1.0 m, direction 2.0 m, confidence 0.5) on vertices / edges; UncAD-style trajectory scoring from vertex validity; UMPE-style log-confidence attention bias; PLUTO-style route-filtered DFS reference lines; DDC / LK / DAC sub-scores on nuScenes; MAT-style path-aware token ordering for the route head (our DAG can enumerate root-to-leaf paths).
- **Adapt code only for formats, not for training:** PlanTF / Diffusion Planner feature dicts, QCNet dataset class, LaneGCN graph tensors. The nuPlan devkit, mmcv / mmdet3d and PyG-pinned code do not install in the gcar environment.
- **Baselines to cite, not rerun:** UniAD, VAD, SparseDrive, DiffusionDrive, Hydra-MDP, the NAVSIM leaders (other sensors and data).
- **External consumer on nuScenes worth testing:** HiVT / DenseTNT through the Gu et al. repositories, after checking that the large pre-built data sets (about 500 GB each) can be avoided by generating own inputs.

## 5. Open questions and gaps

- No paper measures closed-loop (or PDMS-like) effects of a perceived lane graph with topology in a nuPlan-style or rule-based planner; the oracle / degraded graph swap into such a planner is untested in the literature.
- No planner-aware map-quality metric exists beyond PKL (detections) and A-mAP (directional split, evaluated with a prediction model); no weighting of vertices by route relevance.
- No work back-propagates planning into topology; whether it would help is unknown (our E121 / E119: no).
- Route attribution: only MAT (hard, road-level); SMART leaves the route unused. How nuPlan-trained planners behave with a lane-level route instead of a roadblock-level one is not studied.
- How nuPlan planners degrade with false forks, missing segments or fragmented lanes is not reported (PlanTF has no map augmentation; PLUTO only agent / traffic-light negatives).
- Calibration of per-vertex uncertainty is listed as future work by Gu et al.; UncAD thresholds are unspecified.
- Not determined from the pages: PlanTF PointsEncoder internals and loss terms; GameFormer-Planner N and P; PDM-Closed centre-line search algorithm and scorer weights; VADv2 navigation encoding; UniAD stage-2 stop-gradient details; QCNet point counts per polygon; PARA-Drive module ablations (PDF not readable).
- Numbers differ across versions: PlanTF Val14 (own table 84.83 / 76.78 vs PLUTO's table 85.30 / 77.07); TransFuser navtest PDMS 84.0 (NAVSIM) vs 78.0 (Hydra-MDP); RAP navhard EPDMS 36.9 (own) vs 39.6 (pseudo-simulation snapshot); straight-driving share 73.9 % vs 87 % (BEV-Planner); planner count 86 vs 83 in the pseudo-simulation paper.

## 6. Search log

WebSearch (20 queries; standard unless marked extended):
1. QCNet CVPR 2023 query-centric trajectory prediction map polygons relative position encoding
2. end-to-end autonomous driving navigation route OpenStreetMap SD map route as input planner lane-level route conditioning nuScenes NAVSIM (extended)
3. navigation-guided sparse scene representation end-to-end SSR; route-aware map elements; route lane graph perception
4. NAVSIM navtest PDMS state-of-the-art 2026 end-to-end planner camera-only (extended)
5. nuScenes open-loop planning L2 collision state of the art 2026 SparseDrive DiffusionDrive successor map instances
6. impact of online HD map construction errors on downstream planning closed-loop planning-aware map metric lane graph PDM nuPlan (extended)
7. planning-oriented online mapping evaluate map quality by planner performance, oracle vs predicted map degradation nuScenes
8. planning loss backpropagated into lane graph perception differentiable end-to-end lane graph planner gradients map head (extended)
9. Perception in Plan coupled perception and planning VeteranAD planning anchors guided perception map nuScenes
10. route-conditioned lane graph tokens navigation route attribution onto perceived lane graph planner SD map route end-to-end driving 2025 (extended)
11. trajectory prediction robustness to map noise missing lanes false lanes lane graph perturbation map dropout
12. end-to-end driving diffusion planner conditioned on predicted lane centrelines map queries lane-guided trajectory generation (extended)
13. nuPlan planner with perceived map instead of ground-truth map noisy lane centerlines online mapping closed-loop robustness (extended)
14. "planning-oriented" online map construction metric planner-aware map benchmark nuScenes planning impact of mapping errors
15. ScenarioNet nuScenes MetaDrive closed-loop replay lane graph map format
16. nuScenes closed-loop planning evaluation benchmark reactive agents map expansion lane graph
17. PARA-Drive parallelized architecture mapping module ablation (snippets only)
18. route-aware lane detection navigation path prior which lane to follow at intersections lane-level route reasoning (extended)
19. lane topology graph end-to-end planning head uses centerline connectivity successor lanes planner nuScenes OpenLane-V2 (arxiv.org only)
20. learned planner robustness to perceived map errors, replace GT HD map with online mapping output, closed-loop degradation (arxiv.org only)

WebFetch (arXiv HTML / ar5iv unless noted): 2309.10443 (+ PDF attempt), 2404.14327 (+ tail), 2501.15564, 2303.05760, 2310.05885, 2212.10156 (two passes), 2303.12077, 2402.13243, 2405.19620, 2411.15139, 2406.06978, 2406.15349, 2306.07962, 2312.03031, 2305.10430 (html + abs), 2504.12826, 2509.13926, 2403.16439, 2007.13732, 2005.04259, 2209.13508, 2106.15004, 2306.10508, 2502.04329, 2403.10521, 2311.04079, 2304.08481, 2605.02762, 2512.19150 (two passes), 2004.08745 (abstract), 2409.14170, 2211.06031, 2508.11488, 2504.12667, 2507.07487 (two passes), 2608.18035, 2609.06055, 2510.04333, 2506.04218, 2106.11810 (abstract), 2407.06683, 2609.22582, 2511.22865, 2408.01471, 2404.10295 (abstract), 2503.07656, 2503.12820, 2503.05689, 2409.18341, 2207.07601, 2304.02009, 2603.15185.
GitHub pages (README or raw source files): jchengai/planTF (README, nuplan_feature_builder.py, nuplan_feature.py, map_encoder.py, planning_model.py), jchengai/pluto (README, pluto_feature_builder.py, pluto_model.py, scenario_manager.py, route_manager.py), ZhengYinan-AIR/Diffusion-Planner (README, map_process.py, diffusion_planner.py, encoder.py), MCZhi/GameFormer-Planner (README, data_utils.py), autonomousvision/tuplan_garage (README, route_utils.py, pdm_path.py, pdm_closed_planner.py, abstract_pdm_closed_planner.py), autonomousvision/navsim (README, agents.md, metrics.md), hustvl/VAD (plan_loss.py, VAD_base_e2e.py), ZikangZhou/QCNet (README, argoverse_v2_dataset.py, qcnet_map_encoder.py), ZikangZhou/HiVT (README, argoverse_v1_dataset.py), WallelWan/OMA-MAT (README), OpenDriveLab/UniAD, uber-research/LaneGCN, nachiket92/PGP, NVlabs/trajdata, alfredgu001324/MapUncertaintyPrediction (READMEs).
