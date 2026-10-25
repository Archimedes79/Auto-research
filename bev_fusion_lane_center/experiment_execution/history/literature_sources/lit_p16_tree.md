# Literature review: verified-ray hypothesis tree and spray-and-link (loop 16)

Date: 2026-10-23. Scope: design of `reports/Hypothesis_Tree_Spec_and_Tests_EN.md` and `reports/Growth_and_Link_Architectures_EN.md` (4.2a, 5). Searches were run with web search and abstract pages in this session; no files were downloaded.

**Status markers.** *checked* = the abstract or listing page (arXiv, CVF, ICLR/NeurIPS proceedings) was seen in this session, either opened directly or as a search-result summary of that page. *not verified* = from memory or from a third-party summary only. Facts marked "(from memory)" inside a checked row were not on the page seen. No numbers are given unless they were on a page seen in this session.

**Caveat on OpenLane-V2 numbers.** OpenLane-V2 changed its TOP metric definition between dataset/devkit versions (v1.x vs v2.1, from memory), so OLS / TOP_ll values from different papers are not necessarily comparable. Subset A = Argoverse 2, subset B = nuScenes (checked in the UniTopo abstract: "two subsets built on Argoverse2 and nuScenes").

---

## Q1. Iterative / autoregressive lane- or road-graph growth

| paper | year / venue | link | what it does, relevant to our design | status |
|---|---|---|---|---|
| RoadTracer (Bastani et al.) | CVPR 2018 | [arXiv 1802.03680](https://arxiv.org/pdf/1802.03680) | Aerial road graph by iterative search guided by a CNN decision function; per step: walk direction or stop; fixed step length and angle bins (from memory). +45 % more junctions captured at 5 % error vs segmentation (abstract). Not end to end; trained with a dynamic-label oracle (from memory). | checked (abstract) |
| DAGMapper (Homayounfar et al.) | ICCV 2019 | [arXiv 2012.12377](https://arxiv.org/abs/2012.12377v1) | Highway lane boundaries with forks/merges as inference in a DAG graphical model; nodes carry geometry + topology (fork/merge state); 89 % correct topology on two highways (abstract). Per-step decoding details not re-checked. | checked (abstract) |
| iCurb (Xu, Sun, Liu) | RA-L 2021 | [arXiv 2103.17118](https://arxiv.org/pdf/2103.17118) | Curb graph grown vertex by vertex from aerial images; per step next vertex + stop flag; **DAgger** (restricted + free exploration) because plain behaviour cloning fails to recover from own errors. Not differentiable through the rollout. | checked |
| RNGDet / RNGDet++ (Xu et al.) | T-GRS 2022 / RA-L 2023 | [2202.07824](https://arxiv.org/pdf/2202.07824), [2209.10150](https://arxiv.org/pdf/2209.10150) | DETR-style agent explores the road network from candidate initial vertices; per step a **set** of next vertices (DETR queries, so multiple outputs at junctions, from memory); trained by behaviour cloning (teacher-forced), DAgger avoided for sampling efficiency. ++ adds instance segmentation and multi-scale features. | checked |
| CenterLineDet (Xu et al.) | ICRA 2023 | [arXiv 2209.07734](https://arxiv.org/pdf/2209.07734) | Vehicle-sensor version of RNGDet: multi-frame BEV (6 cameras + lidar), DETR-like decision transformer traces centerlines from candidate vertices, imitation learning, nuScenes. Closest prior to our G architecture; not end to end through the trace. | checked |
| RNTR (Lu et al.) | ICCV 2023 | [arXiv 2402.08207](https://arxiv.org/pdf/2402.08207) | Road network as integer "RoadNet Sequence" from surround cameras on nuScenes; AR, semi-AR (sub-trees from key points) and non-AR variants; NAR is faster and more accurate than AR. Introduces **Landmark P/R and Reachability P/R** metrics (landmarks include fork/merge points). | checked |
| LaneGraph2Seq (Peng et al.) | AAAI 2024 | [arXiv 2401.17609](https://arxiv.org/pdf/2401.17609) | Lane graph DAG serialized as vertex-centric DFS + edge partition sequence; language-model decoder; classifier-free guidance + nucleus sampling to improve connectivity; nuScenes, Argoverse 2. Discrete sampling, not differentiable end to end. | checked |
| SeqGrowGraph (Xie et al.) | ICCV 2025 | [arXiv 2507.04822](https://arxiv.org/abs/2507.04822) | Grows the graph one vertex (junction) per step: adjacency matrix expands n -> n+1, centerline shapes as quadratic Beziers; DFS order best; handles loops and bidirectional lanes; centerlines split only at junctions incl. forks and merges. Teacher forcing not stated on the abstract page. nuScenes + AV2, SOTA claimed. | checked |
| TopoGPT (Fu et al.) | ECCV 2026 (arXiv 2606.31814) | [arXiv 2606.31814](https://arxiv.org/abs/2606.31814) | Lane-graph tokenizer + AR transformer pre-trained on 3.3 M map scenes (scene-conditioned next-token), then a perception adapter from multi-view BEV. OpenLane-V2: average +6.4 lane-level, +11.6 point-level over prior methods (abstract; absolute values not on page). Authors note latency and early-error propagation. | checked |
| Mastering Spatial Graph Prediction (Anagnostidis et al.) | arXiv 2022 | [arXiv 2210.00828](https://arxiv.org/abs/2210.00828) | Aerial road graph grown by adding edge sequences; RL agent with non-continuous rewards; tree-based search at inference improves results. The only tree search over graph growth found, but RL, not differentiable. | checked |
| DeH4R (Gong, Ji) | T-GRS (arXiv 2025/2026) | [arXiv 2508.13669](https://arxiv.org/abs/2508.13669) | Hybrid: candidate vertices + adjacent-vertex prediction + initial graph + graph expansion; +4.62 APLS over RNGDet++ on CityScale, about 10x faster. Shows "generate then grow from dangling ends" = our hybrid H. | checked |

Summary for Q1: every growing method found predicts per step a next vertex (or set of vertices) plus stop, trained by **behaviour cloning / teacher forcing**, DAgger, RL, or next-token likelihood; none is differentiable through the rollout. Forks: RNGDet-type via several DETR outputs per step (from memory), SeqGrowGraph/RNTR/LaneGraph2Seq via the serialization (branching in DFS order), DAGMapper via a fork state per node.

## Q2. Online lane topology on nuScenes / OpenLane-V2

| paper | year / venue | link | what it does | numbers seen (subset) | status |
|---|---|---|---|---|---|
| TopoNet (Li et al.) | arXiv 2023 | [arXiv 2304.05277](https://arxiv.org/pdf/2304.05277v1) | Centerline queries + scene graph NN with knowledge graph over lane / traffic-element relations. Standard baseline. | TOP_ll 4.1 (A, quoted by TopoMLP, old metric) | checked |
| TopoMLP (Wu et al.) | ICLR 2024 | [arXiv 2310.06753](https://arxiv.org/pdf/2310.06753) | Strong 3D lane detector + two MLP topology heads. | OLS 38.2, TOP_ll 7.2 (A, R50, no extras); 41.2 OLS with extras | checked |
| LaneSegNet (Li et al.) | ICLR 2024 | [arXiv 2312.16108](https://arxiv.org/abs/2312.16108v1) | Lane segments (centerline + boundaries + topology) as one output; **lane attention** over long-range regions with identical-initialised reference points. | +6.9 DET_l, +5.6 mAP lane segment (gains); absolute not on page | checked |
| TopoLogic (Fu et al.) | NeurIPS 2024 | [arXiv 2405.14747](https://arxiv.org/pdf/2405.14747) | Connectivity from **geometric endpoint distance** mapped by a learned function + query similarity; can be added post hoc to trained models. | TOP_ll 23.9 vs 10.9, OLS 44.1 vs 39.8 (A) | checked |
| CGNet (Han et al.) | ECCV 2024 | [arXiv 2407.11337](https://arxiv.org/abs/2407.11337) | Junction-aware query enhancement, Bezier-space connection for continuity, iterative topology refinement (GCN + GRU). nuScenes + AV2 centerline graphs. | not on page | checked |
| LaneGAP (Liao et al.) | ECCV 2024 | [arXiv 2303.08815](https://arxiv.org/pdf/2303.08815) | Paths (not pieces) as primitives, Path2Graph to recover the graph; argues pieces break at junctions. | +1.6 mIoU vs TopoNet on OLV2 | checked |
| Topo2Seq (Yang et al.) | AAAI 2025 | [arXiv 2502.08974](https://arxiv.org/pdf/2502.08974) | Training-only topology sequence decoder (randomised keypoint order) regularises a DETR lane-segment decoder; no inference cost. | +2.0 AP lane segment, +2.7 topology (third-party summary) | checked (abstract), numbers not verified |
| SMART | ICRA 2025 | [arXiv 2502.04329](https://arxiv.org/pdf/2502.04329) | Map prior from SD + satellite maps trained on large geo-referenced HD maps; plugged into online topology models. | up to +28 % (relative) on OLV2 | checked (abstract) |
| TopoPoint (Fu et al.) | arXiv 2025 | [arXiv 2505.17771](https://arxiv.org/abs/2505.17771) | Explicit **endpoint detection**, point-lane attention with distance mask, point-lane GCN, inference-time endpoint matching. | OLS 48.8 (subset not stated), DET_p 52.6 vs 45.2 | checked |
| TopoFG (Xu et al.) | AAAI 2026 | [arXiv 2511.12590](https://arxiv.org/abs/2511.12590) | Fine-grained queries from BEV-mask and in-lane keypoint priors, boundary-point topology reasoning with topological denoising. | OLS 48.0 (A), 45.4 (B) | checked |
| UniTopo (Li et al.) | IEEE TCSVT (arXiv 2605.08911, May 2026) | [arXiv 2605.08911](https://arxiv.org/abs/2605.08911) | Models each lane jointly with its predecessor/successor lanes ("reasoning-by-detection" replaced). Newest TOP_ll found. | TOP_ll 30.1 (A), 31.8 (B); +6.0 / +8.6 over T²SG | checked |
| TopoBDA (Kalfaoglu et al.) | Neurocomputing 2026 | [arXiv 2412.18951](https://arxiv.org/abs/2412.18951) | Bezier deformable attention for thin elongated centerlines; reports gains from adding **radar**, lidar, SD map. | SOTA claimed, no numbers on page | checked |

Best numbers found: OLS 48.8 (TopoPoint, subset unclear), 48.0 / 45.4 (TopoFG, A / B); TOP_ll 30.1 / 31.8 (UniTopo, A / B). DET_l absolute values: not found in this session.

## Q3. Multi-hypothesis / beam / tree search that stays trainable

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| Continuous relaxation of beam search (Goyal et al.) | AAAI 2018 (arXiv 2017) | [arXiv 1708.00111](https://arxiv.org/abs/1708.00111) | Sub-differentiable surrogate of beam search so the final loss on beam output can be trained; template for a "soft beam" ablation. | checked |
| Fully differentiable beam search decoder (Collobert et al.) | ICML 2019 | [arXiv 1902.06022](https://arxiv.org/abs/1902.06022) | Beam decoder trained through inference (speech), removes exposure and label bias; marginalises over alignments. | checked |
| Differentiable top-k with OT (Xie et al.) | NeurIPS 2020 | [proceedings](https://papers.nips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html) | SOFT top-k by entropic OT; applied to kNN and beam search. Costlier than sorting (reviewer note). Candidate for soft top-b / top-M. | checked |
| Sparse seq2seq / entmax (Peters, Niculae, Martins) | ACL 2019 | [arXiv 1905.05702](https://arxiv.org/abs/1905.05702v1) | alpha-entmax (sparsemax at alpha = 2): exact zeros, gradients defined; makes beam search sometimes exact. Basis for the sparsemax ablation of the child weights. | checked |
| Resilient MCL (Letzelter et al.) / Annealed MCL (Perera et al.) | NeurIPS 2023 / NeurIPS 2024 | [arXiv 2311.01052](https://arxiv.org/pdf/2311.01052), [arXiv 2407.15580](https://arxiv.org/abs/2407.15580) | WTA training of K hypotheses + **learned scoring head** (rMCL); annealing fixes WTA's bad local minima. Directly applicable to b children per tip. | checked |
| Differentiable particle filter via entropic OT (Corenflos et al.) | ICML 2021 | [arXiv 2102.07850](https://arxiv.org/pdf/2102.07850) | Resampling (= beam pruning) replaced by a differentiable transport map; biased but consistent. Relevant if the top-M beam must pass gradients to dropped nodes. | checked |
| Learned duplicate removal (Ding et al., GCPR 2022); FCOS-based end-to-end detector (Wang et al., CVPR 2021) | 2022 / 2021 | [GCPR pdf](https://pages.iai.uni-bonn.de/gall_juergen/download/nonmaximumsuppression_gcpr22.pdf), [arXiv 2012.03544](https://arxiv.org/pdf/2012.03544) | GNN rescoring of detections trained end to end replaces NMS; one-to-one assignment + differentiable 3D max filter. Support for stage C (attention over siblings) instead of hand NMS. | checked (search summary) |
| Relation Networks for Object Detection (Hu et al.) | CVPR 2018 | [arXiv 1711.11575](https://arxiv.org/pdf/1711.11575) | Learned duplicate removal via object relation module. | not verified (checked in an earlier session only) |

## Q4. Survival / transmittance along a line; learned verification and cascades

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| L-CNN, End-to-end wireframe parsing (Zhou et al.) | ICCV 2019 | [arXiv 1905.03246](https://arxiv.org/pdf/1905.03246) | Junction proposals -> line sampler (positives/negatives from GT and predicted junctions) -> **line verification** with LoI pooling (features at points along the segment). Closest analogue to our ray verification, but one binary score per segment. | checked |
| HAWP (Xue et al.) | CVPR 2020 | [ar5iv 2003.01663](https://ar5iv.org/html/2003.01663) | Same LoI verification head, better line proposals. | checked (search summary) |
| Nnet-survival (Gensheimer, Narasimhan) / DeepHit (Lee et al.) | PeerJ 2019 / AAAI 2018 (from memory) | [review arXiv 2305.14961](https://arxiv.org/pdf/2305.14961) | Discrete-time hazard networks: per-interval hazard, survival = product of (1 - hazard), NLL with **right censoring**. Our T_k / P(k) is exactly this model. | checked via review summary; formulas from memory |
| Differentiable raycasting for occupancy forecasting (Khurana et al.) | ECCV 2022 | [arXiv 2210.01917](https://arxiv.org/abs/2210.01917) | Occupancy rendered along rays differentiably to lidar sweeps; self-supervised; up to 17 % fewer planner collisions. Ray-accumulation used for driving perception. | checked |
| UniPAD (Yang et al.) / RenderOcc (Pan et al.) | CVPR 2024 / ICRA 2024 (from memory) | [arXiv 2310.08370](https://arxiv.org/pdf/2310.08370), [arXiv 2309.09502](https://arxiv.org/abs/2309.09502v1) | NeRF-style volume rendering (transmittance along rays) as pre-training / 2D supervision of BEV-voxel features on nuScenes. Transmittance used for supervision, not for hypothesis verification. | checked |
| Cascade R-CNN (Cai, Vasconcelos) | CVPR 2018 | [arXiv 1712.00726](https://arxiv.org/pdf/1712.00726) | Stages trained on the previous stage's output with **increasing IoU thresholds**; same cascade at inference. Template for stages A -> B -> C. | checked |
| DINO (Zhang et al.) | ICLR 2023 | [arXiv 2203.03605](https://arxiv.org/pdf/2203.03605) | Contrastive denoising, mixed query selection, **look forward twice** (later layer's loss also refines the earlier layer's box, no detach). | checked |
| Sparse4D v3 (Lin et al.) | arXiv 2023 | [arXiv 2311.11722](https://arxiv.org/pdf/2311.11722) | Adds a **quality estimation** head (calibrated confidence for ranking) and temporal denoising; relevant to calibrating nu. | checked |

## Q5. Adaptive / deformable / curve-aligned sampling

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| Deformable DETR (Zhu et al.) | ICLR 2021 | [arXiv 2010.04159](https://arxiv.org/pdf/2010.04159) | Reference point + learned offsets (2MK channels) + softmax weights from the query; cost independent of map size. Our teardrop = fixed reference pattern; learned offsets are the natural extension. | checked |
| DAB-DETR (Liu et al.) | ICLR 2022 | [arXiv 2201.12329](https://arxiv.org/pdf/2201.12329v4) | Anchor boxes as queries updated per layer; width/height **modulate** the positional attention (oval prior). Model for gain-modulated pattern length/width. | checked |
| Sparse4D v1-v3 (Lin et al.) | 2022-2023 | [v3 2311.11722](https://arxiv.org/pdf/2311.11722) | Instance anchors with fixed + learned keypoints, multi-view sampling; top-k instance propagation over time. | checked (v3); keypoint details from memory |
| LaneATT (Tabelini et al.) | CVPR 2021 | [arXiv 2010.12035](https://arxiv.org/abs/2010.12035v2) | Line anchors pool features along the anchor; anchor attention for global context. Line-aligned sampling. | checked |
| CLRNet (Zheng et al.) | CVPR 2022 | [arXiv 2203.10350](https://arxiv.org/pdf/2203.10350) | Lane priors refined from high to low feature levels; ROIGather samples points along the prior; Line IoU loss over the whole lane. Cross-level refinement = our stage B. | checked |
| LaneSegNet lane attention; TopoBDA Bezier deformable attention | ICLR 2024; 2026 | see Q2 | Deformable reads steered by the lane (reference points along the segment; Bezier control points drive the offsets). | checked (abstracts; sampling details not on page) |
| PolarMask (Xie et al.); Polar R-CNN | CVPR 2020; arXiv 2024 | [arXiv 1909.13226](https://arxiv.org/pdf/1909.13226), [arXiv 2411.01499](https://arxiv.org/pdf/2411.01499) | Ray lengths from a centre regressed per angle bin (PolarMask); polar anchors for lanes. Precedent for "rays from a point with a learned length", but regression, not survival. | checked |

## Q6. Temporal propagation of map elements / lane graphs

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| StreamMapNet (Yuan et al.) | WACV 2024 | [arXiv 2308.12570](https://arxiv.org/pdf/2308.12570) | Highest-confidence queries carried to the next frame with a learned ego-pose transform + GRU BEV fusion; geographically disjoint resplits of nuScenes/AV2. | checked |
| SQD-MapNet (Wang et al.) | ECCV 2024 | [arXiv 2401.09112](https://arxiv.org/pdf/2401.09112) | **Stream query denoising**: noised previous-frame GT fed as stream queries during training; >3 mAP over StreamMapNet on nuScenes; no inference cost. | checked |
| MapTracker (Chen et al.) | ECCV 2024 | [arXiv 2403.15951](https://arxiv.org/pdf/2403.15951) | Mapping as tracking; raster + vector memory latents, strided memory fusion; consistency-aware mAP; 76.1 mAP nuScenes. | checked |
| TopoStreamer (2025); FASTopoWM (2025) | arXiv 2507.00709; 2507.23325 | [2507.00709](https://arxiv.org/abs/2507.00709v1), [2507.23325](https://arxiv.org/html/2507.23325v1) | Temporal lane-**segment topology** on OpenLane-V2: streaming attribute constraints + segment denoising (+1.7 OLS); fast-slow latent world model (OLS 46.3 vs 41.5). Only temporal topology methods found. | checked (search summaries; numbers differ across arXiv versions) |
| Stability under scrutiny (Shan et al.) | arXiv 2025 | [arXiv 2510.10660](https://arxiv.org/abs/2510.10660) | Presence / localisation / shape stability, mAS; accuracy and stability are largely independent. A metric to report for M6. | checked |

## Q7. End-to-end driving consuming map / graph tokens

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| UniAD (Hu et al.) | CVPR 2023 | [arXiv 2212.10156](https://arxiv.org/abs/2212.10156v2) | Planning-oriented stack; map queries pass to motion/planning via query interfaces. Map-ablation numbers not seen this session. | checked (abstract) |
| VAD (Jiang et al.) | ICCV 2023 | [arXiv 2303.12077](https://www.arxiv.org/pdf/2303.12077) | Vectorised map (boundaries, lane vectors with direction) as instance-level **planning constraints**; -29 % collision rate on nuScenes. | checked |
| SparseDrive (Sun et al.) | ICRA 2025 | [arXiv 2405.19620](https://arxiv.org/html/2405.19620v2) | Sparse detection/tracking/mapping instances feed a parallel motion planner with collision-aware rescoring. | checked |
| Ego status (Li et al.) | CVPR 2024 | [arXiv 2312.03031](https://arxiv.org/pdf/2312.03031) | On nuScenes open-loop, an ego-status-only MLP matches SOTA L2 / collision: **perception (incl. map) quality barely shows up** in these metrics. | checked |
| UncAD (Yang et al.) | ICRA 2025 | [arXiv 2504.12826](https://arxiv.org/abs/2504.12826) | Map uncertainty passed to prediction/planning + uncertainty-aware trajectory selection: up to -26 % collision, -42 % drivable-area conflict on nuScenes. | checked |
| MAP (Yin et al.) | ICCVW 2025 | [arXiv 2509.13926](https://arxiv.org/abs/2509.13926) | Plan-enhancing online mapping + ego-status-guided planning; -56 % off-road vs baseline (DAIR-V2X). | checked |
| VADv2 / NAVSIM | 2024 | [NAVSIM 2406.15349](https://arxiv.org/html/2406.15349v1) | Probabilistic planning (VADv2, not verified); NAVSIM non-reactive simulation metrics as a better evaluation than open-loop L2. | NAVSIM checked, VADv2 not verified |

No paper found that feeds an explicit **lane graph with topology** (nodes + adjacency) into a learned planner and back-propagates the planner loss into the graph. LaneGAP motivates paths by planning needs but does not measure it.

## Q8. Bottom-up keypoint / piece linking; fork-merge evaluation

| paper | year / venue | link | relevance | status |
|---|---|---|---|---|
| FOLOLane (Qu et al.) | CVPR 2021 | [arXiv 2105.13680](https://arxiv.org/abs/2105.13680) | Local keypoints + local geometry head that links keypoints of the same lane; bottom-up decoding. | checked |
| LaneAF (Abualsaud et al.) | RA-L 2021 | [arXiv 2103.12040](https://arxiv.org/pdf/2103.12040) | Horizontal / vertical affinity fields group pixels row by row; variable lane count. | checked |
| RCLane (Xu et al.) | ECCV 2022 | [arXiv 2207.09399](https://arxiv.org/pdf/2207.09399) | Relay chains: per foreground pixel, forward/backward transfer + distance maps to step to the next point; distance to both lane ends. Pixel-level "growth" without an agent. | checked |
| GANet (Wang et al.) | CVPR 2022 | [arXiv 2204.07335](https://arxiv.org/pdf/2204.07335) | Each keypoint regresses offset to its lane start; grouping is parallel. | checked |
| Relationformer (Shit et al.) | ECCV 2022 | [ar5iv 2203.10202](https://ar5iv.labs.arxiv.org/html/2203.10202) | One-stage [obj] + [rln] tokens; set prediction of nodes and relation (edge) classification from pairs; road networks among the tasks. Template for the link head. | checked |
| SAM-Road (Hetang et al.) | CVPRW 2024 | [arXiv 2403.16051](https://arxiv.org/pdf/2403.16051) | Vertices from road/intersection masks by NMS, edges by a small transformer GNN on SAM features; RNGDet++-level accuracy, about 40x faster. Strongest evidence that link-based beats growth on speed. | checked |
| InstaGraM (Shin et al.) | CVPRW 2023 | [arXiv 2301.04470](https://arxiv.org/pdf/2301.04470) | Vertices + edge maps, association by GNN (SuperGlue-like); fast vectorised map. | checked |
| LaneGNN (Büchner et al.) | arXiv 2023 (ICRA 2023, from memory) | [arXiv 2302.06175](https://arxiv.org/abs/2302.06175) | Ego-respective successor lane graphs from aerial images by GNN; overlapping predictions aggregated into a global graph (bottom-up merge). | checked |
| LineGraph2Road (Wei et al.) | arXiv 2026 | [arXiv 2602.23290](https://arxiv.org/abs/2602.23290) | Sparse keypoint graph -> line graph -> graph transformer (edges as nodes); TOPO-F1 and APLS. | checked |
| PolyWorld (Zorzi et al.) | CVPR 2022 | - | Vertex detection + optimal-matching connections (polygons). | not verified |

**Fork/merge metrics found.** (i) Landmark P/R and Reachability P/R (RNTR; also used by SeqGrowGraph): landmark points include forks and merges. (ii) TOPO / GEO P/R with a matching radius, APLS (path lengths). (iii) Junction TOPO (JTOPO) restricted to the subgraph around junctions; SDA (split-detection accuracy within a radius). (ii)-(iii) were seen as search summaries of LaneGNN-line and CGNet-line papers; exact definitions not checked. OpenLane-V2 TOP_ll is an adjacency mAP between matched lanes, not a fork/merge localisation metric.

---

## Implications for the design

1. **Exposure bias is the known failure of growers.** iCurb needed DAgger; RNGDet/CenterLineDet use behaviour cloning; TopoGPT reports early-error propagation. Train the fixed-shape tree on its **own** selections from step 1 (free unroll). Optionally also run a GT-seeded "restricted" unroll, like iCurb's restricted + free exploration. Labels stay geometric (matching-free), so no expert is needed.
2. **Treat the b children as multiple-choice hypotheses.** Within one tip, assign each ground-truth branch to the nearest child (winner-takes-all, b = 2, no Hungarian). Give the children a learned score (rMCL). Use annealed WTA in the first epochs so that both slots do not collapse onto one branch (aMCL).
3. **The survival product is a discrete-time hazard model.** Train it with the survival NLL: P(k) for a ray whose last on-lane sample is k, and T_k only (right-censored) for rays that leave the BEV range or enter an occluded/unobserved region. Censored rays must not be treated as "lane ends". This is the one concrete change to the loss in spec section 5.
4. **Use L-CNN as the verification precedent and baseline.** L-CNN/HAWP verify proposed lines from features pooled along them (LoI pooling). Our per-sample cumulative version should be compared against a **single-score LoI-style verifier** on the same rays. If the cumulative version does not beat it on step-length accuracy and on false "hops" to the neighbouring lane (T06), it is not worth its cost.
5. **Use a Cascade R-CNN-style threshold schedule.** Tighten the positive radius from stage to stage (e.g. A 1.0 m, B 0.5 m, C 0.3 m, from the A/B/C labels of spec section 5). Do not detach stage outputs between stages, so the later stage's loss also moves the earlier positions (DINO's look-forward-twice). Add a calibrated quality head for nu (Sparse4D v3), because nu drives the beam and the alive threshold.
6. **Keep the teardrop as a reference pattern and learn bounded offsets.** In Deformable/DAB-DETR style, learn offsets and a length/width gain per ray (the spec's gain in [0.5, 1.5]). In stage B, sample **along the tip -> child chord or a Hermite/Bezier curve**, not a free small teardrop. LaneSegNet, TopoBDA and CLRNet all gain from curve-aligned reads.
7. **Selection with gradient through the selected items is the field's norm.** DETR query selection, Sparse4D and StreamMapNet top-k propagation all work this way. Soft top-k (OT), continuous beam relaxation, entmax and differentiable resampling exist but cost more. Run at most one of them (entmax over a tip's candidates is the cheapest) as an ablation after the hard-selection version works.
8. **Replace hand NMS with learned rescoring where possible.** Stage C attention over siblings, ancestors and committed points is the lane-graph analogue of learned duplicate removal (Ding et al., Relation Networks). Keep the hand dominance rule of the reference only as an inference fallback, and report how often it still fires.
9. **Add an explicit fork/merge (endpoint/junction) head as seeds and keys.** TopoPoint (endpoint detection), CGNet (junction-aware queries) and SeqGrowGraph (split only at junctions) all gain topology from explicit junction points. The reference shows forks only resolve from about 6 deg and merges are located about 7 m late, so a junction head is the cheapest fix.
10. **Temporal recursion (M6).** Train it with **noised previous-frame ground truth** as prior roots first (SQD-MapNet), then switch to the model's own previous output. Propagate only the top-confidence nodes (StreamMapNet). Report stability (presence / localisation / shape, mAS) next to accuracy, because the two are largely independent.
11. **Baselines to compare against.** (a) Our own E94/E120 systems. (b) An L-architecture baseline with a **TopoLogic-style geometric link** (endpoint distance -> learned link probability), which is cheap and can be applied post hoc. (c) A SAM-Road / Relationformer-style "nodes + pairwise edge classifier" on the same BEV. On nuScenes, the external reference points are RNTR / LaneGraph2Seq / SeqGrowGraph (landmark and reachability metrics) and OpenLane-V2 subset B (OLS, TOP_ll). Our labels and range differ, so these are reference points, not the same benchmark.
12. **Fork/merge metric.** Report landmark P/R for vertices with out-degree 2 (forks) and in-degree 2 (merges) at radii 1 / 2 / 4 m (RNTR landmarks + SDA). Add reachability P/R (RNTR) or APLS for paths and JTOPO around junctions. Also report a fork-angle resolution curve (recall vs divergence angle, as in T07); no paper reports one.
13. **Planner metrics will not show map quality on nuScenes open-loop** (ego-status paper). Measure the effect of graph quality by oracle/degraded-graph swaps into the same planner and by route-level metrics, not by L2 alone. Pass the validity pi into the planner (UncAD shows that uncertainty-aware consumption reduces collisions).
14. **Avoid token-sequence decoding** (LaneGraph2Seq, TopoGPT, SeqGrowGraph AR). It needs sampling/argmax at inference, is not differentiable through the graph, and adds latency. Topo2Seq's *training-only* sequence decoder is the one idea worth borrowing: an auxiliary path-order loss that costs nothing at inference.
15. **Radar is untested in this literature.** Only TopoBDA reports radar gains in its abstract, for lane topology. We cannot borrow radar-specific design choices; our radar ablation is novel evidence.

## Gaps (what no paper found here does)

- **A fixed-shape b-ary hypothesis tree with a per-level beam, unrolled in one batched pass and trained end to end** with gradient through the selected children. Growers are imitation-learned agents (RoadTracer, iCurb, RNGDet, CenterLineDet), RL with tree search (Anagnostidis et al.), or token decoders (RNTR, LaneGraph2Seq, SeqGrowGraph, TopoGPT). None back-propagates a geometric or planner loss through the rollout.
- **Step length chosen by verification.** Survival/transmittance along a proposed step decides how far to go (the longest chord that stays on the lane). Growers use fixed step lengths or regress the next vertex (RoadTracer fixed steps, from memory). Line verifiers (L-CNN, HAWP) give one score per proposed segment, not a distribution over the endpoint. Transmittance in driving perception (UniPAD, RenderOcc, differentiable raycasting) is used to render supervision, not to verify a hypothesis.
- **A discrete-time hazard / censoring view of lane continuation.** Unoccluded lane ends vs censored rays at range or occlusion are not modelled anywhere found.
- **Tree -> DAG conversion by clustering + edge splitting inside a learned system.** Merges come from serialisation (SeqGrowGraph, RNTR), graph-model inference (DAGMapper) or global aggregation of overlapping graphs (LaneGNN), not from converging hypotheses of different roots.
- **Recursive log-odds fusion of node validity over time for a lane graph.** Temporal map methods propagate queries or memory latents (StreamMapNet, MapTracker, TopoStreamer); none keeps explicit per-node validity with ego-warped Bayesian-style fusion.
- **A planner loss that shapes lane-graph topology.** End-to-end planners consume map instances (VAD, SparseDrive, UniAD) or map uncertainty (UncAD). No study back-propagates planning into an explicit graph, and none measures the planning effect of fork/merge errors.
- **A fork-resolution analysis** (minimum divergence angle and localisation offset of forks/merges) is reported by no method; the existing metrics (TOPO, APLS, landmark P/R, TOP_ll) do not isolate it.
- **Camera + radar BEV lane graphs** on nuScenes are essentially unstudied (TopoBDA's radar result is the only one seen).
