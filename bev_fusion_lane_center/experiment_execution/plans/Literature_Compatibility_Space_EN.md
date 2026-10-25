# Literature compatibility space for the loop-16 lane-graph builders

Synthesis of the seven cluster files and the evaluation / code inventory of `report_final/text/lit16/` (301 cards, ids GR / TP / VM / MH / VS / PL / SN, cluster notes of EV) against the loop-16 design (verified-ray hypothesis tree G, spray-snap-link L) - 2026-10-23 - reading and writing only: no experiment, no code run, no web access, no download.

> **How to read the ids.** Card ids (GR07, VS12 ...) point to `lit16/<cluster>.md`; "EV B.4" = part B section 4 of `eval_data_code.md`; "TP-notes (g)2" = cluster compatibility notes of the topology file, item (g)2; "VS-notes 3b.4" = section 3b item 4 of the verification file; "MH-notes 5" = section 5 (open questions and gaps) of the multi-hypothesis file; the same pattern holds for GR, TP, VM, PL, SN. K ids are the technique register of `Loop16_Triage_Plan_EN.md` 10.0, M1 - M15 the mixes of `Concept_Hypothesis_Tree_Lane_Graph_EN.md`. Every number below was copied from a card or a cluster section and carries its id; numbers I computed myself are marked "(own count)" or "(derived)". Nothing was re-fetched from the web by me: all paper facts are **not re-checked** beyond what the cards say. Facts about our code come from reading `scripts/p16_builder.py` (504 lines, saved 19:15), `p16_geom.py` (148, 19:14), `p16_loss.py` (283, saved 19:38) read-only on 2026-10-23, plus a look at `p16_train.py`, `p16_infer.py` and `p16_diag.py` for warm start, assembly and diagnostics. **The code is being edited by another session**: my first reading (a 444-line builder without stage B / C, SPF1 - SPF3, MCL flag and assembly) was out of date within the hour, so every "builder today" statement is dated and must be re-checked before it is acted on.

---

## 1. Purpose, scope, depth of reading

**Purpose.** For every technique family of the literature: what it does at which stage of our pipeline, what it needs from its neighbours, what it excludes, whether we can reuse, adapt or only re-implement it, and which mixes of blocks are therefore feasible. The result is a compatibility space that the loop-16 triage (ladder, flips, defaults) can use without re-reading the clusters.

**Depth of the 301 cards** (own count from `cards.csv`; the depth field is free text, so "mixed" = a card where some of its sources were read at method level and others only at abstract level):

| cluster | cards | M | M+C | mixed M/A (+C) | A or A+C | C only | what could not be opened, main limit |
|---|---|---|---|---|---|---|---|
| GR growth | 32 | 18 | 10 | 1 (GR07) | 3 (GR20, GR27, GR29) | 0 | VecRoad PDF-only; Trexplorer base paper only as MICCAI listing; SeqGrowGraph hyper-parameters; Anagnostidis ablation table garbled; PDFs come back as binary |
| TP topology | 31 | 24 | 7 | 0 | 0 | 0 | supplements of CGNet / TopoLogic, configs and bbox coders, SEPT, Chameleon, TSTGT, 2nd / 3rd place reports 2023; released code differs from paper text in several places (TP-notes 5) |
| VM vector map + temporal | 46 | 42 | 4 | 0 | 0 | 0 | SQD-MapNet has no code and an unreadable supplement; mAS page cut at 100 k of 134 k characters; **42 of 46 cards are "partly" verified** (numbers via a small reader model) |
| MH multi-hypothesis | 58 | 45 | 1 | 7 (MH01, MH06, MH14, MH15, MH46, MH55, MH57) | 5 (MH30, MH33, MH48, MH54, MH58) | 0 | original MCL, QCNet paper, Ding GCPR 2022, Kim MHT (PDF / 403); DynamicViT details; code mostly "not checked" |
| VS verification, sampling | 44 | 33 | 4 | 5 (VS04, VS11, VS29, VS41, VS43) | 2 (VS03, VS05) | 0 | Khurana method section (binary PDFs), DeepHit paper text, HAWPv2 / v3, ACT; many code links are "from memory, not opened" |
| PL planner, navigation | 49 | 36 | 10 | 0 | 2 (PL22, PL23) | 1 (PL40) | nuPlan paper, PKL full text, QCNet CVPR paper, PARA-Drive; several pages cut at 100 k characters |
| SN sensors, BEV | 41 | 34 | 1 | 3 (SN13, SN27, SN32) | 3 (SN28, SN34, SN40) | 0 | per-layer IoU of the foundation unpublished; PDFs as binary; SOLOFusion, StreamPETR, GaussianLSS, MapPrior not read |
| EV data, metrics, code | no cards | Parts A - D at method level for metric definitions (OpenLane-V2 old and new devkit, MapTR AP, TOPO / JTOPO, RNTR landmarks, mAS, PDMS) | | | | | nuScenes terms-of-use page unreadable (script-rendered); web.archive blocked; nuPlan score weights from memory |
| **all** | **301** | **232** | **37** | **16** | **15** | **1** | 42 cards have a code page in their depth field (own count) |

**Honest limits.**
1. "Method level" means a reader model returned a digest of the method section, equations and tables; equations are paraphrased and single numbers can be mis-transcribed (GR, MH, VS, VM coverage statements). Where a card flags a text / table inconsistency it is repeated here only if it matters.
2. No experiment was run by the clusters or by me. All gains below are gains **reported by the papers** on their hosts (trainable encoders, Hungarian training, other ranges); none was measured on the cached GaussianCaR pyramid. Transfer is extrapolation (TP-notes 5, SN-notes 5).
3. `cards_stats.md` counts the first token of the annotated depth / differentiable fields, so it differs slightly from the counts here (it lists M 248 = my 232 + 16 mixed). My counts are re-computed from `cards.csv`.
4. Licence statements are what README pages and the GitHub API summary said (EV Part C / D), not legal advice. Several 2026 arXiv ids (26xx) and their self-reported numbers are as written on the cards.
5. The register ids K01 - K146 and the mixes M1 - M15 are another session's finished work; this file only maps onto them.

---

## 2. Taxonomy of building blocks by stage, and the interface contract against OUR pipeline

**Our side, as fixed by the project context and read from the builder today.** Input = frozen cached BEV pyramid at 0.5 m, +-50 m: f0 200 x 200 (64 ch), f1 100 x 100 (96 ch), f2 50 x 50 (128 ch); the builder projects them to 32 / 32 / 16 channels (80 in total) and reads them with bilinear `grid_sample` at ego-metre positions with **border padding** (`p16_builder.samp`). Output = vertices (ego metres), step token (project: 128-d; the builder's `DefaultCfg.d` is 96), validity / path validity, parent edges, flag (growth direction), plus route probability and planner path downstream. Losses are matching-free (`p16_loss.py`: distance fields, downstream ground-truth set, point-to-polyline both ways, validity-weighted repulsion). Environment: WSL `gcar`, Python 3.12, PyTorch 2.8, no mmcv / mmdet / mmdet3d, RTX 3060 12 GB.

**Stage table.** n = primary stage of the card (first code in the stage field) / any mention (own count from `cards.csv`; the 301 primaries add up).

| stage | n | what the blocks of the literature do (cards) | literature in -> out | contract with our pipeline |
|---|---|---|---|---|
| B encoder, fusion | 33 / 51 | LSS-family lift plus radar raster or pillars (SN04 - SN08, SN25, SN27, SN35, SN36); Gaussian lift (SN01, SN08); query encoders (SN22); dense map decoders (SN15, SN20, SN21); lidar-teacher distillation (SN31) | images + radar -> BEV feature (200 x 200 at 0.5 m is the dense-segmentation convention, SN-notes (a)) | **only SN01 is ours** (frozen, Apache-2.0 repo, non-commercial vendored rasteriser); everything else is idea-level. Possible extras without retraining: native fused 4-scale pyramid, alpha coverage (SN01, SN-notes (a)); PCA to 32 ch discards unknown information |
| D data, visibility | 9 / 14 | piece graph from map expansion (TP29, EV A.1); visibility masks (SN10, SN11, SN39); calibration-only mask (SN13, derived); radar facts (SN14); leakage-free splits (SN03, VM46) | map layers + sensors -> labels + masks | our labels = nuScenes lane / connector pieces in travel direction, scene-disjoint split; no visibility mask exists; all published masks need lidar except SN13 (derived, geometric) |
| S1 seeds | 9 / 16 | heat-map peaks (GR01 - GR03, VM38), skeleton end points (GR05), start poses (GR08), endpoint detector with incidence (TP02), keypoint queries (TP05), polar poles (VS28), start-point proposals (VS33), previous-frame dead ends (GR02) | heat map -> point set (positions, headings) | direct: lattice 5 m (jittered in training) or 128 anchors, learned snap and direction (`seed_snap`, `seed_dir`), K0 = 64 after NMS 3 m. VS28 (every pole proposes its nearest lane) is the closest published form of "no peak picking" |
| S2 sampling, readers | 22 / 33 | reference points + offsets + weights (VS20, VS23, VS25), lane / curve attention (TP10, VS31, VS32, TP15), line pooling (VS26, VS27), window lookup (VS21), keypoint pattern scaled by instance size (VS24), row anchors (VS33, VS34), yaw fans (VS36), polar rays (VS39), FPN level rule (VS42) | feature map + hypothesis -> sampled feature vector | direct: one `grid_sample` per ring point; teardrop = 12 rays x <= 5 rings. Image-plane parts (VS36 - VS38, VS40) lose their projection; row-wise parametrisations (VS33, VS34) do not apply (VS-notes 3b.6) |
| S3 verification | 16 / 26 | hazard / survival with censoring (VS01 - VS04), ray compositing (VS06, VS09), line verifiers (VS12 - VS15, VS17), observability masks (VS43) | samples along a line -> score or endpoint law | direct for the algebra (nu_k, T_k, P(k)); **the survival-with-censoring combination is unprecedented in perception** (VS-notes 5); line verifiers are one-score baselines |
| S4 generation, selection | 53 / 77 | per-step agents with DETR slots or one step (GR01 - GR08, GR19, GR32), token growth (GR13 - GR16), WTA / anchor slots (MH01 - MH22), duplicate removal (MH31 - MH33), differentiable selection (MH34 - MH48), beam-aware training (MH41, MH42, MH49, MH50), trees (MH55 - MH58, PL07) | hypothesis set {(h_k, s_k)} in a metric frame | direct for set shape and scores; training rules need the adapters of section 3; **no paper trains a b-ary geometric tree with a beam through the selection** (MH-notes 3.0 item 5) |
| S5 refinement, cascade | 7 / 35 | cascade thresholds (VS18), look-forward-twice (VS19), residual updates (VS21, VS22), position refinement by test-time optimisation (GR12), diffusion refiners (GR31, TP21, VM43) | hypothesis + features -> corrected hypothesis | direct for VS18, VS19, VS21; diffusion refiners are compute-bound (conflict). Our E120-type revisit exists; stage B (`refine_children`: centre + 8 points at 1.5 m around each child, "no detach between the stages") and stage C (`rescore`: attention over parent and siblings, validity logit + delta) are coded behind the flags `stage_b` / `stage_c` (19:15 version) |
| S6 topology, linking | 18 / 37 | geometric endpoint prior (TP01, TP13, TP14), endpoint tokens (TP02, TP05), connected-pair objects (TP06), corruption-trained re-scorer (TP07), pairwise heads on nodes (GR10, TP24, TP30), lane-graph convolutions (PL35, PL37) | pieces / nodes -> adjacency scores | adapter: pieces vs vertices via incidence matrices (TP-notes (a)); our L-head (`link_head`) is a TopoLogic-shaped scorer with a verified connector; learned link heads were within 0.01 of rules in E42 - E56 |
| S7 assembly | 2 / 10 | path-wise graph and merge (TP04), adjacency expansion (TP23), DOGE operators (GR12), closure by proximity (GR23), loop guards (GR04), sub-graph aggregation (GR08) | paths / proposals -> merged graph | rule-based in all papers (merge radius, graph-distance guard); tree -> DAG lives in `p16_infer.assemble` (inference-side numpy: path union in descending path validity, snap onto an existing edge within 1.0 m = edge split, reuse a vertex within 1.5 m, then bubble clean-up 2.5 m and spur removal < 5 m); its docstring records that the legacy cluster merge (1.5 m, 30 degrees) produced a mesh of spurious forks and merges on D3 (split precision 0.006) |
| S8 time | 24 / 43 | query / instance propagation (VM01, VM03, VM06 - VM09, VM11), BEV memories (VM16 - VM21), raster memories (VM13 - VM15), log-odds on grids (VM12), set carry-over (GR02, GR30), encoder-side fusion (SN33) | previous set / features + pose -> prior | adapter: ego warp of vertices and tokens (VM01, VM04, VM08); **no vertex-level log-odds validity memory exists** (VM-notes 5); global-pose memories conflict with "no pose given" (VM K2) |
| S9 route, SD map | 8 / 12 | hard SD-road to centre-line assignment after perception (PL01), SD priors in perception (PL13, PL41 - PL44, TP17), differentiable localisation (PL46) | SD vectors / raster + perceived lanes -> route attribute | adapter: ours is soft route probability per vertex by cross-attention to SD raster tokens (E104b); PL01 is the only attribution-after-perception precedent and is hard and road-level |
| S10 planner interface | 30 / 38 | polygon tokens with route flag (PL03), route-filtered reference lines (PL04), lane tensors (PL05), instance queries (PL08 - PL10, PL30), graph convolutions (PL35, PL37), rule-based centre-line planner (PL02), uncertainty into planning (PL09, PL13, PL15), dense cost volumes (PL34) | lane polylines + attributes + route -> trajectory | adapters A1 - A6 of PL-notes 3b (graph -> 20-point pieces, route flag, validity mask or log-confidence bias); traffic lights, boundaries, speed limits absent (PL-notes C1, C2) |
| L losses, labels | 39 / 91 | WTA family (MH01 - MH08), nearest-slot labels (MH10 - MH13, MH16), quality-argmax labels (MH19, MH20, MH29), one-to-many auxiliaries (MH24 - MH26, MH28, TP12), duplicate labels (MH31), soft targets (VM34), raster losses (VM32, VM33), point-to-polyline relatives (VM26, GR26), consistency losses (VM23, VM24) | prediction + GT -> loss without or with assignment | direct for everything that needs only a distance per (hypothesis, GT element) (MH-notes 3.1); Hungarian variants conflict |
| T training regime | 19 / 62 | DAgger and dynamic labels (GR03, GR04, GR27), scheduled sampling (MH51), beam-aware (MH49, MH50), one-mode closed loop (MH53), denoising (VM02, VM05, VM06), chained training (GR07), LP-FT and adapters (SN28, SN29) | regime on top of a loss | direct (our labels are defined on arbitrary states); **free roll-out from step 1 with several hypotheses has no precedent** (MH-notes 3.0 item 4) |
| E metrics, protocols | 12 / 37 | OpenLane-V2 DET_l / TOP_ll (TP29, EV B.1), landmark / reachability (GR28, EV B.4), TOPO / JTOPO / SDA / APLS (EV B.3), stability (VM44, VM45), planner-centric swaps (PL23, PL24), PDM-type scores (PL20, PL21) | graph or instances -> number | adapters G1 - G6 of EV B.8 on our vertices / edges / validity; section 4(k) |
| X other | 0 / 11 | synthesis cards and context | - | - |

**Contract summary.** What flows between stages in the literature is always a *set of hypotheses with a score in one metric frame* (MH-notes 3.1). Our contract is the same set plus a parent pointer and a 128-d token, so the stage interfaces S2 -> S3 -> S4 -> S5 are direct. The three places where the contract needs real adapters are (i) the **loss side** (Hungarian-based recipes and score semantics, conflicts 1 and 4 of section 3c), (ii) **S6 / S7** (piece graph vs vertex graph, merge rules) and (iii) **S8 / S9 / S10** (identity, pose and route formats).

---

## 3. Compatibility matrix: our stages x technique families

Legend: **D** direct (works on our interface as published) - **A** adapter (named) - **C** conflict with reason tag (diff = differentiability, reg = regime / training rule, int = interface, sen = sensor, cmp = compute or environment, geom = geometry) - **U** unknown, nothing found - **-** not applicable. Derived from the `compat_needs` / `compat_conflicts` fields of the cards and the cluster notes; a letter pair means "D for part, A for the rest".

Families (columns): **G** growth agents (GR01 - GR08, GR19, GR20, GR32, MH55, MH56) - **Q** token / sequence decoders (GR13 - GR16, GR18, GR22, GR23, TP22, TP23, VM31) - **H** Hungarian set / query decoders (TP02 - TP15 hosts, VM26, VM27, VM40, MH23 - MH26, MH28, VS16, VS22, SN16, SN17) - **K** link heads and geometric priors (TP01, TP07 - TP09, TP12 - TP14, TP24, TP25, TP30, GR10, PL35) - **V** survival, ray and line verification (VS01 - VS17, VS43, VS44) - **M** multi-hypothesis training and scoring (MH01 - MH22, MH29 - MH33) - **B** differentiable selection and beam training (MH34 - MH54) - **R** readers, cascades, refiners (VS18 - VS42, TP10, TP15, VM39, GR12) - **T** temporal (VM01 - VM25, TP19, TP20, TP31, GR30, SN33) - **P** route, SD map, planner (PL01 - PL49, TP17).

### 3a. Columns G, Q, H, K, V

| stage | G | Q | H | K | V |
|---|---|---|---|---|---|
| B / D cached pyramid, scene-disjoint | D: step heads read BEV heat maps (GR02, SN19) | C diff: frozen-encoder regime exists (SN02) but tokens need argmax | A: 39 -> 256 ch adapter, crop 200 x 100 (SN16, SN17, SN37); C reg: Hungarian | D for the BEV-only lane branch (SN37); C int: perspective parts (TP08, TP27) | D: needs only `grid_sample` (VS-notes 3g); C int: trainable 3-D volumes (VS08, VS10) |
| S1 seeds | D: peaks (GR01 - GR03), skeleton ends (GR05); A: DeH4R candidate masks (GR06) | - | A: ordinal keypoint tokens with matching-free denoising adjacency (TP05); C reg: vanilla branch | A: junction / endpoint heat map as roots (TP03 JAQ is a heat-map branch, TP02) | A: every pole proposes its nearest lane, then verify (VS28) |
| S2 readers | D: fan of bilinear samples (E94; GR06 l x l grid) | - | A: reference point + offsets + weights = pattern + learned offsets (VS20, VS31) | A: distance-biased attention between vertex tokens (TP13, TP14) | D: shared rings compare ring by ring (VS-notes 3e); common radii have no precedent |
| S3 verification | A: stop / validity heads (GR01, GR04) are event classes, replace by survival | C diff: premature EOS is the main failure (GR-notes (e)8) | - | A: verified connector instead of one score (L head); VS12 is the one-score baseline | D: algebra VS01 - VS04, baselines VS12 - VS15; C reg: single-event assumption (VS01); U: calibration of products (VS-notes 5) |
| S4 children, beam | A: b children; slots need Hungarian (GR01, GR06, GR07) -> nearest / sector labels (GR04, GR09); beam untested (GR-notes (e)7) | C diff | C reg: per-tip Hungarian (GR06, GR07, MH55); A: sector slots (GR09) | A: geometry gates which endpoints may connect (TP01) | D: endpoint law P(k) gives the pool; U: no top-b from a survival law found (VS-notes 5) |
| S5 refine, cascade | A: chained multi-step training (GR07, Hungarian), short unroll (GR05) | - | C reg: main queries of DINO / Sparse R-CNN (VS19, VS22); denoising part direct | - | A: re-verify after each stage, thresholds tighten (VS18) |
| S6 link | A: expansion from dangling ends (GR06), sub-graph step (GR08) | C diff: adjacency inside tokens (TP23) | C reg: labels re-indexed by the assignment (TP08, TP09); A: pair heads on vertices | D: TP01 mapping, TP13 contrast; geometry priors replace each other (TP-notes (e)2) | A: connector samples (GR10: SAM-Road++, LineGraph2Road); U: against a per-sample product |
| S7 assembly | A: radius merge + graph-distance guard (GR04, GR06, GR23), DOGE operators (GR12) | - | A: Path2Graph merge is a rule on top of any host (TP04) | A: soft merge weight with learned bandwidth (TP01); U: bandwidth at 0.5 m vertex error | - |
| S8 time | A: previous dead ends as seeds (GR02), border hand-over (GR30) | - | C reg: temporal parts portable, Hungarian main loss is not (VM K1) | A: stream adjacency (VM09) | U: no vertex-level log-odds validity memory (VM-notes 5); grid-level exists (VM12) |
| S9 route | - | - | - | A: path-aware attention over root-to-leaf paths (PL01); needs the DAG paths | - |
| S10 planner | - | - | C reg: instance-query planners sit on matched map heads (PL08 - PL10, PL30); A: pooling (PL-notes A5) | D: our edges = LaneGCN pre / suc (PL35) | - |
| L losses, labels | D: dynamic labels from any state (GR03, GR04) | C diff: CE on quantised tokens; training-only use (GR17) | C reg: assignment costs; A: nearest / threshold rules (VM-notes 3(b) A3) | A: labels = endpoint coincidence, geometry decides (TP-notes (d)); corruption labels (TP07) | A: conditional (masked) hazard labels (VS01) |
| T regime | D: DAgger, restricted / free switch (GR03) | C diff: only retraining on own outputs (GR18) | C reg | A: simulated corruptions (TP07) | U: free roll-out with a survival read-out has no precedent |

### 3b. Columns M, B, R, T, P

| stage | M | B | R | T | P |
|---|---|---|---|---|---|
| B / D cached pyramid | D: heads read any features (MH16 samples BEV at coordinates) | D: acts on scores only | A: pure-PyTorch deformable core (VS20, EV C.4); C int: polar lifts (VS40) | D head-side set recursion; C int: encoder-side fusion needs a trainable encoder (SN33) | D: SD raster tokens (PL42); C sen / int: satellite + GPS pose (PL41) |
| S1 seeds | A: k-means anchors in the tip frame (MH12, MH16) = the `anchors` option | D: seed NMS as hard selection with local labels | A: non-uniform yaw fan (VS36), polar poles (VS28) | A: stored predictions or dead ends as roots (VM03, GR02) | - |
| S2 readers | D: bearings x ranges are the anchor set (MH10, MH12) | - | D: bounded offsets (VS24, VS25), gain modulation (VS23); RAFT window is isotropic (VS21) | - | D: sample BEV along the planned path (PL29) |
| S3 verification | A: WTA scores mean cell mass, not "on a lane": add a validity head (MH01, MH29) | - | A: per-stage radius schedule on verification targets (VS18) | A: observability flag as input (SN13, VS43) | - |
| S4 children, beam | A: nearest-child WTA + BCE score (MH01); two branches need ordered assignment (own inference, MH-notes 3.4); U: b = 2 evidence | D: hard top-M + local labels + survivor weights; C geom: soft / OT selection blends positions (MH41, MH44) | A: sibling rescoring (MH26 diagnosis, MH31) | A: carry top-validity nodes (VM01, about 30 %) | A: pre-generated tree + learned cost (MH57, PL07) |
| S5 refine, cascade | D: detached proposals as anchors (MH14) | A: gold-in-beam reset (MH49, MH50) | D: VS18, VS19, VS21, VS26; C cmp: GR12 runs 300 iterations per image | - | - |
| S6 link | A: valid-pairs-only / one-to-many positives live inside the assignment (TP12): label-density idea only | - | A: attention-derived links (TP25); U | A: stream adjacency (VM09) | D: pre / suc edges (PL35), typed edges (PL37) |
| S7 assembly | A: learned duplicate removal, best-per-GT label (MH31 - MH33) | A: merging equal states by logadd needs a clustering rule (MH42) | A: merge / split / prune operators (GR12) | - | A: de-duplication 0.5 m in route DFS (PL04) |
| S8 time | A: stream-query denoising, stored predictions as prior (VM02, VM03) | A: confidence decay of carried instances (VM06: x 0.6) | A: pose-conditioned token update (VM01, VM04, VM08) | D: the family itself; C reg: global-pose memories (VM13, VM19, VM03, VM12) | U: carry route state |
| S9 route | - | - | - | - | A: hard -> soft (PL01); log-confidence bias D (PL13); localisation volume costs memory (PL46) |
| S10 planner | A: path candidates by anchors / vocabulary (PL25, PL27) | - | - | - | D / A: formats A1 - A6 (PL-notes 3b); C sen: boundaries, lights, speed limits absent; U: gradient into topology (E121 null, no precedent) |
| L losses, labels | D: needs only a distance per (hypothesis, GT element) (MH-notes 3.1) | A: gold prefix = nearest-child sequence (MH49) | D: stage-wise positives (VS18) | D: raster consistency after warp, matching-free form (VM23, VM24) | A: hinge constraints (PL08), uncertainty (PL09) |
| T regime | U: free roll-out x multi-hypothesis; MH53 executes one mode and detaches | A: scheduled sampling, BSO ramps (MH41, MH49, MH51) | A: credit feedback by look-forward-twice (VS19) | A: dual-mode training, noised GT, own predictions (VM02, VM03) | A: train planner on out-of-fold graphs (plan data hygiene, SN34) |

### 3c. The 15 most important conflicts

| # | conflict | cards | class | what survives |
|---|---|---|---|---|
| 1 | Hungarian assignment inside set / query decoders | TP02 - TP15 hosts, VM26, VM27, VM40, MH23 - MH26, MH28, GR01, GR06, GR07, MH55, VS16, VS22, PL10, PL11, PL30 | reg | head formulas, priors, label-density ideas; not the training recipe |
| 2 | token / sequence decoders: tokenisation and argmax | GR13 - GR16, GR18, GR22, TP22, TP23, VM31 | diff | training-only auxiliaries (GR17; TP16 disagrees, section 9) and the landmark metrics (EV B.4) |
| 3 | soft / OT selection blends node positions; entmax gives exact zeros and one winner per tip | MH41, MH44, MH35 | diff, geom | weight-carrying survivors (MH45, MH47); entmax only inside a tip with a two-hot target |
| 4 | score semantics differ: WTA responsibility (MH01) vs existence validity vs path score | MH01, MH29, MH41, MH42, MH49 (MH-notes 3.1) | reg | separate heads; a fork has two true children, so normalised scores need a two-hot target |
| 5 | free roll-out from step 1 with several hypotheses has no precedent; detaching cuts the path to early children | MH41, MH49, MH51, MH17, MH53, MH14 | reg | warm start / ramp; one-mode rollout with detach (MH53) |
| 6 | single-event survival model vs dashes, occlusion, crossings; border or zero padding reads "no lane" at the raster edge | VS01, VS-notes 3b.1 - 3b.4 | reg | masked chain with a censoring / observability flag |
| 7 | perspective-image components | TP08, TP27, SN22, SN26, VS36 - VS38, VS40 | int, sen | anchor / query / loss ideas only |
| 8 | gains that need a trainable encoder | TP16, TP11 (36.1 vs 34.1 mAP with / without gradient into the encoder), TP21, SN29 | int | head-side modulation only; LP-FT order if ever unfrozen (SN28) |
| 9 | lidar-dependent labels, masks, teachers | VS43, SN10, SN11, SN31, SN24 | sen | SN13 (calibration-only mask, derived) |
| 10 | global-pose memories vs "no pose given" | VM13, VM19, VM03, VM12, PL45 | reg | rolling local memory from inter-frame ego motion (VM K2) |
| 11 | mm-stack repositories, PyTorch 1.9, custom CUDA ops | every lane-topology and vector-map repo (EV finding 5; TP-notes (b)4) | cmp | read and re-implement; pure-PyTorch cores |
| 12 | GPL-3.0 and no-licence repositories | RNGDet++, CenterLineDet, iCurb / Topo-boundary, StreamMapNet, SQD-MapNet, InstaGraM; TopoLogic, TopoPoint, TopoFG, UniTopo, TopoStreamer, SeqGrowGraph, DeH4R, aMCL, PlanTF, PLUTO, Diffusion Planner (EV Part D) | legal | idea level; code stays read-only |
| 13 | track identity undefined at forks / merges; fixed query count and 20 points per element vs a variable DAG | VM04, VM14, VM15, VM45, VM26, VM40 (VM K5, K7) | int | identity by geometry (GT pairing), not by inheritance |
| 14 | row-wise and single-curve parametrisations | UFLD, CondLaneNet, LaneATT, CLRNet offsets, BezierLaneNet (VS33, VS34, VS27, VS26, VS30); cubic Bezier cannot branch (VS29) | int | pooling along the hypothesis; Bezier only along one chord |
| 15 | RL, policy gradient, sampling; test-time optimisation | GR11, MH39, MH40, MH52; GR12 (300 iterations per image), TP21 (45 U-Net passes), VM43 | diff, cmp | deterministic expectation; DOGE operators as rules |

### 3d. The 15 most useful adapters

| # | adapter | cards | connects | note |
|---|---|---|---|---|
| 1 | query <-> bilinear sample (reference point + offsets + weights = pattern + learned offsets, `grid_sample` at ego metres) | VS20, VS31, VS32, TP10, VM01, VM39, EV C.4 | every query-based reader to our pyramid | pure-PyTorch deformable core exists (`ms_deform_attn_core_pytorch`); offsets are in level cells (0.5 / 1 / 2 m) |
| 2 | Hungarian -> matching-free label (nearest hypothesis, nearest slot, quality-argmax per GT, threshold rule, identity for noised queries, point-to-polyline) | MH01, MH02, MH10, MH12, MH16, MH19, MH20, MH29, VM02, TP05, VM38, E120 | all set-prediction recipes to our losses | VM-notes 3(b) A3, MH-notes 3.2 |
| 3 | survival algebra: nu_k = 1 - h_k, T_k, endpoint law P(k), per-metre hazard for unequal gaps, expected loss over P(k) | VS01, VS04, VS06 | rings 4 / 8 / 12 / 18 / 24 m to NeRF / PonderNet / pycox | VS-notes 3a |
| 4 | censoring / observability: in-range flag, calibration-only mask, mask the loss and keep the chain | SN13, VS43, SN10, SN39 | rays leaving the raster or unseen areas to the hazard loss | no occluder mask exists for camera + radar (VS-notes 5) |
| 5 | soft <-> hard selection: soft training with annealed eps and hard inference, or hard forward with survivor weights | MH34, MH41, MH45, MH47 | beam cut to gradient | carry weights, never blend positions |
| 6 | normalised <-> independent scores | MH16, MH29, MH35 | slot scores to validity | sigmoid + multi-hot, or Fenchel-Young + two-hot |
| 7 | geometric prior -> vertex graph: TopoLogic mapping as attention bias or soft merge weight, bandwidth re-fitted | TP01, TP02, TP13, TP14 | link / merge to our vertices | released bandwidth: half value at about 2.8 m (TP-notes (g)2) |
| 8 | polyline set <-> lane graph: Graph2Path / Path2Graph, arc-length resampling, piece <-> vertex incidence | TP04, VM27, TP23, TP-notes (a) | paths and pieces to the DAG | merge radius must be set relative to vertex spacing and error (TP-notes (g)1) |
| 9 | pieces for external metrics: cut at junctions, confidence = min validity, resample to 11 / 20 / 100 points | EV B.8 G2, G3 | our graph to OLS / TOP / MapTR AP | fix the cut rule and confidence once; they change the numbers |
| 10 | ego warp + token update: MLP over [token, T], motion-aware layer norm, bilinear raster warp | VM01, VM04, VM08, VM16, VM17 | previous set to current frame | condition on the RELATIVE pose (TP20) |
| 11 | vertex set <-> raster: soft rasteriser exp(-D / tau) | VM32, VM33, VM13, VM14, VM23, VM24 | raster memory and consistency loss | pure PyTorch at 200 x 200 |
| 12 | graph -> planner tokens: 20-point pieces, on-route flag from route probability, validity mask or log-confidence bias | PL03, PL05, PL13, PL-notes 3b A1 - A3 | our output to polygon / lane-tensor consumers | lane-level to roadblock-level is easy, the reverse is not |
| 13 | route attribution: path-aware attention over root-to-leaf paths of the DAG; route-filtered DFS | PL01, PL04, PL03 | S9 to S10 | needs the DAG; PL01 is hard and road-level |
| 14 | BEV window and channel adapters (crop 200 x 100 or 120 x 60, 39 -> 256 ch) | SN16, SN17, SN-notes (a) | query-head baselines on our cache | no resampling needed for the 200 x 100 crop |
| 15 | Bezier <-> polyline (Bernstein matrix, pseudo-inverse, pinned end points) | VS30, TP-notes (b)1, VS29 | connectors (E53) and curve reads | fine for one chord, not for branching |

---

## 4. What the literature says about our open design questions

Each item: what is published (with ids), what the builder does today (as read), and a **Consequence for loop 16** line.

### (a) Hazard / survival verification along a ray: masking, censoring

**Published.** The algebra appears under four names: logistic hazard with right censoring (VS01), recurrent hazard (VS02), halting probability with an expected loss (VS04), volume-rendering weights (VS06). With nu_j = P(sample j on the lane | j-1 was), T_k = prod nu_j and P(last verified = k) = T_k (1 - nu_{k+1}) they coincide (VS-notes 3a). Rules that matter: (1) the label of sample k is *conditional*: BCE only while all earlier samples were alive (pycox cumulative BCE up to the event or censoring index, VS01); a marginal "near a lane" label per sample is the wrong object for a product (VS-notes 3b.1). (2) The censoring-interval convention differs: Nnet-survival credits the interval only if censoring falls in its second half, logistic hazard counts it as survived, the PMF form excludes it, PonderNet gives the remainder to the last step (VS01, VS04; VS-notes 3b.2). (3) Occlusion is not range: mask the hazard loss on unobserved samples, keep the chain, give an observability flag as input (own derivation, VS-notes 3b.3; Occ3D evaluates only observed voxels, PanopticBEV predicts an occlusion class, Anchor3DLane / LATR train per-point visibility, VS43). (4) Zero padding at the map edge reads "no lane", so a range flag is needed (VS-notes 3b.4). (5) Unequal gaps (4 / 4 / 4 / 6 / 6 m) are consistent under a per-metre hazard nu = exp(-sigma delta) (VS-notes 3a). (6) PonderNet's expected loss sum_k P(k) l_k + beta KL(P || geometric prior), beta = 0.01 there, gives every candidate endpoint a gradient without argmax (VS04); the Mip-NeRF 360 distortion term (weight 0.01) compacts the endpoint law but also penalises true bimodality (VS07). Calibration: with nu = 0.95 per ring the product at ring 5 is 0.77, calibration of P(k) is untested anywhere read (VS-notes 5); D-calibration (VS05, abstract only) is the only evaluation idea. **No perception paper trains a hazard along a ray with censoring** (VS-notes 5); the nearest are per-row existence in CondLaneNet (VS33), the no-lane cell of UFLD (VS34) and termination heads (VS44).

**Builder today.** `p16_loss.level_losses(hazard=True)` multiplies the BCE weight by the cumulative product of the *label* shifted by one: samples after the first label failure get weight 0, the failing sample itself counts. That is the VS01 / pycox "event interval counted" convention. `p16_diag` already reports an ECE of the path validity against "within 1 m of a lane" (5 bins). Not in the code as read: (i) a censoring flag - samples beyond +-50 m are read with border padding (`p16_builder.samp`) for features and labels alike, so they copy the border cell and are labelled by it instead of being censored; (ii) the mask ignores `tolerant`: with `tolerant=True` the read-out may skip one sample, while the loss never trains samples behind a failure (to be checked).

**Consequence for loop 16.** Keep the strict masked BCE (it matches VS01). Before the SPF3 flip (reach 24 m vs 12 m) add an in-range multiplier on the loss weight and read out-of-range rings as "alive": a tip within about 24 m of the raster edge (derived) sends forward rays out of the raster. Make the tolerant variant's mask consistent or drop it. Extend the existing path-validity ECE by a per-ring ECE of nu_k (the product of five conditionals is the quantity the beam ranks by). Use the per-metre form if the ring set changes between flips (SPF3 now changes it through the `reach` switch). VS04's expected loss is an optional term; VS07 only if the endpoint law turns out bimodal.

### (b) Detach versus gradient between refinement stages

**Published.** Everyone detaches the geometry that picks the next read: Deformable DETR blocks the gradient at the previous box (refinement 45.4 vs 43.8 AP, the cut itself not ablated), Sparse R-CNN blocks box gradients (6 stages 42.3 vs 3 stages 39.9 AP), RAFT detaches `coords1` every iteration and still improves with more updates at test (8 -> 32 updates: 1.61 -> 1.43 Sintel clean EPE, no divergence to 200), CLRNet detaches priors before pooling (3 stages 54.74 vs 1 stage 52.80 mF1), Polar R-CNN stops the one-to-one branch gradient (71.68 -> 80.81 F1@50) (VS-notes 3d, VS19, VS21, VS26). Only DINO feeds credit back: look-forward-twice supervises layer i through layer i+1's update of the *undetached* previous box (+0.4 AP, 47.0 -> 47.4), while the box passed on is detached (VS19). QCNet detaches proposals before refinement (MH14); the multi-hypothesis closed-loop paper cuts gradients at executed transitions (MH53). Cascade thresholds: iterative single regressor 35.4 vs cascade 38.9 AP (VS18). **No lane or polyline evidence for full no-detach on point positions** (VS-notes 5).

**Builder today.** Child positions keep their gradient into the next level's sampling coordinates; selection indices are detached; the path record `ppos` is detached; warm-start states are stop-gradient; stage B refines the undetached child (its comment: "no detach between the stages") and stage C rescores with a residual on the validity logit.

**Consequence for loop 16.** The default "no detach" of the concept is backed by literature only in the DINO form (+0.4 AP). Add the RAFT form (stop gradient on coordinates, keep token gradient) as a cheap flip, log the gradient norm per level in gate G2, and fall back to the loss-side look-forward-twice form if norms explode.

### (c) b children per tip, idle second hypotheses, duplicate labels

**Published.** Matching-free patterns (MH-notes 3.4 a): nearest-hypothesis WTA with a BCE score head (rMCL, MH01), softmin with cooling (aMCL, MH02), catch-up for the farthest member (MH03), eps floor 0.05 (MH04), k winners shrinking to 1 (MH05), nearest fixed anchor with CE / BCE (MH10, MH12, MH16), quality-argmax per GT (MH19, MH20), policy over successor edges (MH17). The named failure is starvation: members without training signal (MH03), dead hypotheses and centroid bias (MH04), a toy collapse to 1 - 2 modes (MH10), collapse for some seeds (MH02). All remedies were shown with at least 3 hypotheses on regression problems, **none for b = 2 with mostly single-GT inputs** (MH-notes 3.0 item 1, section 5 item 2). Diversity repulsion destabilises if active from the start and is used in 10 fine-tuning epochs (MH08). aMCL's schedule is T0 = 0.5, rho = 0.95 per epoch over 1000 epochs (MH-notes 3.8). Scores mean responsibility; existence-and-quality targets are separate: Align-DETR t = exp(-r/tau) p^a u^(1-a) (MH29), POTO alpha 0.8 (MH19); duplicate labels: among the detections of one GT the best-scored is positive, the rest are duplicates (MH31; duplicate removal 29.6 NMS, 30.2 Soft-NMS, 30.5 learned, 31.0 end to end mAP). Number of positives has a sweet spot: RATopo positives per lane 1 / 2 / 3 / 4 / 5 give TOP_ll 30.4 / 32.2 / 32.2 / 31.7 / 30.8, naive one-to-many 27.0 (TP12); Co-DETR K = 6 is worse than K = 2 (MH25). Static intention queries beat learnable ones (0.3059 vs 0.2633 mAP, MH12). A straight lane lies on the border of two sectors, so slot labels need overlap or soft targets (MH-notes 3.4, own inference).

**Builder today.** Children come from the ray pool by validity x direction prior after a dominance rule and greedy NMS (2.0 m, segment 1.2 m); by default the position loss pulls *every* kept child to the downstream GT set (no assignment); the validity target is 1 for any child within 0.8 m with the chord on a lane. This is form A1 of MH-notes 3.4 (a pool with validity and duplicate suppression, b as a cap), not the WTA form A2. The HF3 switch `mcl` makes the child nearest to the downstream set carry the position loss and the others a share eps = 0.1: that is **relaxed WTA (MH04)**, not rMCL / aMCL - there is no cell-mass score head and no cooling. A duplicate label appeared in `p16_loss` while this file was being written (flag `dup_label`, default off in `level_losses`): a child that lies within 1.5 m of the chord of a longer valid sibling is a negative. It is a geometric, label-stable cousin of MH31's rule (MH31 picks the best-*scored* detection per GT, so its labels drift with the current scores, MH-notes 3.1) and it covers siblings of one tip only; duplicates across tips are left to `dup_geo` (validity-weighted repulsion under 1.5 m) and to the assembly.

**Consequence for loop 16.** A1 is already coded and is the safer choice; keep WTA out of the default at b = 2 (no evidence). The `mcl` arm is MH04 with a larger eps than the paper's 0.05 and MH04 / MH02 name its cost (ghost predictions from centroid bias, "permanent" bias); cheap upgrades are cooling eps -> 0 within the run (the aMCL idea, schedule compressed from 1000 epochs to unit-run steps) or the catch-up term (MH03). Replace the binary 0.8 m validity target by a closeness target (MH29, VM34), cap positives per branch at 2 - 3 (TP12), extend the coded same-tip duplicate label to the candidates of the beam cut across tips (MH31 / MH32 rank-embedding rescorer), and log an idle-child diagnostic: share of fork tips whose second child has validity < 0.1 (spec test T27).

### (d) Hard top-M beam versus soft selection

**Published.** Hard top-k with a local label per candidate needs no gradient to dropped nodes (all detectors, MH-notes 3.4 c R1). Beam-aware sequence losses add gold-in-beam: BSO margin loss with reset (word ordering BLEU 31.0 -> 34.3, MH49), Negrinho's "continue" roll-in (MH50), gold paths in the normaliser (beam 500 mostly; 35 / 90 / 180 min per epoch at beam 500 / 1000 / 2000, MH42). Soft beam: CCG tag accuracy 80.35 greedy, 82.42 hard beam, 85.78 soft-beam training, beam 3, alpha annealed, cross-entropy warm start (MH41) - but it averages states. OT resampling blends positions (MH44), SOFT top-k and soft sort carry O(eps) bias and a train / test mismatch (MH34, MH36), entmax gives exact zeros and one winner (MH35), hard forward with multiplicative survivor weights keeps consistency (MH45, MH47); resampling during training hurt tracking success in one filter study (79 -> 75 %, MH43). **All demonstrations are on discrete or latent states; none on geometric nodes** (MH-notes 3.4 c, section 5 item 4). For M x b <= about 100 candidates every operator is cheap next to BEV sampling (MH-notes 3.4 f).

**Builder today.** `beam_topm` on detached path validity, one beam per growth direction; gradient reaches survivors through the validity price in the recall loss (`graph_losses`: price 2 (1 - pi)), i.e. the MH45 / MH47 pattern. Path validity is `pi = min(parent, child)` (`p16_geom.path_validity`). The SPF2 keep gate is applied *after* the tree: a hard-concrete gate z on every node token (stretched concrete with gamma = -0.1, zeta = 1.1, beta = 2/3, sampled in training) multiplies the path validity, and `l0_penalty` prices the expected number of open gates of plausible nodes per 100 m against a target of 14.3; SPF1 `vc_penalty` is a soft vertex count against the same target. The beam itself is not gated.

**Consequence for loop 16.** Stay hard. The entmax ablation of HF3 is meaningful only *inside a tip* with a two-hot fork target, not as a beam operator (MH35). For the coded gate lit16 offers only cautions, no card: routing papers warn that a uniform-usage prior is wrong for rare slots (MH47) and that a keep mask needs a target ratio (MH48, abstract level); a gate applied after the beam cannot save a node the beam already cut, whereas over-complete-first, prune-late is the pattern of the sparsification review (concept 4b S5, S6). Gold-in-beam is the only beam-aware ingredient worth a flip: measure the share of levels where the GT-nearest node is among the M kept; if it is near 1 the reset is pointless. Test a log-sum or product path score against the min: all beam-aware papers sum log scores (MH41, MH42, MH49, MH50, MH57), min passes gradient only to the bottleneck node, and calibration of products is unstudied (MH-notes 5 item 3).

### (e) Free roll-out versus teacher forcing, scheduled sampling, warm start

**Published.** iCurb: restricted run (follow the expert when the prediction strays more than 15 px) then free runs, expert label computed *after* the prediction as the nearest GT point ahead of a progress index; paper N = 5 free runs, README says 1 is enough; F1@5 px full 0.883, without free exploration 0.750, without restricted 0.865, without aggregation 0.865, without history 0.818 (GR03). RoadTracer: a static oracle drifts, dynamic multi-hot labels over all unexplored branches fix it (GR04). VectorMapNet retraining on predicted keypoints +5.1 / +8.5 mAP, SAM-Road++ node-guided resampling about +1.25 F1 (GR-notes (e)2). DAGMapper trains a 5-step unroll with softmax relaxation and Chamfer loss (F1@2 px 76.3, GR05); the RNGDet family uses noise only (GR01). Scheduled sampling needs a decaying coin; "always sampling" collapses (captioning BLEU-4 11.2 vs 28.8 baseline vs 30.6 scheduled, MH51); BSO fails from random initialisation and needs CE pre-training and a beam curriculum 2 +1 every 2 epochs (MH49); soft beam starts from CE (MH41); PGP trains 100 epochs on GT traversals then 100 on sampled paths (MH17); Trexplorer Super chains sub-trajectories to cure duplicate branches and early stops (GR07, MH55); the one-mode closed-loop rollout lowers collisions 3.95 -> 3.08 % on nuScenes at 0.5 s replanning (MH53). **No paper trains a multi-hypothesis model free-running from step 1** (MH-notes 3.0 item 4).

**Builder today.** Labels are defined for any state (distance fields; off-lane tips weigh 0.3). Warm start replaces a random share of tips by the nearest GT point (< 2.5 m) with the GT tangent, stop-gradient (`expert_state`); the share ramps linearly from 0.5 to 0 over the first 40 % of the iterations (`p16_train`, `warm_frac` = 0.4) and is 0 for the designs with `warm=False`. 2.5 m is close to iCurb's 15-px switch of about 2.3 m (derived from the 15.2 cm per pixel in GR03).

**Consequence for loop 16.** The ramp the literature asks for (MH51, MH49, MH41, MH17) is already coded, so factor F is "ramp vs none", and the free designs (D1, D4, D5, D8) are exactly the no-precedent case: do not read their rung-0 failures as architectural while the control curve is still rising (plan section 9). What the literature adds is a level or beam curriculum (BSO beam 2 +1 every 2 epochs, Trexplorer chained sub-trajectories) and the ramp length as a tunable (40 % is untested against others); end with a phase at share 0 and full gradients, as coded.

### (f) Duplicate suppression and tree -> DAG assembly

**Published.** Path2Graph merges vertices of different paths closer than 0.15 m, equal to its vertex spacing; the merged vertex inherits predecessors and successors (TP04); path count at +-50 m is not reported. Radii in the growers: RoadTracer 36 m with graph distance >= 72 m (3 D and 6 D, D = 12 m), RNGDet 5 px, CenterLineDet 1 m (0.25 m grid), DeH4R 10 px, NTG 5 m, DOGE 4 m, LaneGNN lateral 20 px = 3 m (GR-notes (a), (g)). DOGE's operator set: merge nodes closer than 4 m, snap a node onto a nearby edge and split it (T junction), collinear merge with refit, prune short or thin edges; without TopoAdapt TOPO-F1 / APLS 82.80 / 69.45 vs 84.58 / 73.48 (GR12). LaneGNN: naive aggregation TOPO precision 0.366 vs 0.481 with weighted aggregation (GR08). Tiny loops: RoadTracer's graph-distance guard, NTG closure by proximity, PolyMapper loop removal, LaneGNN depth cap 10 (GR-notes (a), GR32). Learned alternatives: relation modules and graph-based rescoring (MH31 - MH33), soft merge weight in the TopoLogic form (TP-notes (g)1).

**Builder today.** During training duplicates are handled by NMS at selection and `dup_geo` / the learned pair head (k = 8 nearest nodes, label: both within 1 m of a lane and within 1.5 m along the label graph). The assembly is inference-side (`p16_infer.assemble`): path union in descending path validity, node snapped onto an existing edge within `cov_r` = 1.0 m (edge split), vertex reuse within `vtx_r` = 1.5 m, skip an edge if a path a -> c already exists, then `simplify` with bubbles (two branches that leave a vertex and meet again within `bubble_r` = 2.5 m: the weaker one is removed) and spurs (dead-end chains shorter than 5 m on a junction); a snapped vertex with in = out = 1 is removed again.

**Consequence for loop 16.** The rule set the literature recommends is coded; what remains is calibration and order. The four radii (1.0, 1.5, 2.5 m, 5 m) lie inside the published range: 0.15 m (LaneGAP) and 1 m (CenterLineDet) for BEV lane graphs, 4 - 10 m for aerial road graphs (DOGE 4 m, NTG 5 m, DeH4R 10 m, RoadTracer 36 m); the vertex error is 0.43 - 0.54 m (project). Sweep them on VAL as one grid, not one by one, and log how often each rule fires (T14). The legacy merge with a fixed radius and heading tolerance failed on D3 (split precision 0.006, `assemble_cluster` docstring), which is the loop and mesh problem RoadTracer, NTG and LaneGNN guard against with a minimum graph distance (RoadTracer ratio 72 m / 36 m, derived: about twice the merge radius); the path union with edge snapping is the DOGE / Path2Graph form. DOGE's order (merge, T split, collinear merge, prune) and its ablation of the whole operator set (82.80 / 69.45 without vs 84.58 / 73.48 with, GR12) support keeping a clean-up pass; these are export rules, training sees only validity.

### (g) Link / topology heads: what really gives gains

**Published (OpenLane-V2 subset A, v2.1 metrics unless stated).** TopoLogic ablation TOP_ll: pair MLP 10.8, similarity 12.9, endpoint-distance mapping 20.1, both 23.9; mapping function tanh 15.1, sigmoid 19.1, Gaussian 21.7, learned exponent 23.9; post hoc on TopoNet 10.9 -> 22.3 (TP01). Geometry priors replace rather than stack: CoPo with the fixed mapping added -0.5, UniTopo +0.5 (TP-notes (e)2). Released bandwidth: half value at about 2.8 m (TP-notes (g)2). Endpoint tokens: TopoPoint's distance-biased coupling +3.4 TOP_ll and +5.0 DET_p (23.8 -> 27.2, 44.8 -> 49.8), full model 28.7; 100 -> 200 point queries DET_p 49.7 -> 51.8 (TP02). TopoFG 23.9 -> 30.8, of which the topology block with denoising adjacency +4.0 (TP05). CGNet junction-aware queries +0.8 TOPO F1 and +1.7 APLS, Bezier-space continuity +1.7 / +2.3 (TP03). Connected-pair objects: UniTopo connection queries 12.4 -> 23.3, +10.9 over a matched-parameter baseline (TP06). Label density: RATopo valid-pairs-only +3.3, full 32.2 (TP12). The link head is not the bottleneck: true topology on predicted lanes adds only +0.3 .. +0.6 TOP_ll (old metric, TP08). Corruption-trained re-scoring lifts edge-set quality (TJS_ll TopoNet 16.3 -> 32.9, TopoMLP 18.5 -> 59.7); on TOP_ll plain BCE 12.1 vs BCE + focal 22.2 vs per-relation normalised BCE 21.8, i.e. class balance carries the gain (TP07). Pairwise node heads with interior / extension samples: SAM-Road++ APLS +1.6, 14 - 50 x faster than RNGDet++ (GR10, GR-notes (b)). The literature predicts our E42 - E56 outcome: the label is endpoint coincidence, so geometry decides (TP-notes (d)).

**Consequence for loop 16.** Expect little from M2 / B1 beyond the rules (E45 already shows it); do not stack TP01, TP13, TP14. The consistent gains are explicit junction / endpoint supervision (M1; our own E44 / E53 went 0.34 / 0.38 -> 0.71 / 0.64 split / merge F1) and a corruption-trained, class-balanced verifier on the learned graph, whose regime (1.8 x too many junctions) matches TopoEnhance (TP07, E113 / E116). Re-fit any link bandwidth to sub-metre vertices.

### (h) Temporal recursion and validity memory

**Published.** Only 8 - 33 % of the previous set is carried; TopoStreamer's optimum is 30 % (10 / 30 / 50 / 75 % -> 34.8 / 36.6 / 34.3 / 32.9 mAP, VM09, VM-notes O1); condition on the relative pose, trajectory conditioning hurts by 5 mAP (TP20). The source of the prior in training decides: single frame 59.9, plain self-fed stream 59.2, normal denoising 62.7, stream denoising 63.9 mAP (VM02) - most of the gain is denoising; stored real predictions with dual-mode training p = 0.5 (VM03); pose-noise training and distance-strided memories at 1 / 5 / 10 / 15 m (VM04); test-time pose noise costs 67.2 -> 63.8 mAP at 0.2 m / 0.02 rad (HRMapNet, VM-notes O2). Matching-free consistency between the warped previous output and the current one: MapTCL +0.6 .. +3.7 mAP and +0.5 .. +6.2 C-mAP on three hosts (VM23), DTCLMapper +1.1 / +1.5 mAP (VM24, VM-notes O7). Validity memory forms that exist: bounded additive counter (VM13), decayed rasters 0.9 / 0.95 (VM14, VM15), confidence decay x 0.6 (VM06), visibility count (VM16), GRU, confidence-weighted merge (VM22), hand-specified log-odds with a distance weight on grids (VM12: Bayesian only 35.06, with confidence weight 35.57 global IoU). **Does not exist:** a vertex-level log-odds recursion, a Hungarian-free temporal vector-map model, pose-error robustness studies, a stability metric for graph outputs (VM-notes 5). Gains on strong single-frame results are small (E41 +0.008 F1; TP31 +0.5 M-F), stability gains are large (native temporal mAS 91.9 / 90.4 vs single-frame 71.6, VM44).

**Consequence for loop 16.** Measure stability next to F1 in rung 3. Carry about 30 % by validity, relative pose, training mix of stored own predictions + noised GT + single-frame mode; noise magnitudes must be chosen by us (only TopoStreamer box shift 0.2 / label flip 50 %, Mask2Map class flip 0.2, Uni-PrevPredMap displacement within +-6 m are given, VM-notes 5). Ablate log-odds against decay; it is new at vertex level.

### (i) Route / SD-map attribution after perception, planner interface

**Published.** The only attribution after perception is MAT (PL01): hard, unique, road-level assignment; path-aware attention 74.1 vs 61.8 for a plain backbone; val NR-F1 78.7 vs 72.9; test on perceived maps 41.9 / 45.0 / 54.9 vs 36.3 / 39.1 / 50.8; SD shift of 10 / 20 / 50 % gives val 77.5 / 76.0 / 74.6. SMART, SMERF and OpenLane-V2-OSM use SD maps as a prior without a route (PL-notes 1). Planners take the route as a per-polygon on-route flag (PlanTF, roadblock level, PL03), route-filtered DFS reference lines (PLUTO, depth 15 lanes, 0.5 m de-duplication, PL04), 25 route lanes through adaLN (PL05) or command embeddings that are weak conditioning (PL49); lane-level to roadblock-level is easy, the reverse is not (PL-notes 3b A2). Graph quality is invisible to nuScenes open-loop L2: GT vs predicted topology gives identical 0.62 m / 0.16 % (PL14), an ego-status MLP reaches 0.29 - 0.35 m (PL19); DenseTNT on an online map beats the GT map (PL24). Planning loss into map structure reaches only map-point coordinates: VAD hinge constraints 0.76 -> 0.72 m L2 and 0.28 -> 0.22 % collision (PL08), UncAD DACR 1.39 -> 0.80 % (PL09), MAP explicitly (PL12); **no work reaches edges or topology** (PL-notes 3d.4). Soft validity enters attention as a log-confidence bias (UMPE, PL13).

**Consequence for loop 16.** Keep the cross-attention route head (E104b). Judge graph quality by oracle / degraded graph swaps into the same E117 and a PDM-type IDM test bed (PL02, PL24), not by open-loop L2. Pass the validity pi to the consumer (PL09, PL13). Do not expect a gradient from the planner into the tree to help (E121, PL08's small gain); an optional MAT-style ordering of root-to-leaf paths of the DAG is the cheap route-side addition (PL01).

### (j) Frozen cached BEV encoders: leakage, train / test gap, location overlap

**Published.** On the official nuScenes splits 79.4 % (val) and 85.5 % (test) of samples lie within 5 m of a training sample (AV2 45.0 / 41.9 %); location-disjoint splits drop online-mapping scores by more than 35 mAP on average (MapTR R50 48.8 / 59.9 -> 19.0 / 19.7) (SN03); MapTR 48.7 -> 20.9 on the StreamMapNet split (EV B.2); RNTR landmark F 54.1 -> 36.5 (SN02). A frozen map-pretrained encoder with a trained head is indistinguishable from unfreezing on lane graphs (landmark F 38.6 vs 39.1, reachability 48.6 vs 49.8, SN02); a frozen generic encoder is much worse for LSS (drivable 61.6 vs 72.9, SN23); LP-FT (head first) if ever unfrozen (SN28); adapters inside the backbone beat head-only adaptation under shift (SN29, needs inner layers, not in the cache); a second stage should be trained on held-out outputs (SN34, abstract level). **No paper measures the feature gap of a frozen map-pretrained BEV encoder** between its training scenes and unseen ones (SN-notes 5); the project's 0.93 vs 0.82 F1@1 is neither explained nor bounded.

**Consequence for loop 16.** Our split is scene-disjoint, not location-disjoint: run the location-overlap check (5 m rule, effort S, SN03 / VM46) before any comparison with published numbers, report seen / unseen scenes separately if the foundation's training list is obtainable, and keep the route head and consumer on out-of-fold builder outputs (plan data hygiene, SN34). BEV-space augmentation of cached raster and labels together and feature noise are untested suggestions (SN-notes (d)9).

### (k) Metrics: OpenLane-V2 versions, TOPO / JTOPO, landmark thresholds, mapping to OURS

**Published.** OpenLane-V2 devkit v1.0.0 / v2.0.0 ("x.0") and v1.1.0 / v2.1.0 ("x.1") compute TOP differently: the same model gives TOP_ll 4.1 vs 10.9 (TopoNet) and 7.1 vs 21.7 (TopoMLP), OLS 35.6 vs 39.8 and 38.3 vs 44.1; DET_l / DET_t are identical across versions; no constant conversion exists; TOP ranks only candidates above the 0.5 edge threshold, so validity must be calibrated (EV B.1, B.9). TOPO / JTOPO exist in two protocol forms: LaneGAP text 0.15 m interpolation, 0.45 m match, 7.5 m sub-graph; CGNet code 0.2 / 0.8 / 40 m (units inferred); SDA is an F1 at threshold 1 with Hungarian matching in the CGNet code, an accuracy within a radius R elsewhere (EV B.3, B.9). RNTR landmarks: thresholds 0.5 - 5.0 m in 0.5 m steps (paper), reachability 0.5 - 2.5 m, paths of at most 5 edges; the released code uses 5 + 5 thresholds, non-one-to-one nearest matching (duplicates all count as true positives) and silently skips frames without predictions (EV B.4).

| their metric | closest metric of ours | what differs | comparable? |
|---|---|---|---|
| DET_l / OLS (Frechet AP, 1 / 2 / 3 m, 11 points, confidence-ranked) | point F1@1, lanes found | instance level and confidence-ranked vs pooled points; direction-aware; window x +-50, y +-25 m | no; computable after G2 (EV B.8) |
| TOP_ll (adjacency AP, x.0 vs x.1) | none; split / merge F1 and route success are related | adjacency between instances; version-dependent | no |
| MapTR mAP (Chamfer 0.5 / 1 / 1.5 m, 60 x 30 m) | point F1 at 0.5 / 1 / 1.5 m | same thresholds, instance ranking vs pooled points; direction-agnostic | partly |
| STSU pooled P / R, detection ratio | point F1, lanes found | thresholds up to 5 m, front camera | closest family |
| GEO F1 (1:1 matching, 0.15 m interpolation) | F1@0.5 | one-to-one vs nearest-point P / R | similar idea |
| TOPO / JTOPO F1 | split / merge F1 + F1 | sub-graph connectivity around each matched vertex (7.5 m) vs key-point detection | complementary |
| SDA (CGNet code, F1 at 1) | split / merge F1 at 6 m | threshold 1 vs 6 m | compute ours at 1 m for a like-for-like value |
| RNTR landmark P / R / F (forks, merges, roots, 0.5 - 5 m) | split / merge F1 at 6 m | threshold curve vs one radius; roots included | ours is looser; add the curve |
| RNTR reachability, APLS | route success | landmark-pair paths vs a driven route | related |
| mAS, C-mAP | temporal consistency | needs ego-warp pairs, Hungarian pairing, identity | related; formula of mAS inconsistent as printed (EV B.5) |
| open-loop L2 / collision, PDMS, EPDMS, nuPlan CLS | planner path error 5 - 40 m | time-indexed vs distance-indexed; need agents or simulator | weak signal / not computable (EV B.6) |
| (none) | vertex error, direction accuracy, false length fraction, vertices per 100 m | published protocols have no vertex-position or vertex-count measure | ours only (count-aware scores exist only for polygons, concept 4b S8) |

**Published numbers that are NOT comparable with ours** (EV B.9, TP-notes (h), VM 3(a)-C): OpenLane-V2 TOP / OLS across devkit versions or when the version is unstated (UniTopo); anything on the leaking nuScenes split (74 - 77 mAP) vs location-disjoint (27 - 46, VM-notes 3(a)-C); RNTR "default" vs PON split (about 20 points of landmark F); TOPO / JTOPO between the LaneGAP and CGNet parameter sets, or between 30 x 60 m (CGNet, JTOPO 34.1 - 43.3) and 50 x 100 m (LaneGAP 0.392 junction F1); APLS of aerial vs lane graphs (63.14 vs 41.4); SDA in pixels vs metres; instance AP without a per-instance confidence. All published ranges (60 x 30, 100 x 50, x +-50 / y +-25, RNTR 96 x 64 m) differ from our +-50 m square. Hence external numbers are context only; the only like-for-like comparison is to compute their metric on **our** output with adapters G1 - G6 and a stated parameter set.

**Consequence for loop 16.** `p16_diag` already reports landmark P / R of splits and merges at 1 / 2 / 4 / 6 m, a one-to-one F1 and the path-validity ECE, so the main protocol gap is small. Missing against the published protocols, in value order (EV B.8): the location-overlap check; the thresholds 0.5, 3 and 5 m so that a mean over 0.5 - 5 m can be quoted next to the 6 m point; SDA as an F1 with Hungarian matching at 1 m and the degree rule of the CGNet code (our landmark P / R at 1 m is close but not identical); roots in the landmark set; reachability; Chamfer AP on pieces; lane-only OLS-lite with the x.1 definition after calibrating validity; stability once time is on. Say whether matching is one-to-one (the released RNTR code is not) and score every frame.

---

## 5. Reuse / re-implement / adapt decision table (52 rows, the most relevant techniques)

Decision vocabulary of SCHEMA: reuse_idea, reimplement, adapt_code, baseline, not_suitable. Effort S < 1 day, M days, L a week or more. "Licence / framework" is the EV verdict (Part C / D) where EV covers the repository; "not recorded" means EV did not list it. **Read-only source** = the licence page forbids or does not grant reuse, so the code is read and the idea re-implemented (EV Part D). Sources of the decisions are the cards' `fit_to_us` fields and the "Reuse versus re-implement" sections of the clusters; where I changed a decision, the blocker column says why.

| # | cards | technique | decision | effort | licence / framework verdict | blocker |
|---|---|---|---|---|---|---|
| 1 | VS01 | logistic-hazard BCE with censoring (pycox / Nnet-survival pattern) | reimplement | S | pycox `loss.py` read as a page; licence not recorded | convention at the BEV edge and in occlusion; single-event assumption |
| 2 | VS04 | PonderNet expected loss + KL to a geometric prior | reimplement | S | no official code found | sharp endpoint at inference needs calibrated P(k) |
| 3 | VS07 | distortion regulariser on ray weights | reimplement | S | multinerf URL from memory, not opened | penalises true bimodality (forks inside one ray) |
| 4 | VS12, VS13 | L-CNN / HAWP line-of-interest verifier (baseline B3) | baseline | S - M | MIT, PyTorch no pin; APH needs MATLAB (EV C.4) | one score cannot say where a long line stops; sample count (32) never ablated |
| 5 | VS14, VS15 | DeepLSD inlier-fraction verifier (zero-parameter B3 variant) | reuse_idea | S | code URL from memory | hard threshold on the fraction |
| 6 | VS18, VS19 | cascade radius schedule, look-forward-twice | reimplement | S | DINO Apache-2.0, compiled ops; Cascade R-CNN not recorded | LFT is loss-side credit only; Hungarian in the host |
| 7 | VS20 | Deformable DETR reader, pure-PyTorch `grid_sample` core | adapt_code | S | Apache-2.0; use `ms_deform_attn_core_pytorch`, not the CUDA op (EV C.4) | unbounded offsets leave the raster; Hungarian in the decoder |
| 8 | VS24, VS25, VS23 | Sparse4D bounded offsets, DAT tanh, DAB gain | reuse_idea | S | Sparse4D MIT; others not recorded | pattern scaling needs per-tip (length, width) |
| 9 | VS21 | RAFT window lookup (stop-gradient coordinates) | reuse_idea | S | not recorded | isotropic window vs anisotropic teardrop |
| 10 | VS26 | CLRNet ROIGather, Line-IoU, deepest-first order | reimplement | S - M | Apache-2.0, compiled extension (EV C.4) | row-wise offsets invalid for BEV graphs |
| 11 | VS29, VS31, VS32 | curve-aligned reads (BezierFormer, LaneSegNet, TopoBDA) | reuse_idea | M | LaneSegNet Apache-2.0 on the mm stack; TopoBDA no model code (EV C.2) | cubic Bezier cannot branch; Hungarian decoders |
| 12 | MH01, MH02 | rMCL, aMCL (WTA with score head, cooling) | reimplement | S | rMCL MIT toy code; aMCL no licence (EV C.5) | no evidence for b = 2; published schedule is 1000 epochs; the code has only relaxed WTA (MH04) |
| 13 | MH03 - MH05 | catch-up term, eps floor, EWTA | reimplement | S | code not checked | redundancy (MH03), blur (MH04) |
| 14 | MH10, MH12, MH16 | fixed slots / anchors with nearest-slot CE / BCE | reuse_idea | S | MTR needs custom ops (from memory); DiffusionDrive repo licence not recorded | straight lane on a sector border |
| 15 | MH19, MH29, VM34 | quality-aware score targets (POTO, Align-DETR, DAMap) | reuse_idea | S | DeFCN Detectron2-based; Align-DETR, DAMap not checked | needs a closeness measure in [0, 1] |
| 16 | MH31, MH32 | relation-network duplicate removal, best-per-GT label (a geometric same-tip variant is coded as `dup_label`) | reimplement | S | code not checked | label depends on current scores |
| 17 | MH34, MH35 | SOFT top-k, entmax / sparsemax | reimplement | S | entmax package not in the environment (MH-notes 4) | blends positions; exact zeros; two-hot for forks |
| 18 | MH45, MH47 | survivor-weight gradient, load balance | reuse_idea | S | not checked | dropped nodes get nothing; rare forks vs uniform usage |
| 19 | MH41, MH42, MH49, MH50 | beam-aware training (soft beam, gold-in-beam, BSO, continue) | reuse_idea | M | not checked | needs a gold prefix inside the tree |
| 20 | MH51 | scheduled sampling coin / ramp | reuse_idea (linear share ramp already coded in `p16_train`) | S | - | no gradient through the sampled input |
| 21 | MH53 | one-mode on-policy rollout with detach | reuse_idea | M | CL-Traj URL, licence not read | detach cuts credit to early children |
| 22 | MH57, PL07 | pre-generated tree + learned cost (DTPP) | reuse_idea | M | DTPP repo needs nuPlan devkit (Python 3.9); licence not recorded | rule-generated tree, agent futures |
| 23 | GR03 | iCurb restricted / free labels, dynamic expert | reimplement | S - M | GPL-3.0 (Topo-boundary) -> read-only source | single-branch; forks need one progress index per branch |
| 24 | GR04 | RoadTracer multi-hot unexplored-branch label, merge guard | reimplement | S | licence not recorded | angle / sector discretisation |
| 25 | GR05 | DAGMapper short unroll + Chamfer | reuse_idea | M | no code found | RoI rotated by the predicted direction |
| 26 | GR06 | DeH4R rounds, radius merge, grid ROI | reuse_idea | M | no licence file -> read-only; SAM2 weights; repo name differs between GR06 and EV | Hungarian set loss |
| 27 | GR10 | SAM-Road / SAM-Road++ vertex NMS + pairwise edge head (arm B2) | adapt_code | M | SAM-Road MIT, SAM-Road++ no licence (EV C.1) | SAM encoder replaced by our pyramid |
| 28 | GR12 | DOGE merge / split / collinear / prune operators | reuse_idea | M | no code (announced) | the test-time optimiser is not used; most operators are already in `p16_infer.simplify` / `assemble`, so the card serves as the order and calibration reference |
| 29 | GR01, GR02 | RNGDet++ / CenterLineDet agents | baseline | M | GPL-3.0 (RNGDet++ adds academic-only wording); Docker, python-2 evaluation image (EV C.1) | not differentiable; Hungarian |
| 30 | TP01, TP13, TP14 | TopoLogic mapping, CoPo geometry + contrastive loss, T2SG distance bias | reimplement | S | TopoLogic no licence; CoPo Zenodo release; T2SG not recorded | bandwidth 2.8 m vs vertex error 0.5 m; priors do not stack |
| 31 | TP02, TP05 | TopoPoint coupling, TopoFG keypoint tokens + denoising adjacency | reuse_idea | M | no licence (read-only) | joint token decoder; Hungarian in the vanilla branch |
| 32 | TP03 | CGNet junction heat map, Bezier-space continuity | reuse_idea | S | Apache-2.0, bundled mmdetection3d | continuity needs matched pairs |
| 33 | TP04 | LaneGAP Graph2Path / Path2Graph | reimplement (path union is coded in `p16_infer.assemble`) | S | Apache-2.0 (API), repository holds no code (EV C.2) | path count at +-50 m unknown |
| 34 | TP07 | TopoEnhance corruption-trained re-scorer | reimplement | M | no code found | class balance per relation; node set from our own graph |
| 35 | TP12 | RATopo valid-pairs-only, positives per lane | reuse_idea | S | repository homothetic/RATopo, licence not recorded | lives inside the assignment |
| 36 | VM01, VM08 | pose-conditioned token update, transformation loss, MLN | reimplement | S | StreamMapNet GPL-3.0 -> read-only; StreamPETR custom licence | needs element identity |
| 37 | VM02, VM05 | stream-query denoising, noise taxonomy | reimplement | S - M | SQD-MapNet GPL-3.0, no code released | noise magnitudes unknown |
| 38 | VM12, VM13, VM16 | distance-weighted log-odds, bounded counter, visibility count | reimplement | S - M | L2G-Map announced only | no vertex-level version exists |
| 39 | VM32, VM33 | soft rasteriser, global raster loss | reimplement | S - M | MapVR repo URL on the card | order independence only by construction |
| 40 | VM44, VM45 | mAS, C-mAP adapted to GT pairing | reimplement | M | toolkit announced; MapTracker GPL-3.0 | printed mAS formula inconsistent; identity at forks |
| 41 | PL01 | MAT path-aware attention over SD roads | reuse_idea | M | OMA-MAT repository, licence not recorded | hard unique assignment |
| 42 | PL02, PL21 | PDM-style IDM test bed, DAC / DDC / LK sub-scores | reimplement | M | NAVSIM Apache-2.0; nuPlan devkit Python 3.9 (EV C.8) | no agents, speed profile needed |
| 43 | PL08, PL09, PL13 | VAD hinge constraints, UncAD scoring, UMPE log-confidence bias | reimplement | S | VAD Apache-2.0; others not recorded | open-loop signal weak |
| 44 | PL03, PL04 | PlanTF polygon contract, PLUTO route-filtered DFS | adapt_code (format only) | M | no licence (read-only), nuPlan devkit | 100 m range, roadblock-level route |
| 45 | SN13, SN10, SN39 | calibration-only observability mask for censoring | reimplement | S | - | no occluder mask without lidar |
| 46 | SN03, VM46 | location-overlap check of the split | reimplement | S | geographical-splits repository URL on the card | needs global ego poses |
| 47 | EV B.1 | OpenLane-V2 `evaluate.py` (x.1) | adapt_code | S - M | Apache-2.0; pins numpy < 1.24, scipy == 1.8.0 (EV C.8) | state the version; calibrated validity |
| 48 | EV B.3 | CGNet `topo_eval` (GEO / TOPO / JTOPO / SDA / APLS) | adapt_code | S - M | Apache-2.0, pure Python; old networkx / scipy pins | parameter set differs from the LaneGAP text |
| 49 | EV B.4, GR28 | landmark / reachability (our own variant) | reimplement | S - M | SeqGrowGraph evaluation code has no licence -> read-only | released code is non-one-to-one and skips empty frames |
| 50 | SN01 | GaussianCaR extra outputs (fused 4-scale pyramid, alpha coverage) | adapt_code | S | Apache-2.0; vendored rasteriser non-commercial; FP32 only | PCA to 32 channels discards unmeasured information |
| 51 | GR13 - GR16, TP22, TP23, VM31 | token / sequence decoders | not_suitable (RNTR, SeqGrowGraph kept for metrics only) | - | RNTR MIT but mmdet3d 1.4 stack; SeqGrowGraph no licence | tokenisation, argmax |
| 52 | TP16, GR17 | Topo2Seq training-only sequence decoder | not_suitable (TP16) vs reuse_idea M (GR17): see section 9 | - | no code found | stated mechanism is gradient into a trainable encoder |

---

## 6. Mixing rules and feasibility of mixes

### 6a. Compatibility rules that follow from the matrix

| rule | statement | basis |
|---|---|---|
| R1 matching | A block that needs a one-to-one assignment contributes its head formula, prior or label-density idea, never its training recipe; replace the assignment by a nearest-hypothesis / nearest-slot / quality-argmax label (adapter 2). | section 3 conflict 1; MH-notes 3.2; VM-notes 3(b) |
| R2 score semantics | One score, one meaning: validity (independent sigmoid, existence and closeness), responsibility (normalised, WTA) and path score (additive or multiplicative along the tree) must not share a head. WTA can feed validity only through a separate head. | MH-notes 3.1, 3.6; MH01 conflicts |
| R3 selection | Hard top-M / NMS with local labels and survivor weights stacks with everything. Soft or OT selection may act on scores but never on node positions; entmax only inside a tip with a two-hot target. | MH-notes 3.3 - 3.4; MH41, MH44, MH35, MH45 |
| R4 verification x hypotheses | Survival verification (pool and endpoint law) stacks with WTA / quality labels for training the pool; no paper combines a survival law with top-b selection, so the pair is untested. | VS-notes 5; MH-notes 3.7 |
| R5 geometry priors | Geometric link / merge priors replace each other (TopoLogic, CoPo, T2SG, an own geometry encoding); a learned semantic term stacks on one of them (+3.8 TOP_ll, TP01). | TP-notes (e)2; TP13 |
| R6 junction supervision | Junction / endpoint supervision stacks with a link prior (TopoFG and TopoPoint contain both). | TP05, TP02; TP-notes (c) |
| R7 denoising | Denoised ground truth trains refinement, rescoring and verification with known targets; it does not train the generation of new hypotheses. Combine with the pool / MCL training for generation. | MH27; VM02; TP07 |
| R8 temporal | Carry top-validity items only (about 30 %), relative pose, mixed prior sources in training; global-pose memories are excluded; denoising stacks with a stream (SQD) and with consistency losses (MapTCL). | TP19, TP20, VM02, VM23; VM K2 |
| R9 regime | Warm start / scheduled sampling / curricula stack with any block. Free roll-out from step 1 excludes no block but has no precedent with hypotheses; detaching interacts with gate G3 (planner reach). | MH-notes 3.0, 3.6; MH53 |
| R10 frozen interface | Anything whose gain comes from gradient into a trainable encoder is excluded or reduced to its head-side idea; LP-FT order if the foundation is ever unfrozen. | TP11, TP16, TP21, SN28, SN29 |
| R11 code and licence | mm-stack code and GPL / unlicensed repositories are read and re-implemented; only plain-PyTorch cores (Deformable DETR `grid_sample` path, DINO components, SAM-Road head, evaluation code under Apache-2.0 / MIT) can be ported. | EV Part C, D |
| R12 evaluation | Any block is judged with the same adapters G1 - G6 and a stated parameter set; external numbers are context. | EV B.8, B.9 |

### 6b. The mixes M1 - M15 of the concept document

| mix | verdict | adapter or conflict | literature support and caveat |
|---|---|---|---|
| M1 tree + junction head | feasible with adapter | junction / endpoint heat map as extra dense-head channel or reuse of the E44 key-point head as roots and stage-C keys; TopoPoint-style coupling = distance-biased attention between vertex and edge tokens (TP02); TopoFG tokens need the denoising-adjacency target (TP05) | strongest consistent family: CGNet JAQ / BSC, TopoPoint +3.4 TOP_ll, TopoFG, LaneGAP junction F1 0.564 vs 0.496 (TP-notes (e)3, (g)3); our E44 / E53; small-angle labels are noisy (concept); published junction F1 not comparable |
| M2 tree + geometric link | feasible with adapter | soft merge weight in the TopoLogic form, bandwidth re-fitted to 0.5 m vertices; connector verified by survival | E45 and E42 - E56 say rules decide; geometry priors do not stack (CoPo -0.5); double counting with M1 |
| M3 MCL children | feasible, evidence missing | A2 needs an ordered assignment for two branches (own inference, MH-notes 3.4); score semantics conflict (R2); the coded `mcl` flag is relaxed WTA with eps 0.1 (MH04), not rMCL / aMCL | rMCL / aMCL shown with >= 3 hypotheses only (MH-notes 5 item 2); published cooling runs 1000 epochs (MH-notes 3.8); the concept names rMCL + aMCL, the code implements something weaker |
| M4 censored survival | feasible, half direct | masked BCE already coded; censoring flag by adapter 4 (SN13, VS43); no occluder mask for camera + radar | no perception precedent (VS-notes 5); border padding in `samp` is not censoring |
| M5 curve-aligned stage B | feasible with adapter | the coded stage B reads a free pattern (centre + 8 points at 1.5 m) around the child; the curve-aligned variant reads along the parent -> child chord (one segment, so the no-branching limit of Bezier does not bite, VS29, VS30); Hungarian-based decoders excluded | BezierFormer five reference points +1.5 to +1.7 F1 (VS-notes 3d); effect on a frozen pyramid unknown |
| M6 cascade schedule | feasible | tighten positives per stage (VS18); "no detach" only in the DINO form (VS19) | cascade 38.9 vs iterative 35.4 AP (VS18); our E39 / E120 gains were small (+0.001, +0.012 F1) |
| M7 denoising roots | feasible with adapter | noise taxonomy rotate / shift / scale / curvature (VM05); targets known for refine and rescore only (MH27) | SQD: 59.2 -> 62.7 by denoising alone (VM02); magnitudes must be chosen by us (VM-notes 5) |
| M8 path-order auxiliary | feasible, low expected value | TP16 says not suitable (mechanism = gradient into a trainable encoder), GR17 says reuse M | the +2.0 AP of the concept is a third-party number; frozen encoder removes the stated mechanism (section 9) |
| M9 predecessor / successor attention | feasible, direct | typed edges (pre, suc, sibling) as attention bias in `CtxLayer` (QCNet typed edges PL37, LaneGCN pre / suc PL35) | UniTopo's +10.9 comes from connection queries inside a Hungarian query host (TP06); the idea transfers, the architecture does not |
| M10 verified link family | feasible with adapter | SAM-Road head (MIT) on our pyramid; connector samples (GR10) plus survival | hybrid decode + expansion 81.21 vs 81.02 / 79.18 alone (GR06); linking 14 - 50 x faster than growth (GR-notes (b)); "may simply win" |
| M11 time | feasible with adapters | ego warp + token update (VM01, VM04, VM08), log-odds vertex memory (new), mixed prior sources (VM02, VM03) | gains on strong single-frame models are small (E41, TP31), stability large (VM44); global-pose methods excluded |
| M12 uncertainty to planner | feasible, direct | per-vertex validity as feature and log-confidence bias (PL09, PL13) | UncAD DACR 1.39 -> 0.80 % (PL09); open-loop L2 hides the effect (PL14) |
| M13 expected-count budget | coded (SPF1 `vc_penalty`, SPF2 hard-concrete gate + `l0_penalty`); unknown within lit16 | no hard-concrete card; nearest: dynamic vertex count (VM28), keep ratio masks (MH48, abstract), load balance conflicts with rare forks (MH47) | evidence lives in the sparsification review (K129, K134, K135), not re-checked here |
| M14 Imai-Iri labels | unknown within lit16 | no paper chooses the step by verification (GR-notes 5); PivotNet's DP alignment is the nearest (VM28); VecRoad flexible step is abstract only (GR20) | sparsity oracles are K142, K144 (outside lit16) |
| M15 many positives (RNGDet-style) | feasible, direct | nearest-assignment labels need only distances | sweet spot 2 - 3 positives per branch (TP12), K = 2 best in Co-DETR (MH25), dynamic-k (MH22) |

### 6c. New mixes the literature suggests (at most 8; the concept document does not have them)

| id | components | expected effect (hypothesis, not a finding) | test needed | cost |
|---|---|---|---|---|
| N1 closeness-target validity + calibration | MH29 (p^a u^(1-a)), MH19, VM34, VM17 (soft geometric targets), VS05 (D-calibration), temperature (K104, E109) | validity means closeness, the beam ranks by something calibrated, thresholds at export become stable | ECE of nu_k and path validity; beam recall at equal M; F1@Sp17 | S |
| N2 duplicate label across tips + sibling rescorer before the beam cut | MH31 (rank embedding, best detection per GT positive), MH32, MH33 (abstract); the coded same-tip `dup_label` and `dup_geo` as baselines | fewer parallel branches and lower vertices per 100 m without a hand radius | duplicate rate, vertices per 100 m at equal F1@1, gold-in-beam rate | S - M |
| N3 gold-in-beam reset and normaliser | MH49 (BSO reset), MH50 (continue), MH42 (gold in the normaliser) | the GT-nearest node is never lost at a cut; gives gradient to scores of otherwise cut nodes | gold-in-beam rate per level before / after; F1 at equal training | M |
| N4 level / beam curriculum on top of the existing share ramp | MH49 (beam 2 +1 every 2 epochs), GR07 (chained sub-trajectories), MH51 and GR03 (ramp, restricted / free switch at 15 px, about 2.3 m: already coded as the 0.5 -> 0 share), GR04 (dynamic labels) | free roll-out becomes trainable at low fidelity and the 40 % ramp length is no longer a guess | F1 and drift at rung-0 fidelity vs control; ramp length 20 / 40 / 60 % | S |
| N5 corruption-trained, class-balanced verifier on tree outputs | TP07 (corrupted GT vertex sets, per-relation normalised BCE), TP13 (one positive, three hard negatives), TP05 (denoising adjacency), MH27 | fewer false forks / merges; the E113 / E116 regime (1.8 x junctions) is the TopoEnhance regime | split / merge F1, edge-set Jaccard (TJS-type), with and without class balance | M |
| N6 dead-end carry-over + re-expansion as roots | GR02 (dead ends of frame t-1 as seeds), GR06 (re-expansion from degree-1 vertices, 1 - 3 rounds), VM03 (stored own predictions, dual mode) | seed-limited recall is repaired without more peaks; time and growth share one mechanism | lanes found at equal false length; recall of lanes with no peak | M |
| N7 raster memory of the own output + matching-free consistency | VM13 / VM14 / VM16 (bounded counter, decayed masks, visibility count), VM32 / VM33 (soft rasteriser), VM23 / VM24 (consistency), VM12 (log-odds with distance weight) | second temporal channel that does not need vertex identity; stability up at equal F1 | adapted mAS / C-mAP, F1; vertex memory vs raster memory vs both | M |
| N8 planner-aware graph test bed | PL02 (IDM along the best route path), PL21 (DAC / DDC / LK sub-scores), PL24 (input swaps), PL19 (ego-status ablation, turning subset), PL18 (forward half) | graph quality judged by driving-relevant error; replaces open-loop L2 as the arbiter between designs | oracle / predicted / degraded graph into the same IDM and E117 | M |

Considered and not proposed: slot-anchored children (6 sectors of 60 degrees as in Sat2Graph GR09, or the E94 fan as anchors with overlapping targets, MH10) - the concept's pool-and-suppress form already covers the same job; diffusion refiners (GR24, GR31, TP21, VM43) - compute-bound and stochastic; token decoders as blocks (conflict 2).

---

## 7. Bridge to the technique register K01 - K146 (the register is not edited here)

### 7a. Literature cards -> existing K ids

"exact" = the card is the paper the K entry names; "related" = same technique family, different paper or only part of it. K66 (nuScenes) maps to EV A.1. K ids with no card in lit16: K01, K16 - K19, K26 (internal, classical or from memory); K62, K63, K67, K69 - K74, K76, K77, K81, K82, K85, K91 - K93, K96, K100, K101, K104 - K106, K111 - K113, K115, K119, K120, K124, K125, K129 - K131, K134, K135, K138 - K142, K144 - K146.

| K id | card(s) | match | K id | card(s) | match |
|---|---|---|---|---|---|
| K02 survival along a ray | VS01, VS02, VS04, VS06 | related (algebra; NeRF = VS06 exact) | K03 n best endpoints, CenterLineDet | GR02, SN19, GR01 | exact |
| K04 Deformable DETR | VS20 | exact | K05 DAB-DETR | VS23 | exact |
| K06 Sparse4D | VS24, VM06, VM07 | exact | K07 RAFT | VS21 | exact |
| K08 integral regression, soft-argmax | TP26 | related only | K09 Relation Networks | MH31 | exact |
| K10 LaneSegNet | TP10, VS31, SN17 | exact | K11 DAGMapper | GR05 | exact |
| K12 LaneGAP | TP04 | exact | K13 InstaGraM, SAM-Road | GR10 (SAM-Road) | partial; InstaGraM only in EV C.1 |
| K14 RoadTracer | GR04 | exact | K15 matching-free point-to-polyline | VM26, VM27, GR26, GR21 | related |
| K20 sparsemax | MH35 | exact | K21 Soft-NMS | MH31 (numbers only) | no card |
| K22, K42 cascade | VS18 | exact | K23 scheduled sampling, DAgger | MH51, GR27, GR03 | exact |
| K24 log-odds fusion | VM12 | related (published grid recursion) | K25 Douglas-Peucker | VM28 | related only |
| K30 TopoLogic | TP01 | exact | K31 TopoPoint | TP02 | exact |
| K32 CGNet | TP03 | exact | K33 TopoFG | TP05 | exact |
| K34 UniTopo | TP06 | exact | K35 Topo2Seq | TP16, GR17 | exact |
| K36 lane / Bezier-aligned attention | TP10, VS31, TP15, VS32 | exact | K37 CLRNet, LaneATT | VS26, VS27 | exact |
| K38 rMCL, aMCL | MH01, MH02 | exact | K39 entmax, soft top-k | MH35, MH34 | exact |
| K40 L-CNN, HAWP | VS12, VS13 | exact | K41 hazard with censoring | VS01 (VS02 - VS05) | exact for the method; K41 cites a review |
| K43 DINO, look forward twice | VS19, MH27 | exact (DN-DETR = MH27) | K44 Sparse4D v3 quality head | VM06, VS24 | exact |
| K45 StreamMapNet, SQD, MapTracker | VM01, VM02, VM04 | exact | K46 mAS | VM44 | exact |
| K47 RNTR landmarks | GR14, SN02, EV B.4 | exact | K48 SAM-Road, Relationformer | GR10 | partial; Relationformer in EV C.1 |
| K49 UncAD, ego-status caveat | PL09, PL19 | exact | K50 iCurb | GR03 | exact |
| K51 DeH4R | GR06 | exact | K52 SMART | PL41, TP17 | exact |
| K61 GaussianCaR | SN01 | exact | K60 CRN | SN05 | exact |
| K64 BEVFormer | SN22 | exact | K65 LSS | SN23 | exact |
| K68 LP-FT | SN28 | exact | K75 Online Navigation Refinement | PL01 | exact |
| K78 HRMapNet | VM13 | exact | K79 TopoStreamer, FASTopoWM | VM09, VM10 (TP19, TP20) | exact |
| K80 Uni-PrevPredMap | VM03 | exact | K83 Sat2Graph | GR09 | exact |
| K84 RNGDet | GR01 | exact | K86 SeqGrowGraph | GR13, TP23 | exact |
| K87 map uncertainty into prediction | PL15 | exact | K88 UniAD | PL11 | exact |
| K89 VAD | PL08 | exact | K90 SparseDrive | PL10, VM11 | exact |
| K94 GoalFlow | PL28 | exact | K95 MAP | PL12 | exact |
| K97 Diffusion Planner | PL05 | exact | K98 UMPE | PL13 | exact |
| K99 OrienterNet | PL46 | exact | K102 PredMapNet, MapTCL | VM14, VM23 | exact |
| K103 video lane graph | TP31 | exact | K107 learned NMS (Hosang) | MH32 | exact |
| K108 AD-MLP, BEV-Planner, NAVSIM | PL19, PL20 | exact | K109 MapTR, MapTRv2 | VM26, VM27 | exact |
| K110 VectorMapNet | VM31, GR18 | exact | K114 Gumbel-softmax, straight-through | MH46 | exact |
| K116 MapVR | VM32 | exact | K117 MambaMap | VM21 | exact |
| K118 LaneDiffusion, MapDiffusion | GR24, TP21, VM43 | exact | K121 PivotNet | VM28 | exact |
| K122 BeMapNet | VM29 | exact | K123 GeMap | VM36 | exact |
| K126 Mask2Map | VM38 | exact | K127 VADv2 | PL27 | exact |
| K128 H-DETR, Co-DETR | MH24, MH25 | exact | K132 sMCL | MH06 | exact |
| K133 OpenLane-V2 | TP29, EV B.1 | exact | K136 DynamicViT | MH48 | exact (abstract level) |
| K137 Gumbel top-k, soft ranking | MH39 (Kool), MH36 (OT sort), MH38 | partial | K143 VecRoad | GR20 | exact (abstract + project page) |

### 7b. Cards that deserve a NEW register entry (proposal only; the fork session owns the register, the user decides)

"checked" = the card says the page was read (`verified: checked`); "partly" = partly verified; the link is the one printed on the card.

| proposed id | topic | card(s) | source and link from the card | status | why |
|---|---|---|---|---|---|
| K147 | conditional (masked) logistic-hazard BCE with an explicit censoring convention | VS01 | Gensheimer and Narasimhan arXiv 1805.00917; Kvamme and Borgan 1910.06724 | checked (method + code page) | K41 cites only a review; this is the loss as coded in `p16_loss` |
| K148 | PonderNet expected loss under the halting law + KL to a geometric prior | VS04 | arXiv 2107.05407 | checked | optional endpoint-law loss without argmax |
| K149 | distortion regulariser on ray weights | VS07 | Mip-NeRF 360, arXiv 2111.12077 | checked | bimodality control of the endpoint law |
| K150 | inlier-fraction line verifier without a network | VS14, VS15 | DeepLSD arXiv 2212.07766 | checked | zero-parameter variant of baseline B3 |
| K151 | tanh-bounded sampling offsets | VS25 | DAT arXiv 2201.00520 | checked | bounded learned deviation of the teardrop |
| K152 | observability / visibility masks for censoring | VS43, SN13, SN10, SN39 | Occ3D 2304.14365; PON 2003.13402; radar mask 2604.01921 | partly | censoring flag for M4; all but SN13 need lidar |
| K153 | catch-up term for idle hypotheses | MH03 | DiverseNet arXiv 2008.10634 | checked | second-child starvation |
| K154 | relaxed WTA / evolving WTA | MH04, MH05 | arXiv 1612.00197; 1906.03631 | checked | eps floor, k schedule |
| K155 | closeness-aware score targets | MH29, MH19 | Align-DETR 2304.07527; DeFCN 2012.03544 | checked | validity as closeness (N1) |
| K156 | beam-aware training with gold in the beam | MH49, MH50, MH42, MH41 | arXiv 1606.02960; 1811.00512; 1902.06022; 1708.00111 | checked (MH50 partly) | N3 |
| K157 | survivor-weight gradients, routing load balance | MH45, MH47 | arXiv 2106.10314; 2101.03961 | checked | hard forward, weights carry gradient |
| K158 | one-mode on-policy rollout with detach | MH53 | arXiv 2603.23393 (card: v2 Sept 2026) | checked | free roll-out with several hypotheses |
| K159 | pre-generated hypothesis tree with learned cost | MH57, PL07 | DTPP arXiv 2310.05885; TPP 2301.11902 | checked | nearest published tree-with-scoring |
| K160 | dynamic multi-hot labels for all unexplored branches | GR04 | RoadTracer arXiv 1802.03680 | checked | extends K14 with the label rule |
| K161 | short-unroll training with Chamfer loss | GR05 | DAGMapper arXiv 2012.12377 | checked | extends K11 with the training recipe |
| K162 | chained sub-trajectory training of a tree tracker | GR07, MH55 | Trexplorer Super arXiv 2507.10881 | checked | duplicates and early stops |
| K163 | successor sub-graph per step with weighted aggregation | GR08 | LaneGNN arXiv 2302.06175 | checked | merge by weighted aggregation |
| K164 | DOGE topology operators (merge, T split, collinear merge, prune) | GR12 | arXiv 2511.19850 | checked | tree -> DAG operator set |
| K165 | beam over vertex sequences, edge check by heat-map integral | GR32 | PolyMapper arXiv 1812.01497 | checked | only published graph-growth beam |
| K166 | corruption-trained re-scorer with per-relation normalised BCE | TP07 | TopoEnhance arXiv 2610.11058 | partly | N5 |
| K167 | redundancy assignment, valid-pairs-only supervision | TP12 | RATopo arXiv 2508.15272 | checked | positives per branch |
| K168 | geometry-biased attention + relational contrastive loss | TP13, TP14 | CoPo 2506.13553; T2SG 2411.18894 | partly / checked | link and merge priors |
| K169 | direction-tensor field as conditioning | TP11 | Lane Field arXiv 2610.25284 | partly | dense head extension |
| K170 | physical-prior noise taxonomy for denoising | VM05, VM02 | SparseMeXt 2505.08808 | partly | M7 noise types |
| K171 | grid log-odds with distance weight, visibility count, decayed masks | VM12, VM16, VM14, VM15 | arXiv 2503.02578; 2409.18737; 2602.16669; 2503.07168 | partly | N7 and the vertex memory |
| K172 | matching-free temporal consistency losses | VM24, VM23 | arXiv 2405.05518; 2608.05209 | partly / checked | N7 |
| K173 | soft closeness targets for validity | VM34, VM17 | DAMap 2510.22675; InteractionMap 2503.21659 | partly | N1 |
| K174 | location leakage and geographic splits | SN03, VM46 | Lilja et al. arXiv 2312.06420 | checked | split check |
| K175 | rule-based IDM test bed and map-sensitive sub-scores | PL02, PL21 | PDM arXiv 2306.07962; pseudo-simulation 2506.04218 | checked | N8 |
| K176 | forward-half metric and input-swap protocols | PL18, PL24 | AMap arXiv 2512.19150 | checked / partly (synthesis) | N8 |
| K177 | graph evaluation code (OpenLane-V2 x.1, CGNet `topo_eval`, landmark protocol) | EV B.1, B.3, B.4 | repositories listed in EV C.2, C.8 | checked as pages | metrics adapters |
| K178 | consumer reads cached BEV features at vertices | PL16 | Gu et al. ECCV 2024, arXiv 2407.06683 | checked | consumer variant not yet tried |
| K179 | out-of-fold features for second-stage training | SN34 | arXiv 2206.08473 | partly (abstract) | route head / consumer hygiene |

---

## 8. Findings that should change loop-16 decisions now (12)

1. **Censoring is missing from the hazard loss; calibration is only partly measured.** The masked BCE matches the pycox convention (VS01), but rays that leave the 50 m raster are read with border padding - they copy the border cell and are labelled by it, they are not censored (`p16_builder.samp`) - and the tolerant read-out is trained with a strict mask. Add an in-range multiplier on the loss weight and an "alive" read-out beyond the raster before the SPF3 flip (the `reach` switch exists now), and add a per-ring ECE of nu_k to the path-validity ECE that `p16_diag` already has (VS01, VS43, SN13; VS-notes 3b.2 - 3b.4, 5; VS05).
2. **Path validity is a bottleneck min; every beam-aware paper sums log scores.** The min gives gradient only to the weakest node on the path and its calibration is unstudied. Add a product / log-sum variant as a one-line flip and log the gold-in-beam rate per level before spending effort on a reset (`p16_geom.path_validity`; MH41, MH42, MH49, MH50, MH57; MH-notes 5 item 3).
3. **WTA / MCL stays out of the defaults, and the coded `mcl` flag is the weak form.** The remedies were shown with >= 3 hypotheses; the second slot starves exactly at ordinary tips (MH03, MH04, MH10; MH-notes 5 item 2). The flag implements relaxed WTA with eps 0.1 (MH04: paper eps 0.05, ghost predictions from centroid bias; MH02: the bias is permanent), not the rMCL + aMCL of the concept. Upgrade it by cooling eps to 0 within the run (published schedule 1000 epochs, MH02) or the catch-up term (MH03) before drawing a conclusion on b = 3 + MCL. Improve the coded default (pool + validity + suppression) with a closeness target for validity (MH29, VM34), the duplicate label extended from siblings of one tip (coded as `dup_label`, a geometric variant of MH31's rule) to the candidates of the beam cut across tips (MH31, MH32), positives per branch capped at 2 - 3 (TP12, MH25) and the idle-second-child diagnostic (T27).
4. **The warm-start ramp exists; test its length and add a curriculum.** The coded ramp (share 0.5 -> 0 over 40 % of the iterations) is the pattern of all precedents (MH51, MH49, MH41, MH17); free exploration is critical once it is added (iCurb 0.750 -> 0.883 F1@5 px, GR03); nothing trains several hypotheses free from step 1 (MH-notes 3.0), which is what the `warm=False` designs D1, D4, D5, D8 do. Sweep the ramp length (20 / 40 / 60 %) on one design and add a level / beam curriculum (MH49, GR07); read rung-0 failures of the free designs with the fidelity caveat.
5. **"No detach" is literature-backed only in the DINO form.** All other cascades detach the coordinates that decide the next read (VS-notes 3d); stage B and C are coded without detach. Add the RAFT form (stop gradient on coordinates, keep token gradient) as a flip and log gradient norms per level next to the per-group norms of gate G2 (VS19, VS21).
6. **Keep selection hard and scope the soft ablation; the coded keep gate acts after the beam.** Hard top-M with local labels and survivor weights is the pattern with evidence (MH45, MH47); soft / OT selection blends positions (MH41, MH44); entmax only inside a tip with a two-hot fork target (MH35). For the SPF2 hard-concrete gate lit16 has no card: the only cautions are the uniform-usage prior of routing (MH47) and keep-ratio masks at abstract level (MH48); the evidence base is the sparsification review (K129, K134, K135), not re-checked here. A gate after the beam cannot rescue a node the beam cut: compare with gating before the cut in one flip.
7. **The assembly rules are coded; calibrate them as one grid.** `cov_r` 1.0 m, `vtx_r` 1.5 m, `bubble_r` 2.5 m and `spur_len` 5 m sit inside the published range (0.15 m LaneGAP, 1 m CenterLineDet, 4 - 10 m aerial; GR-notes (g), TP04); sweep them together on VAL, log how often each rule fires and the number of root-to-leaf paths per frame (Graph2Path may explode at +-50 m, TP04, TP-notes 5), and keep a minimum graph distance of about twice the merge radius against loops (RoadTracer ratio 72 / 36, GR04). The failed legacy merge (split precision 0.006 on D3) is the literature's loop / mesh warning (GR04, GR23, GR08). Order after DOGE: merge, T split, collinear merge, prune (GR12). All of this is export-side; training sees validity only.
8. **Junction supervision is the best-supported gain and E44 already exists.** Feed the E44 key-point heat map as roots and as stage-C keys (M1) at rung 0 rather than only as a later flip; J carries 0.20 of the composite (plan section 5). Do not stack TopoLogic, CoPo and T2SG priors, and expect little from M2 / B1 beyond the rules (TP02, TP03, TP05, TP04; TP13; E45).
9. **Train the verifier on corruption with class balance.** The learned graph with 1.8 x too many junctions is the TopoEnhance regime; the lever is class balance (plain BCE 12.1 -> normalised 21.8 TOP_ll) and hard negatives (one positive, three negatives), not a deeper GNN (TP07, TP13; E113 / E116).
10. **Time recursion defaults for rung 3 are now known:** carry about 30 % by validity (36.6 vs 34.8 / 34.3 / 32.9 mAP at 10 / 50 / 75 %), relative-pose conditioning, training mix of stored own predictions + noised GT + single-frame mode, pose-noise training; report stability, because F1 gains are small (E41 +0.008) while mAS gaps are large (VM09, TP20, VM02, VM03, VM04, VM44).
11. **Evaluation: the remaining protocol gaps are small.** `p16_diag` has landmark P / R at 1 / 2 / 4 / 6 m, a one-to-one F1 and an ECE. Still missing, in value order: location-overlap check of the split, thresholds 0.5 / 3 / 5 m (to quote a 0.5 - 5 m mean next to the 6 m point), SDA as a Hungarian F1 at 1 m with the degree rule, roots and reachability, Chamfer AP on pieces, lane-only OLS-lite with x.1 after calibrating validity; every frame scored, one-to-one matching stated (the released RNTR code is non-one-to-one and skips empty frames) (EV B.4, B.8, B.9, SN03).
12. **Make baseline B3 fair.** The survival read-out sees 5 samples per ray; published verifiers pool 32 (L-CNN, HAWP) or count 50 (DeepLSD) samples and neither ablates the count (VS12, VS14, VS-notes 3f, 5). Run B3 in two forms on rays densified to 32 samples: the LoI single score and the zero-parameter inlier fraction (> 50 % of samples with distance field < 1.5 and angle within pi/9, VS14). B2 should use the SAM-Road graph head (MIT, light port), B1 the TopoLogic form with a re-fitted bandwidth (GR10; TP01).

---

## 9. Gaps, contradictions, and the tests that would answer them

### 9a. Contradictions and unreconciled items

| # | item | where | handling |
|---|---|---|---|
| 1 | Topo2Seq: not_suitable (mechanism = gradient into a trainable encoder) vs reuse_idea, effort M (auxiliary path loss costs nothing from the cached pyramid) | TP16 vs GR17 | not reconciled; treat M8 as low expected value, test only if cheap |
| 2 | SeqGrowGraph code: "none found" vs a repository (MIV-XJTU/SeqGrowGraph, no licence, mmdet3d 1.4 stack, holds the landmark / reachability evaluation code) | GR13 vs EV (EV says it was corrected late) | take EV |
| 3 | DeH4R repository: 7777777FAN/DeH4R (card) vs godx-7/DeH4R (EV); no licence stated in either | GR06 vs EV C.1 | not reconciled; read-only anyway |
| 4 | LaneGAP code: "released with MapTRv2" (card) vs standalone repository empty and not verified inside MapTR (EV) | TP04 vs EV C.2 | Path2Graph is re-implemented from the paper |
| 5 | Concept defaults vs evidence: "no detach" (W6, M6) and "MCL children default" (M3) are stronger than the cluster evidence | Concept section 4, 6 vs VS-notes 3d, MH-notes 5 | sections 4(b), 4(c), 8 items 3, 5 |
| 6 | Evidence quality differs by cluster: VM has 42 of 46 cards "partly" verified, VS / MH / TP / PL mostly "checked" | cards.csv | numbers from VM carry a higher transcription risk |
| 7 | TopoBDA appears with several configurations (R50 subset B camera 49.9 OLS / 35.6 TOP_ll, + radar 54.8 / 38.6, main Swin-B result 54.3 / 34.0): same paper, different rows | SN09, TP15, TP-notes (h) | no contradiction; do not quote a gain across rows |
| 8 | Third-party numbers: the +2.0 AP of Topo2Seq in the concept (W8) is not a primary number | concept 4, TP16 | label as third-party |

### 9b. Gaps: what no paper does, and the test that would answer it

| gap | evidence | test |
|---|---|---|
| hazard along a ray with censoring, calibrated products | VS-notes 5 | toy scenes with rays leaving the raster (T26); ECE per ring and for the path validity |
| WTA specialisation speed of the second slot at b = 2 | MH-notes 5 items 2, 9 | fork scene T27 and collapse rate vs cooling schedule |
| gradient to dropped geometric nodes | MH-notes 5 item 4 | hard + local labels vs survivor weights vs entmax-in-tip, same budget |
| free roll-out from step 1 with several hypotheses | MH-notes 3.0 | factor F: ramp vs no ramp, ramp length 20 / 40 / 60 %, fidelity curve of the control |
| detach vs no detach for point positions | VS-notes 5 | per-level gradient norms; RAFT-form flip |
| lane graph beam / b-ary tree end to end | GR-notes 5, MH-notes 5 item 1 | the ladder itself (this is the contribution) |
| step length by verification; ring count and spacing | GR-notes 5; VS-notes 5 (5 vs 12 vs 24 rings) | SPF3 and B3 on densified rays |
| fork resolution (minimum angle) | GR-notes 5 | T28 small forks, fork-angle resolution curve |
| path count and merge radius at +-50 m | TP-notes 5 | log root-to-leaf paths per frame; radius sweep in T14 |
| vertex-level log-odds validity memory; pose-error robustness of temporal fusion | VM-notes 5 | N7 / M11 ablation with injected pose jumps (project failure protocol) |
| stability metric for graph outputs | VM-notes 5; EV B.5 | adapted mAS / C-mAP with GT pairing |
| feature gap seen vs unseen scenes of a frozen map-pretrained encoder | SN-notes 5 | report by scene group if the foundation's list is obtainable; location-overlap check |
| radar-specific evidence for far lanes; range-binned lane IoU | SN-notes 5 | range-binned F1 of our output, with the radar-off synthetic failure |
| occluder mask for camera + radar | VS-notes 5; SN-notes 5 | foundation alpha coverage (SN01) or a learned visibility head |
| planner effect of false forks, fragmented lanes, missing segments; gradient planner -> topology | PL-notes 5 | N8 swaps with degraded graphs; gate G3 |
| count-aware lane metric (vertices per 100 m against quality) | EV B.7 (no counterpart); concept 4b S8 | frontier F1@1 against vertices per 100 m (plan 10.9) |
| lane-level route attribution with nuPlan-trained planners | PL-notes 5 | not needed unless a nuPlan consumer is adopted |

---

## 10. Appendix: depth and property counts per cluster (own count from `cards.csv`)

Depth classes as in section 1. "differentiable" is the first word of the card field; the 19 cards in the last column are n/a 7, unknown 1, blank 8 (GR20, GR22, GR27 - GR30, VS05, VS44) and three free-text answers (PL27 planning gradient not stated, PL28 hard scoring / selection, VM12 closed-form log-odds sum, differentiable).

| cluster | n | M | M+C | mixed M/A (+C) | A or A+C | C only | yes | partial | no | n/a, unknown, blank, other |
|---|---|---|---|---|---|---|---|---|---|---|
| GR growth | 32 | 18 | 10 | 1 | 3 | 0 | 6 | 5 | 15 | 6 |
| TP topology | 31 | 24 | 7 | 0 | 0 | 0 | 5 | 23 | 2 | 1 |
| VM vector map + temporal | 46 | 42 | 4 | 0 | 0 | 0 | 7 | 32 | 5 | 2 |
| MH multi-hypothesis | 58 | 45 | 1 | 7 | 5 | 0 | 27 | 25 | 6 | 0 |
| VS verification, sampling | 44 | 33 | 4 | 5 | 2 | 0 | 21 | 20 | 1 | 2 |
| PL planner, navigation | 49 | 36 | 10 | 0 | 2 | 1 | 28 | 10 | 3 | 8 |
| SN sensors, BEV | 41 | 34 | 1 | 3 | 3 | 0 | 26 | 10 | 5 | 0 |
| **all** | **301** | **232** | **37** | **16** | **15** | **1** | **120** | **125** | **37** | **19** |

| cluster | fit: reuse_idea | reimplement | adapt_code | baseline | not_suitable | verified: checked | partly | blank |
|---|---|---|---|---|---|---|---|---|
| GR | 29 | 0 | 0 | 1 | 2 | 21 | 4 | 7 |
| TP | 21 | 2 | 0 | 5 | 3 | 24 | 7 | 0 |
| VM | 41 | 0 | 0 | 1 | 4 | 4 | 42 | 0 |
| MH | 47 | 5 | 0 | 5 | 1 | 50 | 8 | 0 |
| VS | 37 | 1 | 0 | 1 | 5 | 40 | 4 | 0 |
| PL | 36 | 2 | 2 | 6 | 3 | 41 | 8 | 0 |
| SN | 32 | 1 | 1 | 4 | 3 | 31 | 10 | 0 |
| **all** | **243** | **11** | **3** | **23** | **21** | **211** | **83** | **7** |

Notes. (1) The `fit_to_us` first word is the decision; 243 of 301 cards are "reuse the idea", only 14 are reimplement or adapt_code, which matches the environment (no mm stack, cached features, Hungarian excluded). (2) adapt_code: PL03, PL05, SN01; reimplement: MH01, MH02, MH03, MH31, MH34, PL02, PL35, SN13, TP01, TP13, VS01. (3) not_suitable: GR13, GR15, MH23, PL22, PL45, PL48, SN22, SN26, SN31, TP16, TP22, TP27, VM31, VM41, VM42, VM43, VS08, VS10, VS11, VS16, VS40. (4) Abstract-level cards (A or A+C): GR20, GR27, GR29, MH30, MH33, MH48, MH54, MH58, PL22, PL23, SN28, SN34, SN40, VS03, VS05. (5) Cards with `differentiable: no`: 37 (GR 15, VM 5, MH 6, PL 3, SN 5, TP 2, VS 1); in GR they are the growth agents, token decoders and rule-based tracers, in MH only the sampling or rule-based items (MH09, MH30, MH39, MH52, MH56, MH58), while the selection family MH34 - MH48 is yes or partial except MH39. (6) Raw `verified` field (`cards_stats.md`): checked 207 + "checked." 4, partly 48 + "partly." 35, blank 7; the table folds the variants into their base words. (7) The stage counts of section 2 are first-code counts of the stage field (own count).

