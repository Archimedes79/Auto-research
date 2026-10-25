# Literature scan: lane-graph / centerline online mapping, SD-map priors, temporal fusion, robustness (2024 - mid 2026)

Date of scan: 2026-10-15. Method: every entry was checked on its arXiv abstract page; numbers marked "(html)" were additionally read from the arXiv HTML full text. Nothing was downloaded to disk. Numbers are copied as reported; datasets/splits differ between papers, so do not compare across rows. Abbreviations: OLS = OpenLane-V2 score, OV2 = OpenLane-V2, Cl = centerline.
Tags: (a) map-independent low-error lane hypothesis, (b) temporal lane match/propagation, (c) SD-map to lane-level assignment, (d) E2E interface to planning.

## A. Key papers (18)

**1. SMART: Advancing Scalable Map Priors for Driving Topology Reasoning.** Ye, Paz, Zhang, Guo, Huang, Christensen, Wang, Ren; ICRA 2025; arXiv 2502.04329.
Idea: a map-prior model trained on SD map + satellite, supervised by geo-referenced HD maps (sensor independent), outputs a lane graph (200 centerline queries, DETR decoder + GCN, lane-lane topology matrix). Its features are frozen and plugged into online methods (replace TopoNet BEV queries; cross-attention for TopoMLP).
Numbers (html, OV2): TopoNet DET_l 28.6 -> 46.1, TOP_ll 10.9 -> 27.5, OLS 39.8 -> 51.1; TopoMLP OLS 44.1 -> 53.1. Prior alone (offline): DET_l 37.9, TOP_ll 31.9. Modality ablation OLS: SD only 44.1, satellite only 48.1, both 51.1. No SD-noise experiment in the paper.
Borrow: (c) SD map -> lane graph needs only prior channels; (a) the prior model is exactly the thing we must not trust blindly; use its output only as a candidate to verify.

**2. TopoSD: Topology-Enhanced Lane Segment Perception with SDMap Prior.** Yang et al.; arXiv 2411.14751 (Nov 2024, no venue listed).
Idea: SD map as 6-channel raster canvas (roads 6 m wide, 0.125 m/cell) plus vector tokens, injected into BEV features and queries; topology-guided decoder.
Numbers (html, OV2 subset A): LaneSegNet mAP 33.5 -> 40.2, TOP_lsls 25.4 -> 34.5. Raster-only 36.8, tokens-only 37.2 mAP. Noise robustness (Gaussian shift std 5 m, rot <=5 deg, prob 0.5): clean-trained model 40.2 -> 23.6 mAP (-41.3%); noise-trained model 35.1 clean, 34.6 noisy (-1.4%). So noise augmentation costs about 5 mAP clean.
Borrow: (c) the exact noise recipe and the clean-vs-robust trade-off; (a) noise-trained model as the "safe" variant.

**3. Online Navigation Refinement (MAT / OMA).** Wan et al.; arXiv 2507.07487 (v5 Jan 2026, no venue).
Idea: associate SD-map roads with online-perception lane vectors. Vectors (start, end, direction) -> MLP; path-aware attention (tokens ordered by topological path) + spatial attention (Hilbert-curve order); each centerline gets a softmax over roads; topology-constrained beam search; many lanes to one road. Dataset OMA: 30K scenarios, 2.6M lane vectors.
Numbers (html): NR-F1 val: MAT-L 78.7 vs EAM3 72.9 vs HMM 70.1 (34-70 ms vs 345-465 ms). Test on predicted maps (MAT-L): MapTR 41.9, MapTRv2 45.0, SeqGrowGraph 54.9. Appendix robustness numbers were not read.
Borrow: (c) the task definition (lane -> road assignment, NR-F1) and an HMM-style map-matching baseline; (d) output = lane path with road id.

**4. NavMapFusion: Diffusion-based Fusion of Navigation Maps.** Monninger, Zhang, Staab, Ding; WACV 2026; arXiv 2512.03317.
Idea: OSM road polylines enter by cross-attention; prior/perception mismatch is treated as diffusion noise, so agreeing regions are reinforced and outdated segments suppressed. BEV-feature dropout p=0.30 forces use of the prior.
Numbers (html): +21.4% relative at 100 m (22.4 -> 27.2 mAP). Gain grows with range: 60x30 m +0.0% (35.6), 120x60 m +52.5% (11.8 -> 18.0; variant without dropout). Alignment matters little: trained aligned 26.6/26.0 (aligned/unaligned test) vs trained unaligned 26.8/26.5. Training noise sigma=1 m holds up to eval sigma=3 m (curve only).
Borrow: (a)/(c) prior value is at long range; near range the sensor-only hypothesis is already good. Our +-50 m grid is mid-range, so expect small SD gain close to the ego.

**5. Coherent Online Road Topology Estimation and Reasoning with SD Maps.** Pham, Witte, Behley, Betz, Stachniss; IROS 2025; arXiv 2507.01397.
Idea: lane segments + topology + boundaries with hybrid lane-segment encodings (prior + denoising queries) and past frames for temporal consistency, in one model. Abstract gives no numbers.
Borrow: (b)+(c) the SD prior and previous frame enter the same query set; denoising training stabilises both.

**6. Enhancing Lane Segment Perception with Crowdsourcing Trajectory Priors (TrajTopo).** Jia et al.; arXiv 2411.17161 (v2 Apr 2025).
Idea: heatmap + vector-token prior, a spatial alignment module (predicted per-position offsets, bilinear warp) and a confidence fusion y = a*F_bev + b*F_prior with a+b=1 (1x1 conv + softmax).
Numbers (html, OV2): LaneSegNet AP_ls 32.30 -> 42.30, TOP_lsls 25.40 -> 34.65; alignment module +1.17 AP (41.13 -> 42.30). No noise-robustness test.
Borrow: (a)/(c) per-pixel softmax gate between sensor branch and prior branch is a 10-line module for a U-Net.

**7. MapTracker: Tracking with Strided Memory Fusion.** Chen, Wu, Tan, Ma, Furukawa; ECCV 2024 oral; arXiv 2403.15951.
Idea: mapping as tracking. Raster (BEV) and vector latents are kept in a memory; tracked element latents are propagated by PropMLP conditioned on ego motion (quaternion + translation); BEV memory of 4 past frames at 1/5/10/15 m strides is fused; Gaussian noise is added to the pose during training.
Numbers (html, consistent GT): nuScenes mAP/C-mAP: MapTRv2 69.5/50.5, StreamMapNet 70.4/56.4, MapTracker 76.1/69.1; Argoverse2 76.9/68.3. Geo-disjoint 100x50 m nuScenes: StreamMapNet 23.0/14.6 -> 36.2/27.5. Stride {1,5,10,15} m 76.1 vs no strides 74.9.
Borrow: (b) distance-strided history (not just last frames) plus pose-noise training; C-mAP-style consistency metric.

**8. HRMapNet: Enhancing Vectorized Map Perception with Historical Rasterized Maps.** Zhang et al.; ECCV 2024; arXiv 2409.00620.
Idea: accumulate all past predictions into a global raster (cell +30 if predicted, -1 if not; no confidence filtering, ~1 MB/km), retrieve the local crop and feed it to BEV features and query init.
Numbers (html, nuScenes 24 ep): MapTRv2 mAP 61.5 -> 67.2; StreamMapNet 60.4 -> 66.3. Pose noise sigma_t 0.2 m: 66.2; sigma_r 0.02 rad: 64.6 -> 63.8, still above baseline 61.5. Rainy: 50.8 -> 62.6 even with empty initial history; night 39.6 -> 42.6.
Borrow: (b) the asymmetric evidence update is the cheapest error-damping temporal filter for a raster head; directly usable on our 200x200 grid.

**9. TopoStreamer: Temporal Lane Segment Topology Reasoning.** Yang et al.; arXiv 2507.00709 (v5 Nov 2025, no venue).
Idea: BEV features warped by relative pose and fused by GRU; top-k (30% of 200) confident queries propagated via MLP(concat(Q, pose)); supervision by tracking instance IDs; lane denoising (shift factor 0.2, 50% label flip).
Numbers (html, OV2 A): mAP 32.6 (LaneSegNet) -> 36.6; TOP_lsls 25.4 -> 28.5; Cl OLS 40.7 -> 44.4. Propagated fraction sweep mAP: 10% 34.8, 30% 36.6, 50% 34.3, 75% 32.9.
Borrow: (b) propagate only a minority of hypotheses; too much history hurts (over-reliance).

**10. FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models.** Yang et al.; arXiv 2507.23325 (v4 Nov 2025).
Idea: stream (slow) and newly initialised (fast) queries are supervised in parallel; pose-conditioned latent world models propagate queries and BEV; at inference the single-frame output is used if pose is missing or inaccurate.
Numbers: abstract mAP 37.4 vs 33.6 prior SOTA; OLS 46.3 vs 41.5 (subset ambiguous). html: temporal 37.4 / TOP 29.6 vs same model single-frame 34.1 / 26.6. No pose-noise experiment.
Borrow: (b)+(a) explicit fallback to the prior-free head when pose/history is untrusted.

**11. Uni-PrevPredMap.** Peng et al.; RA-L 2026 (per arXiv page); arXiv 2504.06647.
Idea: one model for no-prior, previous-prediction prior and prior+HD-map; tri-mode training, sampling ratio N:T:T&M = 0.5:0.3:0.2; ~half of prior vectors displaced (mean ~3 m) during training.
Numbers (html): map-absent mAP nuScenes 74.0 (24 ep) / 77.0 (72 ep), Argoverse2 72.3. Perturbation table (mAP): perfect 85.6, instance displacement +-3/6/9 m 81.9/80.9/80.3, deletion 82.5, whole-frame displacement +-3/9 m 77.0/74.6, empty map 74.0. Whole-frame shifts are far more harmful than per-instance noise.
Borrow: (a)+(b) prior dropout in training so that the head stays valid without prior; test global shift, not only local jitter.

**12. Is Your HD Map Constructor Reliable under Sensor Corruptions? (MapBench).** Hao et al.; NeurIPS 2024 D&B; arXiv 2406.12214.
Idea: 29 camera/LiDAR corruptions, 31 models. Numbers (html): MapTR camera-only 50.3 -> 15.0 (camera crash) / 14.2 (frame lost); StreamMapNet 63.4 -> 13.4 / 15.5; camera snow 50.3 -> 3.8. Temporal vs non-temporal StreamMapNet (retrained, other split): mRR 55.5 vs 47.1; camera-crash mAP 10.3 (of 28.0) vs 5.0 (of 23.8). Text and table disagree for HIMap camera-missing.
Borrow: (a) corruption protocol (camera crash / frame lost) to evaluate any hypothesis head; temporal gain under failure is real but single-family evidence.

**13. LaneGAP (Lane Graph as Path).** Liao et al.; ECCV 2024; arXiv 2303.08815. Paths (traffic flow) instead of pieces, Path2Graph recovers the graph; abstract: +1.6 mIoU over TopoNet on OV2. Borrow: (d) path is the planner-facing primitive.

**14. CGNet: Continuity Preserving Online CenterLine Graph Learning.** Han, Yu, Li; ECCV 2024; arXiv 2407.11337. Junction-aware queries, Bezier-space continuity between connected segments, memory-based GNN topology refinement; abstract has no numbers. Borrow: (d) continuity at junctions, which planners need.

**15. TopoMaskV2.** Kalfaoglu et al.; ECCV 2024 VCAD workshop; arXiv 2409.11325. Cl as instance masks + quad-direction flow labels, mask-to-centerline post-processing. OLS 44.1 -> 49.4 (A), 44.7 -> 51.8 (B). Borrow: (a)/(d) raster head with direction channels fits our U-Net directly.

**16. E2E systems and their map interface.** VAD (Jiang et al., ICCV 2023, 2303.12077), SparseDrive (2405.19620), DiffusionDrive (Liao et al., CVPR 2025, 2411.15139).
VAD (html): 100 map queries x 20 points; ego query cross-attends map queries; training-time constraints: boundary distance (<1.0 m hinge), lane direction vs lane-divider vector within 2.0 m, both on map elements with confidence >0.5. Ablations: no ego-map interaction L2 0.86 m / col 0.29% vs full 0.72 / 0.22%; raster instead of vector map col 0.39% vs 0.22%. SparseDrive: map instance features + 20-pt polylines join agent-map cross-attention with the ego; collision-aware rescore 0.12 -> 0.08%. DiffusionDrive: trajectory features cross-attend agent/map queries (nuScenes, on SparseDrive); NAVSIM variant has no map head, PDMS 88.1.
None of these consumes an explicit lane graph; interface = confidence-filtered query features/polylines.
Borrow: (d) export per-lane polyline + confidence + (new) uncertainty, not a raw mask.

**17. UncAD.** Yang et al.; ICRA 2025; arXiv 2504.12826. Laplace NLL per map point (mu, b); b-encoded uncertainty queries concatenated to map queries; trajectories whose NLL risk against boundary ellipses exceeds a threshold are zeroed. nuScenes (html): VAD collision 0.23 -> 0.17%, drivable-area conflict 1.39 -> 0.80%; SparseDrive 0.10 -> 0.07%, 1.06 -> 0.88%; +1.9% params. Related: 2507.18498 (IROS 2025) finds uncertainty helps mainly at abrupt kinematic changes, up to 23.6%. Borrow: (d) calibrated uncertainty as the planner interface.

**18. LaneDiffusion.** Wang et al.; arXiv 2511.06272 (page states ICCV 2025; unverified). Diffusion in BEV-feature space produces a lane-centerline prior injected into BEV features; decoded to centerlines + topology. Reported gains over previous SOTA (abstract): GEO F1 +4.2, TOPO F1 +4.6, JTOPO F1 +4.7, APLS +6.4. Borrow: (a) a generative lane-prior head working directly on BEV features (our setting) for occluded centerlines.

## B. Also checked (abstract verified)
Priors/SD: SMERF 2311.04079 (up to 60% gain), P-MapNet 2403.10521 (+18.73 mIoU / +8.50 mAP), 2408.01471 IROS 2024 (+30% Cl mAP), SDTagNet 2506.08997 NeurIPS 2025 (+5.9 mAP over no prior, +3.2 over SD-prior baselines), D2HDMap 2606.20725 (IV 2026; noise-aware training, 44.8 mAP geo-disjoint), Bateman 2406.01961 (sim-to-real gap of synthetic prior perturbation), SD-map survey 2409.02415. Lane graph: SeqGrowGraph 2507.04822 (ICCV 2025), TopoGPT 2606.31814 (ECCV 2026; +6.4 lane / +11.6 point), TopoFG 2511.12590 (OLS 48.0 A / 45.4 B), UniTopo 2605.08911, TopoLogic 2405.14747 (TOP_ll 23.9, OLS 44.1 A). Temporal: StreamMapNet 2308.12570, SQD-MapNet 2401.09112, MemFusionMap 2409.18737, MambaMap 2507.20224, voxel history map 2409.00593 (RA-L 2025), stability benchmark 2510.10660 (mAP and stability are largely independent). Robustness: SafeMap 2507.00861 (ICML 2025, missing cameras), FlexMap 2601.22376. Radar: TopoBDA 2412.18951 (OV2 subset B, Cl OLS_l camera 49.9 -> camera+radar 54.8, camera+LiDAR 61.6; radar fused in voxel space). E2E: MAP 2509.13926, VADv2 2402.13243, Hydra-MDP 2406.06978, UniAD 2212.10156.

## C. Ranked: five ideas for the 3-hour budget (frozen 39-ch 200x200 BEV, 0.5 m/px, small U-Net)
1. **Prior dropout + noise-augmented prior channels.** Add SD raster channels; train with modes {no prior 50%, prior 50%} (Uni-PrevPredMap ratio idea), prior jitter Gaussian shift up to 5 m, rot <=5 deg, p=0.5 (TopoSD) plus occasional whole-frame shift. Evaluate a corruption sweep. By-product: a prior-free head = hypothesis generator (a). Cheapest, most evidence.
2. **Pose-warped previous-prediction raster with evidence accumulation.** HRMapNet update (+30/-1 per cell), warp with ego pose, add as channels; train with pose noise (sigma_t 0.2 m, sigma_r 0.02 rad) and history dropout; fast/slow fallback to the single-frame output (FASTopoWM) when pose is unreliable (b). Test with MapBench-style frame-lost/camera-crash on the frozen features.
3. **Per-pixel gated fusion + learned offset warp.** Softmax gate (alpha+beta=1) between sensor branch and prior branch with predicted offset field (TrajTopo); check that gate mass moves to sensor branch under shifted priors (a)(c).
4. **Agreement/uncertainty output for planning.** Compare prior-free vs prior-conditioned heads (disagreement = mismatch signal, NavMapFusion view) and add Laplace/NLL or entropy output; evaluate calibration and precision-at-agreement; export polyline + confidence + uncertainty (d).
5. **Non-learned SD->lane assignment baseline.** Skeletonise predicted centerline raster to polylines, assign to SD roads by HMM/beam search under road-connectivity constraints, score NR-F1-style (c); add direction channels (TopoMaskV2) as an auxiliary target if time remains.

## D. Claims I could NOT verify
- Only the arXiv abstract and, where marked "(html)", the HTML text via a summarising fetch tool were read; table values were not cross-checked against PDFs. Hence possible summariser slips.
- Inconsistencies seen: Uni-PrevPredMap displacement range ([0,6] vs [-6,6] m) and its Table 6 mAP (85.6) uses a different setting than the map-present 80.9, so only relative patterns are reported; MapBench text vs table for HIMap camera-missing; FASTopoWM Cl OLS 46.3 is called subset A in the abstract but subset B in the html; TopoStreamer abstract numbers changed between versions.
- LaneDiffusion venue "ICCV 2025" appears on the fetched page but the arXiv id is Nov 2025; treat as unverified. Uni-PrevPredMap RA-L 2026 vol/issue taken from the arXiv page.
- ONR robustness to SD shift/noise (its appendix G.2/G.3) was not read; SMART has no SD-noise test; TrajTopo and FASTopoWM have no pose/prior-noise numbers.
- UniAD's map-to-planner interface (query based) is confirmed only at abstract level; Hydra-MDP and VADv2 map interfaces not verified in full text. NMP (CVPR 2023), MapEX, SAM-Road++ (2411.16733, satellite road graphs), RoadNet, TopoMLP++, RelTopo, GeoReFormer were not opened.
- No paper found that studies camera+radar lane-graph prediction with temporal fusion; radar evidence is TopoBDA (one table, nuScenes subset B) only. No benchmark found for lane-topology (not vector-map) under sensor failure.
- No paper found that injects history-pose error into a temporal lane-graph model with numbers, except HRMapNet (vectors) and MapTracker/ HRMapNet noise training.
- Housekeeping: one WebFetch of an arxiv /pdf URL auto-cached a 3.2 MB PDF in the tool-results folder; I deleted it. No other files were saved.
