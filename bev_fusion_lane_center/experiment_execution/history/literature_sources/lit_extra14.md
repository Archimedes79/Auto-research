# Additional literature (2025 - mid 2026), 14 verified entries

Scope: camera+radar BEV lane-centreline graph, route/SD-map assignment to lane elements,
temporal fusion, planner-side consumer. Not repeated here: the classic works already cited in loops 1-13.
Each entry was opened on its arXiv page (or the official proceedings page) on 2026-10-19.
Title, authors, date and venue field were read there. Relevance sentences are our own wording.
"arXiv only" means the arXiv page lists no venue.

## 1. Table

| key | authors | venue, year | URL | relevance to our building blocks | theme |
|---|---|---|---|---|---|
| MAP25 | Yin et al. | ICCV Workshops 2025 (accepted, per arXiv comment) | https://arxiv.org/abs/2509.13926 | Feeds online-map features plus ego state into the planner and trains both together; a gate weights the map branch by ego state. | a |
| UncAD25 | Yang et al. | ICRA 2025 | https://arxiv.org/abs/2504.12826 | Online map comes with a per-coordinate uncertainty that steers prediction and planning and ranks candidate trajectories by collision risk; trained inside E2E planners on nuScenes. | a, d |
| FASTopoWM25 | Yang et al. | arXiv only, 2025 | https://arxiv.org/abs/2507.23325 | Lane-segment topology with a fast and a slow latent memory (query-level and BEV-level) carried across frames, aimed at E2E driving. | c, a |
| TopoStreamer25 | Yang et al. | arXiv only, 2025 | https://arxiv.org/abs/2507.00709 | Streaming lane-segment topology with cross-frame attribute constraints and lane-segment denoising queries on OpenLane-V2. | c, d |
| SafeMap25 | Hao et al. | ICML 2025 | https://arxiv.org/abs/2507.00861 | Online map stays usable when camera views are missing: missing regions are reconstructed and the BEV is corrected by distilling from a full-view model. | c |
| SEFMAP26 | Fu et al. | arXiv only, 2026 | https://arxiv.org/abs/2602.21589 | Camera+LiDAR map prediction with expert fusion, uncertainty-aware gating between sensors, and masked-input training for degraded sensors. | c, d |
| ONR25 | Wan et al. | arXiv only, 2025 | https://arxiv.org/abs/2507.07487 | Learns to associate an SD-map route (road level) with online-perceived lanes to give lane-level guidance; transformer with path-aware attention, own benchmark and metric. | b |
| CoherentSD25 | Pham et al. | IROS 2025 | https://arxiv.org/abs/2507.01397 | Lane segments and their topology predicted with SD-map prior encoded into queries, denoising queries and past-frame consistency in one model. | b, c, d |
| NavMapFusion25 | Monninger et al. | WACV 2026 | https://arxiv.org/abs/2512.03317 | Coarse navigation map treated as a noisy prior that a diffusion process refines together with sensor features for online vectorised mapping. | b |
| LaneDiffusion25 | Wang et al. | ICCV 2025 | https://arxiv.org/abs/2511.06272 | Centreline graph learning where a generated prior is injected at BEV-feature level before centreline and connectivity decoding. | d, b |
| UncMatters26 | Gogoi et al. | arXiv only, 2026 | https://arxiv.org/abs/2603.20076 | Online map with a structured (correlated) covariance across points instead of independent per-point noise, for downstream motion prediction. | d |
| NaviHydra25 | Wu et al. | arXiv only, 2025 (v2 2026) | https://arxiv.org/abs/2512.10660 | Shows planners often ignore the given route command at intersections; adds a binary compliance metric, a counterfactual-command dataset and a scoring planner trained to follow commands. | e |
| GoalFlow25 | Xing et al. | CVPR 2025 | https://arxiv.org/abs/2503.05689 | Picks a goal point from a candidate set first, then generates trajectories toward it with flow matching; goal conditioning keeps trajectories from drifting. | e |
| SpaRC-AD25 | Wolters et al. | arXiv only, 2025 | https://arxiv.org/abs/2508.10567 | Camera+radar fused in sparse queries for detection, tracking, map polylines, motion and planning; first radar-camera E2E baseline we found that includes online mapping. | f, a |

Note on NaviHydra25: the arXiv title of the current version is "Closing the Navigation Compliance Gap in
End-to-end Autonomous Driving". Version 1 carried the name NaviHydra in the title.

## 2. What this means for our design

- **Assignment head (ONR25, CoherentSD25, NavMapFusion25).** ONR25 is the closest published relative of our
  route-to-lane assignment. It also learns road-to-lane association from noisy, misaligned data. We can reuse its idea
  of a dedicated association metric and test misaligned routes as a stress case. CoherentSD25 and NavMapFusion25 use the
  SD map as an input prior. We use it as an output attribute. A clean ablation is prior-as-input versus
  assignment-as-output, with a shuffled or shifted route as control to detect shortcut use.
- **Denoising queries and frame consistency (TopoStreamer25, CoherentSD25).** Both use denoising queries on lane
  segments. This supports our query denoising and verification blocks. TopoStreamer25 also adds a cross-frame
  constraint on attributes. A consistency loss on the route attribute across frames fits our temporal fusion cheaply.
- **Memory design (FASTopoWM25).** A fast and a slow memory path is a small change to test against our single
  memory. It also gives a natural place to hold the route attribute slowly while geometry updates quickly.
- **Failure training (SafeMap25, SEFMAP26).** Both train with missing or masked inputs. SEFMAP26 adds a gate that
  weighs sensors by confidence. Our failure cover and chain-under-failure tests can adopt the same training-time
  masking protocol, with radar dropout as the analogue to their LiDAR or camera masking. The gate output is a
  ready-made per-sensor trust signal. Both papers use camera or LiDAR, not radar, so the radar case stays our own result.
- **Trust signals to the planner (UncAD25, UncMatters26).** UncAD25 passes map uncertainty to the planner and uses it
  to select trajectories. UncMatters26 argues that errors along one lane are correlated. Our per-node and per-edge
  trust is independent by construction. A low-rank-plus-diagonal term along each lane would be a small extension and a
  fair comparison point. A planner-side selection rule that uses trust, as in UncAD25, is a concrete consumer test.
- **Consumer with navigation attribute (NaviHydra25, GoalFlow25).** NaviHydra25 treats command compliance as its own
  failure mode with a binary metric and counterfactual commands. This matches our counterfactual-route test and gives
  us a metric to copy. GoalFlow25 conditions on a goal chosen from candidates. The end of the route-assigned lane set is
  a natural differentiable goal candidate for our consumer.
- **Joint training with a map interface (MAP25, UncAD25, SpaRC-AD25).** These papers pass map features or
  polylines to the planner and train end to end. None of the three, as far as we saw, hands over a graph with a
  per-element navigation attribute. We should state this as "among the works we found", not as a general claim.
  MAP25's ego-state gate is a cheap idea to gate trust by ego motion.
- **Radar expectations (SpaRC-AD25).** The reported mapping gain from radar is small (about 1.8 mAP) while detection
  and tracking gain much more. For lane geometry we should expect a modest radar benefit and a larger one in degraded
  conditions, so report both clean and degraded input.
- **Alternative to edge editing (LaneDiffusion25).** It improves the centreline graph by generating a prior in BEV
  space. This is heavier than our sparse verify/prune/merge on graph elements. It is a useful baseline for a
  "generate first, edit later" comparison and not a replacement.

## 3. Not verified or not used

Not used because the check failed or the fit was poor:

- Plug-and-Play Traffic Element Awareness for E2E driving (arXiv 2608.18035, ECCV 2026): page exists and was read,
  but the abstract is about traffic lights and signs, and it is dated August 2026, after the requested window.
  A search snippet mentioned lane adjacency; this was not confirmed in the abstract.
- OLRA, "Effective and Low-cost Lane-based Map Localization for Vehicle-Centric Route Generation"
  (arXiv 2606.16101): exists, but it is about localisation, under review, and off topic.
- Driver2Map (listed as arXiv 2608.01338 in a search result): not opened, ID and date not checked, dropped.
- ReSMap (NeurIPS 2026 poster page in a search result): not opened, venue and date unconfirmed, dropped.
- DiffSemanticFusion (arXiv 2508.01778), MapFusion (arXiv 2502.04377), SUPER-AD (arXiv 2511.22865),
  GraphWorld (arXiv 2606.16274): seen only as search results, abstracts not opened, dropped.
- The venue of FASTopoWM25, TopoStreamer25, ONR25, SEFMAP26, UncMatters26, NaviHydra25 and SpaRC-AD25 is not listed
  on arXiv. They may have been published later. We cite them as arXiv preprints.
- Radar-specific online lane-graph work: no paper found (2025 to mid 2026). The camera+radar entry is SpaRC-AD25 only.
  Camera+radar BEV segmentation work (see reserve) mostly targets vehicles, not lanes.
- No paper found that combines ADASIS-style horizon data with lane-level BEV perception.

Verified reserve (opened, in scope, not put in the table to keep it at 14):

| key | authors | venue, year | URL | note |
|---|---|---|---|---|
| SMART25 | Ye et al. | ICRA 2025 | https://arxiv.org/abs/2502.04329 | Learned map prior from SD and satellite data, used to lift existing topology models. (b) |
| SEPT25 | Pei et al. | IEEE RA-L, 2025 | https://arxiv.org/abs/2505.12246 | SD-map prior for perception and topology, with an intersection keypoint auxiliary task. (b) |
| TopoPoint25 | Fu et al. | NeurIPS 2025 | https://papers.nips.cc/paper_files/paper/2025/hash/ab6022d3d669b5baafa24c91d7c407a6-Abstract-Conference.html | Explicit lane endpoint detection and joint endpoint-lane reasoning for connectivity. (d) |
| MapUnc25 | Zhang et al. | IROS 2025 | https://arxiv.org/abs/2507.18498 | Map uncertainty helps mostly in non-steady driving; ego-kinematics gate. (d) |
| RoboMap25 | Hao et al. | IROS 2025 | https://arxiv.org/abs/2507.01484 | Camera+LiDAR map construction with modality-dropout training. (c) |
| SNG26 | Hua et al. | ICRA 2026 | https://arxiv.org/abs/2604.12208 | Argues E2E models underuse global navigation input; language-based guidance. (e) |
| NavigScene25 | Peng et al. | ACM MM 2025 | https://arxiv.org/abs/2507.05227 | Language navigation context paired with driving data. (e) |
| CaR1-25 | Montiel-Marin et al. | arXiv only, 2025 | https://arxiv.org/abs/2509.10139 | Camera-radar BEV baseline, vehicle class only. (f) |
| GaussianMap26 | Lyu et al. | arXiv only, 2026 | https://arxiv.org/abs/2606.31177 | Gaussian-primitive BEV for online vector maps; camera and camera+LiDAR, no radar. (f) |
| RESAR-BEV25 | Zeng et al. | IEEE T-ITS, accepted 2026 | https://arxiv.org/abs/2505.06515 | Camera-radar BEV segmentation, 7 classes (class list not confirmed in abstract). (f) |
