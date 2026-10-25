# PL02 - PDM-Closed / PDM-Open / PDM-Hybrid: full-text extraction

| field | value |
|---|---|
| paper | Parting with Misconceptions about Learning-based Vehicle Motion Planning. Daniel Dauner, Marcel Hallgarten, Andreas Geiger, Kashyap Chitta (Tuebingen AI Center, Robert Bosch GmbH). CoRL 2023. arXiv 2306.07962 [cs.RO]; v1 2023-06-13, v2 2023-11-02 (the text read here). Links opened: arxiv.org/abs/2306.07962, arxiv.org/html/2306.07962v2 (experimental HTML, main text only), github.com/autonomousvision/tuplan_garage (README, LICENSE, source files, YAML configs, commit list). Cross-reads for results the paper does not contain: PlanTF (arxiv.org/html/2309.10443, Table VI), interPlan (arxiv.org/abs/2404.07569 and /html/, Table III), NAVSIM metrics page (github.com/autonomousvision/navsim/blob/main/docs/metrics.md). |
| axis, tier, K ids | A7 (navigation context to the planner), tier 1 core; no K id exists for it (register.md has no PDM / IDM entry, checked); a new id is proposed in section 5 |
| level-1 note(s) | [PL02](../papers/PL02_pdm-closed-pdm-open-pdm.md), [C17](../papers/C17_pdm-closed-centre-line-of.md); related: [PL03](../papers/PL03_plantf.md), [PL04](../papers/PL04_pluto.md), [PL20](../papers/PL20_navsim-v1.md), [PL24](../papers/PL24_input-swap-evaluation-protocols.md) |
| code extraction | [tuplan_garage](../code/tuplan_garage.md) (Apache-2.0) |
| sources read | see "paper" row. Version v2 of the HTML (main text, Tables 1 - 4, no appendix). The supplementary PDF (danieldauner.github.io/assets/pdf/Dauner2023CORL_supplementary.pdf, 633 KB) came back as binary: **not read**; the scoring details that the paper defers to it were taken from the code instead (section 4). Note: the fetch tool stored a copy of that PDF in its own tool-results folder; I did not open it. All read 2026-10-25. |
| reading marks | Every statement about the paper and the repository is **digest** (seen through the summarising fetch tool; no raw page; numbers may be mis-transcribed; the tool refused to reproduce passages verbatim). Statements about our own project are **checked** (local files read). Statements marked **memory** were not seen. Section 5 is my analysis, not the paper's content. |
| extracted by, date | extraction agent (Claude, sub-agent of the literature session), 2026-10-25 |

## 1. Problem and claim
- Task: nuPlan planning (8 s ego trajectory from ego state, route, map, tracked agents). The paper argues that two jobs are mixed in the benchmark and should be solved separately: short-term planning (what the closed-loop score rewards) and long-horizon ego forecasting (what the open-loop score rewards). The open-loop score (OLS) and the closed-loop scores (CLS-R reactive agents, CLS-NR non-reactive) are misaligned.
- Claim 1: a rule-based planner on top of one input, the **centre-line of the route lane sequence** (found by a graph search), reaches the best closed-loop score when its proposals are simulated and scored (PDM-Closed, Val14 CLS-R / CLS-NR 92 / 93, learned planners 50 - 72).
- Claim 2: for open loop, an MLP on **only** the centre-line (+ 2 s ego history) is the best (PDM-Open, OLS 86); the centre-line alone is nearly the answer.
- Claim 3: PDM-Hybrid = PDM-Closed up to a correction horizon C = 2 s, then a learned offset: keeps the closed-loop score (92 / 93) and gives OLS 84. A preliminary version (with GC-PGP forecasting) won the 2023 nuPlan challenge (first of 25).
- Limits the authors state: no lane changes, needs HD map and precise (privileged) perception, no localisation error, map imperfection or misdetection is considered.

## 2. Method as built (digest, from the paper and the code)

### Pipeline
```
route roadblock ids (nuPlan, one list per scenario)
   | iteration 0 only: route_roadblock_correction (BFS repairs, loop cut)
   v
route roadblocks -> route_lane_dict {lane id -> lane}   (all lanes of the route roadblocks)
ego rear axle -> start lane (on-route lane under the rear axle, smallest heading error)
   |  Dijkstra over route lanes only, edge cost = lane baseline length,
   |  goal = any lane of the LAST roadblock of a window of <= 30 route roadblocks from the ego's roadblock
   v
lane chain -> centre-line = concatenated baseline_path.discrete_path (x, y, heading)  -> PDMPath (progress, linear interp1d, LineString)
   | iteration 0 only (the released code never re-plans, see section 4)
   v
3 lateral paths: centre-line, +1 m and -1 m (constant shift along the left normal)
   x 5 IDM policies (target speed = 0.2 / 0.4 / 0.6 / 0.8 / 1.0 x speed limit of the current lane)  ->  N = 15 proposals
observation: tracked objects (<= 50 vehicles, 25 pedestrians, 10 bicycles, 50 static, within 50 m)
   -> constant-velocity forecast, occupancy maps every 0.2 s over 8 s, red-light lane polygons as static obstacles
per proposal: IDM rollout of (progress, velocity) along its path, 4 s at 10 Hz  -> state array (N, 81, 11), the first 41 steps filled
simulate: batched one-step LQR tracker + kinematic bicycle, 10 Hz, 4 s -> states (N, 41, 11)
score: prod(no at-fault collision, drivable area, driving direction) x (5 progress + 5 TTC + 2 comfort) / 12    -> (N,)
select: argmax; extend the winner by the same IDM to 8 s (80 steps at 10 Hz);
        emergency brake replaces it if an at-fault collision is expected within 2 s AND ego speed <= 5 m/s
-> InterpolatedTrajectory, 8 s, 10 Hz
PDM-Hybrid: + MLP on (history, centre-line, PDM-Closed trajectory at 0.5 s) -> offset added after t0 + 2 s
PDM-Open:   MLP on (history, centre-line at 1 m up to 120 m) -> 16 waypoints at 0.5 s
```
Measured on Val14 without a stated machine: PDM-Closed 91 ms, PDM-Hybrid 96 ms, PDM-Open 7 ms (Table 2); the PDM code runs on CPU (numpy, shapely).

### Inputs, outputs
- **Inputs of PDM-Closed (what the code reads):** ego state at the current step (rear-axle pose, velocity, acceleration, steering angle and rate; nuPlan Pacifica vehicle parameters); the route as a list of roadblock / roadblock-connector ids; the map through the nuPlan map API (roadblocks, roadblock connectors, their lanes with baseline paths, polygons, successor edges, speed limit; carpark areas; the intersection layer; traffic-light status per lane connector); tracked objects with type, box, velocity (ground truth in the benchmark). Full list in section 5.1.
- **Output:** 8 s trajectory, 80 steps of 0.1 s (x, y, heading, velocity, acceleration, steering), i.e. pose-level, not only points.
- **PDM-Open / Offset feature frame:** all centre-line and trajectory points are expressed relative to the ego rear-axle pose; ego history is 10 past poses at 0.2 s plus the current one (11 states, each position, velocity, acceleration as SE(2) triples = 99 numbers); centre-line 120 samples at 1 m starting at the ego's projection on the centre-line, (x, y, heading) = 360 numbers.

### Equations
- **E1 (paper Eq. 1, IDM, as printed):** dv/dt = a (1 - (v / v0)^delta - (s* / s)^2); a acceleration limit, v0 target speed, s* safety margin (dynamic desired gap), delta exponent, s the gap to the leader; all set by hand.
- **E1 as implemented (code, `BatchIDMPolicy.propagate`):** s* = s0 + v T + v (v - v_lead) / (2 sqrt(a b)); s = max(x_lead - x - l_rear_lead, s0); dv/dt = a (1 - (v / v0)^10 - (s* / s)^2), **exponent hard-coded to 10** (a TODO says to make it a parameter); dv/dt clipped to [-b, a]; explicit Euler for progress and velocity. Defaults from the YAML: s0 = 1.0 m, T = 1.5 s, a = 1.5 m/s^2, b = 3.0 m/s^2, fallback target speed 15 m/s when a lane has no speed limit; v0 = fraction x speed limit with fractions [0.2, 0.4, 0.6, 0.8, 1.0].
- **E2 (paper Eq. 2, PDM-Hybrid):** w_Hybrid^t = w_Closed^t + 1[t > C] phi_Offset^t(w_Closed, c, h), C = 2 s by default; c centre-line, h ego history. In the code the cut is a splice: PDM-Closed states up to t0 + C, the offset model's states after it.
- **PDM-Open (unnumbered):** w_Open = phi_Open(c, h), L1 loss. Layers (code): state encoder Linear(99 -> 512) + ReLU, centre-line encoder Linear(360 -> 512) + ReLU, head Linear(1024 -> 512), Dropout 0.1, ReLU, Linear(512 -> 512), ReLU, Linear(512 -> 48) reshaped to 16 x (x, y, heading). About 1.05 M parameters (my sum of the layer sizes). PDM-Offset adds a trajectory encoder Linear(48 -> 512), head input 1536, and **adds the output to the flattened PDM-Closed trajectory** (about 1.34 M parameters, my sum).
- **Score (code, `PDMScorer`):** S_i = [prod over {NC_i, DAC_i, DDC_i}] x (5 P_i + 5 TTC_i + 2 C_i) / 12. Terms and thresholds in section 2 "Scoring".

### Scoring (from `pdm_scorer.py` and its helpers; the paper only says "closely resembles the nuPlan evaluation metrics" and defers weights to the supplement)
- **Ego area flags** per proposal and step (ego box = 4 corners + centre, centre of the box ahead of the rear axle): `MULTIPLE_LANES` = at least two drivable polygons contain a corner and no polygon contains all four; `NON_DRIVABLE_AREA` = at least one corner lies in no drivable polygon; `ONCOMING_TRAFFIC` = the box centre lies in no polygon of an **on-route** lane (despite the name there is no travel-direction test; the lane direction enters only through the route set).
- **NC no at-fault collision (multiplicative):** per step the ego polygon is intersected with the forecast occupancy map; objects already overlapping the ego at t = 0 are ignored for the whole scenario; collision type from geometry (stopped ego, stopped track, rear, front, lateral); at fault = front collision or collision with a stopped track, or lateral collision while the ego is in multiple lanes / off the drivable area. Score 0 for an agent, 0.5 for a static object, else 1.
- **DAC drivable area (multiplicative):** 0 if any step has `NON_DRIVABLE_AREA`, else 1.
- **DDC driving direction (multiplicative):** cumulative displacement of the box centre while flagged `ONCOMING_TRAFFIC` (largest contiguous segment): 1.0 below 2.0 m, 0.5 below 6.0 m, else 0.0.
- **P progress (weight 5):** projection of the box centre at the last and first step onto the **centre-line**, difference clipped at 0, multiplied by the multiplicative score, then **divided by the maximum over the proposals** (if that maximum exceeds 0.1 m; otherwise 1, forced to 0 where the multiplicative score is 0). Progress is relative to the best proposal, not to a fixed reference.
- **TTC (weight 5, binary):** the ego box is projected forward at constant speed at offsets 0, 0.3, 0.6, 0.9 s (code indices 0, 3, 6, 9 at 0.1 s); 0 if an object ahead is hit, or any object not behind is hit while the ego is in multiple lanes, off the drivable area or inside the intersection layer; skipped when ego speed < 0.005 m/s.
- **C comfort (weight 2, binary):** 1 only if all six checks hold over the whole horizon: longitudinal acceleration in [-4.05, 2.40] m/s^2, lateral acceleration within 4.89, jerk magnitude 8.37, longitudinal jerk 4.13, yaw rate 0.95 rad/s, yaw acceleration 1.93 rad/s^2 (Savitzky-Golay smoothed derivatives). A violation only removes the 2/12 term.
- **No speed-limit term** exists in the PDM scorer (the string "limit" does not occur in `pdm_scorer.py`); the speed limit enters only through the target speeds (fraction <= 1.0 of the limit). The paper's description of the nuPlan CLS includes speed-limit compliance; PDM's own scorer drops it.
- Selection is `argmax` over the 15 scores. If all scores are 0, `argmax` returns proposal 0 = centre-line at 20 % of the speed limit (my reading of the enumeration order: lateral index outer, longitudinal index inner; the centre-line is lateral index 0).

### Losses, labels, training
- **PDM-Closed:** none, no parameters.
- **PDM-Open and PDM-Offset:** L1 on the future waypoints (paper: "L1 loss"); about 177 k training scenarios (all 70 scenario types, at most 4 k per type); target = the logged expert trajectory. The loss weights of PDM-Offset, epochs, optimiser, learning rate, batch size: **not found** in the pages read (the supplement may have them).
- Feature definition for the learned parts: section 2 "Inputs, outputs".

### Inference (hidden rule steps)
1. Route repair once per scenario (iteration 0): if the ego's roadblock is not on the route, backward BFS from the first route roadblock (depth 15), else forward BFS from the ego's roadblock towards the first three route roadblocks (depth 30); every consecutive pair of route roadblocks that are not linked by an incoming edge is bridged by a forward BFS (depth 30, accepted if the path has at least 3 entries); the route is cut at the first roadblock connector whose polygon overlaps an earlier connector by more than 1 area unit (loop removal).
2. Start lane: among on-route lanes whose polygon contains the rear axle, the smallest absolute heading error to the nearest baseline point; else the lane containing the box centre; else the closest lane polygon (no distance limit).
3. Dijkstra over **route lanes only**; if the goal is never popped, the returned path ends at the deepest, then cheapest, expanded lane and the `path_found` flag is not used.
4. Proposals and centre-line are built **only when the iteration counter is 0**; later steps reuse them and only refresh the speed limit (a TODO says "find additional conditions to trigger re-planning").
5. The path end is a virtual stationary leader: with no object ahead, the IDM leader progress is set to the path length and its rear length to half the vehicle length.
6. Leading-agent lookup is refreshed every 2 steps (0.2 s) and uses a corridor of the ego width along the path up to progress + max target speed x 8 s.
7. Emergency brake (only if the best proposal predicts an at-fault collision within 2.0 s **and** the ego speed is at most 5.0 m/s): a proportional braking profile towards zero speed (gain 10 above 0.2 m/s) with the command clamped to [-4.05, 2.40] m/s^2. The tool's reading of one line suggests a sign oddity at speeds between 0.2 and about 0.4 m/s (negative displacement); not verified.

## 3. Numbers (digest; table numbers as printed in v2 unless another paper is named)

Table 1 (nuPlan, OLS-CLS trade-off; columns: centre-line, history used; CLS-R / CLS-NR / OLS):

| method | centre-line | history | CLS-R | CLS-NR | OLS |
|---|---|---|---|---|---|
| IDM (a = 1.0) | yes | - | 77 | 76 | 38 |
| IDM (a = 0.1) | yes | - | 54 | 66 | 48 |
| PDM-Open | no | no | 51 | 50 | 69 |
| PDM-Open | no | yes | 38 | 34 | 72 |
| PDM-Open | yes | no | 54 | 53 | 85 |
| PDM-Open | yes | yes | 54 | 50 | 86 |

The centre-line is worth 16 OLS points (69 -> 85) for the learned model; history adds 1; history alone makes closed loop worse (38 / 34).

Table 2 (Val14, 1,118 scenarios = 100 per scenario type x 14 types, some types have fewer; Time in ms, machine not stated):

| method | representation | CLS-R | CLS-NR | OLS | ms |
|---|---|---|---|---|---|
| Urban Driver (open-loop re-implementation) | polygon | 50 | 53 | 82 | 64 |
| GC-PGP | graph | 55 | 59 | 83 | 100 |
| PlanCNN | raster | 72 | 73 | 64 | 43 |
| IDM | centre-line | 77 | 76 | 38 | 27 |
| PDM-Open | centre-line | 54 | 50 | 86 | 7 |
| PDM-Closed | centre-line | 92 | 93 | 42 | 91 |
| PDM-Hybrid | centre-line | 92 | 93 | 84 | 96 |
| PDM-Hybrid* (GC-PGP forecasting, challenge entry) | graph | 92 | 93 | 84 | 172 |
| Log replay | GT | 80 | 94 | 100 | - |

The centre-line and graph variants of PDM-Hybrid score the same; the centre-line one is faster. Log replay stays below 100 CLS because the LQR tracker drifts from the logged path (authors' explanation, paraphrased).

Table 3 (2023 nuPlan challenge, test; score = mean of CLS-R, CLS-NR, OLS): PDM-Hybrid* 93 / 93 / 83, score 90 (rank 1 of 25); hoplan 89 / 88 / 85 (87); pegasus_multi_path 82 / 85 / 88 (85); Urban Driver 68 / 70 / 86 (75); IDM 72 / 75 / 29 (59). The final centre-line version could not be tested (leaderboard closed).

Table 4 (ablations):
- (a) PDM-Hybrid correction horizon C: 0.0 s -> CLS-R 58, OLS 87; 2.0 s -> 92, 84; 2.5 s -> 92, 84; 3.0 s -> 92, 72.
- (b) PDM-Closed without a module (CLS-R, ms): base 92, 91; no lateral offsets 89, 55; no longitudinal proposals (single IDM speed) 88, 64; no forecasting of other agents 86, 90.
- (c) PDM-Open OLS: baseline 86; shorter centre-line 84; coarser centre-line 86; smaller MLP 84 (the lengths and strides of "shorter" / "coarser" were not in the text read).

Results of other papers (different protocol each; not comparable with Table 2 or with us):
- Test14 (PlanTF Table VI, columns OLS / NR-CLS / R-CLS, time in PlanTF's own setup): Test14-random (280 scenarios, 20 per type, random): IDM 34.15 / 70.39 / 72.42; PDM-Closed 46.32 / 90.05 / 91.64 (140 ms); PDM-Hybrid 82.21 / 90.20 / 91.56. Test14-hard (280 scenarios; the 20 worst per type of PDM-Closed's own run on 100 per type): IDM 20.07 / 56.16 / 62.26; PDM-Closed 26.43 / 65.07 / 75.18; PDM-Hybrid 73.81 / 65.95 / 75.79. Test14-hard is selected with PDM-Closed itself, so it is biased against PDM-Closed.
- interPlan (arXiv 2404.07569, Table III, 80 hand-built long-tail scenarios, 8 categories of 10): PDM-Closed 42 (Val14 92), IDM 31, PDM-Open 25, GC-PGP 10, Urban Driver 4. PDM-Closed scores 0 on the accident-site category and 0 on the goal sub-score of the lane-change scenarios ("by design unable to perform lane changes").
- NAVSIM (metrics page, v1): the PDM score is NC x DAC x (weighted TTC, EP, C) / sum of weights with a 4 s non-reactive simulation of the planned trajectory by an LQR controller; the numeric v1 weights were not printed on the page read (the project's own note lit8_b records 5, 5, 2).

Compute: see Table 2. Hardware: not stated in the paper. Breakdown: the lateral offsets (3 paths instead of 1) cost 36 ms (91 vs 55), the five speeds 27 ms (91 vs 64).

Caveats: the benchmark uses ground-truth map and ground-truth tracked objects; Val14 is a subset of the nuPlan validation scenarios; reactive agents (CLS-R) follow IDM strictly along centre-lines and are unrealistically passive, non-reactive agents (CLS-NR) ignore the ego (authors' limitation paragraph).

## 4. Paper versus code (digest of the code, details in [tuplan_garage](../code/tuplan_garage.md))
- Paper: "graph search"; code: Dijkstra at lane level over route lanes (cost = baseline length) after a BFS repair at roadblock level.
- Paper: emergency brake on an expected collision within 2 s; code: additionally requires ego speed <= 5 m/s.
- Paper: IDM with exponent delta; code: exponent 10, hard-coded.
- Paper: scoring "closely resembles nuPlan metrics", weights in the supplement; code: NC, DAC, DDC multiplicative; progress 5, TTC 5, comfort 2; no speed-limit term; driving direction implemented as "box centre outside the route lanes".
- Paper: the centre-line is the plan's reference; code: the centre-line and the proposals are computed once per scenario (iteration 0), never re-planned.
- Paper: forecasting "constant velocity, 8 s, 10 Hz"; code: positions extrapolated linearly from a velocity along the travel heading, stored every 0.2 s (`observation_sample_res` 2) and indexed by the 10 Hz step.
- Code details not in the paper: the path end acts as a standing leader; objects colliding with the ego at t = 0 are ignored for the rest of the scenario; the object caps (50 / 25 / 10 / 50); `map_radius` 50 m; PDM-Hybrid runs its MLP on CPU and builds its centre-line feature from the *current* lane at every step although the closed-loop proposals stay those of iteration 0.
- Checkpoints of PDM-Open / PDM-Offset: Google Drive folder named in the README (file names not listed there).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Per frame, in the ego frame: lane centre-lines as ordered polylines with heading (1 m spacing is enough: PDM-Open uses 1 m, and a coarser stride did not hurt it, Table 4c), the directed successor relation between them, a route attribute that defines **one** lane chain from the ego lane forward, lane polygons (route lanes and all other drivable area), the ego state (velocity, acceleration, steering; nuScenes CAN bus), a speed limit, tracked objects (only for the collision / TTC terms). No images, no lidar, no global pose. |
| what it gives us | A7: a **consumer test bed** and a rule-based **baseline** for a perceived graph with a route flag, and a planner-aware map-quality metric (the project has none, OVERVIEW section 3, item 5): swap GT / perceived / degraded graph into the same planner and compare a driving-relevant error (path error, DAC, DDC, progress). It also shows how much of the driving score the centre-line alone explains (Table 1). |
| conflicts | Not differentiable (Dijkstra, shapely geometry, argmax over proposals): can only be an evaluation consumer, never on the training path. Released code needs the nuPlan devkit and the nuPlan map API (lane / roadblock objects), pinned `nuplan-devkit==1.2.2`; our env is Python 3.12, PyTorch 2.8. No simulator exists for nuScenes: the planner's internal 4 s simulation is available, a closed-loop rollout is not. The route enters as a **hard** lane set; there is no confidence input. No lane changes. Needs agents for the collision terms (our chain has no detector). |
| reuse mode | Re-implement from the description (numpy / torch, no nuPlan types); Apache-2.0 would allow adapting code with attribution, but the code is welded to nuPlan objects. |
| effort | S: centre-line + IDM baseline (about 1 day). M: PDM-Closed-lite with lateral offsets, speeds, LQR + bicycle simulation, raster scorer, emergency brake (about 1 week). L: original code behind fake nuPlan map objects (not recommended). Blockers: agents, speed limit, lane polygons (section 5.1). |
| expected gain and risk | Gain: a driving-relevant arbiter between graph designs (the project note N8 in Literature_Compatibility_Space_EN.md wants exactly this); published open-loop L2 hides graph quality (PL24: GT and predicted topology give identical L2). Risk: PDM's strength is its privileged map and its simulation score, not the centre-line alone (IDM on the same centre-line gets 77 / 76, PDM-Closed 92 / 93); a score gap between graphs may be dominated by the scorer's polygons, not by the centre-line; ego-status shortcuts do not apply to the rule-based parts. |
| feeds which experiment / K id | Proposed new technique, to be registered before use (not registered by me): **K-new "PDM-type rule-based consumer (route-lane centre-line, IDM proposals, simulated score) as graph test bed and baseline"**. Feeds the oracle / perceived / degraded swap next to E117 / E119. |

### 5.1 What map information PDM-Closed consumes, and what our graph can supply
(code reading, digest; "route set" = `route_lane_dict`)

| PDM input (nuPlan object / attribute) | used for | our source | verdict |
|---|---|---|---|
| route roadblock ids (list of str) | route lanes; goal of the search; route repair | per-vertex route probability from the route head (SD route) | replace: no roadblocks in our graph; route lanes = edges with route probability above a threshold (a rule step, fine for an evaluation consumer), the goal = farthest vertex of the chain |
| roadblock `interior_edges` (all parallel same-direction lanes), `incoming_edges`, `outgoing_edges`, `polygon` | route lane set, BFS repair, loop cut | none | not needed after the replacement above; the repair solves a problem of the nuPlan route list (route off the ego's roadblock, gaps), not of a perceived graph |
| lane / lane connector `baseline_path.discrete_path` (x, y, heading list), `baseline_path.length` | centre-line, Dijkstra cost | vertices + directed edges, resampled to 0.5 - 1 m, heading from the tangent | **available** |
| lane `outgoing_edges` | Dijkstra successor relation | directed edges with forks and merges | **available** (our strength) |
| lane `get_roadblock_id()` | window start and goal test | none | replace by chain length / route probability |
| lane `polygon` (route and non-route) | DAC, DDC, MULTIPLE_LANES, start-lane search | no width in the graph | **missing**: buffer the centre-line by a default half-width, or take the drivable area from the foundation's cached map logits (GaussianCaR cache holds 7 layers incl. `lane`, `road_segment`, `carpark_area`, SN01) as a raster; PDM tests four corners and the centre of the box against polygons, which is a mask lookup on a raster (whether our `road_segment` layer equals the "drivable area" class of the GaussianCaR paper is not verified) |
| `speed_limit_mps` of the current lane | target speeds | none perceived | **missing**: SD-map attribute if our simulated SD map has one (not checked), else the code's own fallback of 15 m/s |
| layers `ROADBLOCK`, `ROADBLOCK_CONNECTOR`, `CARPARK_AREA` within 50 m | drivable-area map | `lane` / `road_segment` / `carpark_area` logits (200 x 200, 0.5 m, +-50 m: the same 50 m radius) | available as raster, probabilistic |
| layer `INTERSECTION` (`is_in_layer`) | TTC rule only | junction head, SD map | optional; drop the rule or approximate |
| traffic-light status per lane connector | red-light obstacle polygons | none | missing; drop (no detector) |
| tracked objects (type, box, velocity) | forecast, NC, TTC, leading-agent IDM | none in the chain | missing: nuScenes GT boxes (privileged, label as such) or none (then NC and TTC are always 1 and the test bed measures map terms only) |
| ego state | start state, LQR, comfort | nuScenes CAN bus / ego pose | available |
| crosswalks, stop lines, lane markings, lane types | not used | - | not needed |

### 5.2 How robust is the centre-line extraction to missing, short or false lanes?
Not tested in the paper (no map imperfection is considered); the table is my reading of the code, no experiment.

| fault in the perceived graph | what the released code does | consequence | adapter rule |
|---|---|---|---|
| no route lane at all | `_get_starting_lane` returns None; the next line reads `speed_limit_mps` of it | exception | guard: fall back to a stopping trajectory |
| ego not on any route lane (lateral registration error) | picks the closest route lane polygon, no distance limit | centre-line starts beside the ego; LQR pulls back; offsets +-1 m cover only small errors | report the ego-to-lane distance next to the score |
| gap in the chain between route lanes | Dijkstra never pops the goal; returns the path to the deepest, then cheapest, expanded lane; `path_found` unused | truncated centre-line, no error | acceptable: "better short than false" holds |
| short centre-line (shorter than the stopping distance) | path end is a virtual stationary leader (leader progress = path length) | IDM brakes to stop at the end; progress is normalised by the best proposal, so no absolute penalty | conservative; at 50 m BEV range and 15 m/s the horizon already clips the plan |
| missing junction connector (foundation has no connector layer) | chain ends at the junction mouth | stop before the junction, zero progress through it | needs our junction / fork head or a straight-through fill |
| fork with both branches in the route set | Dijkstra takes the shorter branch (cost = length only) and ignores any probability | arbitrary branch | give the search a single-path route set, or cost = length - log(route probability) |
| spurious lane (false positive) outside the route set | ignored by the search | enters the drivable-area map: weakens DAC, and DDC treats it as non-route | harmless to the centre-line, can hide off-road driving |
| spurious lane inside the route set | can be chosen if shorter | wrong path | threshold route probability high, keep validity |
| lateral shift of the whole graph (1 m) | LQR follows the shifted reference; if the polygons come from the same graph, DAC / DDC stay consistent | the error shows only when the plan is judged against the true map | judge against GT polygons, not the perceived ones |

Because the original never re-plans (iteration 0 only), a per-frame perceived graph needs re-extraction every frame, and the "route repair" step is not needed.

### 5.3 Route flag: binary or soft; confidence
- PDM-Closed consumes the route as a **hard set** of roadblock ids; PDM-Open / PDM-Hybrid consume only the resulting polyline (the route is implicit). No variant has a confidence input.
- Binary is what the scorer needs: the route lane polygons decide DDC. A soft route probability has to be turned into (a) a threshold (evaluation-only rule step), or (b) an edge cost for the search, or (c) a multiplier on the proposal score (score x probability of the chosen chain). Idea (not in the paper, not tested): several route hypotheses as extra paths in the proposal set (K paths x 3 offsets x 5 speeds), with the final score multiplied by the chain's route probability, so that the scorer resolves a fork by simulation and the probability breaks ties; the route lane set must then be the union of the hypotheses.
- Validity / confidence of vertices has no input either; use it to cut the chain where validity drops below a threshold (the "short is better than false" rule) and report the cut length.

### 5.4 Adapter options
1. **A0 centre-line baseline (S):** follow the highest-route-probability chain through our directed edges from the ego's lane (or the nearest vertex), resample to 1 m, run IDM with fixed fractions, produce path points at 5 ... 40 m, compare with E117's 0.34 m path error and with the oracle-route 0.316 m. Answers "how far does a route-chain + IDM go without learning" (C17's idea).
2. **A1 PDM-Closed-lite (M):** A0 plus lateral offsets, five speeds, batched LQR + bicycle simulation, raster scorer (DAC / DDC / progress / comfort; collision and TTC only with GT agents), emergency brake. Torch or numpy; the IDM (about 215 lines in the original), the one-step LQR, the bicycle model and the comfort checks are small and have no nuPlan dependency apart from the vehicle parameters; the scorer's shapely polygon tests become mask lookups. Evaluate the winner against GT polygons and the future ego path from the nuScenes poses.
3. **A2 original code behind fake nuPlan objects (L):** build `Lane`, `RoadBlock`, `AbstractMap`, `EgoState`, `DetectionsTracks`, `TrajectorySampling` stand-ins; couples us to `nuplan-devkit==1.2.2`; not recommended.

### 5.5 Proposed experiment (not run; for the loop owner's card)
Arms on the same VAL frames (two seeds of the route head, paired scene bootstrap, noise band 0.010): (0) GT lane graph + GT route -> A1; (1) perceived graph + hard route (threshold 0.5) -> A1; (2) perceived graph + route cost in the search; (3) degraded graphs (drop 20 % of edges, truncate to 25 m, add a false branch) -> A1; (4) E117 on the same graphs. Metrics: path error at 5 ... 40 m, DAC / DDC / progress of the chosen proposal judged on GT polygons, ms per frame (CPU and GPU), chain length at the cut. Prediction to write down before the run: the GT-to-perceived gap in A1 is larger than in E117 (hard rule consumers amplify graph errors, learned consumers average them), and the short-chain behaviour keeps DAC high for truncated graphs.

## 6. Not found / open
- The supplement (binary PDF, not read): scoring details were taken from the code; any weights or thresholds that differ in the supplement are unknown.
- Hardware and the per-module runtime breakdown behind the 91 ms; PDM-Offset loss weights, epochs, optimiser, learning rate, batch size; the lengths of the "shorter" / "coarser" centre-line variants in Table 4c.
- No analysis of centre-line errors, no robustness test; Val14 / Test14 / interPlan all use the privileged map. No result with a perceived or noisy map exists in the pages read.
- Exact figure of the nuPlan speed-limit coverage (how many lanes have `speed_limit_mps`), and the lane-width distribution of nuPlan, are not given.
- The NAVSIM v1 weights could not be confirmed on the page read.
- HTML artefacts: the supplement is referenced but absent, some table references render blank.

## 7. Our experiments that touch it
- None runs a rule-based consumer (PL02 note: "none").
- E117: the learned consumer (93 k parameters) on the chain: 0.34 m path error with ego motion and route probability (system A 0.36 m). E119: the oracle route gives 0.316 m, the planner loss into the route head gives no gain. E121: planner loss through positions, no gain. These numbers are the targets of the A0 / A1 comparison above; they are path errors in metres, not PDM scores.
- The Literature_Compatibility_Space note (adapter N8, "planner-aware graph test bed") already proposes PL02 as the arbiter; this extraction supplies the details and the missing inputs.

## Corrections to the level-1 note (PL02 and C17)
1. C17 reports PDM-Closed OLS 44; Table 2 and the README give **42** (PL02 has 42).
2. PL02 (Fit) says the centre-line comes from "dynamic programming over route probability"; the method is **Dijkstra over route lanes with lane length as cost** after a BFS repair of the roadblock route; the route is a hard set, there is no probability anywhere.
3. PL02: "score weights in the supplement (not read)": they are in the code: multiplicative NC, DAC, DDC; weighted progress 5, TTC 5, comfort 2 (sum 12); there is no speed-limit term.
4. PL02 and C17: "emergency brake if a collision is expected within 2 s": the code also requires ego speed <= 5 m/s.
5. PL02 (Conflicts): "centre-line quality is the single point of failure": the scorer depends as much on lane **polygons** (drivable area, route lanes) as on the centre-line; the centre-line feeds only the proposals and the progress term.
6. New facts the notes lack: the released code computes the centre-line once per scenario (iteration 0); the path end acts as a standing leader; the IDM exponent is 10; Test14 and interPlan results (section 3).
7. C17 gives the effort as S, PL02 as M: S holds for the centre-line + IDM baseline, M for the PDM-Closed-lite with scorer (section 5.4).
