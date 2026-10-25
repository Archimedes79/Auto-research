# Verified-ray hypothesis tree: refined description and spec tests

Camera + radar BEV, lane centre-line graph · specification note, 2026-10-23 · refines section 4.2a of `Growth_and_Link_Architectures_EN.md` · feeds the gate of `Loop16_Triage_Plan_EN.md`

> **Status.** A numpy reference implementation with an *oracle* verification (distance to drawn toy lanes) exists and passes 18 spec tests (`scripts/p16_ref.py`, `scripts/test_p16_spec.py`). The network version is not written; its tests are specified in section 7. Numbers in this note come from the reference on toy scenes, not from a trained model.

## 1. What the reference changed in the earlier description

Writing the reference and its tests corrected five points of section 4.2a:

| earlier text | corrected | found by |
|---|---|---|
| likelihood p = P(last verified = k) times a direction prior, ranks the hypotheses | **validity nu = T_k* x direction prior** (survival up to the chosen endpoint, k* = most probable last verified sample); ranking by nu. P(k) only chooses the endpoint along a ray. Short certain stubs would otherwise win | beam figure (stubs and the backward ray won), T04, T06 |
| near duplicates suppressed by endpoint distance (3 m) | **dominance by segment:** a candidate whose endpoint lies within 1.2 m of the segment tip-to-endpoint of a longer candidate with at least half its validity is redundant; remaining endpoints closer than 2 m are duplicates (higher nu wins) | T09 (order invariance) |
| path likelihood as a product | **bottleneck:** pi(node) = min over the path of nu; a product would punish lanes with many short steps | design review of the curved branch |
| the merge of two paths appears by itself | **tree -> DAG needs two steps:** cluster living nodes closer than 1.5 m with similar heading, then **split every edge at a vertex that lies on it** (within 1 m, not near the ends). Without the split the merge vertex sat 18 m behind the true merge | T14 |
| a tree of depth D with b children has b^D nodes | in the reference **21 of 1022 slots are alive**: the tree is a fixed *index template*; compute only on the best M nodes per level (beam), see section 4 | T13, T14 |

Two limits the tests measured (oracle width sigma = 0.6 m, rays at +-9 deg): a fork is resolved as two hypotheses from about **6 deg** divergence (not at 3 deg or less); forks and merges are located where the lanes come within about **1 m** of each other (fork at 25 m -> vertex at 24 m, merge geometrically at 48 m on a tangential merge -> vertex at 41.6 m, i.e. 6.4 m early). The step length grows with the radius: first step 4, 4, 8, 8, 24 m for R = 10, 15, 30, 100, infinity.

## 2. Specification

Notation: tip = (position p, heading theta, token s, path validity pi); frame = ego BEV, x forward, y left, metres; phi = bearing of a ray relative to the tip heading. All defaults are in `CFG` of `p16_ref.py`.

| module | definition | default parameters |
|---|---|---|
| **M1 pattern** | 12 rays at phi in {0, +-9, +-20, +-50, +-90, +-125, 180} deg; length `r(phi) = max(r_floor, r_cap / (1 + (phi / phi_c)^2))`; samples on the shared rings, clipped to the ray: `{ring < r} + {r}`; 28 points. Seeds without heading: two-lobed version (the envelope is taken from the angular distance to the nearer axis direction 0 / 180 deg, rays at phi and phi + 180) | r_cap 24 m, r_floor 4 m, phi_c 20 deg, rings 4 / 8 / 12 / 18 / 24 m |
| **M2 verification and survival** | per sample k of a ray a score v_k in (0, 1) (oracle: `exp(-(d / sigma)^2)` clipped to [0.02, 0.99]; network: small causal 1-D convolution or recurrence over the ray tokens). `T_k = v_1 ... v_k`; `P[0] = 1 - v_1`, `P[k] = T_k (1 - v_(k+1))`, `P[K] = T_K`; `sum P = 1`. Endpoint k* = argmax over k >= 1 of P (expectation over k as the soft variant). Tolerant variant for gaps: `T_k = max(T_(k-1) v_k, skip x T_(k-2) v_k)`, endpoint = last k with T_k >= alive | sigma 0.6 m, skip 0.5 |
| **M3 hypotheses** | per ray a candidate (endpoint, nu) with `nu = T_k* x prior(phi)`, `prior = max(floor, exp(-(phi / 45 deg)^2))` (learned in the network); coverage mask: endpoint within 2.5 m of a committed point -> nu x 0.02; redundancy by dominance and duplicate suppression (section 1); the **b best** become children (b >= 2), ordered left to right by bearing, padded with dummies so that the shape is fixed; a child lives if nu >= 0.3. Child heading = direction tip -> endpoint | b = 2, alive 0.3, cov 2.5 m, nms_seg 1.2 m, nms_r 2 m |
| **M4 tree** | levels d = 1 ... D; node i of level d has parent `parent[i]` of level d - 1; dead parents give dummy children; `pi = min(pi_parent, nu)`; node lives if pi >= alive. **Merge:** cluster living nodes (distance < 1.5 m, heading difference < 30 deg, position = pi-weighted mean), edges between clusters, then split edges at vertices on them (snap 1 m, ends 1.5 m). Beam budget: keep the M living nodes with the highest pi per level (selection only) | D = 8 in the reference (steps up to 24 m), M = 16 per root / 256 per frame planned |
| **M5 cascade** (network) | per level three stages: **A generate** (M1 - M3), **B refine** (a small teardrop, r_cap 6 m, around every child with its own heading: dp <= 1 m, dtheta, re-verification), **C rescore** (attention over siblings, the ancestors up to the root and committed points: final nu and the 128-d token). Weights shared over the levels, separate per stage; the pattern gain gets a depth embedding and the parent's pi, because uncertainty grows with depth (ancestors are hypotheses themselves). Deep supervision on A, B and C | gain in [0.5, 1.5] x default |
| **M6 time** | previous set (positions, headings, pi, tokens) warped by the ego motion: `p_t = R(-psi)(p - d)`, `theta_t = theta - psi`; the warped nodes are prior roots and extra keys of stage C; `pi_t = sigmoid(a logit(pi_(t-1)) + b logit(nu_t))` | a 0.9, b 1.0 (learned in the network) |

Shapes of the network version (per frame, R roots, N_d nodes at level d, J = 12 rays, K <= 5 samples): pattern positions `[N_d, J, K, 2]`, features `[N_d, J, K, C]`, verification v, survival T `[N_d, J, K]`, P `[N_d, J, K+1]`, endpoints `[N_d, J, 2]`, candidates nu `[N_d, J]`, children `[N_d, b, (2 + 1 + 1 + d_tok)]`. With R about 300 living roots, b = 2 and a beam of 256 nodes per level the load is 256 x 28 sample reads per level, about 60 k sample reads per frame at D = 8 (estimate; each read fetches from up to three pyramid levels).

## 3. Three axes: depth, stage, time

![tree and axes](assets/p16_tree.png)

- **Depth (steps along the lane).** The tree is generated in a fixed shape: level d holds the b children of every tip of level d - 1 (b >= 2, so a fork is always representable); all nodes of a level form one batch, no dynamic frontier, no sequential loop over single tips. Because most slots are padding, the implementation keeps the M best nodes per level (a beam); the full template of b^d nodes is only an indexing scheme.
- **Stage (chained hypothesis networks).** Generate -> refine -> rescore is a cascade as in cascaded detectors: every stage sees the output of the previous one and gets its own loss; the same three stages repeat at every level (shared weights across levels, as a recurrence; per-level weights are a later ablation).
- **Time (frames).** The set of the previous frame is warped and becomes prior roots and context; the validity is fused in log-odds. Iterating over time is the third recurrence and is switched on last (rung 3 or a later loop); everything else is specified so that it does not change when time is added.

## 4. Why a pre-generated tree with b >= 2 per step

- **Forks need two slots.** With b = 1 a fork has to be found by a second pass; with b = 2 both branches are children of the same tip (T07). The second slot is padding or a dead hypothesis on a single lane (T04), which costs nothing but a mask.
- **Fixed shape.** Batched tensors per level, static memory, no data-dependent loops; the hard parts (top-b, NMS, top-M) are selections with gradient through the selected items, as allowed by the gate.
- **No averaging of modes.** Children are separate endpoints; the final graph keeps them as soft-valid points, and later levels, the merge and the rescore stage decide.
- **Cost.** Full b^D would be 510 nodes per root at D = 8, b = 2 (1022 for two roots in the reference, 21 alive); the beam M bounds this. b = 3 is a refinement option.
- **Merging is part of the design.** Hypotheses of different roots converge on the same lane; the tree is a hypothesis space, the lane graph is a DAG after clustering and edge splitting.

## 5. Losses and labels (matching-free)

- *Verification labels from geometry:* per ray the furthest sample such that all samples up to it lie within 0.5 m of a ground-truth lane; BCE per sample on `v_k` with the cumulative target (T-consistent); no expert, no matching.
- *Validity label:* a node is positive if its position is within 0.5 m of a ground-truth lane and connected to the parent's ground-truth path; BCE on nu and on the rescored validity.
- *Geometry:* point-to-polyline distance both ways, weighted by validity (E120 loss), at the end of every stage.
- *Optional:* sibling repulsion hinge (siblings farther than 1.5 m apart unless both are near ground truth); density term (soft occupancy <= 1).
- *Downstream:* route head and consumer on the final set, validity added to the attention logits of the route head.

## 6. Spec tests

Scenes are drawn in `test_p16_spec.py` (straight lane, arcs R = 10 ... infinity, neighbouring lane at 3.5 m, forks, lane end, gap, toy road with one fork and one merge). **Oracle mode** = tolerances of the reference. **Network mode** = the same scenes with a learned verification: a test passes if at least 95 % of 200 randomised instances (offsets, headings, widths) meet the oracle bound relaxed by a factor 1.5; the exact algebraic tests (T01 - T03, T13, T22) stay exact.

| id | level | what is tested | expectation (oracle mode) | status |
|---|---|---|---|---|
| T01 | geometry | 12 rays, 28 samples, envelope, rings | counts; r(0) = 24, r(+-20) = 12, r(+-50) ... = 4; non-increasing in |phi|; samples on rings or at r | pass |
| T02 | geometry | equivariance and mirror | pattern(c, theta) = R(theta) pattern(0, 0) + c to 1e-9; phi -> -phi mirrors y | pass |
| T03 | algebra | survival | sum P = 1, P >= 0, T non-increasing; v = 1 -> P[K] = 1; v_1 = 0 -> P[0] = 1 | pass |
| T04 | one tip | straight lane | one living child at (24, 0) +- 0.5 m with nu >= 0.9, second slot present but dead | pass |
| T05 | one tip | step against curvature | first step non-decreasing with R (4, 4, 8, 8, 24 m); vertex on the lane within 0.8 m | pass |
| T06 | one tip | neighbouring lane at 3.5 m | all living children on the own lane (|y| < 1 m) | pass |
| T07 | one tip | fork | two living children on the two branches for 18, 12 and 6 deg; one for 3 and 0 deg; one living child on a single lane | pass |
| T08 | one tip | coverage and backward | backward ray nu < 0.3; no child within 2.5 m of a committed point | pass |
| T09 | selection | redundancy | stub on a longer segment and near duplicate removed, order invariant over 10 shuffles | pass |
| T10 | tree | lane end | no living node beyond the lane end + 0.5 m; the lane is followed up to its end (>= 7 m) | pass |
| T11 | one tip | gap of 3 m | strict stops before the gap (x <= 9 m); tolerant crosses it (x >= 23 m) at lower nu | pass |
| T12 | seed | no heading | two living children, +-24 m on the lane axis, headings 180 deg apart | pass |
| T13 | tree | fixed shape, locality | level sizes 1, 2, 4, 8; parent index order; dummies dead; a lane 40 m away changes nothing | pass |
| T14 | DAG | toy road with fork and merge | recall >= 0.85, false length <= 0.10 (measured 1.00 / 0.00); a vertex with out-degree 2 at x in [18, 36], in-degree 2 at x in [38, 50] | pass |
| T15 | DAG | duplicates | no two vertices closer than 1.0 m | pass |
| T16 | beam | budget | at most M kept per level; every kept node has pi >= every dropped living node of its level | pass |
| T21 | time | recursion | true node validity rises monotonically (to > 0.9 in 5 frames), planted false node falls (< 0.1) | pass |
| T22 | time | ego warp | rigid (distances preserved), invertible to 1e-9, heading shifts by -psi, identity for zero motion | pass |

## 7. Tests for the network version (to write with the builder)

| id | what | criterion |
|---|---|---|
| T17 | gradient reach (gate G2) | geometry + validity + link + route + planner loss; every parameter group (stem / readers, seeds and snap, pattern, verification, stages A / B / C, token heads, route head, consumer) has a finite, non-zero gradient |
| T18 | position sensitivity (G4) | autograd slope of the loss with respect to a pattern gain and a seed offset within 20 % of central differences |
| T19 | overfit one scene | 200 steps on one frame: loss below 5 % of the start, AUC of the verification (on / off lane samples) above 0.99 |
| T20 | labels | the geometric verification labels equal the oracle's furthest verified sample on 100 random rays |
| T23 | stage B | refinement lowers the mean point-to-polyline distance of the children and moves them by at most 1 m |
| T24 | deep supervision | the loss of each stage falls during the overfit run; stage C never worse than stage A on the training scene |
| T04n - T12n | network mode of T04 - T12 | 95 % of 200 randomised instances within the relaxed bound |
| T14n | toy road | recall >= 0.8, false length <= 0.15 on 50 random toy roads |
| T26 | censoring (planned, reference first) | a ray that leaves the BEV range or enters a masked region is not a lane end; the survival likelihood uses T only |
| T27 | MCL children (planned) | on the 18 deg fork scene the two children cover different branches in >= 95 % of 200 instances after a short toy training |
| T28 | small forks with junction head (planned) | 2 - 4 deg forks give two hypotheses with the junction head M1, one without (reference limit 6 deg) |
| T29 | merge location (planned) | merge vertex within 2 m of the geometric merge with M1 / M2 (reference without: 6.4 m early) |
| T25 | time with the network | five frames of a static scene with moving ego: validity of true nodes rises, planted false nodes fall, nodes match the lane after the warp within 0.3 m |

## 8. Run and reproduce

```bash
wsl bash -c 'cd /mnt/c/ML-Experiment/scripts && source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate gcar && python test_p16_spec.py'
```

Files: `scripts/p16_ref.py` (reference), `scripts/test_p16_spec.py` (18 tests, `-v` for tracebacks, also collected by pytest), `scripts/fig_p16_tree.py` (figure from the reference run). Thresholds of T09 and T14 were set from the reference's behaviour after two test assumptions were found wrong (the longer of two collinear candidates wins; merges are located where the lanes come within about 1 m); this is documented in the table rows and the section 1 table.

## 9. Open points

- The direction prior and the coverage mask are hand-set in the reference; in the network they are learned and must reproduce T06 and T08.
- The tolerance sigma = 0.6 m sets the fork resolution (about 6 deg) and the merge location; the network's effective tolerance is learned and has to be measured with the same scenes.
- Beam budget M and depth D are untested on real frames; D = 8 is a guess (steps are up to 24 m on straight lanes but 4 - 8 m on curves, so curved lanes need more levels).
- Stage B and C, shared against per-level weights and the time recursion have no reference implementation yet, only the tests of section 7.
