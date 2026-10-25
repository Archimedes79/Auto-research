# Concept: an end-to-end differentiable lane graph built from verified-ray hypotheses

Camera + radar BEV, lane centre-line graph with navigation route (nuScenes) · concept note, 2026-10-23 · collects the design note (`Growth_and_Link_Architectures_EN.md`), the spec (`Hypothesis_Tree_Spec_and_Tests_EN.md`), the plan (`Loop16_Triage_Plan_EN.md`) and the literature review (`report_final/text/lit_p16_tree.md`) into one concept that can be tested

> **Status.** Concept and specification; a numpy reference of the geometry with an oracle verification passes 18 spec tests. No network of this concept has been trained. Numbers marked *(measured)* come from earlier experiments of this project (E94, E120, loop 15) or from the reference on toy scenes.

## 1. Goal and constraints

- Output: a lane centre-line graph (vertices, edges, validity, 128-d tokens) with a route attribute per edge, consumed by a learned planner.
- Every module connected into one network and trainable end to end: positions, existence and topology receive gradient from the geometry, route and planner losses. Allowed non-differentiable operations: selections with gradient through the selected items (top-b, beam, candidate lists), thresholds at export.
- Built on what exists: frozen camera + radar BEV features, the dense head (heat map, direction), the input memory, the route head E104b (cross-attention to SD-map tokens), the consumer E117.
- Weaknesses of the current learned chain B that the concept has to address *(measured)*: argmax steps and teacher forcing (E94), 22 vertices per 100 m against 14.3 for a 0.3 m Douglas-Peucker oracle, split / merge F1 0.36 / 0.46 against 0.69 / 0.65 for the rule system, seeds as a limit (192 peaks), TEST F1@1 0.851 (0.863 after the E120 refinement).

## 2. The concept on one page

```
frozen BEV features + dense head (heat map, direction)
        |
seeds everywhere: lattice points snap onto the lane, two-lobed teardrop, validity      (no peak picking)
        |
hypothesis tree, level by level (fixed shape, beam of M nodes per level):
   A generate : teardrop of 12 rays (short in all directions, long and narrow forward, up to 24 m),
                samples on rings 4 / 8 / 12 / 18 / 24 m, verification of every sample,
                survival along the ray -> endpoint; b >= 2 best endpoints become children
   B refine   : small teardrop around every child: dp <= 1 m, dtheta, re-verification
   C rescore  : attention over siblings, ancestors and committed points -> validity, token
        |
tree -> DAG: cluster near duplicates, split edges at vertices on them (forks and merges appear)
        |
revisit (E120-type refinement of all vertices), keep gate
        |
route head (SD-map tokens, cross-attention) -> planner consumer -> path
        |
time: the set of frame t-1, ego-warped, is prior roots and context of frame t; log-odds fusion of validity
```

Key ideas and where they come from:

| idea | effect | origin |
|---|---|---|
| teardrop pattern (short rays wide, long rays narrow, full circle at 4 m) | reach 24 m with the sample budget of E94; seeds and heading errors covered | user sketch, 2026-10-23 |
| long rays verified by their intermediate samples (survival) | step length = longest chord that stays on the lane (Douglas-Peucker emerges); no hopping onto neighbouring lanes; labels from geometry | user, refined in the spec |
| n best endpoints as hypotheses, b >= 2 per step, pre-generated tree | forks without event classes, no averaging of modes, fixed tensor shapes | user, refined in the spec |
| cascade generate -> refine -> rescore | chained hypothesis networks with deep supervision | user ("hintereinandergehaengte Hypothesennetze"), cascaded detectors |
| tree -> DAG by merge and edge split | merges at the right place | found by spec test T14 |
| iteration over time | robustness, validity accumulates | user, Sparse4D-style instance propagation |
| matching-free losses, validity as soft mask | gradient to positions, existence and topology without Hungarian matching | E120 (loop 15) |

## 3. Components and their status

| component | specified in | reference-tested | network built | first experiment |
|---|---|---|---|---|
| teardrop pattern M1 | spec section 2 | T01, T02 | no | rung 0 (factor B) |
| verification and survival M2 | spec section 2 | T03 - T05, T11 | no | rung 0 |
| hypothesis selection M3 | spec section 2 | T06 - T09, T12 | no | rung 0 |
| fixed-shape tree, beam M4 | spec section 2 | T10, T13, T16 | no | rung 0 (G designs) |
| tree -> DAG (merge, edge split) | spec section 1 / 2 | T14, T15 | no | rung 0 |
| cascade B / C (M5) | spec section 2 | no | no | refinement flips, rung >= 1 |
| time (M6) | spec section 2 | T21, T22 | no | rung 3 or later |
| spray-snap-link family L | design note section 5 | partly (ray tests) | no | rung 0 (factor A) |
| route head, consumer | loops 10 - 15 | - | yes (E104b, E117) | rung 3 |

## 4. Winning concepts from the literature

Source: `report_final/text/lit_p16_tree.md` (about 75 papers, abstracts or listing pages seen on 2026-10-23 unless marked). Ids K30 onwards are in the technique register of the plan (section 10.0). OpenLane-V2 numbers are from different devkit versions and are not strictly comparable; subset B is nuScenes.

| # | concept that works | evidence (as seen) | ids | what it means for us |
|---|---|---|---|---|
| W1 | **explicit endpoints / junction points** as queries or priors | TopoPoint (endpoint detection, OLS 48.8, subset not stated), CGNet (junction-aware queries), TopoFG (keypoint priors, OLS 48.0 A / 45.4 B), SeqGrowGraph (splits only at junctions) | K31 K32 K33 | our weakest spot (forks resolved only from about 6 deg, merges 6 m early in the reference, split / merge F1 0.36 / 0.46 in E94) is exactly what these fix |
| W2 | **geometric link prior**: endpoint distance and heading mapped by a learned function to a link probability | TopoLogic: TOP_ll 23.9 vs 10.9 on A, also post hoc on trained models | K30 | a cheap, differentiable link head for L and for joining dangling tree ends |
| W3 | **joint predecessor / successor modelling** | UniTopo: TOP_ll 30.1 (A) / 31.8 (B), newest found | K34 | stage C should attend explicitly along parent / child relations, not only to siblings |
| W4 | **curve-aligned deformable reads** | LaneSegNet lane attention, TopoBDA Bezier deformable attention (radar gains claimed), CLRNet ROIGather along the prior | K36 K37 | stage B should sample along the tip -> child chord or a Hermite curve, not a free small teardrop |
| W5 | **denoising training** (noised ground truth as extra queries) | DINO contrastive denoising, TopoFG topological denoising, SQD-MapNet stream-query denoising (> 3 mAP over StreamMapNet), TopoStreamer segment denoising | K43 K33 K45 | noised ground-truth paths as extra roots of the tree; noised previous-frame ground truth for the time recursion |
| W6 | **cascade with stricter thresholds, no detach, quality head** | Cascade R-CNN, DINO look-forward-twice, Sparse4D v3 quality estimation | K42 K43 K44 | stages A / B / C with 1.0 / 0.5 / 0.3 m positives; calibrated validity because it drives the beam |
| W7 | **multiple-choice learning with a score head** | rMCL (learned scoring), aMCL (annealing against collapse) | K38 | training rule for the b >= 2 children: winner-takes-all per ground-truth branch, annealed |
| W8 | **training-only sequence decoder** | Topo2Seq (+2.0 AP lane segment, third-party number) | K35 | an auxiliary root-to-leaf path-order loss, no inference cost |
| W9 | **nodes + pairwise edge classifier** (fast, strong) | SAM-Road (RNGDet++-level, about 40x faster), Relationformer, InstaGraM | K48 K13 | the strongest alternative to growth; must be a baseline |
| W10 | **line verification by pooling along the segment** | L-CNN, HAWP (LoI pooling, one score per segment) | K40 | precedent and baseline for our per-sample survival |
| W11 | **temporal propagation of the best instances** | StreamMapNet (top-confidence queries + ego transform), MapTracker (memory, 76.1 mAP nuScenes), stability metric mAS | K45 K46 | the time recursion M6 propagates only top-validity nodes; report stability |
| W12 | **map uncertainty consumed by the planner** | UncAD (up to -26 % collisions on nuScenes) | K49 | the validity pi goes into the planner; graph quality is measured by graph swaps, because open-loop L2 barely reflects perception (ego-status study) |
| W13 | **discrete-time hazard with censoring** | survival-network literature (review seen; formulas from memory) | K41 | rays that leave the BEV or enter occlusion are censored, not lane ends |
| W14 | **generate, then grow from dangling ends** | DeH4R (+4.62 APLS over RNGDet++ on CityScale, about 10x faster) | K51 | supports the hybrid H |

What the literature does **not** support, and why the concept keeps it anyway or drops it:
- Token-sequence decoders (RNTR, LaneGraph2Seq, SeqGrowGraph, TopoGPT) need sampling or argmax at inference and are not differentiable through the graph: **not adopted** (only W8 as a training-only loss).
- Every growing method is imitation-learned, RL or next-token trained; TopoGPT and iCurb report error propagation. Our tree is trained on its own selections from the first step; this is **new and the main risk** (exposure bias was the reason for DAgger in iCurb).
- No paper chooses the step length by verification, converts converging hypotheses into a DAG in a learned system, fuses node validity over time in log-odds, or back-propagates a planner loss into an explicit lane graph. These are the open contributions and also unproven.

### 4b. Sparsification-specific literature (`report_final/text/lit_p16_a_sparsification.md`, 30 rows SP01 - SP30, register K134 - K146)

| # | finding | evidence | what it means for us |
|---|---|---|---|
| S1 | the vertex count can be a differentiable expected quantity | hard-concrete L0 (closed-form expected count, K129), CoFi constraint to a target sparsity (K134), PTDNet expected kept edges (K135) | SPF1 / SPF2: penalty with a Lagrangian pull to a target density instead of one hand-tuned weight; none of these has been used on vertices of a lane graph (gap, also our contribution) |
| S2 | a simplification oracle is an accepted target | PivotNet pivots from Visvalingam-Whyatt (K121), MachMap: more than 93 % fewer points with AP still 97 - 100 % at 0.2 - 0.5 m (K142) | the Douglas-Peucker label is legitimate; our own count: Douglas-Peucker 14.1, Imai-Iri 13.2, greedy longest chord 13.2 per 100 m on 60 test frames at 0.3 m (K144), so the floor is about 13, not 14.3 |
| S3 | n hypotheses with learned validity exist | RNGDet: 10 vertex queries per step, validity BCE, branching, no merge, no tolerance rule for the step (K84) | closest published form of M3; our additions are the verification-chosen step and the DAG merge |
| S4 | many positives in training, lean one-to-one head at test | Co-DETR, H-DETR, Group DETR (K128) | label every endpoint inside the tolerance as positive in training (M3), small b at test |
| S5 | selection in practice is hard, the scorer is trained by auxiliary supervision | Sparse DETR, Focus-DETR, SparseOcc (K138, K139, K141) | validity and keep gate by BCE plus the count term; do not expect gradient through the selection |
| S6 | over-complete first, prune late | Focus-DETR cascaded keep ratio, SparseOcc cannot recover pruned cells | keep a generous b and beam at the first levels, cut by validity and budget at the end |
| S7 | duplicate control has three forms | learned re-scoring (GossipNet, K107), score decay (Soft-NMS, K130), exact zeros (sparsemax / entmax, K39, K130) | factor D levels and the soft-selection ablation |
| S8 | the lane and road metrics opened ignore vertex count | OpenLane-V2 DET_l / TOP_ll, TOPO, APLS; only polygon work has count-aware scores (PolyWorld C-IoU, N ratio, K145) | the frontier of the plan (F1@1 against vertices per 100 m, F1@Sp17, FL@R85) |

## 5. Mixing well-working concepts

All mixes keep the gate of the plan (no argmax on the training path, selection only with gradient through the selected items). Each mix states what it combines, the hypothesis, where it enters the ladder, and how it is tested.

| mix | combines | hypothesis | enters as | test / metric | risk |
|---|---|---|---|---|---|
| **M1 tree + junction head** | our tree + W1 (TopoPoint / CGNet endpoints) | a junction heat map (fork / merge points) as extra roots and as keys of stage C lifts fork recall below 6 deg and moves merges to the right place | refinement flip of every surviving G design (rung >= 1); default in H | spec T07 extended to 2 - 4 deg forks (new T28), fork / merge landmark P/R at 1 / 2 / 4 m, fork-angle resolution curve | junction labels at small angles are noisy |
| **M2 tree + geometric link** | tree + W2 (TopoLogic) + K02 verification of the connector | dangling ends of the tree are joined by a learned link prior on endpoint distance / heading, verified along the connector; replaces the hand clustering and edge split of the reference | part of H; refinement flip "learned DAG conversion" for G | T14 / T15 without the hand rules; how often the hand rule still fires | double counting with M1 |
| **M3 MCL children** | b >= 2 children + W7 (rMCL + aMCL) | winner-takes-all per ground-truth branch with a learned score and annealing prevents both slots collapsing on one branch | **default** for G from rung 0 (cheap, literature-backed) | new T27: on the fork scene both children cover different branches after training; collapse rate | WTA instability early in training |
| **M4 censored survival** | survival read-out + W13 | the survival NLL with censoring at range and occlusion gives calibrated lane ends and fewer false stops at the BEV border | **default** loss from rung 0 | new T26: a ray leaving the BEV is not a lane end; false stops near the border | needs an occlusion mask (radar / camera visibility) |
| **M5 curve-aligned stage B** | stage B + W4 (Hermite / Bezier deformable reads) | sampling along the chord / curve refines the child better than a free small teardrop | refinement flip "stage B on" uses this variant | spec T23; vertex error across / along | none specific |
| **M6 cascade schedule** | A / B / C + W6 (thresholds, no detach, quality head) | stricter positives per stage and a calibrated validity give a better beam | flip "stage C on" uses this schedule | T24; calibration (ECE) of the validity; beam recall | longer training |
| **M7 denoising roots** | tree + W5 (noised ground-truth paths as extra roots; for time noised previous-frame ground truth) | denoising stabilises free-unroll training like it stabilises DETR and streaming map training | refinement type S (stabilise) and default for the time recursion | training curves; F1 at equal training length | train / test mismatch if overused |
| **M8 path-order auxiliary** | tree + W8 (training-only path sequence loss) | an auxiliary root-to-leaf ordering loss improves topology at no inference cost | flip type D for junction diagnoses | TOP-like adjacency precision, landmark P/R | small effect |
| **M9 predecessor / successor attention** | stage C + W3 + E120 graph-biased attention | explicit parent / child relations in the attention improve validity at junctions | part of stage C | split / merge F1 | overlap with M1 |
| **M10 verified link family** | L family + W9 (node + pairwise edge classifier) + W2 + K02 | SAM-Road-style nodes and edges with a geometric prior and survival verification along each candidate edge are the strongest non-growth design | new **baseline arm B2** and a candidate design for a second array | same metrics as the ladder | it may simply win; then G is not needed |
| **M11 time** | M6 time recursion + W11 + W5 | top-validity nodes propagated with ego warp, trained with noised previous ground truth, fused in log-odds, improve accuracy and stability | rung 3 / next loop | T21, T22, T25; stability mAS; accuracy | drift without the denoising stage |
| **M12 uncertainty to the planner** | route head / consumer + W12 | passing pi into the planner reduces route / path errors from false lanes | rung 3 | oracle / degraded graph swaps into the same planner (graph quality effect), path error with and without pi | open-loop metrics barely react (ego-status study) |

| **M13 expected-count budget** | SPF1 / SPF2 + K129 / K134 (hard-concrete expected count with a Lagrangian pull) | a count term toward the oracle density (about 13 - 14 per 100 m) removes redundant vertices without a hand rule and keeps F1@1 | rung 1 forks SPF1 / SPF2 | vertices per 100 m, F1@Sp17, FL@R85 | count term can drive false stops; balance by the task loss |
| **M14 Imai-Iri labels** | verification-chosen step + K144 (valid-chord graph, shortest path) | training the step against the Imai-Iri path (about 6 % fewer vertices than Douglas-Peucker) gives the sparsest correct chain | SPF1 level 2 variant | vertex count against the oracle floor | validity of a chord does not imply validity of shorter chords, the target can be non-greedy |
| **M15 RNGDet-style validity + many positives** | M3 + K84 + K128 | every endpoint inside the tolerance is a positive in training, small b at test | part of M3 training | T27, collapse rate | none specific |

Mixes that are **not** made: token decoders with our tree (not differentiable), Hungarian set decoders (excluded by the gate, E32 already lost 0.058 F1), soft top-k by optimal transport and differentiable resampling before the hard-selection version works (cost; one of them, entmax, stays as the single soft-selection ablation).

## 6. Test plan for the concept

**Defaults changed by the literature (from rung 0, all designs where applicable):** M3 MCL training of the children, M4 censored survival loss, no detach between stages (look-forward-twice; E120 already showed detach does not matter for positions). These are recorded in the run ledger with their basis ids.

**Baseline arms (run beside the ladder, not halved):**

| arm | what | basis | why |
|---|---|---|---|
| B0 control R | E94 + E120 (existing) | K16 K15 | noise band, fidelity curve, project reference |
| B1 geometric link on pieces | L with a TopoLogic-style link prior only | K10 K30 | the cheap link baseline the literature says is strong |
| B2 nodes + edges | SAM-Road / Relationformer-style vertex heat map + pairwise edge classifier with survival verification (M10) | K48 K30 K02 | the strongest non-growth alternative |
| B3 single-score verifier | the same rays, one LoI-style score per ray instead of the per-sample survival | K40 | is the survival worth its cost? |

**Where the mixes enter the ladder of the plan:** M3 / M4 as defaults; M1, M5, M6, M8, M9 as refinement flips of the surviving designs (types F and D, chosen by the diagnosis table of the plan); M2 and the W14 generate-then-grow idea inside the hybrid challenger H; M7 as the stabilise type; M10 as baseline arm B2 (and the first row of a second array if it wins); M11 and M12 in rung 3 or the next loop.

**New spec tests to add before the loop** (numpy reference first, then network mode):
- T26 censoring: a ray that leaves the BEV range or enters a masked region is not counted as a lane end (survival likelihood uses T only).
- T27 MCL: on the 18 deg fork scene, after a short training of a toy verifier the two children cover different branches in at least 95 % of 200 random instances.
- T28 small forks: with the junction head (M1) forks of 2 - 4 deg give two hypotheses; without it one (reference: 6 deg limit).
- T29 merge location: on the toy road the merge vertex lies within 2 m of the geometric merge with M1 / M2 (reference without them: 6.4 m early).

**New metrics in the triage score and the report:** fork and merge landmark precision / recall at 1 / 2 / 4 m (RNTR landmarks, split-detection-style radii), reachability P/R or APLS, a fork-angle resolution curve, validity calibration (ECE), stability (mAS) once time is on, and for the planner oracle / degraded graph swaps instead of open-loop L2 alone. The composite S of the plan keeps its weights; J (split / merge F1) is computed from the landmark P/R at 2 m.

**External reference points (not the same benchmark):** OpenLane-V2 subset B (nuScenes) OLS 45.4 (TopoFG), TOP_ll 31.8 (UniTopo); RNTR / LaneGraph2Seq / SeqGrowGraph landmark and reachability metrics on nuScenes. Our labels, range and sensors (camera + radar) differ; the comparison is qualitative.

**Order of work:** (1) add T26 - T29 to the reference and the spec; (2) harness and gate as in the plan; (3) ladder rungs 0 - 2 with the defaults and the baseline arms B0 - B3; (4) flips M1 / M5 / M6 / M9 on the survivors; (5) H with M2; (6) rung 3 with M11 / M12. Every run carries its basis ids (K01 - K52) in the run ledger, so the report can say what was tried on which basis.

## 7. Limits

- The literature numbers are from abstracts; several OpenLane-V2 values are on different devkit versions.
- The concept's distinctive parts (fixed-shape tree trained end to end, verification-chosen step length, tree -> DAG, log-odds over time, planner loss into the graph) have no precedent; their value is unknown until the ladder runs.
- Small training set (the matched set decoder E32 lost 0.058 F1 here); free-unroll training without imitation is the main risk.
- Camera + radar lane graphs are almost unstudied, so radar-specific choices cannot be borrowed.
