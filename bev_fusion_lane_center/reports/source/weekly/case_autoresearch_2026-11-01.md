# The case for Autoresearch - evidence for the meeting of 2026-11-01 / 02

Version 2, 2026-11-01, after an independent sceptical review of version 1 (twelve overclaims and several gaps were found, checked against the files and corrected;
section 12 lists them). Every number comes from a stored file (section 11 lists the files); a number that waits for a run says PENDING and is filled before the meeting.

**Definitions.** BEV project: VAL = 1000 keyframes spread over 82 of the 150 official nuScenes validation scenes; it has no TEST split and none was used. Lane project:
VAL = 873 frames and TEST = 21 scenes of its own scene-disjoint split of 255 scenes; TEST was used in loop 16 and in loop 18 (logged: once per finished design, with the
person's OK; the TEST numbers of loops 1 - 15 were not logged at the time). "Like for like" = a published system run by our own harness on the same frames with the same metric.

## 0 The short version

Autoresearch is a research loop: an agent proposes and runs experiments on ONE RTX 3060, every result goes into a ledger and into git, a rule written before the run decides
(for most runs, section 3), and a person keeps the decisions. After three weeks it has

1. **produced**: 22 loops (21 filed, one running), 221 lane experiment folders and 14 BEV experiments, 540 papers triaged (404 notes, 38 deep extractions), 390 commits,
   77 lane launcher runs since loop 18, 30 decisions asked of the person (22 answered);
2. **found causes, not only knobs**: three stories (section 2). The better parts are borrowed from published models (BEVFusion's map-trained features, GaussianCaR's
   features, DualBEV's heads). What the loop added is the diagnosis by elimination, the controlled swap, the like-for-like harness and the integration (box in 2.1);
3. **kept a record that shows its checks and its lapses** (section 3): a reviewed judge, a prediction and a rule in the ledger for 63 of 77 executed lane runs, measured noise
   bands, and reviews that found real bugs and process lapses;
4. **measured itself against the state of the art like for like** where a published system could be run on our frames, with named limits (section 4);
5. **and is NOT state of the art on any public leaderboard.** Say this first. What it shows today is the measured distance, a ranked path, and a test that decides whether
   the path is realistic (section 5).

The value is not that an AI writes code. It is a loop that records everything, tests one block at a time against a control, and makes its own checks visible, including where
they failed.

## 1 It produces (counts from the repository, 2026-11-01)

| what | number | source |
|---|---|---|
| days since the first loop (11 Oct) | 22 | `reports/README.md` (week 1 = 11 - 17 Oct) |
| loops | 21 filed + loop 22 running | `HANDOVER.md` |
| experiment folders, lane project (catalogue ids E15 - E267) | 221 | `experiments/` |
| experiment folders, BEV project (t1 - t11, since 25 Oct) | 14 | `bev_fusion/experiments/` |
| launcher runs since loop 18 (lane ledger, executed rows) | 77, of which 67 ok, 6 failed, 2 stalled, 1 stopped, 1 blocked; 26.7 job-hours | `LEDGER.csv` |
| ... of these with a prediction and a rule in the ledger | 63 of 77 | `LEDGER.csv` |
| launcher runs of the BEV project (ledger since 31 Oct) | 4 rows; 2.9 job-hours (BEV trainings before D23 were not under the launcher: unmetered) | `bev_fusion/records/LEDGER.csv` |
| idea to verdict, the two latest cards | t10: card committed 19:30, result committed 23:24 on 31 Oct (3 h 54 min, including 115 min of training); t11 velocity cause: card 00:10, verdict 01:17 on 1 Nov | `git log`, card headers |
| papers triaged / notes / deep extractions | 540 / 404 / 38 | `literature/papers.csv`, `papers/`, `deep/` |
| commits (git started on 25 Oct) / tagged runs | 390 / 46 | `git log`, `git tag` |
| decisions asked of the person (D1 - D31, no D12) | 30; 22 answered, 8 open | `DECISIONS.md` |
| tools / agent roles / suites | 16 / 7 / 2 (evalsuite, trainsuite) | `autoresearch/` |
| hardware | one RTX 3060, 12 GB | `PROBLEM.md` |

Reading: output is high, and the verdicts are not all wins. Since loop 18 the ledger holds 29 experiments, 18 of them with a verdict by tool: 5 KEEP, 8 CUT, 5 TIE
(`metrics.py`). Read this with care: six of the CUT / TIE belong to one family (the recall-weight arms of E219) and two KEEPs are data checks (E235.chk3, E251.chk). The
prediction written before the run was right in 8 of 15 cases that carried one.

## 2 It finds causes (three stories)

### 2.1 The maps of the BEV system

- **Question** (Mirko, 30 Oct): "Look exactly why performance is not SOA, find out." The gap was measured like for like first: the published map model GaussianCaR
  (6 cameras + radar, trained for maps, 40 epochs on 4 x A100) scored with our metric on our frames reached 0.85 / 0.59 / 0.66 / 0.54 / 0.69 / 0.63 IoU (drivable, walkway,
  carpark, crossing, lane divider, road divider); our chain reached 17 - 55 % of that on everything except the drivable area (85 %) (`t6_soa_gap`).
- **The loop removed candidates one at a time, each with a control**: not the range (the gap is as large at 0 - 20 m), not the threshold (best threshold moves ours by
  <= 0.04), not over-fitting for the big areas (our TRAIN IoU is already below the SOA's VAL IoU; for thin classes partly: lane divider 0.346 train against 0.274 val), no
  block exchange helped (11 exchanges: 7 ties, 4 worse; within a 0.054 band, so "no lever found", not "proved irrelevant"), task competition is small (a maps-only run gains
  0.015 - 0.051, 7 - 12 % of the gap), the labels are not the cause (the SOA scores 0.69 on the same labels). What stayed standing: the image features are frozen, trained for
  DETECTION, at stride 16 (a linear probe: lane dividers 0.031 against a prior of 0.018 are not even readable), plus a training budget 9 - 27 x smaller than the SOA's.
- **The test** (one day: port, cache of 58,101 frames, two seeds and a same-code control): swap the image features for the map-trained stride-8 features of BEVFusion. It is
  a bundle (backbone, stride, training objective and depth head all differ), so it tests "the source of the image features" as one block; it was planned in plan 2.8 with a
  qualitative rule ("if the maps move towards the SOA, the encoder is the cause"), not with a numeric card. Maps nearly doubled (lane divider 0.191 -> 0.351 / 0.345,
  walkway 0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165, drivable 0.716 -> 0.803 / 0.809) and the objects HALVED (front-sector mAP2 0.367 -> 0.183 / 0.180,
  pedestrians gone). The loop wrote that down as a result, not as a failure.
- **The next hypothesis, built and run the same night** (both feature sets in one chain, `t10_dual`, two seeds): maps kept (lane divider 0.335 / 0.350, walkway 0.340 / 0.354,
  crossing 0.154 / 0.173, drivable@0.5 0.806 / 0.807; the control: 0.191 / 0.180, 0.178 / 0.176, 0.002 / 0.000, 0.716 / 0.716) and vehicle + pedestrian detection kept (front mAP2
  0.361 / 0.364 against the control's 0.367 / 0.368: -0.004 to -0.006, bootstrap intervals touching zero, inside the 0.015 seed spread; pedestrian AP@2 0.199 / 0.202 against
  0.210 / 0.213). The rare two-wheeler class and the 3-class mAP are noisy and nothing is claimed about them (two-wheeler AP@2: control 0.121 / 0.051, dual 0.016 / 0.061;
  3-class mAP 0.251 / 0.235 against 0.221 / 0.234). The prediction (written first) hit on four of five numbers; the fifth (pedestrian AP@2) came out better than predicted (0.199
  against 0.15). Rule E1 held on both seeds. The seed twin was killed once by the out-of-memory killer and resumed on a later code snapshot; it agrees with the first seed.
- **What it means**: a feature swap beat the start recipe trained with 4.7 x the optimiser steps (7000 x accumulation 2; a different config and a larger frame set) on all six
  map layers (map-trained features alone) and on five of six (dual chain). The saving is partly borrowed compute: the better features come from a map-trained network that
  somebody else trained.

| maps, VAL, `seg_fov` (IoU) | drivable | walkway | carpark | crossing | lane divider (0.5 m) | road divider (0.5 m) |
|---|---|---|---|---|---|---|
| start (`t2_ref_s0`, 3000 steps) | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| start, 4.7 x the training (`t1_r4_soa`, 7000 x 2 steps, 3109 frames) | 0.698 | 0.244 | 0.162 | 0.093 | 0.274 | 0.322 |
| both feature sets (`t10_dual_s0`, 3000 steps) | 0.756 | 0.340 | 0.144 | 0.154 | 0.335 | 0.442 |
| both feature sets, seed twin (`t10_dual_s1`, 3000 steps) | 0.757 | 0.354 | 0.139 | 0.173 | 0.350 | 0.442 |
| same-code control, two seeds (`t8_ctl_s0` / `s1`, 3000 steps) | 0.668 / 0.668 | 0.178 / 0.176 | 0.115 / 0.106 | 0.002 / 0.000 | 0.191 / 0.180 | 0.224 / 0.227 |

**What is borrowed and what is ours** (said openly, because the first objection is "the wins are other people's models"):
- Borrowed from published work: the foundation features of the lane project (GaussianCaR camera + radar BEV features: +0.24 lane F1, the largest single effect of that
  project), the map-trained features of BEVFusion (maps x1.8 - 1.9), the DualBEV image heads and BEV encoder, the CenterPoint head.
- Added here: finding WHICH piece limits (elimination with controls), the controlled swap and the merge of two feature sets, the harness that scores two published systems on
  our frames, the differentiable lane-graph builder, fork pruning, map registration with a trust gate, the diagnosis of the velocity layer, and the loop with its ledger.

### 2.2 The velocity branch that learned nothing in 24 of 29 runs

- **Symptom** (first seen in the eight-try report, weak spot 2): in every 3000-step run the velocity error equals the zero predictor (2.39 against 2.39 m/s); a velocity loss
  weight of 10 changed nothing (try 9).
- **Cause**, found by reading stored checkpoints and detections (no GPU): the velocity branch of the object head is a dead ReLU layer. Its output is its bias (1e-7 m/s);
  0.011 % of its hidden units are active at the supervised centres, against 13 - 50 % for the other branches of the same head. 24 of the 29 runs it classified are dead; the
  runs with an effective batch of 6 or more are alive (skill 0.20 - 0.25 where measured; one batch-6 run stayed weak): an association, not yet a result. A linear read-out of
  the trunk features gives skill 0.019 (predicted >= 0.15, so that prediction FAILED): the trunk holds no velocity information because nothing ever asked it to. The CenterPoint
  reference head (mmdet3d) has a BatchNorm in each branch; ours has none.
- **What the input holds**: the radar Doppler at the GT vehicle boxes, with the box heading and a closed form, recovers 79 % of the velocity of moving vehicles (error 1.37
  against 6.42 m/s for the zero predictor; an upper bound, it uses the GT heading); our best learned heads reach 22 - 25 %.
- **Status: a diagnosis, not yet a fix.** Queued, not run (about 2 GPU hours): B' (fresh branch + a moving / static gate) and B2 (B' with gradient accumulation 2). BatchNorm is
  the first-ranked candidate port; it has no card or config yet. The dead branch sat in every 3000-step run of the BEV project (the BEV project started on 25 October).
- **For a team**: a dead branch looks like "the model does not learn velocity" for as long as nobody opens the checkpoint. The loop did, from files, in hours.

### 2.3 The lane-centre graph: equal Q to a 628 ms network, in 24 ms with cached features

| step | result | source |
|---|---|---|
| the right frozen foundation (published GaussianCaR features) | lane F1 0.624 -> 0.867 VAL (0.598 -> 0.835 TEST), +0.24 | E27 |
| learned trace network, no hand-built tracer or assembler | F1 0.883 VAL / 0.851 TEST; 628 ms per frame | E94 |
| differentiable sparse-graph builder, chosen from eight designs by successive halving | champion Q 0.735 against 0.737 of the control, lanes found 0.695 against 0.800, R@1 far 0.709 against 0.811; 24 ms against 628 ms (26 x), cached foundation features, the foundation network not included | E123, `PROBLEM.md` |
| fork pruning | junction score J 0.265 -> 0.409 (control 0.408); paired scene bootstrap +0.144 [+0.104, +0.182] | E215 |
| rough localisation + trust gate for the SD map | pose error 2.55 -> 0.82 m at a 2 m map error; its own S gate (+0.015) was missed by 0.004, the pose and wrong-map gates were met | E213.a |

TEST (21 scenes, no intervals; the control's numbers come from loop 16's TEST use, the champion's from loop 18's): champion S 0.748, J 0.422, lanes found 0.653 against the
control 0.702 / 0.404 / 0.762. It wins on S and J and **loses on far-range coverage**: gap 1 of `PROBLEM.md`.
**Named gaps, reported every time**: the delivered graph still passes rule steps (path union, fork pruning, Douglas-Peucker simplification), the route head is detached, and the
planner-level path error sits at a floor of 0.27 m whatever the graph (gap 3); only the TRAINING path is fully differentiable. The data are partly simulated (the SD map) and use exact poses.
Five questions are closed with evidence so nobody re-runs them: a route loss through the builder (no, three times), larger recall weights (no beyond weight 3), anchoring
settings (no), three GPU jobs at once (no, they stall), a matching-free Poisson set loss (Q -0.073).

## 3 It keeps a record of its own checks, and of where they failed

**The rules, and where they hold.**
- A judge that is reviewed and re-checked, not one that cannot change: golden numbers, and a stored number is reproduced before a new harness is believed (the re-evaluation of
  the reference reproduced lane divider 0.181 / 0.179 and drivable 0.714 / 0.711 exactly). The hook protects the truth files (golden numbers, TEST log, champion entries,
  `project.json`), not the evaluator code itself (D21); two findings of review pass 1 (L1, L2) were bugs in that code.
- A rule written before the run, for most runs: 63 of 77 executed lane runs carry a prediction and a rule in the ledger. Exceptions the reviews found: the pivotal t8 feature swap
  (a qualitative rule, no numeric card), E257's rule relaxed after the sweep, E259.b0's prediction written after E259.a0 finished, three experiment files (E215, E219, E232)
  re-encoded from the loop-18 record. Where a card exists a tool decides (`decide.py`: KEEP / CUT / TIE / INCONCLUSIVE) inside a measured band (Q seed SD 0.0025; mAP2 seed
  spread 0.015, decision band 0.054 set by the walkway).
- The TEST split once per finished design, logged (loops 16 and 18); numbers from files by script; a fresh reviewer for every report. The review of the week-2 report still found
  4 wrong numbers, 8 mixed-split cases and 16 overclaims in about 420 claims. That is why the review exists, and why this case file went through one too.

**What the checks caught, and where they were bypassed** (each is in a record with its fix):

| incident | how it was found | consequence |
|---|---|---|
| an arm of the first study ran with the camera ON (`camera_off` ignored by the cached model) | a stage probe of the supposed floor model | two conclusions retracted; fixed with a self-test (`t3_improve_loop` report, section 9) |
| the pre-trained FPN of the BEV trunk was loaded with its inputs in the wrong order | the DualBEV port produced no signal | fixed behind a flag; effect on our chain: a tie (0.367 against 0.375 / 0.359) |
| training GT built from live code, so runs of different days used different targets | the lane chat noticed the cache signatures | same-code controls (`t8_ctl`), a decision for one GT version (D29, open) |
| velocity branch dead in 24 of 29 classified runs | system-review lens, from checkpoints | section 2.2 |
| three review passes in one evening: pass 1 listed eight high-severity findings (a TEST guard that accepted a substring, a null baseline that inherited the first frame's tags, a GT cache written non-atomically, ...), pass 3 nine more bugs | fresh reviewers reading code | six of the nine repaired in the pass with a regression test that fails on the old code, two by other chats |
| process lapses: E257's rule relaxed after the sweep, E259.b0's prediction written late | the loop-21 reviewer | recorded as caveats in the summaries |
| a full C: disk crashed a run at a checkpoint write (4.5 idle GPU hours); the seed twin of the dual chain was killed once by the out-of-memory killer tonight | the run logs | disk cleaned (C: 4.1 GB -> 36 GB free), jobs resumed |
| my own predictions: DualBEV front-only mAP2 "0.40 - 0.55" (measured 0.250); zero-camera mAP3 "0.20 - 0.26" (measured 0.013) | the measurement | recorded as misses |

**Outside context.** A survey of AI-scientist systems (arXiv 2608.05179, 29 June 2026) names the bottleneck: not whether agents can run research, but whether their claims
can be verified; only 38 % of the runnable systems it coded release seeds or execution traces, and of nine closed-loop systems none shows an externally validated oracle
inside the loop. Autoresearch puts exactly that into the loop: two published systems (DualBEV, GaussianCaR) run on our harness as external references, and its own lapses are
in the record.
Other loops report results too (primary pages read on 2026-11-01): Karpathy's `autoresearch` (one GPU, a fixed five-minute run per experiment, about 12 experiments per hour or
100 per night by its README), AlphaEvolve (a scheduling heuristic that recovers 0.7 % of Google's worldwide compute, a 48-multiplication 4 x 4 complex matrix product), AIDE
with o1-preview on MLE-bench (bronze medal or better in 16.9 % of 75 Kaggle competitions). None of these is a perception problem; they show that loops with a fixed judge find
real improvements, and that the judge is where they differ.

## 4 Where it stands against the state of the art (like for like, with its limits)

| area | SOA reference | SOA | ours | reading |
|---|---|---|---|---|
| detection, front sector 0 - 50 m, mAP2 (vehicle + pedestrian) | DualBEV, a port written by us from its checkpoint, front camera, single frame; NOT verified against its published 36.8 mAP (six cameras, 360 degrees, ten classes) | 0.250 | camera only 0.242 (tie); camera + radar 0.359 / 0.375 (3000 steps), 0.400 (7000 x 2); dual chain 0.361 | camera path equal; ours with radar is +0.11 - 0.15 above it; radar's own gain at equal steps is +0.117 / +0.133. Our system also uses three frames and heads tuned on this metric. No SOA camera + radar detector was run on our frames (CRN is on disk) |
| maps, IoU per layer (drivable / walkway / carpark / crossing / lane divider / road divider) | GaussianCaR scored by our harness on 400 frames | 0.845 / 0.594 / 0.659 / 0.536 / 0.691 / 0.634 | dual chain on 1000 frames: 0.756 / 0.340 / 0.144 / 0.154 / 0.335 / 0.442 = 89 / 57 / 22 / 29 / 48 / 70 % of the SOA (start recipe: 79 / 32 / 16 / 0 / 26 / 35 %) | INDICATIVE: the frame sets differ (about +-0.03); the same-frames re-score is queued behind the trainings. Four of six layers are below 60 % |
| lane-centre graph | none on our metric. Closest published: TopoBDA camera + radar on OpenLane-V2 subset B (nuScenes), R50 | DET_l 49.4, TOP_ll 38.6 | Q 0.735, topo F1 0.741 | NOT comparable: other annotation, other metric, Hungarian matching. A bridge is a plan (section 5), not a result |
| speed | GaussianCaR: 81.1 ms (maps, RTX 4090, 6 cameras, 448 x 800, its paper) | 81.1 ms | BEV chain 50.7 ms (BEV network) + 17.2 ms (image heads) on an RTX 3060, batch 1, idle GPU, one front camera; lane graph 24 ms with cached foundation features | same order of magnitude, not like for like |
| box velocity | published mAVE (all classes), e.g. CRN | 0.344 | the zero predictor | not learned yet (section 2.2) |

**Not claimed**: any number on a public leaderboard; six cameras; ten classes; NDS; 20 epochs of training; a generalisation beyond nuScenes (non-commercial licence: research
use only; the GaussianCaR weights and rasteriser are non-commercial too).

## 5 The path to the state of the art, ranked by information per cost

Every lever below has a measured effect size from this repository; the sums are hypotheses to test, not forecasts.

| # | lever | measured so far | cost | what it would show |
|---|---|---|---|---|
| 1 | dual features, longer (6000 steps first), then all 700 scenes | the old features gained +0.053 on the lane divider from 3000 to 6000 steps and +0.09 to 7000 x 2 | 3 - 4 GPU hours for 6000 steps | the slope of the NEW features: how much of the remaining map gap training closes |
| 2 | stride-4 detail branch on top | on the old features: lane divider +0.065 / +0.042, road divider IoU@1 +0.119 / +0.096 (two seeds) | 2 GPU hours per seed | whether the thin classes add up |
| 3 | velocity: fresh branch, moving / static gate, then BatchNorm and a Doppler read-out | closed-form bound 0.79 against 0.25 learned | 1 - 3 GPU hours | box velocity at all |
| 4 | a SOA camera + radar detector (CRN) on our frames | the checkpoint is on disk | half a day of porting | turns "no number" into a number |
| 5 | bridge the lane graph to OpenLane-V2 subset B (DET_l, TOP_ll) | nothing yet; needs the annotation files and the official evaluator (a download: needs Mirko's OK) | 1 day for a feasibility check | the first number a stranger can compare |
| 6 | a confirm evaluation on the 68 official val scenes never used for a decision (if their caches exist) | - | 1 GPU hour | removes the selection effect of reusing the same 82 scenes |

**Compute reality.** GaussianCaR was trained for 40 epochs with an effective batch of 16 on 4 x A100 80 GB. For the old features the remaining gap needed 4 - 10 more doublings
of the steps (16 - 1000 x the compute; `t6_soa_gap` section 5). With the new features the slope is unknown; lever 1 measures it: the dual chain at 6000 steps against the matched
6000-step reference of the old features (lane divider 0.234) and against its own 3000-step run (0.335). **Rule, proposed now:** a gain of at least +0.04 on the lane divider over
the 3000-step dual chain = continue with levers 1 and 2; below +0.02 = drop the claim that the SOA is reachable on this hardware and keep "SOA-level on camera detection, partial
on maps".

## 6 Making the hobby meaningful: three outputs that outlive it

1. **A method package others can run**: the loop, the judge, the tools and the rules (`autoresearch/`: launcher with a git tag per run, ledger, decision tool, evalsuite,
   trainsuite, seven agent roles), with a short write-up "a verification-first research loop for perception" (not yet written). It is problem-independent: the second problem (BEV
   fusion) was plugged in on 31 October in one day.
2. **One public number per problem** (levers 4 and 5): a result a stranger can check, with the harness as the artefact.
3. **A lessons-learned list that saves a team months**: frozen detection-trained image features cap map quality (here by a factor of two); a velocity branch can be dead from the
   first steps; the cache that builds your GT can silently change between runs; a seed twin is worth a ledger row. Each lesson has its number and its file.

**Boundaries, stated up front.** The data (nuScenes) and the weights used (GaussianCaR, DualBEV, BEVFusion) are non-commercial; the lane project uses a simulated SD map and exact
poses; the BEV system sees one front camera; the BEV claims now have two seeds (control and dual chain), while the like-for-like maps, the lane results and the 7000 x 2 reference rest on one; everything ran on one machine under
one lead chat. It is a private research project on
public data. A company pilot needs its own clearance of IP, company time, data governance and the external agent tool (an LLM service reads the code), before any of this is used there.

## 7 The hardest questions

1. **"Is this an AI inventing numbers?"** Tables come from `metrics.json` through scripts; a decimal that exists in no source file is listed by `check_numbers.py`; a fresh
   reviewer re-reads primary files. The record still shows errors: 4 wrong numbers, 8 mixed-split cases and 16 overclaims in about 420 claims of the week-2 report, found by
   review, and twelve overclaims in the first draft of this file, found by a sceptical reader. That is the argument for the review, not against it.
2. **"Aren't the wins just other people's models?"** The better parts are borrowed (GaussianCaR, BEVFusion, DualBEV, CenterPoint). The contribution is finding which piece
   limits, testing it with a control, merging two sets, the harness, the lane-graph builder, and the diagnosis of the dead layer (box in 2.1).
3. **"What does the person do?"** Sets the question, the rules and the stop time, answers 30 decision entries (champion promotion, TEST use, protected files, deletions,
   downloads), reads the ledger. The agent does the coding, the running and the first reading.
4. **"Why not an existing tool (MLflow, W&B, Optuna)?"** Those track runs and tune knobs. This also writes the hypothesis and the decision rule before the run, runs the controls,
   and diagnoses causes. It could sit on top of them; it was not benchmarked against them.
5. **"Could this be luck?"** The t8 and t10 claims have a same-code control with two seeds and the dual chain has two seeds; the maps replicate within 0.02 across the seeds. The
   dual seed twin ran on a later code snapshot and was resumed once. Every BEV selection used the same 82 val scenes (a confirm run on the other 68 is lever 6); the lane TEST
   has 21 scenes and no intervals; the like-for-like maps and the 7000 x 2 reference are single runs.
6. **"Is it state of the art?"** Not on a public benchmark. Like for like: equal to the SOA camera detector (a port we wrote), +0.11 - 0.15 with radar; maps at 22 - 89 % of the
   SOA by layer (indicative); lane graph: no comparable number yet.
7. **"What does it cost?"** Known: one RTX 3060, 29.6 job-hours under the launcher since loop 18 (BEV trainings before D23 unmetered), 30 decision entries, 1 - 4 hours from card to
   verdict for the two latest cards. Unknown: agent tokens or money, human hours, and what a team does today for the same questions. The pilot is designed to measure the last one.
8. **"Does it transfer, and is it product-relevant?"** The tools are problem-independent (a metric-card writer and a launch command are the plug). Not tested on another
   organisation's data; the data and weights are non-commercial; the lane project is partly simulated; see the boundaries in section 6.
9. **"Where did it fail?"** Five closed questions; 13 of 18 verdicts CUT or TIE (six from one family); a velocity head dead in 24 of 29 runs; a feature swap that halved the
   objects; predictions missed by factors of 1.6 to 15; rule and prediction lapses found by review.
10. **"What can go wrong?"** A full C: disk crashed a run once; the seed twin was OOM-killed tonight; code changed under a running job; one lead chat and eight open decisions;
    the loop wrote and found its own bugs. Mitigations: GPU gate and slots, one lead per loop, noise bands, a clean-up rule that moves and never deletes, reviews.
11. **"How will we know it worked?"** Criteria and a stop rule written now (section 8 and the rule in section 5).

## 8 What to ask for

1. Time on the GPU at night and at weekends inside the written rules (stop time, GPU limit, no TEST without OK).
2. A named person who signs off every claim that leaves the team and reads the ledger once a week.
3. A second reviewer for headline numbers; budget for the agent tool.
4. A one-month pilot with a second team, designed so that it can fail: two weeks measuring that team's current process first (experiments per week, wrong-number findings per 100
   claims), then four weeks with the loop. Stop rule: no gain in experiments per week, or more wrong numbers than the baseline = stop and report. Budget: one GPU, one reviewer
   day a week.
5. Before any company use: clearance of IP, company time, data governance and the external agent tool.
6. (Mirko's own decision) permission for two downloads (OpenLane-V2 annotations and its evaluator) to build the first public lane number.

**Success criteria, written now, checked at the next review**: (a) every map layer at 50 % or more of the SOA like for like and the drivable area at 90 % (today 22, 29, 48, 57,
70 and 89 %, indicative); (b) box velocity with skill 0.5 or more on moving vehicles; (c) one public-benchmark number per problem; plus the rule of section 5. Missing them is a
result too.

## 9 Ten minutes, in order

1. The claim and its limit (section 0). 2. The loop in one picture. 3. Maps: question, eliminated causes, the test, the result (2.1), what is borrowed. 4. Velocity: the dead
layer (2.2). 5. Lane graph: equal Q at 26 x (2.3), with its gaps. 6. What the checks caught, and where they were bypassed (3). 7. Like-for-like SOA table (4). 8. Cost,
boundaries and the path with its stop rule (5, 6). 9. The ask (8).

## 10 Open points before the meeting

- PENDING: like-for-like maps (a 400-frame re-score needs about 8 GB of host memory and cannot run next to a training on this box; a GPU variant, a few minutes per model, is
  prepared for the quiet window after tonight's round), the lane prototype of the overnight round, lever 1 (the 6000-step dual run: card, config and start line are ready; the queue owner judged that it does not fit before tonight's
  planned shutdown, so it is the next GPU item and needs Mirko's go).
- Decisions that wait for Mirko: D4, D17, D18, D19, D25, D29, D30, D31; permission for the two downloads (section 8, item 6).

## 11 Sources

`bev_fusion_lane_center/PROBLEM.md`, `experiment_execution/{LEDGER.csv, DECISIONS.md, closed_questions.md}`, `evaluation/TEST_USAGE_LOG.md`, `experiments/INDEX.csv`,
`reports/source/weekly/{review_week2.md, s21_review.json}`, `bev_fusion/records/LEDGER.csv`, `bev_fusion/plan.md` (2.8),
`bev_fusion/experiments/{t3_improve_loop, t4_soa_front, t6_soa_gap, t7_lane_branch, t8_bevfeat, t10_dual_features, t11_vel_cause}/`, `bev_fusion/runs/*/eval/metrics.json`,
`bev_fusion/runs/t1_r4_soa/eval/latency.json`, `autoresearch/history/repo_review_2026-10-31*.md`, `bev_fusion/temp/pitch/{facts.json, ladder.md, indicative.txt}` (counts and
ladders made by scripts), `bev_fusion_lane_center/literature/papers/{SN01_gaussiancar.md, RB14_gaussiancar-no-failure-evaluation-exists.md, SN09_topobda-the-only-camera-radar.md}`.
External, primary pages read on 2026-11-01: arXiv 2608.05179 (the verification-gap survey); arXiv 2410.07095 (MLE-bench); the AlphaEvolve post of Google DeepMind; the README of
github.com/karpathy/autoresearch.

## 12 What the sceptical review of version 1 changed

Twelve claims were stronger than their evidence and are corrected above: the definition of VAL / TEST (BEV has no TEST; lane TEST used in loops 16 and 18); "every result with a rule
before the run" (63 of 77, with named exceptions); "13 of 18 by rule" (composition shown); "one cause at a time" (one block at a time; the t8 swap is a bundle); "a judge the loop cannot
change" (a reviewed judge; the hook protects truth files, not the evaluator); "not the BEV network" and "not over-fitting" (no lever found within the band; partly for thin classes);
"objects kept" (vehicle + pedestrian; the two-wheeler class fell); "same chain trained 4.7 x longer" (a different config and frame set; borrowed features); "never learned / dead for three
weeks" (dead in 24 of 29 classified runs; three live runs; BatchNorm not queued; the linear read-out prediction failed); "same quality / 24 ms" (equal Q, lanes found lower; cached
foundation features); "TEST once" (loops 16 and 18; the control's numbers are from loop 16); "no number typed" (reviews still found errors); "4.1 GB free" (now 36 GB); "six-page write-up"
(not written). Added: what is borrowed, cost known and unknown, boundaries, the end-to-end gaps of the lane graph, speed, the compute reality with a stop rule, a confirm run, the pilot design.
After the two-seed evaluation of 03:15 two statements of this file's own were corrected in the other direction: a gain of "pedestrian segmentation 0.000 -> 0.091" was seed noise (the control's
second seed scores 0.090 too) and was dropped; the "two-wheeler / 3-class mAP loss" of the dual chain lies inside the control's own seed spread (two-wheeler AP@2 0.121 / 0.051 for the control,
0.016 / 0.061 for the dual chain; 3-class mAP 0.251 / 0.235 against 0.221 / 0.234).
