# The case for Autoresearch - evidence for the meeting of 2026-11-01 / 02

Written 2026-11-01 for Mirko. Every number comes from a stored file (section 11 lists the files); a number that waits for a run says PENDING and is filled before
the meeting. VAL = the official validation scenes of nuScenes, nothing was tuned on TEST. "Like for like" = the other system scored by our own harness on the
same frames with the same metric.

## 0 The short version

Autoresearch is a research loop: an agent proposes and runs experiments on ONE RTX 3060, every result goes into a ledger and into git, a rule written BEFORE the run
decides, and a person keeps the decisions. After three weeks it has

1. **produced**: 22 loops (21 filed, one running), 221 lane experiment folders and 14 BEV experiments, 540 papers triaged (404 notes, 38 deep extractions),
   385 commits, 75 launcher runs since loop 18, 30 decisions asked of the person (22 answered);
2. **found causes, not only knobs**: three stories in section 2 (maps, velocity, lane graph);
3. **kept itself honest**: 13 of the 18 experiments decided by rule since loop 18 ended CUT or TIE; its own reviews found and repaired real bugs (section 3);
4. **measured itself against the state of the art like for like**, where a published system can be run on our frames (section 4);
5. **and is NOT yet state of the art on any public leaderboard.** Say this first. What it can show today is the distance to the SOA, measured, and a ranked,
   evidence-based path to close it (section 5).

The value is not that an AI writes code. It is a loop that never forgets, never skips the log, tests one cause at a time, and cannot grade its own homework.

## 1 It produces (counts from the repository, 2026-11-01)

| what | number | source |
|---|---|---|
| days since the first loop (11 Oct) | 22 | `reports/README.md` (week 1 = 11 - 17 Oct) |
| loops | 21 filed + loop 22 running | `HANDOVER.md` |
| experiment folders, lane project (catalogue ids E15 - E267) | 221 | `experiments/` |
| experiment folders, BEV project (t1 - t11, since 25 Oct) | 14 | `bev_fusion/experiments/` |
| launcher runs since loop 18 (lane ledger) | 75, of which 65 ok, 6 failed, 2 stalled, 1 stopped, 1 blocked; 25.8 job-hours | `LEDGER.csv` |
| launcher runs of the BEV project (ledger since 31 Oct) | 4 rows, 3 ok, 1 failed; 2.9 job-hours (the BEV trainings before D23 were not under the launcher) | `bev_fusion/records/LEDGER.csv` |
| papers triaged / notes / deep extractions | 540 / 404 / 38 | `literature/papers.csv`, `papers/`, `deep/` |
| commits (git started on 25 Oct) / tagged runs | 385 / 45 | `git log`, `git tag` |
| decisions asked of the person (D1 - D31, no D12) | 30; 22 answered, 8 open | `DECISIONS.md` |
| tools / agent roles / suites | 16 / 7 / 2 (evalsuite, trainsuite) | `autoresearch/` |
| hardware | one RTX 3060, 12 GB | `PROBLEM.md` |

Reading: output is high, and so is the share of "no". Of the 29 experiments of loops 18 - 21, 18 were decided by their written rule: 5 KEEP, 8 CUT, 5 TIE
(`metrics.py`). The prediction written before the run was right in 8 of 15 cases that carried one. A loop that says "no" in 72 % of its decisions and misses its own
prediction half of the time is not selling anything; that is the point of the rules.

## 2 It finds causes (three stories)

### 2.1 The maps of the BEV system: from 17 - 55 % of the SOA to a measured, larger share

- **Question** (Mirko, 30 Oct): "Look exactly why performance is not SOA, find out." The gap was measured like for like first: the published map model GaussianCaR
  (6 cameras + radar, trained for maps) scored with our metric on our frames reached 0.85 / 0.59 / 0.66 / 0.54 / 0.69 / 0.63 IoU (drivable, walkway, carpark,
  crossing, lane divider, road divider); our chain reached 17 - 55 % of that on everything except the drivable area (85 %) (`t6_soa_gap`).
- **The loop removed candidates one by one, each with a control**: not the range (the gap is as large at 0 - 20 m), not the threshold (best threshold moves ours by
  <= 0.04), not over-fitting (our TRAIN IoU is already below the SOA's VAL IoU), not the BEV network (11 block exchanges, none better than the reference: 7 ties, 4 worse),
  not task competition (a maps-only run gains 0.015 - 0.051, 7 - 12 % of the gap), not the labels (the SOA scores 0.69 on the same labels).
  What was left, and measured: the image features are frozen, trained for DETECTION, at stride 16 (a linear probe: lane dividers 0.031 against a prior of 0.018 are not
  even readable), plus a training budget 9 - 27 x smaller than the SOA's.
- **The test** (one day: port, cache of 58,101 frames, two seeds and a same-code control): swap the image features for the map-trained stride-8 features of BEVFusion.
  Maps nearly doubled (lane divider 0.191 -> 0.351 / 0.345, walkway 0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165, drivable 0.716 -> 0.804 / 0.809) and the
  objects HALVED (front-sector mAP2 0.367 -> 0.183 / 0.180, pedestrians gone). The loop wrote that down as a result, not as a failure.
- **The next hypothesis, built and run the same night** (both feature sets in one chain, `t10_dual`): maps kept (lane divider 0.335, walkway 0.340, drivable 0.806),
  objects kept (front mAP2 0.361 against 0.367, inside the 0.054 noise band, pedestrian AP@2 0.199 against 0.210). Its prediction (written first) hit on four of five numbers;
  the fifth (pedestrian AP@2) came out better than predicted (0.199 against 0.15). Rule E1 held: seed twin next (`t10.dual_s1` is training).
- **What it means**: the 3000-step dual chain beats the same chain trained 4.7 x longer on five of six map layers (ladder below). Diagnosis beat compute.

| maps, VAL, `seg_fov` (IoU) | drivable | walkway | carpark | crossing | lane divider (0.5 m) | road divider (0.5 m) |
|---|---|---|---|---|---|---|
| start (`t2_ref_s0`, 3000 steps) | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| start, 4.7 x the training (`t1_r4_soa`, 7000 x 2 steps, 3109 frames) | 0.698 | 0.244 | 0.162 | 0.093 | 0.274 | 0.322 |
| both feature sets (`t10_dual_s0`, 3000 steps) | 0.756 | 0.340 | 0.144 | 0.154 | 0.335 | 0.442 |

### 2.2 The velocity that was never learned

- **Symptom** (first seen in the eight-try report, weak spot 2): in every 3000-step run the velocity error equals the zero predictor (2.39 against 2.39 m/s); a
  velocity loss weight of 10 changed nothing (try 9).
- **Cause**, found by reading stored checkpoints and detections (no GPU): the velocity branch of the object head is a dead ReLU layer. Its output is its bias (1e-7 m/s);
  0.011 % of its hidden units are active at the supervised centres, against 13 - 50 % for the other branches of the same head. 24 of the 29 runs it classified are dead;
  an effective batch of 6 or more separates the live ones (association, not yet a result). The CenterPoint reference head (mmdet3d) has a BatchNorm in each branch; ours has none.
- **What the input holds**: the radar Doppler at the GT vehicle boxes, with the box heading and a closed form, recovers 79 % of the velocity of moving vehicles
  (error 1.37 against 6.42 m/s for the zero predictor); our best learned heads reach 22 - 25 %. Fresh branch, BatchNorm and a moving / static gate are queued as arms.
- **For a team**: a dead branch looks like "the model does not learn velocity" for weeks. The loop found it in hours, from files.

### 2.3 The lane-centre graph: the same quality at 26 x the speed, fully differentiable

| step | result | source |
|---|---|---|
| the right frozen foundation | lane F1 0.624 -> 0.867 VAL (0.598 -> 0.835 TEST), +0.24: the largest single effect of the project | E27 |
| learned trace network, no hand-built tracer or assembler | F1@1 0.883 VAL / 0.851 TEST; 628 ms per frame | E94 |
| differentiable sparse-graph builder, chosen from eight designs by successive halving | champion Q 0.735 against 0.737 of the 628 ms control, 24 ms per frame (26 x) | E123, `PROBLEM.md` |
| fork pruning | junction score J 0.265 -> 0.409 (control 0.408); paired scene bootstrap +0.144 [+0.104, +0.182] | E215 |
| rough localisation + trust gate for the SD map | pose error 2.55 -> 0.82 m at a 2 m map error; its own S gate (+0.015) was missed by 0.004, the pose and wrong-map gates were met | E213.a |

TEST was used once per finished design and logged: champion S 0.748, J 0.422, lanes found 0.653 against the control 0.702 / 0.404 / 0.762. It wins on S and J and **loses
on far-range coverage**; that is gap 1 of `PROBLEM.md`, ranked first.
Five questions are closed with evidence so nobody re-runs them: a route loss through the builder (no, three times), larger recall weights (no beyond weight 3),
anchoring settings (no), three GPU jobs at once (no, they stall), a matching-free Poisson set loss (Q -0.073).

## 3 It can be trusted (and what happens when it cannot)

**The rules.** A judge the loop cannot change (`evaluation/`, golden numbers, a stored number reproduced before a new harness is believed: the re-evaluation of the
reference reproduced lane divider 0.181 / 0.179 and drivable 0.714 / 0.711 exactly); a rule written before the run and applied by a tool (`decide.py`: KEEP / CUT / TIE /
INCONCLUSIVE); a measured noise band (Q seed SD 0.0025; mAP2 band 0.054 from two seeds); the TEST split once per finished design and logged; protected files behind a hook;
numbers from files, never typed; a fresh reviewer for every report.

**What these rules caught** (each is in a record with its fix):

| incident | how it was found | consequence |
|---|---|---|
| an arm of the first study ran with the camera ON (`camera_off` ignored by the cached model) | a stage probe of the supposed floor model | two conclusions retracted; fixed with a self-test (`t3_improve_loop` report, section 9) |
| the pre-trained FPN of the BEV trunk was loaded with its inputs in the wrong order | the DualBEV port produced no signal | fixed behind a flag; effect on our chain: a tie (0.367 against 0.375 / 0.359) |
| training GT built from live code, so runs of different days used different targets | the lane chat noticed the cache signatures | same-code controls (`t8_ctl`), a decision for one GT version (D29) |
| velocity branch dead in 24 of 29 classified runs | system-review lens, from checkpoints | section 2.2 |
| three review passes in one evening: pass 1 listed eight high-severity findings (a TEST guard that accepted a substring, a null baseline that inherited the first frame's tags, a GT cache written non-atomically, ...), pass 3 nine more bugs | fresh reviewers reading code | six of the nine repaired in the pass with a regression test that fails on the old code, two by other chats |
| my own predictions: DualBEV front-only mAP2 "0.40 - 0.55" (measured 0.250); zero-camera mAP3 "0.20 - 0.26" (measured 0.013) | the measurement | recorded as misses; they are why decisions follow numbers, not hunches |

**Outside context.** A survey of AI-scientist systems (arXiv 2608.05179, 29 June 2026) names the bottleneck: not whether agents can run research, but whether their claims
can be verified; only 38 % of the runnable systems it coded release seeds or execution traces, and of nine closed-loop systems none shows an externally validated oracle
inside the loop. Autoresearch puts exactly that into the loop: two published systems (DualBEV, GaussianCaR) run on our harness as external references.
Other loops report results too (primary pages read on 2026-11-01): Karpathy's `autoresearch` (one GPU, a fixed five-minute run per experiment, about 12 experiments per hour or
100 per night by its README), AlphaEvolve (a scheduling heuristic that recovers 0.7 % of Google's worldwide compute, a 48-multiplication 4 x 4 complex matrix product), AIDE
with o1-preview on MLE-bench (bronze medal or better in 16.9 % of 75 Kaggle competitions). None of these is a perception problem; they show that loops with a fixed judge find
real improvements, and that the judge is where they differ.

## 4 Where it stands against the state of the art (like for like)

| area | SOA reference we ran on our frames | SOA | ours | reading |
|---|---|---|---|---|
| detection, front sector 0 - 50 m, mAP2 (vehicle + pedestrian) | DualBEV, pre-trained, front camera only | 0.250 | camera only 0.242 (tie); camera + radar 0.359 / 0.375 (3000 steps), 0.400 (7000 x 2), dual chain 0.361 | the camera path equals the SOA camera path; radar adds +0.11 - 0.15 (interval excludes 0). No SOA camera + radar detector was run on our frames (CRN is on disk) |
| maps, same 400 frames, IoU | GaussianCaR | PENDING | PENDING | PENDING (the like-for-like re-score of the dual chain is running on the CPU) |
| lane-centre graph | none on our metric. Closest published: TopoBDA camera + radar on OpenLane-V2 subset B (nuScenes), R50: DET_l 49.4, TOP_ll 38.6 | - | Q 0.735, topo F1 0.741, 24 ms | NOT comparable: other annotation, other metric, Hungarian matching. A bridge is a plan (section 5), not a result |
| box velocity | published mAVE (all classes) e.g. CRN 0.344 | - | the zero predictor | not learned yet (section 2.2) |

**Not claimed**: any number on a public leaderboard; six cameras; ten classes; NDS; 20 epochs of training; a generalisation beyond nuScenes (non-commercial licence: research use only).

## 5 The path to the state of the art, ranked by information per cost

Every lever below has a measured effect size from this repository; the sums are hypotheses to test, not forecasts.

| # | lever | measured so far | cost | what it would show |
|---|---|---|---|---|
| 1 | dual features, longer (6000 - 7000 x 2 steps) and all 700 scenes | the old features gained +0.09 on the lane divider from 3000 to 7000 x 2 steps | 4 - 9 GPU hours | how much of the remaining map gap training alone closes |
| 2 | stride-4 detail branch on top | on the old features: lane divider +0.065 / +0.042, road divider IoU@1 +0.119 / +0.096 (two seeds) | 2 GPU hours per seed | whether the thin classes add up |
| 3 | velocity: fresh branch, BatchNorm, moving / static gate, Doppler read-out | closed-form upper bound 0.79 against 0.25 learned | 1 - 3 GPU hours | box velocity at all |
| 4 | a SOA camera + radar detector (CRN) on our frames | the checkpoint is on disk | half a day of porting | turns "no number" into a number |
| 5 | bridge the lane graph to OpenLane-V2 subset B (DET_l, TOP_ll) | nothing yet; needs the annotation files and the official evaluator (a download: needs Mirko's OK) | 1 day for a feasibility check | the first number a stranger can compare |
| 6 | end-to-end image encoder with both losses (plan t9) | designed, not run | 2 - 3 days | whether features trained for both tasks beat two feature sets |

## 6 Making the hobby meaningful: three outputs that outlive it

1. **A method package others can run**: the loop, the judge, the tools and the rules (`autoresearch/`: launcher with a git tag per run, ledger, decision tool, evalsuite,
   trainsuite, seven agent roles), with a six-page write-up "a verification-first research loop for perception". It is problem-independent: the second problem (BEV fusion)
   was plugged in on 31 October in one day.
2. **One public number per problem** (levers 4 and 5): a result a stranger can check, with the harness as the artefact.
3. **A lessons-learned list that saves a team months**: frozen detection-trained image features cap map quality (here by a factor of two); a velocity branch can be dead
   from the first steps; the cache that builds your GT can silently change between runs; a seed is worth a ledger row. Each lesson has its number and its file.

A pilot with a second team (one month, compare experiments per week and wrong-number findings against today) is the cheapest way to turn (1) into evidence.

## 7 The ten hardest questions

1. **"Is this an AI inventing numbers?"** No number in the reports is typed: tables come from `metrics.json` through scripts; a decimal that exists in no source file is listed
   by `check_numbers.py`; a fresh reviewer re-reads primary files. The record of what went wrong (section 3) is the evidence the checks work.
2. **"What does the person do?"** Sets the question, the rules and the stop time, answers 30 decision entries (champion promotion, TEST use, protected files, deletions,
   downloads), reads the ledger. The agent does the coding, the running and the first reading. This division is the design, not a shortcut.
3. **"Why not an existing tool (MLflow, W&B, Optuna)?"** Those track runs and tune knobs. This also writes the hypothesis and the decision rule before the run, runs the
   controls, and diagnoses causes. It could sit on top of them; it was not benchmarked against them.
4. **"Could this be luck?"** A seed twin or a same-code control exists for every claim in section 2 except where stated (the seed twin of the dual chain is training); the
   decision uses a measured band, and a difference inside it is a TIE.
5. **"Is it state of the art?"** Not on a public benchmark. Like for like: equal to the SOA camera detector, +0.11 - 0.15 with radar; maps: PENDING; lane graph: no comparable number yet.
6. **"What does it cost?"** One RTX 3060 and a few hours of attention a day; 28.7 job-hours under the launcher since loop 18 (earlier loops were not metered); the agent tool budget
   is the open item.
7. **"Does it transfer to our data?"** The tools are problem-independent (a metric-card writer and a launch command are the plug). Not tested on a second organisation's data.
8. **"Where did it fail?"** Five closed questions, 13 of 18 rule decisions CUT or TIE, a velocity head dead for three weeks, a feature swap that halved the objects, predictions
   missed by a factor of 1.6 - 15.
9. **"What can go wrong?"** A full C: disk crashed a run at a checkpoint write once (4.5 idle GPU hours) and is at 4.1 GB free tonight; two chats edit shared code; the loop
   can chase noise. Mitigations: the GPU gate and slots, one lead per loop, the noise band, a clean-up rule that moves and never deletes.
10. **"How will we know it worked?"** Three success criteria, written now: (a) every map layer at 70 % or more of the SOA like for like; (b) box velocity with skill 0.5 or more on moving
    vehicles; (c) one public-benchmark number per problem. Missing them by the next review is a result too.

## 8 What to ask for

1. Time on the GPU at night and at weekends inside the written rules (stop time, GPU limit, no TEST without OK).
2. A named person who signs off every claim that leaves the team and reads the ledger once a week.
3. A second reviewer for headline numbers; budget for the agent tool.
4. Permission for two downloads (OpenLane-V2 annotations and its evaluator) to build the first public number.
5. A one-month pilot with a second team.

## 9 Ten minutes, in order

1. The claim and its limit (section 0). 2. The loop in one picture. 3. Maps: question, eliminated causes, the test, the result (2.1). 4. Velocity: the dead layer (2.2).
5. Lane graph: same quality at 26 x (2.3). 6. What the rules caught (3). 7. Like-for-like SOA table (4). 8. The path (5). 9. The ask (8).

## 10 Open points before the meeting

- PENDING: like-for-like maps (CPU, running), seed twin `t10.dual_s1` (training), second control seed `t8_ctl_s1` (evaluation queued), the lane prototype of the overnight round.
- Decisions that wait for Mirko: D4, D17, D18, D19, D25, D29, D30, D31; the 13 GB of probe dumps on C:.

## 11 Sources

`bev_fusion_lane_center/PROBLEM.md`, `experiment_execution/{LEDGER.csv, DECISIONS.md, closed_questions.md}`, `experiments/INDEX.csv`, `bev_fusion/records/LEDGER.csv`,
`bev_fusion/experiments/{t3_improve_loop, t4_soa_front, t6_soa_gap, t7_lane_branch, t8_bevfeat, t10_dual_features, t11_vel_cause}/`, `bev_fusion/runs/*/eval/metrics.json`,
`autoresearch/history/repo_review_2026-10-31*.md`, `bev_fusion/temp/pitch/facts.json` (counts, made by `facts.py`), `ladder.md` (made by `ladder.py`),
`bev_fusion/temp/soa_gap/map_ref_*_400.json` (like-for-like maps). External, primary pages read on 2026-11-01: arXiv 2608.05179 (the verification-gap survey); arXiv 2410.07095
(MLE-bench); the AlphaEvolve post of Google DeepMind; the README of github.com/karpathy/autoresearch.
