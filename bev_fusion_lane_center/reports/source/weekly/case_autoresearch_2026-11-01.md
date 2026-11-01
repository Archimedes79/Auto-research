# The case for Autoresearch - evidence for the meeting of 2026-11-01 / 02

Version 4, 2026-11-01 09:45 after the numbers review of 09:35 (new since version 3: the like-for-like map scores on identical frames, the velocity fix, the lane prototype with its TEST use, the amended 6000-step run; section 12). Version 1 went through a sceptical review (twelve numbered overclaims), version 2 through a fresh numbers review against the primary files (eight wrong or stale
numbers, nine mixed sets, several statements stronger than their source); everything found was checked and corrected (section 12). Every number comes from a stored file (section 11); counts
are dated, because the repository keeps moving; a number that waits for a run says PENDING.

**Definitions.** BEV project: VAL = 1000 keyframes spread over 82 of the 150 official nuScenes validation scenes; it has no TEST split and none was used. Lane project: VAL = 873 frames and TEST =
21 scenes of its own scene-disjoint split of 255 scenes; TEST was used in loop 16, loop 18 and loop 22 (logged: once per finished design, with the person's OK or grant; the TEST numbers of loops 1 - 15, among
them E27 and E94, were not logged at the time). "Like for like" = a published system run by our own harness on the same frames with the same metric. IoU tolerance is 0 m except the dividers (0.5 m);
"drivable@0.5 m" is the same layer at 0.5 m tolerance.

## 0 The short version

Autoresearch is a research loop: an agent proposes and runs experiments on ONE RTX 3060, every result goes into a ledger and into git, a rule written before the run decides (for most runs,
section 3), and a person keeps the decisions. After three weeks it has

1. **produced**: 22 loops (all filed), 222 lane experiment folders and 14 BEV experiments, 540 papers triaged (404 notes, 36 deep extractions), 418 commits, 100 lane launcher runs
   since loop 18, 34 decisions asked of the person (29 answered, 5 open) (counts of 09:20);
2. **found causes, not only knobs**: three stories (section 2). The better parts are borrowed from published models (BEVFusion's map-trained features, GaussianCaR's features, DualBEV's heads). What
   the loop added is the diagnosis by elimination, the controlled swap, the like-for-like harness and the integration (box in 2.1);
3. **kept a record that shows its checks and its lapses** (section 3): a reviewed judge, a prediction and a rule in the ledger for 86 of 100 executed lane runs, measured noise bands, and reviews that
   found real bugs and process lapses;
4. **measured itself against the state of the art like for like** where a published system could be run on our frames, with named limits (section 4);
5. **and is NOT state of the art on any public leaderboard.** Say this first. What it shows today is the measured distance, a ranked path, and a test that decides whether the path is realistic
   (section 5).

The value is not that an AI writes code. It is a loop that records everything, tests one block at a time against a control, and makes its own checks visible, including where they failed.

## 1 It produces (counts from the repository, 2026-11-01 09:00)

| what | number | source |
|---|---|---|
| days since the first loop (11 Oct) | 21 (the 22nd day) | `reports/README.md` (week 1 = 11 - 17 Oct) |
| loops | 22 filed (loop 22 closed this morning) | `HANDOVER.md` |
| experiment folders, lane project (catalogue ids E15 - E268) | 222 | `experiments/` |
| experiment folders, BEV project (t1 - t11, since 25 Oct) | 14 | `bev_fusion/experiments/` |
| launcher runs since loop 18 (lane ledger, executed rows) | 100: 90 ok, 6 failed, 2 stalled, 1 stopped, 1 blocked; 32.2 job-hours | `LEDGER.csv` |
| ... of these with a prediction and a rule in the ledger | 86 of 100 | `LEDGER.csv` |
| launcher runs of the BEV project (ledger since 31 Oct) | 12, all ok; 7.8 job-hours (BEV trainings before D23 were not under the launcher: unmetered) | `bev_fusion/records/LEDGER.csv` |
| idea to verdict, the three latest cards | t10: card committed 19:30, result committed 23:24 on 31 Oct (3 h 54 min, two commits, 115 min of training inside); t11 velocity cause: about 1 h by the card's own header (00:10 to 01:17 on 1 Nov; one commit holds card and results); t11 velocity hurdle: card 23:45 on 31 Oct, summary 06:10 on 1 Nov (6 h 25 min, two 1000-step trainings and their evaluations inside) | `git log`, card headers |
| papers triaged / notes / deep extractions | 540 / 404 / 36 | `literature/papers.csv`, `papers/`, `deep/` |
| commits (git started on 25 Oct) / tagged runs | 418 / 76 | `git log`, `git tag` |
| decisions asked of the person (D1 - D35, no D12) | 34; 29 answered, 5 open (D4, D17, D29, D30, D31), counts of 09:20 | `DECISIONS.md` |
| tools / agent roles / suites | 15 / 4 / 2 (evalsuite, trainsuite; the four roles merged at 08:19 today) | `autoresearch/` |
| hardware | one RTX 3060, 12 GB | `PROBLEM.md` |

Reading: output is high, and the verdicts are not all wins. Since loop 18 the ledger holds 35 experiments (loops 18 - 22); `metrics.py` counts 23 verdict events on them: 8 KEEP, 10 CUT,
5 TIE. Read this with care: fourteen events come from `decide.py` files (five of the first nine files were re-encoded from the record afterwards: E215, E219 x3, E232) and nine from "rule met / not met"
text in ledger rows; six of the CUT / TIE belong to one family (the recall-weight arms of E219), two KEEPs are data checks (E235.chk3, E251.chk) and one is a tool KEEP that the lead did not adopt
(E264, a guard missed). Of the 16 events where the tool could read a hit or a miss of the written prediction, 8 are hits (the loop-22 page itself counts four hits and two misses by its own tolerances;
the tool read one written miss there).

## 2 It finds causes (three stories)

### 2.1 The maps of the BEV system

- **Question** (Mirko, 30 Oct): "Look exactly why performance is not SOA, find out." The gap was measured like for like first: the published map model GaussianCaR (6 cameras + radar, trained for
  maps, 40 epochs on 4 x A100) scored with our metric on 400 of our frames reached 0.84 / 0.59 / 0.66 / 0.54 / 0.69 / 0.63 IoU (drivable, walkway, carpark, crossing, lane divider, road divider);
  our chain of that day (7000 x 2 steps) reached 17 - 55 % of that on everything except the drivable area (85 %) (`t6_soa_gap`).
- **The loop removed candidates one at a time, each with a control**: not the range (the gap is as large at 0 - 20 m), not the threshold (best threshold moves ours by <= 0.04), not over-fitting
  for the big areas (our TRAIN IoU is already below the SOA's VAL IoU; for thin classes partly: lane divider 0.346 train against 0.274 val), no BEV-network lever found (the trunk, fusion,
  temporal and radar blocks were swapped eight times: all ties inside a 0.054 band, so "no lever found", not "proved irrelevant"; the study's eleven exchanges, which also count sensor
  removals and camera fixes, give 7 ties and 4 worse), task competition is small (a maps-only run gains 0.015 - 0.051, 7 - 12 % of the gap), the labels are not the cause (the SOA scores 0.69
  on the same labels). What stayed standing: the image features are frozen, trained for DETECTION, at stride 16 (a linear probe: lane dividers 0.031 against a prior of 0.018 are not even
  readable), plus a training budget 9 - 27 x smaller than the SOA's.
- **The test** (one day: port, cache of 58,101 frames, two seeds and a same-code control): swap the image features for the map-trained stride-8 features of BEVFusion. It is a bundle (backbone,
  stride, training objective and depth head all differ), so it tests "the source of the image features" as one block; it was planned in plan 2.8 with a qualitative rule ("if the maps move
  towards the SOA, the encoder is the cause"), not with a numeric card. Lane divider, walkway and road divider rose x1.8 - 2.0 and the crossing from 0.002 to 0.145 / 0.165 (lane divider
  0.191 -> 0.351 / 0.345, walkway 0.178 -> 0.339 / 0.342); drivable rose x1.1 (0 m) and carpark x1.5. The objects HALVED (front-sector mAP2 0.367 -> 0.183 / 0.180, pedestrians gone). The loop
  wrote that down as a result, not as a failure.
- **The next hypothesis, built and run the same night** (both feature sets in one chain, `t10_dual`, two seeds): maps kept (lane divider 0.335 / 0.350, walkway 0.340 / 0.354, crossing 0.154 /
  0.173, drivable@0.5 m 0.806 / 0.807; the control: 0.191 / 0.180, 0.178 / 0.176, 0.002 / 0.000, 0.716 / 0.716) and vehicle + pedestrian detection nearly kept: front mAP2 0.361 / 0.364 against
  the control's 0.367 / 0.368, i.e. 0.003 - 0.007 lower in every pairing, a small consistent shortfall that sits inside the rule's band (0.054) with bootstrap intervals touching zero (the
  0.016 seed spread belongs to the older t2 pair, not to these runs); pedestrian AP@2 0.199 / 0.202 against 0.210 / 0.213. The rare two-wheeler class and the 3-class mAP are noisy and nothing
  is claimed about them (two-wheeler AP@2: control 0.121 / 0.051, dual 0.016 / 0.061; 3-class mAP 0.251 / 0.235 against 0.221 / 0.234). The prediction (written first) hit on four of six
  numbers; the other two came out better than predicted (pedestrian AP@2 0.199 against 0.15, vehicle AP@2 0.670 against 0.62). Rule E1 held on both seeds. Caveats on the seeds: the dual twin was killed once by the
  out-of-memory killer and resumed on a later code snapshot; the control's second seed was also resumed after a CUDA crash (its last 500 steps ran on another snapshot); none of these resumes was
  shown to be neutral (the BEVFusion-only seeds were resumed too: s0 after a CUDA error at step 1740, s1 after a 0-byte partial GT cache file, `plan.md` section 8; so four of the six runs behind the
  maps story were resumed), and the seed pairs agree with their first seed.
- **What it means**: a feature swap beat the start recipe trained with 4.7 x the training samples (7000 steps at accumulation 2: 2.3 x the optimiser steps at double the batch; a different
  config and a larger frame set) on five of six map layers clearly (map-trained features alone; the carpark is level: 0.160 against 0.173 / 0.168 on the same 1000 frames) and on five of six (dual chain; below on the carpark). The saving is partly borrowed compute: the better features come from a
  map-trained network that somebody else trained.

| maps, VAL, `seg_fov` (IoU) | drivable (0 m) | walkway | carpark | crossing | lane divider (0.5 m) | road divider (0.5 m) |
|---|---|---|---|---|---|---|
| start (`t2_ref_s0`, 3000 steps) | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| start, 4.7 x the training samples (`t1_r4_soa`, 7000 x 2 steps, 3109 frames) | 0.698 | 0.244 | 0.162 | 0.093 | 0.274 | 0.322 |
| both feature sets (`t10_dual_s0`, 3000 steps) | 0.756 | 0.340 | 0.144 | 0.154 | 0.335 | 0.442 |
| both feature sets, seed twin (`t10_dual_s1`, 3000 steps) | 0.757 | 0.354 | 0.139 | 0.173 | 0.350 | 0.442 |
| same-code control, two seeds (`t8_ctl_s0` / `s1`, 3000 steps) | 0.668 / 0.668 | 0.178 / 0.176 | 0.115 / 0.106 | 0.002 / 0.000 | 0.191 / 0.180 | 0.224 / 0.227 |

**Against the published maps model, on identical frames** (new in version 4; run `pitch.maps_lfl`: 400 frames of 43 val scenes, GaussianCaR and our chains scored by the same harness with the same
mask, the same tolerances and the 0.5 threshold; table in section 4): the dual chain reaches 94 % of GaussianCaR on the drivable area, 72 - 76 % on the road divider, 57 - 62 % on the walkway,
50 - 52 % on the lane divider, 31 - 34 % on the carpark and 26 - 31 % on the crossing (two seeds); the image-only control 82 / 33 / 27 / 0 / 28 / 38 % (drivable, walkway, carpark, crossing, lane divider,
road divider).

**What is borrowed and what is ours** (said openly, because the first objection is "the wins are other people's models"):
- Borrowed from published work: the foundation features of the lane project (GaussianCaR camera + radar BEV features: +0.24 lane F1, the largest single effect of that project; its TEST numbers
  are from loops 1 - 15, not logged at the time), the map-trained features of BEVFusion (lane divider, walkway, road divider x1.8 - 2.0), the DualBEV image heads and BEV encoder, the CenterPoint head.
- Added here: finding WHICH piece limits (elimination with controls), the controlled swap and the merge of two feature sets, the harness that scores two published systems on our frames, the
  differentiable lane-graph builder, fork pruning, map registration with a trust gate, the diagnosis of the velocity layer, and the loop with its ledger.

### 2.2 The velocity branch that learned nothing in 28 of 30 batch-3 runs, and the fix that revived it

- **Symptom** (first seen in the eight-try report, weak spot 2): in every 3000-step run the velocity error equals the zero predictor (2.39 against 2.39 m/s); a velocity loss weight of 10 changed
  nothing (try 9).
- **Cause**, found by reading stored checkpoints and detections (no GPU): the velocity branch of the object head is a dead ReLU layer. Its output is its bias (1e-7 m/s); 0.011 % of its hidden
  units are active at the supervised centres, against 13 - 50 % for the other branches of the same head. In the velocity session's table of 36 evaluated checkpoints
  (`temp/box_vel/out/alive_table.txt`) 28 of the 30 runs with an effective batch of 3 are dead (one tiny, one "alive" only as a maps-only run whose untrained object head outputs 2.97 m/s); the six rows with an effective batch of 6 or more (four distinct runs) are
  alive by output size (skill 0.254 for `t1_r4_soa`, 0.203 for `t1_r0`, but only 0.025 for the 6000-step `t1_r3_combined`): an association, not a proof. A linear read-out of the trunk features gives skill 0.019 (predicted
  >= 0.15, so that prediction FAILED): the trunk holds no velocity information because nothing ever asked it to. The CenterPoint reference head (mmdet3d) has a BatchNorm in each branch; ours has none.
- **What the input holds**: the radar Doppler at the GT vehicle boxes, with the box heading and a closed form, recovers 79 % of the velocity of moving vehicles (error 1.37 against 6.42 m/s for
  the zero predictor). That is an upper bound on 300 frames of 30 scenes, with GT association and GT heading; the older learned heads reached 0.20 - 0.25 only after 5000 - 7000 steps with a larger batch (0.203 for `t1_r0` at 5000 steps, 0.254 for `t1_r4_soa` at 7000 x 2), on matched detections of other frame sets.
- **The fix and its result** (card written 23:45 on 31 Oct before any code; verdict 06:10 on 1 Nov): a moving / static logit next to the velocity regression, the regression trained on moving centres
  only, and a decode gate (velocity 0 where the logit says static); 1000 steps warm-started from the dual chain (`t10_dual_s0`) with a fresh velocity branch, scored on cold VAL (1000 frames of 82
  scenes). Arm B' (batch 3): vehicle velocity skill 0.470 (on the matched true positives: about 1100 moving vehicles in the 1000 frames); arm B2 (accumulation 2, effective batch 6): 0.566 (the dual chain without the fix: 0.000). Vehicle AVE 2.39 -> 1.19 m/s (on the moving
  cars 6.58 -> 2.86 m/s). The written prediction was skill 0.25 (range 0.12 - 0.40): both arms beat it (a miss on the good side). Verdict by the card's rule: V1, the velocity learns, for B2; B' misses the
  card's clause on the error at static cars (0.336 against <= 0.30 m/s). The price: front mAP2 -0.014 (interval [-0.020, -0.006], inside the 0.054 band; vehicle AP@0.5 0.254 -> 0.207), the error at
  static cars rises from 0.03 to 0.28 m/s, drivable, walkway and lane divider within 0.01 of the dual chain (the carpark is +0.03: more samples or the hurdle, not separable), speed within 1 - 2 ms (indicative). One seed per arm; the from-scratch test is the 6000-step run (lever 1, section 5);
  pedestrian velocity stays near 0 (skill 0.014). Next candidates for the moving error: the radar Doppler at the boxes (bound 0.79) and the fused flow pyramid.
- **For a team**: a dead branch looks like "the model does not learn velocity" for as long as nobody opens the checkpoint. The loop did, from files, in hours, and had a fix with a verdict, by a rule
  written first, the next morning. The dead branch sat in 28 of the 30 batch-3 runs of the BEV project (the exceptions: `t2_fusion_cmx` with a tiny output and the maps-only run `t6_maps_only`).

### 2.3 The lane-centre graph: equal Q to a 628 ms network, in 24 ms with cached features

| step | result | source |
|---|---|---|
| the right frozen foundation (published GaussianCaR features) | lane F1 0.624 -> 0.867 VAL (0.598 -> 0.835 TEST, loop 1 - 15 TEST use, not logged then), +0.24 | E27 |
| learned trace network, no hand-built tracer or assembler | F1 0.883 VAL / 0.851 TEST (same caveat); 628 ms per frame | E94 |
| differentiable sparse-graph builder, chosen from eight designs by successive halving | champion Q 0.735 against 0.737 of the control, lanes found 0.695 against 0.800, R@1 far 0.709 against 0.811; 24 ms against 628 ms (26 x), cached foundation features, the foundation network not included | E123, `PROBLEM.md` |
| fork pruning | junction score J 0.265 -> 0.409 (control 0.408); paired scene bootstrap +0.144 [+0.104, +0.182] | E215 |
| rough localisation + trust gate for the SD map | pose error 2.55 -> 0.82 m at a 2 m map error; its own S gate (+0.015) was missed by 0.004, the pose and wrong-map gates were met | E213.a |
| the prototype of the overnight round `recall_sp0_dp` (recall weight 3, fork pruning 0.4, spurs kept, output-stage simplification; one command, CPU, cached network outputs) | VAL Q 0.742, lanes found 0.781, R@1 far 0.779; TEST (one use, 833 frames of 21 scenes) Q 0.727 against the control's 0.712 (paired, +0.015 [+0.000, +0.029]), lanes found 0.740 against 0.762 (-0.021 [-0.037, -0.005]), R@1 far 0.749 against 0.776; 12.6 against 22.0 vertices per 100 m (counted after its output-stage simplification, the control's without one) | E265 - E268, `morning_2026-11-01_loop22.md` |

TEST (21 scenes; the loop-16 and loop-18 numbers carry no intervals, the control's come from loop 16's TEST use, the champion's from loop 18's): champion J 0.422, lanes found 0.653 against the control 0.404 / 0.762; S 0.748
against 0.702, but S rewards sparsity (15.6 against 22.0 vertices per 100 m) and the control's TEST Q was not logged. The champion **loses on far-range coverage**: gap 1 of `PROBLEM.md`. The loop-22 prototype
narrows it (lanes found on TEST 0.740 against the control's 0.762; the same checkpoint's loop-18 TEST number was 0.724, another code version, so an unpaired reference) with Q above the control. Where the gap was closed: on VAL (lanes found / R@1 far) the champion stood at 0.695 / 0.709, the
control at 0.800 / 0.811; the recall weight (training: the rc1 checkpoint, 0.767 / 0.771) closed about two thirds of the gap, the night's chain changes (the spur rule, E265) added +0.014 / +0.008 (about a tenth) for
0.781 / 0.779, and on TEST the prototype is still below the control (lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012]); the seed ring (E266, R@1 far +0.015) is not in the prototype. The prototype's TEST use was logged first, on the person's grant (D28: "One TEST check").
**Named gaps, reported every time**: the delivered graph still passes rule steps (path union, fork pruning, Douglas-Peucker simplification), the route head is detached, and the planner-level
path error sits at a floor of 0.27 m whatever the graph (gap 3); only the TRAINING path is fully differentiable. The data are partly simulated (the SD map) and use exact poses. The results video
and poster show the loop-22 prototype `recall_sp0_dp` (VAL Q 0.742, lanes found 0.781), not the champion of the table.
Five questions are closed with evidence so nobody re-runs them: a route loss through the builder (no, three times), larger recall weights (no beyond weight 3), anchoring settings (no), three GPU
jobs at once (no, they stall), a matching-free Poisson set loss (Q -0.073).

## 3 It keeps a record of its own checks, and of where they failed

**The rules, and where they hold.**
- A judge that is reviewed and re-checked, not one that cannot change: golden numbers, and a stored number is reproduced before a new harness is believed (the re-evaluation of the reference
  reproduced lane divider 0.181 / 0.179 and drivable 0.714 / 0.711 exactly). The hook protects the truth files (golden numbers, TEST log, champion entries, `project.json`), not the evaluator
  code itself (D21); two findings of review pass 1 (L1, L2) were bugs in that code.
- A rule written before the run, for most runs: 86 of 100 executed lane runs carry a prediction and a rule in the ledger. Exceptions the reviews found: the pivotal t8 feature swap (a qualitative
  rule, no numeric card), E257's rule relaxed after the sweep, E259.b0's prediction written after E259.a0 finished, three experiment files (E215, E219, E232) re-encoded from the loop-18
  record. Where a card exists a tool decides (`decide.py`: KEEP / CUT / TIE / INCONCLUSIVE) inside a measured band (Q seed SD 0.0025; the front-sector mAP2 of the same-code controls differs by
  0.001, the older t2 pair by 0.016; decision band 0.054, set by the walkway).
- The TEST split once per finished design, logged (loops 16, 18 and 22); numbers from files by script; a fresh reviewer for every report. The review of the week-2 report still found 4 wrong numbers,
  8 mixed-split cases and 16 overclaims in about 420 claims; the review of version 2 of this file found eight wrong or stale numbers. That is why the review exists.

**What the checks caught, and where they were bypassed** (each is in a record with its fix):

| incident | how it was found | consequence |
|---|---|---|
| an arm of the first study ran with the camera ON (`camera_off` ignored by the cached model) | a stage probe of the supposed floor model | two conclusions retracted; fixed with a self-test (`t3_improve_loop` report, section 9) |
| the pre-trained FPN of the BEV trunk was loaded with its inputs in the wrong order | the DualBEV port produced no signal | a flag for the original order exists (default off); effect on our chain: a tie (0.367 against 0.375 / 0.359) |
| training GT built from live code, so runs of different days used different targets | the lane chat noticed the cache signatures | same-code controls (`t8_ctl`), a decision for one GT version (D29, open) |
| velocity branch dead in 28 of 30 batch-3 runs; a rule-first fix revived it (skill 0 -> 0.566, one seed) | system-review lens, from checkpoints; the card of 23:45 | section 2.2 |
| three review passes in one evening: pass 1 listed eight high-severity findings (a TEST guard that accepted a substring, a null baseline that inherited the first frame's tags, a GT cache written non-atomically, ...), pass 3 nine more bugs | fresh reviewers reading code | six of the nine repaired in the pass with a regression test (the pass states that the tests fail on the old code; not re-run here), two by other chats |
| process lapses: E257's rule relaxed after the sweep, E259.b0's prediction written late | the loop-21 reviewer | recorded as caveats in the summaries |
| a full C: disk crashed a run at a checkpoint write on 30 Oct (4.5 idle GPU hours); the dual seed twin was killed once by the out-of-memory killer and the control's second seed resumed after a CUDA crash | the run logs | disk cleaned, jobs resumed (on the night of 31 Oct C: was at 4.1 GB free, without a crash; 67 GB now) |
| my own predictions: DualBEV front-only mAP2 "0.40 - 0.55" (measured 0.250); zero-camera mAP3 "0.20 - 0.26" (measured 0.013) | the measurement | recorded as misses |

**Outside context.** A survey of AI-scientist systems (arXiv 2608.05179, 29 June 2026) names the bottleneck: not whether agents can run research, but whether their claims can be verified; only
38 % of the runnable systems it coded release seeds or execution traces, and among nine closed-loop systems none of the LLM-era ones shows an externally validated oracle inside the loop.
Autoresearch puts exactly that into the loop: two published systems (DualBEV, GaussianCaR) run on our harness as external references, and its own lapses are in the record.
Other loops report results too (primary pages read on 2026-11-01): Karpathy's `autoresearch` (one GPU, a fixed five-minute run per experiment, about 12 experiments per hour or 100 per night by
its README), AlphaEvolve (a scheduling heuristic that recovers 0.7 % of Google's worldwide compute, a 48-multiplication 4 x 4 complex matrix product), AIDE with o1-preview on MLE-bench (bronze
medal or better in 16.9 % of 75 Kaggle competitions). None of these is a perception problem; they show that loops with a fixed judge find real improvements, and that the judge is where they differ.

## 4 Where it stands against the state of the art (like for like, with its limits)

| area | SOA reference | SOA | ours | reading |
|---|---|---|---|---|
| detection, front sector 0 - 50 m, mAP2 (vehicle + pedestrian) | DualBEV, a port written by us from its checkpoint, front camera, single frame; NOT verified against its published 36.8 mAP (six cameras, 360 degrees, ten classes) | 0.250 | camera only 0.242 (-0.009, interval [-0.021, -0.002]: slightly below, inside the band); camera + radar 0.359 / 0.375 (3000 steps), 0.400 (7000 x 2); dual chain 0.361 / 0.364 | camera path about equal; ours with radar is +0.11 - 0.15 above it (intervals [+0.093, +0.169]); radar's own gain at equal steps is +0.117 / +0.133. Our system also uses three frames and heads tuned on this metric. The DualBEV and older-chain numbers are scored on the earlier GT cache (3061 vehicle / 1435 pedestrian boxes in the area), the dual chain on GT 15efbde2d1 (3060 / 1436; decision D29). No SOA camera + radar detector was run on our frames (CRN is on disk) |
| maps, IoU per layer (drivable / walkway / carpark / crossing / lane divider / road divider) | GaussianCaR scored by our harness on 400 frames (43 val scenes) | 0.845 / 0.594 / 0.659 / 0.536 / 0.691 / 0.634 | dual chain on the SAME 400 frames, two seeds: 0.790 / 0.337 / 0.222 / 0.140 / 0.347 / 0.455 and 0.796 / 0.366 / 0.207 / 0.164 / 0.362 / 0.479 = 94 / 57 - 62 / 31 - 34 / 26 - 31 / 50 - 52 / 72 - 76 % of the SOA | MEASURED like for like (run `pitch.maps_lfl`, 2026-11-01, both systems at the 0.5 threshold, same frames, mask and tolerances): four of six layers at or above 50 % (drivable 94 %), carpark and crossing at about 30 %. With each system at its best of three thresholds the lane divider falls to 46 / 48 % and the road divider to 66 / 70 % (GaussianCaR gains more from its best threshold). BEVFusion features alone are level with the dual chain on five layers and ahead on the carpark (46 %; on 1000 frames 0.17 against 0.14); the image-only control is at 0 - 82 %. The estimate of version 3 for identical frames was within 0.06 on every layer. One run per model except the dual chain (two). Prediction written before the run: seed 1 inside all six ranges, seed 0 inside four (walkway 0.567 against 0.57, crossing 0.26 against 0.28). Limit of the harness: GaussianCaR's 200 x 200 grid was aligned to ours by an orientation search on 40 frames, and a forward offset of +-0.5 m is not resolved (drivable 0.840 at 0 and 0.842 at +0.94; +0.5 m was used): the thin layers at 0.5 m tolerance could move by about one cell, every share carries that uncertainty, and it was not re-scored. A GPU re-score of the stored CPU run reproduces its numbers (check `pitch.maps_lfl_check`, result in the ledger note) |
| lane-centre graph | none on our metric. Closest published: TopoBDA camera + radar on OpenLane-V2 subset B (nuScenes), R50 | DET_l 49.4, TOP_ll 38.6 | Q 0.735, topo F1 0.741 | NOT comparable: other annotation, other metric, Hungarian matching. A bridge is a plan (section 5), not a result |
| speed | GaussianCaR: 81.1 ms (maps, RTX 4090, 6 cameras, 448 x 800, its paper) | 81.1 ms | BEV chain on an RTX 3060, batch 1, one front camera, EXCLUDING the image encoders (their features are cached), one target time: dual features (`t10_dual_s0`): BEV network 71.7 ms + image heads 18.0 ms (that pass was flagged not idle, GPU 18 %); arm B2, idle GPU: 72.2 + 17.9 ms; the 7000 x 2 start recipe: 50.7 + 17.2 ms; lane graph 24 ms with cached foundation features | about 90 ms for the dual chain against GaussianCaR's 81.1 ms (which includes its backbone, six cameras, a 4090): same order of magnitude, not like for like, and the dual chain's BEV network is about 21 ms slower than the start recipe's |
| box velocity | published mAVE (all classes), e.g. CRN | 0.344 | vehicles on matched detections, cold VAL 1000: the zero predictor in the 3000-step dual chain (AVE 2.39 m/s, skill 0); with the moving / static gate (arm B2, warm start, 1000 steps, accumulation 2) skill 0.566 and AVE 1.19 m/s, one seed; skill 0.25 at 7000 x 2 (`t1_r4_soa`) | revived by the fix of section 2.2 (one seed, 1000 steps, front mAP2 -0.014); NOT comparable with mAVE (all classes, another matching) |

**Not claimed**: any number on a public leaderboard; six cameras; ten classes; NDS; 20 epochs of training; a generalisation beyond nuScenes (non-commercial licence: research use only; the
GaussianCaR weights and rasteriser are non-commercial too).

## 5 The path to the state of the art, ranked by information per cost

Every lever below has a measured effect size from this repository; the sums are hypotheses to test, not forecasts.

| # | lever | measured so far | cost | what it would show |
|---|---|---|---|---|
| 1 | dual features, longer: 6000 optimiser steps at accumulation 2 (4x the samples) with the velocity fix from scratch (card 3b of t10), then all 700 scenes | the old features gained +0.053 on the lane divider from 3000 to 6000 steps and +0.09 to 7000 x 2 | about 5.5 - 8.4 GPU hours (3.4 s per step measured at the start; 5.06 s per step by arm B2's ledger row) | the slope of the NEW features and whether the fix also holds from scratch (three things change at once; the card names what it can and cannot show) |
| 2 | stride-4 detail branch on top | on the old features: lane divider +0.065 / +0.042, road divider IoU@1 +0.119 / +0.096 (two seeds) | 2 GPU hours per seed | whether the thin classes add up |
| 3 | velocity: first step DONE (moving / static gate: skill 0.470 at batch 3, 0.566 with accumulation 2); next the seed twin of B2, the from-scratch run (lever 1), then a Doppler read-out for the moving error | closed-form bound 0.79 against 0.566 now | 1 - 3 GPU hours | whether the fix is robust, and where the rest of the moving error sits |
| 4 | a SOA camera + radar detector (CRN) on our frames | the checkpoint is on disk | half a day of porting | turns "no number" into a number |
| 5 | bridge the lane graph to OpenLane-V2 subset B (DET_l, TOP_ll) | nothing yet; needs the annotation files and the official evaluator (a download: needs Mirko's OK) | 1 day for a feasibility check | the first number a stranger can compare |
| 6 | a confirm evaluation on the 68 official val scenes never used for a decision (if their caches exist) | - | 1 GPU hour | removes the selection effect of reusing the same 82 scenes |

**Compute reality.** GaussianCaR was trained for 40 epochs with an effective batch of 16 on 4 x A100 80 GB. For the old features the remaining gap needed 4 - 10 more doublings of the steps
(16 - 1000 x the compute; `t6_soa_gap` section 5). With the new features the slope is unknown; lever 1 measures it: the dual chain at 6000 optimiser steps with accumulation 2 (4x the samples), started at 09:01 on 1 November and paused at step 5030 (84 %) at 14:38, to be resumed (snapshot d14597e2e8, about 5.5 h on the whole GPU at the 3.4 s per step measured in the first steps, 8.4 h at arm B2's ledger rate;
card 3b / 3c of t10, prediction and rule written before the start: lane divider 0.39 with the range 0.355 - 0.44, a gain of +0.05) against its own 3000-step run (0.335, 0.350); the nearest old-feature
reference is the 7000 x 2 start recipe (0.273 on the same 1000 frames, 4.7x the samples). **Rule, written in the card before the run (revision 2, after two critic passes):** a gain of at least +0.04 on the lane
divider over the 3000-step dual chain = continue with levers 1 and 2, provisionally (one seed: the tool never says KEEP on one seed); a gain inside the noise band of 0.029 drops nothing on one run, a second long
run decides: two seeds of this bundle below +0.02 drop only "this bundle is the lever"; the claim that the SOA is reachable on this hardware ("SOA-level on camera detection, partial on maps" stays) is dropped only
if the equal-budget arm without the velocity hurdle also gains less than +0.02.

## 6 Making the hobby meaningful: three outputs that outlive it

1. **A method package others can run**: the loop, the judge, the tools and the rules (`autoresearch/`: launcher that tags the code of a run in git (all 12 BEV runs and 59 of the 100 lane runs carry a tag; the earliest launcher runs predate it), ledger, decision tool, evalsuite, trainsuite, four agent
   roles), with a short write-up "a verification-first research loop for perception" (not yet written). It is problem-independent: the second problem (BEV fusion) was plugged in on 31 October in one day.
2. **One public number per problem** (levers 4 and 5): a result a stranger can check, with the harness as the artefact.
3. **A lessons-learned list that saves a team months**: swapping the source of the frozen image features (a bundle of backbone, stride, training objective and depth head) raised three thin layers (lane divider, walkway, road divider) by x1.8 - 2.0 here, and which
   part of the bundle matters is untested; a velocity branch
   can be dead from the first steps; the cache that builds your GT can silently change between runs; a seed twin is worth a ledger row. Each lesson has its number and its file.

**Boundaries, stated up front.** The data (nuScenes) and the weights used (GaussianCaR, DualBEV, BEVFusion) are non-commercial; the lane project uses a simulated SD map and exact poses; the
BEV system sees one front camera; the BEV claims have two seeds (control and dual chain) while the like-for-like maps have two seeds for the dual chain only (the other chains one run), and the lane results and the 7000 x 2 reference rest on one run; everything ran
on one machine under one lead chat. It is a private research project on public data. A company pilot needs its own clearance of IP, company time, data governance and the external agent tool (an
LLM service reads the code), before any of this is used there.

## 7 The hardest questions

1. **"Is this an AI inventing numbers?"** Tables come from `metrics.json` through scripts; a decimal that exists in no source file is listed by `check_numbers.py`; a fresh reviewer re-reads
   primary files. The record still shows errors: 4 wrong numbers, 8 mixed-split cases and 16 overclaims in about 420 claims of the week-2 report, found by review; twelve overclaims in the first
   draft of this file, found by a sceptical reader; eight wrong or stale numbers in the second, found by a numbers review. That is the argument for the review, not against it.
2. **"Aren't the wins just other people's models?"** The better parts are borrowed (GaussianCaR, BEVFusion, DualBEV, CenterPoint). The contribution is finding which piece limits, testing it with
   a control, merging two sets, the harness, the lane-graph builder, and the diagnosis of the dead layer (box in 2.1).
3. **"What does the person do?"** Sets the question, the rules and the stop time, answers decision entries (29 of 34 so far: champion promotion, TEST use, protected files, deletions, downloads), reads the
   ledger. The agent does the coding, the running and the first reading.
4. **"Why not an existing tool (MLflow, W&B, Optuna)?"** Those track runs and tune knobs. This also writes the hypothesis and the decision rule before the run, runs the controls, and diagnoses
   causes. It could sit on top of them; it was not benchmarked against them.
5. **"Could this be luck?"** The t8 and t10 claims have a same-code control with two seeds and the dual chain has two seeds; the maps agree within 0.02 across the seeds. Both second seeds were
   resumed after a crash on a later code snapshot (not shown neutral). Every BEV selection used the same 82 val scenes (a confirm run on the other 68 is lever 6); the lane TEST has 21 scenes and
   no intervals; the like-for-like maps have two seeds for the dual chain only, and the 7000 x 2 reference is a single run.
6. **"Is it state of the art?"** Not on a public benchmark. Like for like: camera detection about equal to the SOA port (a port we wrote), +0.11 - 0.15 with radar; maps at 26 - 94 % of
   GaussianCaR by layer on identical frames (four of six layers at or above 50 %, carpark and crossing at about 30 %); lane graph: no comparable number yet.
7. **"What does it cost?"** Known: one RTX 3060, 40.0 job-hours under the launcher since loop 18 (lane 32.2 + BEV 7.8, as of 09:00; BEV trainings before D23 unmetered), 34 decision entries, 1 to 6.5
   hours from card to verdict for the three latest cards (1 h 07, 3 h 54, 6 h 25). Unknown: agent tokens or money, human hours, and what a team does today for the same questions. The pilot is designed to measure the last one.
8. **"Does it transfer, and is it product-relevant?"** The tools are problem-independent (a metric-card writer and a launch command are the plug). Not tested on another organisation's data; the
   data and weights are non-commercial; the lane project is partly simulated; see the boundaries in section 6.
9. **"Where did it fail?"** Five closed questions; 10 CUT and 5 TIE of 23 verdict events (six from one family); a velocity head dead in 28 of 30 batch-3 runs (fixed overnight, one seed); a feature swap that halved the objects;
   predictions missed by factors of 1.6 to 15; rule and prediction lapses found by review.
10. **"What can go wrong?"** A full C: disk crashed a run once; the dual seed twin was OOM-killed tonight; code changed under running jobs; one lead chat and five open decisions; the loop wrote and
    found its own bugs. Mitigations: GPU gate and slots, one lead per loop, noise bands, a clean-up rule that moves and never deletes, reviews.
11. **"How will we know it worked?"** Criteria and a stop rule written now (section 8 and the rule in section 5).

## 8 What to ask for

1. Time on the GPU at night and at weekends inside the written rules (stop time, GPU limit, no TEST without OK).
2. A named person who signs off every claim that leaves the team and reads the ledger once a week.
3. A second reviewer for headline numbers; budget for the agent tool.
4. A six-week pilot with a second team, designed so that it can fail: two weeks measuring that team's current process first (experiments per week, wrong-number findings per 100 claims), then four
   weeks with the loop. Stop rule: no gain in experiments per week, or more wrong numbers than the baseline = stop and report. Budget: one GPU, one reviewer day a week.
5. Before any company use: clearance of IP, company time, data governance and the external agent tool.
6. (Mirko's own decision) permission for two downloads (OpenLane-V2 annotations and its evaluator) to build the first public lane number.

**Success criteria, written now, checked at the next review**: (a) every map layer at 50 % or more of the SOA like for like and the drivable area at 90 % (today 26 - 94 % on identical frames, both systems at
the 0.5 threshold: drivable 94, road divider 72 - 76, walkway 57 - 62, lane divider 50 - 52, carpark 31 - 34, crossing 26 - 31); (b) box velocity with skill 0.5 or more on moving vehicles (today 0.566
in one arm, B2: one seed, 1000 warm-start steps; 0 in the 3000-step chain without the fix, 0.25 at 7000 x 2; the seed twin and the from-scratch run are the next checks); (c) one public-benchmark number per
problem; plus the rule of section 5. Missing them is a result too.

## 9 Ten minutes, in order

1. The claim and its limit (section 0). 2. The loop in one picture. 3. Maps: question, eliminated causes, the test, the result (2.1), what is borrowed. 4. Velocity: the dead layer and the fix (2.2). 5. Lane
graph: equal Q at 26 x (2.3), with its gaps. 6. What the checks caught, and where they were bypassed (3). 7. Like-for-like SOA table (4). 8. Cost, boundaries and the path with its stop rule (5, 6).
9. The ask (8).

## 10 Open points before the meeting

- DONE since version 3: the like-for-like maps (section 4; GPU scoring, a few minutes per model, checked against the stored CPU run), the velocity arms (section 2.2), the lane prototype with its one
  TEST use (section 2.3). PENDING: the 6000-step run (below).
- Lever 1 (the 6000-step dual run in the lead's version, with the velocity fix and accumulation 2): STARTED at 09:01 on 1 November, after the restructured autoresearch was merged (08:19) and its smoke
  test passed, by Mirko's decision; **PAUSED at 14:38 at step 5030 of 6000 (84 %) on Mirko's request** (trainsuite stop: a full checkpoint was saved at the next step, exit 75; no steps lost; it had been stopped once before at step 4773 and was restarted by another session at 14:20, which Mirko did not want). It resumes from that checkpoint on Mirko's word;
  its cold VAL 1000 evaluation (chained, about 15 minutes) and the like-for-like map scoring follow the resumed run, about 1 h of training is left. A CUDA error at step 3200 (12:09) had already been resumed once by the launcher
  (same snapshot), so the finished run will be a run resumed three times (fresh data order each time), named so in every table. Until it finishes the deck says "paused at 84 %, result pending"; intermediate checkpoints are not scored
  (the learning rate was 2.2e-05, about 7 % of the peak, at the stop); card 3b / 3c revision 2 and the log 3d of `t10_dual_features`.
- Decisions that wait for Mirko: D4, D17, D29, D30, D31 (D18, D19, D25 and D33 were answered this morning); permission for the two downloads (section 8, item 6).

## 11 Sources

`bev_fusion_lane_center/PROBLEM.md`, `experiment_execution/{LEDGER.csv, DECISIONS.md, closed_questions.md}`, `evaluation/TEST_USAGE_LOG.md`, `experiments/INDEX.csv`,
`reports/source/weekly/{review_week2.md, s21_review.json}`, `bev_fusion/records/LEDGER.csv`, `bev_fusion/plan.md` (2.8),
`bev_fusion/experiments/{t3_improve_loop, t4_soa_front, t6_soa_gap, t7_lane_branch, t8_bevfeat, t10_dual_features, t11_vel_cause, t11_vel_hurdle}/`, `bev_fusion/runs/*/eval/metrics.json`,
`bev_fusion/temp/box_vel/out/alive_table.txt`, `bev_fusion/records/morning_2026-11-01.md`, `bev_fusion/temp/soa_gap/map_ref_<run>_400.json` (runs `pitch.maps_lfl`, `pitch.maps_lfl_check`),
`bev_fusion/runs/t1_r4_soa/eval/latency.json`, `autoresearch/history/repo_review_2026-10-31*.md`, `bev_fusion/temp/pitch/{facts.json, ladder.md, indicative.txt, subset_effect.py, maps_shares.py, maps_lfl.md}` (counts and
ladders made by scripts), `bev_fusion_lane_center/literature/papers/{SN01_gaussiancar.md, RB14_gaussiancar-no-failure-evaluation-exists.md, SN09_topobda-the-only-camera-radar.md}`.
External, primary pages read on 2026-11-01: arXiv 2608.05179 (the verification-gap survey); arXiv 2410.07095 (MLE-bench); the AlphaEvolve post of Google DeepMind; the README of
github.com/karpathy/autoresearch.

## 12 What the two reviews changed

**Version 1, sceptical review (twelve numbered overclaims; fourteen corrections):** the definition of VAL / TEST (BEV has no TEST; lane TEST used in loops 16 and 18); "every result with a rule
before the run" (now a count with named exceptions); "13 of 18 by rule"; "one cause at a time" (one block at a time; the t8 swap is a bundle); "a judge the loop cannot change" (a reviewed judge; the
hook protects truth files, not the evaluator); "not the BEV network" and "not over-fitting" (no lever found within the band; partly for thin classes); "objects kept" (vehicle + pedestrian);
"same chain trained 4.7 x longer" (a different config and frame set; borrowed features); "never learned / dead for three weeks"; "same quality / 24 ms" (equal Q, lanes found lower; cached
foundation features); "TEST once"; "no number typed"; "4.1 GB free"; "six-page write-up" (not written). Added: what is borrowed, cost known and unknown, boundaries, the end-to-end gaps of the lane
graph, speed, the compute reality with a stop rule, a confirm run, the pilot design.

**Version 2, numbers review against the primary files:** wrong or stale: the frame-set effect of the GaussianCaR comparison (not about 0.03: x1.03 - 1.10 for five layers, x1.70 for carpark; the
shares are conservative), the "29 experiments, 18 by rule" counts (18 verdict events on 12 experiments, nine from re-encoded or regex-read text) and the hit rate, "38 deep extractions" (36),
"4.7 x the optimiser steps" (2.3 x the steps, 4.7 x the samples), "eleven block exchanges tied" (7 tied, 4 worse; eight block swaps for the BEV network), "velocity today 0" (0.25 at 7000 x 2),
"queued, not run" (B' trained, B2 running), stale counts (now dated). Mixed sets: "17 - 55 %" is the 7000 x 2-step chain on 400 frames; "inside the 0.015 seed spread" (that is the older t2 pair; the
dual seeds are 0.003 - 0.007 below the controls); the control's second seed was also resumed after a crash; dual seed 1 added; drivable at 0 m against 0.5 m labelled; the Doppler bound's frame set;
the poster shows the `recall` variant; latency is the 7000 x 2 start recipe; "wins on S" (S rewards sparsity). Stronger than the source: "maps x1.8 - 1.9" (lane divider, walkway, road divider;
drivable x1.1, carpark x1.5), "camera detection equals the SOA port" (-0.009, interval excludes 0), the prediction hit count (four of five), "needs Mirko's go" (decided), "fixed judge" (reviewed),
"two-wheelers lost" (noise), "single seeds" (two for BEV), "none of nine" (none of the LLM-era ones), TEST numbers of loops 1 - 15, the t11 timing (card header, one commit), "36 GB free" (67 GB now),
"twelve" corrections (fourteen), dangling references.

**Version 4 (08:30), new evidence, not a review:** the like-for-like re-score on identical frames replaced the indicative shares (carpark 31 - 34 % instead of 21 - 22 %; the other layers within 0.06 of
the estimate of version 3, so the estimate was sound); the velocity status changed from "a diagnosis, not yet a fix" to a fix with a verdict by its own rule (B2 skill 0.566, one seed, a prediction miss
on the good side, front mAP2 -0.014 as the price); the count of dead velocity runs is now 28 of 30 batch-3 runs (the larger table of the velocity session) instead of 24 of 29; the status of the
success criteria (a: four of six layers at or above 50 %; b: met once by one arm) and lever 1 (the lead's version with the fix and accumulation 2, card 3b, rule corrected by the critic in 3c) were brought up to date; the lane prototype of the overnight round and its one TEST use (loop 22, logged) were added to section 2.3 and to the TEST statements. A fresh numbers reviewer then read version 4 against the primary files (09:35): 8 wrong, 5 mixed and 8 overclaiming statements; each was checked against its file and
corrected in this file, in the deck and in the report: the decision counts (34 asked, 29 answered, 5 open: D18, D19, D25 and D33 were answered this morning, D34 and D35 were added); "1 - 4 h from card to verdict" (the
hurdle card took 6 h 25 min: three cards, 1 to 6.5 h); "the dual chain was not timed" (it was: BEV network 72 ms + heads 18 ms, about 90 ms against GaussianCaR's 81 ms, encoders cached); "dead in 28 of 30 runs" without
"batch-3"; "maps within 0.01" for the velocity fix (three layers; carpark +0.03); "a git tag per experiment" (59 of 100 lane runs, all BEV runs); "skill 0.20 - 0.25 in the 7000 x 2 runs" (0.25; 0.203 is a 5000-step run);
GaussianCaR drivable 0.8445 written 0.85 (0.84); the older t2 pair 0.016, not 0.015; one table row that merged two disk incidents; "one-month pilot" (six weeks); 0.335 (1000 frames) set against 0.691 (400 frames) in
the report (now 0.347 on the same frames); VAL and TEST in one column of the lane slide; "like-for-like claims rest on one run" (two seeds for the dual chain); "gap 1 is mostly lost in the chain" (the recall weight closed about
two thirds of the VAL gap, the night's chain changes about a tenth); the frozen-features lesson (a bundle, three thin layers); "beat 4.7x on all six layers" (five, the carpark is level); "keep the maps and detection" (nearly;
carpark -0.029 against the map-trained features alone); the prediction counts (maps re-score: seed 1 six of six, seed 0 four; t10: four of six); the missing resumed seeds of both BEVFusion-only runs; the unresolved +-0.5 m
alignment of GaussianCaR's grid; the GT version of the detection comparison (3061 / 1435 against 3060 / 1436 boxes). The reviewer could not check external pages; the arXiv page of the survey shows "Submitted on 29 Jun 2026".
After the two-seed evaluation of 03:15 two of this file's own statements were corrected in the other direction: a gain of "pedestrian segmentation 0.000 -> 0.091" was seed noise (the control's
second seed scores 0.090 too), and the "two-wheeler / 3-class mAP loss" of the dual chain is of the order of the control's own seed spread (version 4 corrects the earlier wording "inside its range": the dual values lie below the control's range except one).
