# t11 vel cause - why the object head's box velocity is 0 m/s in every main-chain run: a dead hidden layer in the `vel` branch (card written 2026-11-01 00:10, after the stored-file checks below, before the probe and any run)

Part of the t11 velocity group: `../t11_vel_hurdle/experiment.md` (fix: moving / static logit + velocity on moving centres only) and `../t11_motion_field/experiment.md`
(the time mover's motion head). This card is the cause; it changes how those two must start (section 1, decision rule C1).
Scratch code: `../../temp/box_vel/` (`ckpt_weights.py`, `stored_dets.py`, `stored_ave.py`, `probe_features.py`); outputs in `../../temp/box_vel/out/`.

## 0. What the tools run
| field | value |
|---|---|
| id / title | t11_vel_cause / why the box velocity stays at 0 |
| kind | diagnosis (no training arm of its own; one short instrumented run proposed, section 1) |
| metric, direction, band | share of the `vel` branch's hidden units active (pre-activation > 0) at the supervised box centres; vehicle `ave_skill_moving` (1 - AVE / ave0 on moving TPs), higher, band 0.05 (as t11_vel_hurdle) |
| prediction | active share < 1 % at the centres for t8_ctl_s0 (other branches 10 - 50 %); ridge read-out of the trunk features: vehicle skill on moving centres >= 0.15 |
| control | the other branches of the same head (off, size, rot, hm), the zero predictor (ave0), the untrained head of `t6_maps_only` |
| budget_gpu_h, kill | 0 for the checks and the probe (CPU, 2 threads, light devkit ~2.4 GB); 0.4 for the instrumented run (proposal) |
| evaluate | VAL (cold), the stored `scene_records*.pkl` and checkpoints; the probe: 30 val scenes (test) / 40 train scenes (fit) |

## 1. Card
| | |
|---|---|
| lens | system review / training and losses |
| group | objects + motion (velocity of moving objects) |
| sources | review pass 2 section 2 (`../../../autoresearch/history/repo_review_2026-10-31_pass2.md`), the stored detections and checkpoints of t1 - t10, `../t11_vel_hurdle/`, `../t11_motion_field/`, CenterPoint (velocity L1, weight 0.2; its separate head has BatchNorm before the ReLU, our `_branch` has none) |
| status | proposed (layer-0 / 1 evidence in, probe pending) |

**What the stored files already show (layer 0 - 1, nothing trained).**

1. *The branch outputs its bias.* `runs/t8_ctl_s0/ckpt_00003000.pt` (EMA): `obj_head.head.vel.2.bias` max 1.19e-7, and the stored detection velocity is that number
   (`scene_records_n1000.pkl`, scene-0003: vx = 1.193e-7). The last-layer weights still have the size of their initialisation (|W| 0.786 vs 0.816 expected for PyTorch's
   default init of a 64 -> 2 1 x 1 conv; off / size / rot: 0.74 - 0.91, also near init but their units are alive). The same in `t10_dual_s0` (bias 2.5e-7, |W| 0.718),
   `t7_rgb_s0` (6.9e-8, 0.739), `t1_r4_soa` (2.8e-7, 0.816). `ckpt_weights.py`.
2. *The hidden layer is off at every supervised centre.* t8_ctl_s0 on 3 val frames (probe smoke run, `probe_features.py`): the `vel` branch's 64 hidden units are on at
   0.006 - 0.027 % of all cells and at 0 % of the 24 supervised centres (largest pre-activation -0.03, mean -0.67); the other branches of the same head: hm 44 %, rot
   33 %, size 34 %, off 8 % of all cells, 12 - 53 % of the centres. The `vel` term equals the zero predictor's to 6 digits (1.145 / 1.145). With every unit off at the
   centres no gradient passes the ReLU: W0, b0, W2 of the branch, the shared features and the trunk get nothing from the velocity loss, only the bias b2 does. A larger
   weight cannot help (`t3_vel10`: x10, the same dead pattern; with Adam the scale of a branch's only loss barely changes its step).
3. *Every batch-3 main-chain run is the zero predictor; the live ones had larger batches.* `stored_dets.py` (vehicle detections, dt 0) and `stored_ave.py` (the
   evaluation's own `DetMeter.frame_velocity`, front field of view, TP at 2 m, moving = GT >= 0.5 m/s):

   | run | batch x accum, steps | max speed of the detections | distinct (vx, vy) / detections | AVE moving | ave0 moving | skill moving |
   |---|---|---|---|---|---|---|
   | t8_ctl_s0 (base) | 3 x 1, 3000 | 0.024 m/s | 711 / 56,027 | 6.648 | 6.648 | 0.000 |
   | t2_ref_s0 / t3_vel10 / t7_rgb_s0 / t10_dual_s0 | 3 x 1, 3000 | 0.040 / 0.031 / 0.012 / 0.030 | 775 / 55,788 ... 711 / 52,650 | - | - | (zero level) |
   | t3_ref_long | 3 x 1, **6000** | 0.041 | 74 / 56,331 | - | - | (zero level) |
   | t1_r0 | 4 x 4, 5000 | 14.0 | all distinct | 5.246 | 6.582 | 0.203 |
   | t1_r4_soa | 3 x **2**, 7000 | 16.5 | 9,608 / 160,838 | 4.896 | 6.559 | 0.254 |
   | t1_r3_combined (width 0.5, refine) | 2 x **3**, 6000 | 1.40 | 87,875 / 135,566 | 6.420 | 6.585 | 0.025 |
   | t6_maps_only (object head never trained) | - | 2.96 | all distinct | 7.066 | 6.870 | -0.028 |
   | t4_dualbev_front (DualBEV's own head) | - | 13.3 | 7,033 / 7,230 | 3.843 | 6.530 | 0.411 |

   The untrained head (t6) outputs O(1) m/s random values at every detection; training is what drives the branch to its bias. Length alone does not save it
   (t3_ref_long, 6000 steps at batch 3, is dead). The two runs that learned real velocity (skill 0.20 - 0.25) both had an effective batch of 6 - 16, but a larger
   batch is not sufficient: t1_r3_combined (2 x 3) stayed near the zero level (0.025). It looks like a race in the early steps that the small batch loses every time.
4. *The motion head never left its zero start* (also found by `../t11_motion_field/`): `mover.motion.f.net.2.weight` |W| 0.0019 (t8), 0.0017 (t10); v_hat <= 3e-6 m/s at the
   probe's centres, the motion term equal to the zero field's. Its gradient at the trunk feature (`log.jsonl` diag, every 500 steps) is 1e-6 - 1e-8 in every 3000-step run
   (objects 0.07 - 0.33); in t1_r4_soa it grows to 3e-3 - 8e-3 from step 4000 on (|W| 0.45 at step 7000). So the "box velocity from v_hat" test (suggested step 4) gives the
   zero predictor again for every batch-3 run; it is informative only for t1_r4_soa (probe below).

- **Statement (falsifiable).** The box velocity is 0 because the `vel` branch's hidden ReLU layer is dead at the supervised centres, not because the target or the
  evaluation is wrong and not because of its weight: L1 on all centres, whose targets are mostly 0 (64 % of the matched vehicles static, more for pedestrians), pushes every
  output toward 0 with a constant-size gradient; Adam turns that into a fast drift of the hidden pre-activations below 0, and once they are below 0 at the centres the
  ReLU passes no gradient, so the branch never recovers - even when other losses later make the trunk features motion-aware.
- **Mechanism, in more detail.** At the L1 optimum the output is the conditional median, 0 wherever "moving in this direction" is not more likely than not; the cheapest way
  for a ReLU branch to output exactly 0 at every centre is to switch its units off there (the hidden layer is non-negative, so cancellations do not hold across inputs).
  A gradient that fades near the target (L2, Huber) or a regression that does not see the static majority (the hurdle of t11_vel_hurdle) removes the constant push; a
  larger batch dilutes the batches in which no supervised box moves (then the L1 gradient pushes every unit the same way), which fits the live runs of row 3 but
  did not save t1_r3_combined.
- **Prediction.** (P1) t8_ctl_s0, 300 val frames over 30 scenes: share of (unit, centre) pairs active < 1 % for `vel`, 10 - 50 % for off / size / rot / hm. (P2) Ridge
  read-out of the trunk feature S (96 ch) at the vehicle centres, fit on 40 train scenes, tested on the 30 val scenes: vehicle skill on moving centres >= 0.15 (the
  information is in the features, the branch cannot read it); logistic moving / static AUC >= 0.8. (P3) t1_r4_soa: the `vel` units are partly alive at the centres
  (5 - 50 %) and v_hat at the centres has skill > 0.
- **Decision rule.**
  - C1 (P1 holds): the cause is the dead branch. Any run that warm-starts from a batch-3 checkpoint (t11_vel_hurdle arm B as written: `init_from` t10_dual_s0) must
    re-initialise `obj_head.head.vel.*` (and the motion head's last layer is at zero anyway); without that, arm B can only move vel's bias and a "no gain" would be misread
    as "no motion information" (its rule V3). Suggested pair for that card: B' = warm start + fresh vel + hurdle, A' = warm start + fresh vel + today's L1 on all centres
    (P4 below: A' dies again, B' does not).
  - C2 (P2 holds, skill >= 0.15): the trunk features already carry the velocity; a live branch with a non-collapsing loss should get it within ~1000 steps of a warm start.
    P2 fails (skill < 0.05) and the logistic AUC < 0.65: the features carry no linear motion information at the centres; then a warm start cannot show the fix and the test
    must train from scratch (3000 steps, seed pair) or add the input route (radar Doppler skip, t11_vel_hurdle V3).
  - C3 (P1 fails: units alive at the centres): the cause is not the dead layer; back to the median-only explanation of t11_vel_hurdle.
- **Test design.** (a) Done: checkpoints and stored detections of every run (rows 1 - 4). (b) Probe, CPU: `probe_features.py` on t8_ctl_s0 val (300 frames, 30 scenes) and
  train (300 frames, 40 scenes), t1_r4_soa val (200 frames) -> active shares, ridge / logistic / L1 read-outs (`probe_fit.py`, to write). (c) Proposed, GPU, 0.4 h: an
  instrumented run of the base config from scratch, 500 steps, logging per 20 steps the `objhead.vel@+0.0` term, its zero-predictor value and the active share of the `vel`
  units at the centres (P4: below 5 % before step 500 with L1 on all centres; above 30 % with the hurdle or Huber). It needs only a logging hook (a scratch driver around
  `train.train`), no change of `training/`.
- **Cost / kill.** CPU only for (a) - (b); the probe stops when the box has < 4 GB free (other chats' jobs come first). (c) waits for a free GPU slot and the owner of
  t11_vel_hurdle (two velocity arms must not compete for the one GPU).
- **Referee.** -

## 2. Results

**P1 holds** (t8_ctl_s0, step 3000, EMA weights, 300 cold VAL frames spread over 30 val scenes, 2,632 supervised centres; `probe_features.py`,
`p1_report.py` -> `../../temp/box_vel/out/p1_t8_ctl_s0_val.txt`):

| branch of the object head | hidden units active at the supervised centres | over all cells | units on at >= 1 cell of a frame (of 64) |
|---|---|---|---|
| hm | 49.8 % | 44.8 % | 58.8 |
| off | 13.3 % | 8.3 % | 41.1 |
| size | 34.0 % | 34.0 % | 38.4 |
| rot | 31.4 % | 31.9 % | 59.0 |
| **vel** | **0.011 %** (19 of 168,448 unit-centre pairs, in 16 of 299 frames) | **0.014 %** | 16.9 |

- vel's pre-activation at the centres: max 0.09, mean -0.70. The output at the centres is at most 0.013 m/s.
- `objhead.vel@+0.0` is the zero predictor's term (mean 1.9510 vs 1.9510 over the frames, largest per-frame difference 1.7e-3).
- `motion@+0.0` is the zero field's (1.1983 vs 1.1983, 1.3e-6); v_hat at the centres is at most 2.8e-5 m/s.
- Targets at the supervised centres: vehicles 74.2 % static (< 0.5 m/s; 76.7 % below 1 m/s), two-wheelers 52.9 %, pedestrians 39.7 % (52.5 % below 1 m/s). The
  vel map is shared by all groups, so 62 % of all supervised centres are static; the moving vehicles average 6.41 m/s.

**Decision C1** (by the rule as written): the cause is the dead branch. Sent to the owner of t11_vel_hurdle (the overnight lead, 2026-11-01 00:2x), who adopted it:
both t11 arms now start with a fresh `vel` branch (option `init_reset ["obj_head.head.vel."]` with a test, built by that lead's coder), A' = fresh branch +
today's L1, B' = fresh branch + the hurdle; their runs log `objhead.vel@` / `objhead.mov@`. No GPU arm of this card was run (the GPU and the code belong to that
round; the instrumented-run driver `train_diag.py` stays available, smoke-tested on mini).

**P2 fails: the trunk features hold almost no linear velocity information** (t8_ctl_s0, fit on 1,823 train-split vehicle centres of 300 frames / 40 scenes, tested on 1,670 val
vehicle centres of 300 frames / 30 scenes; `probe_fit.py` -> `../../temp/box_vel/out/probe_fit_t8_ctl_s0.txt`; at the GT centres, so an optimistic localisation):

| read-out on the frozen feature | vehicle skill on moving centres (1 - AVE / ave0) | static error (m/s) |
|---|---|---|
| zero predictor = the stored vel output = the stored v_hat | 0.000 (AVE 6.41 m/s) | 0.026 |
| ridge on the trunk feature S (96 ch), lambda by 5-fold CV | 0.019 | 0.407 |
| ridge on S, gated by the logistic moving test | 0.008 | 0.078 |
| linear L1 on S | 0.000 | 0.059 |
| fresh 64-unit branch on the object head's shared feature x, L1 (today's loss), 1500 steps | 0.000 (50 % of its units active) | 0.035 |
| the same with Huber (1 m/s) / L2 / L1 on moving centres only | 0.003 / 0.010 / -0.011 | 0.137 / 0.406 / 1.065 |
| logistic moving / static on S | AUC 0.768; BCE 0.561 against the base-rate entropy 0.571 (2 % below) | |

Reading, with its limits: the trunk of this run never received a velocity gradient (the branch was dead, section 2 above), so nothing asked it to carry motion; the
read-out says what is there, not what the architecture could learn. A live branch on these frozen features learns nothing in 1500 steps whatever the loss, and
the L1 branch does NOT die on them (50 % of its units stay active, it parks its output at 0 through the small last layer), so the "L1 kills the units" form of
the mechanism is not reproduced by the frozen replica; the death happens in the joint training (early steps, trunk drift, batch noise), which the instrumented run
`train_diag.py` is built to show (when the active share falls, and with which loss). C2 by its rule: the features carry no linear motion information (skill 0.02 < 0.05);
the logistic AUC 0.77 is above the 0.65 bar but its BCE gain is 2 %, i.e. what it finds (cars on the road vs parked) is context, not motion.
Consequence for `../t11_vel_hurdle/` (told to its owner): a null result of arm B' after 1000 steps means "not learned in 1000 steps", not "the input lacks motion".

**Alive and dead velocity outputs across the 34 evaluated runs** (`alive_table.py` -> `../../temp/box_vel/out/alive_table.txt`; max predicted vehicle speed at dt 0):

| group | runs | effective batch (batch x accum, clips of 3 keyframes) | steps | max speed |
|---|---|---|---|---|
| dead | 24 (t2_*, t3_vel10, t3_ref_long, t3_img_cam/rad/ref, t3_radar_only, t7_*, t8_*, t10_dual_s0, ...) | **3 x 1** | 3000 (t3_ref_long 6000) | 0.004 - 0.041 m/s (t2_fusion_cmx 0.24) |
| alive, skilled | t1_r4_soa, t3_img_long | **3 x 2** | 7000 | 14.4 - 16.5 m/s (skill 0.25 on moving vehicles) |
| alive, skilled | t1_r0 (width 0.5, seq 1) | 4 x 4 | 5000 | 14.0 m/s (skill 0.20) |
| alive, weak | t1_r3_combined (width 0.5, refine) | 2 x 3 | 6000 | 1.4 m/s (skill 0.025) |
| (untrained head) | t6_maps_only | - | - | 3.0 m/s, skill -0.03 |

An effective batch of 6 or more separates alive from dead in all 29 runs; the steps do not (t3_ref_long: 6000 steps at batch 3, dead; t1_r3_combined: 6000 at batch 6,
alive; t1_r4_soa's config equals t8_ctl_s0's except steps and accum). It is an association over runs of one family, two of the alive ones also trained 7000 steps, and
no seed twin exists at either batch; it is a lead for the cause of the early death, not a result.

## 3. What working systems do (literature; the local notes in `../../literature/papers/`, the CenterPoint head checked against the mmdet3d source)

| system | how velocity is obtained from current + past | velocity result (nuScenes val) | what is portable |
|---|---|---|---|
| BEVDet4D (2203.17054) | the previous BEV feature ego-warped (rotation + translation), concatenated with the current one, two residual units, CenterPoint head; the velocity target is the OFFSET between the two aligned features; training gap sampled 3T - 27T | mAVE 0.909 (no history) -> 0.479 -> 0.376 (rotation) -> 0.328 (gap augmentation); unaligned concat 1.544 and translation-only 1.186 | alignment must be exact incl. rotation (ours is: chained steps, tested); random time gap in training; the pair enters one conv encoder side by side |
| VideoBEV (2303.05970) | one recurrent BEV memory `H = [warp(H_prev); B] * V` (a conv that down-weights older frames) + a time-gap embedding for the velocity head | history 0 / 1 / 4 / 16: mAVE 0.936 -> 0.331; the time embedding removes 25 % of the error rise at 50 % missed frames | an embedding of the frame ages in the velocity branch (ours has the age only as an averaged channel) |
| StreamPETR (2303.11926) | object queries as memory (4 x 256), motion-aware LayerNorm from [ego transform, velocity, dt] | memory 0 / 1 / 2 / 4 frames: mAVE 0.885 / 0.324 / 0.314 / 0.316: ONE past frame gives nearly all; explicit motion compensation alone did not help | object-centric, not a dense grid: limited portability; confirms that one past frame is enough |
| SOLOFusion (2210.02443) / RCBEVDet (2403.16440) | 16 aligned low-res BEV frames stacked; RCBEVDet adds radar with the Doppler speed only as a point feature | mAVE 0.252 / 0.220 (camera-only StreamPETR 0.295, CRN 0.344); "radar does not clearly beat history on mAVE" | heavy; the radar adds little on top of a history in this table |
| CenterFusion (2011.04841) / full-velocity radar returns (2108.10637) | radar returns associated with a detection and its radial Doppler turned into a velocity; the tangential part from optical flow | point-wise error 0.433 m/s (R2P) vs 1.599 radial-only | a Doppler read-out at the detected box (measured for us below) |
| MotionNet (2003.06754) | five ego-aligned sweeps, spatio-temporal pyramid with pseudo-1D temporal convs; per-cell class, a motion/static state and the motion, which is zeroed on background and static foreground | motion error static 0.026 m, slow 0.257 m, fast 1.074 m; ego sync matters (0.257 -> 0.425 without) | the static / moving gate = the hurdle of `../t11_vel_hurdle/` has this precedent |
| CRT-Fusion (2411.03013) | a supervised per-cell velocity head + a velocity-guided warp of the history | +1.7 NDS over CRN; "no isolated ablation of the shift" | this is OUR design (v_hat -> splat); its gain is unproven in the literature |
| CenterPoint head (2006.11275; mmdet3d `SeparateHead`, read 2026-11-01) | per branch conv3x3 -> **BatchNorm2d** -> ReLU -> conv1x1 (final conv with bias); the velocity is two more channels of the same L1 (loss weight 0.25, per-channel code weights from the config) | - | our `_branch` (`object_head.py:22-25`) has conv3x3 -> ReLU -> conv1x1 and NO BatchNorm: the one structural difference of the head |

What this says about our chain:
1. The input structure is not the odd one out: aligned history next to the current frame in a conv encoder before the head is what BEVDet4D, VideoBEV and RCBEVDet do, and our
   exact chained alignment is better than their warps. (Not measured here: that our averaged aggregate `agg` of the K past frames hides the frame difference more than a
   side-by-side concat; a test would be a variant that passes the K aligned frames as separate channels.)
2. Every working system trains the velocity as plain L1 through the detection head from step 1, with its trunk receiving that gradient. In ours the gradient is zero, so the
   trunk has no motion features and every box-velocity result so far is the zero predictor.
3. The dense motion head `v_hat` that moves the state is the unproven part (CRT-Fusion has the same design without an ablation). Its gradient at the trunk feature is 1e-6 - 1e-8,
   its weights have not left the zero start; no working system in the table relies on it for the box velocity. The box velocity should come from the object head's `vel` (as every
   system above), and boxes at dt != 0 can be moved with it (`object_head.propagate_boxes`, S5b), independent of the dense field.
4. Time baseline: ours spans ages 0 / 0.17 / 0.33 s (`cam_stride` 2 at 12 Hz). BEVDet4D trains with gaps of 0.25 - 2.25 s (local note: 3T - 27T, T = 0.083 s); StreamPETR and
   VideoBEV run their histories at keyframe spacing, 0.5 s per step (from memory, NOT in the local notes: to check before relying on it); a
   vehicle at 6.4 m/s moves 1.1 / 2.1 m in our two gaps (2 / 4 cells) against the camera BEV's depth noise. Whether a longer baseline helps is open; the radar Doppler does
   not depend on it.

Ranked candidates to port (each a small card, none run here):
| rank | change | evidence | cost |
|---|---|---|---|
| 1 | BatchNorm in the object-head branches (CenterPoint `SeparateHead`) and a fresh vel branch | every CenterPoint-type head has it; ours is the only one without; the dead branch is the measured failure | 1 line + a gate, then the A' / B' arms of `../t11_vel_hurdle/` plus one arm with BN |
| 2 | the moving / static gate and the velocity on moving centres only | MotionNet; the hurdle card | built by the owner of `../t11_vel_hurdle/` |
| 3 | effective batch 6 (accum 2) for the velocity arms | all 5 alive runs have it, all 24 dead runs lack it (association) | 2 x the step time |
| 4 | random time-gap augmentation and an age embedding in the velocity branch | BEVDet4D 0.376 -> 0.328; VideoBEV | dataset change (random `cam_stride`; the cache holds all 12 Hz frames) |
| 5 | a Doppler read-out at the box (closed form with the box heading) | measured below | a differentiable block of ~30 lines, uses the dataset's radar field |
| 6 | an explicit correlation (cost volume) between the aligned current and previous BEV features (FlowNet / PWC-Net / RAFT; MotionNet's temporal convs) | optical-flow practice; no BEV-detection ablation in the local library | block experiment, only if 1 - 5 fail |
Not worth porting: object-query memories (StreamPETR, Sparse4D: not a grid, would need matching) and 16-frame stacks (SOLOFusion: heavy, saturates).

## 4. Input side: the radar Doppler at the GT vehicle boxes, and the positive control of the probe

**P3 / positive control: t1_r4_soa** (its velocity branch learned; step 7000; 200 cold val frames over 30 scenes, 1,716 supervised centres; the read-out fitted on the first
half of the frames, tested on the second, 461 / 642 vehicle centres; `probe_features.py`, `p1_report.py`, `probe_fit.py` -> `../../temp/box_vel/out/p1_t1_r4_soa_val.txt`,
`probe_fit_t1_r4_soa.txt`):
- The alive branch is sparse too: 0.53 % of the (unit, centre) pairs are active (585 of 109,824; t8_ctl_s0: 0.011 %, 19 of 168,448), in 91 of 198 frames, but those units fire
  hard (pre-activation up to 36 against 0.09), 12 of 64 units are on somewhere in a frame (t8: 17, nearly all of them silent at the centres). `objhead.vel@+0.0` 1.709 against the
  zero predictor's 2.029 (-16 %), `motion@+0.0` 0.932 against 1.177 (-21 %). The prediction "5 - 50 % active" (P3) was wrong in size: a handful of strong units is enough to learn.
- Box velocity from the motion head (the suggested test 4, in the one run where `v_hat` is alive): skill 0.229 at the GT centres, the object head's `vel` 0.220: the same.
- The probe is valid: on these features ridge on the trunk feature S gives skill 0.319 (stored 0.220), the fresh branch 0.26 - 0.37, a moving test AUC 0.787; on t8_ctl_s0 the same
  read-outs give 0.019 and 0.768. The AUC is the same in both runs (context: parked against driving cars), the linear velocity read-out is what separates them. (The logistic BCE of
  this small fit set is above the base-rate entropy, overfitting, only its AUC is used.)
So the C2 reading stands: the t8_ctl_s0 trunk holds no linear motion information because nothing ever asked it to, not because the probe cannot see it.

**The radar input holds the vehicle velocity** (`doppler_at_boxes.py` -> `../../temp/box_vel/out/doppler_at_boxes_val.txt`, `.json`; t8_ctl_s0's dataset, 300 cold val frames over 30
scenes, 1,019 observed GT vehicles in the front sector, 329 of them moving; the radar field is exactly what the radar branch receives: ego-compensated radial velocity along the ray,
positions shifted along the ray to the camera time, returns of the last 0.25 s inside the GT footprint + 1 m). Association by the GT box and the GT heading, so every number is an
UPPER bound of a detector-side read-out. Estimate: `v = (v_r / (h . u)) h`, h = the box axis, median over the returns with |h . u| >= 0.3, 0 where none qualifies.

| reference velocity | radial corr of vr with v_ref . u (moving) | median radial error | moving vehicles with >= 1 return / with an estimate | AVE of the estimate (zero predictor) on all moving | skill | skill where an estimate exists |
|---|---|---|---|---|---|---|
| devkit (annotated positions only, no radar) | 0.985 | 0.18 m/s | 90 % / 83 % | 1.37 (6.42) m/s | **0.787** | 0.920 (n 272) |
| autogt (track velocity fused with the radar Doppler, partly circular) | 0.986 | 0.16 m/s | 90 % / 83 % | 1.33 (6.40) m/s | 0.792 | 0.925 |

- The two references differ by 0.144 m/s on the 329 moving vehicles (autogt vs devkit, skill 0.978): autogt's velocity is essentially the track velocity, and the independent
  devkit reference gives the same Doppler numbers, so the agreement is not an artefact of the GT using the radar.
- By range (devkit reference, skill on all moving): 0 - 20 m 0.89, 20 - 40 m 0.78, 40 - 50 m 0.78; where an estimate exists 0.94 / 0.90 / 0.95.
- Static vehicles: median |vr| 0.07 m/s (90 % quantile 0.22); the moving test (median |vr| > 0.5 m/s) finds 90 % of the moving vehicles at a false-positive rate of 1.4 %; the
  estimate on static vehicles has a mean speed of 0.26 m/s, >= 1 m/s in 1.4 %.
- What it leaves out: 10 % of the moving vehicles have no return in the box and a further 8 % only returns across the line of sight (they count as 0 in the skill above; a camera
  or history cue must cover them); vehicles faster than the sensor's unambiguous interval fold (the circular code of the radar branch exists for that, not used here); the heading
  comes from the GT (a yaw error e changes the estimate by about tan(angle) e); the autogt `track` subset (no radar in its GT, 49 moving vehicles) has a lower skill (0.37 / 0.72
  where an estimate exists) because few of its vehicles have returns (47 %).
- For scale: the learned velocities of the whole repository reach skill 0.22 - 0.25 (t1_r4_soa, t1_r0) and 0.41 (DualBEV's own head on its front view). The input already holds
  0.79 - 0.92 of the velocity with a closed form and the box heading.

**Consequence.** The radar branch delivers the information; what is missing is a velocity gradient that reaches it. Two ports follow from this and from section 3: (a) let the object head
read the Doppler at its centre explicitly (the local Doppler / count channels of the radar raster, or a differentiable closed form with the predicted heading as above) instead of
hoping a dead branch finds it in the trunk feature; (b) the hurdle's moving test has a strong, nearly free cue (median |vr| at the footprint: TPR 0.90 at FPR 0.014). Neither is run here.
