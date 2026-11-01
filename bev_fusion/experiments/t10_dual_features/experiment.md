# t10 dual features - can one trunk use the map-trained BEVFusion features AND the detection-trained DualBEV features? (card written 2026-10-31, before any code or run)

Context: t8 (`../t8_bevfeat/experiment.md`, same-code control, two seeds, cold VAL 1000): frozen BEVFusion camera features (map-trained, stride 8, weak depth)
instead of the frozen DualBEV cache (detection-trained, stride 16, sharp depth) nearly double the maps (lane divider@0.5 0.191 -> 0.351 / 0.345, drivable 0.716 -> 0.804 /
0.809, walkway 0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165) and halve the objects (front mAP2 0.367 -> 0.183 / 0.180, pedestrian AP@2 0.210 -> 0.001 / 0.002).
M6 (the image features limit the maps) is supported; the reverse holds for objects. Neither cache alone gives both. This card is the cheapest test of "both":
the two cached feature sets side by side, no new encoder, no new cache (C: has 27 GB free; both caches exist on D:).
It does not replace t9 (`../t9_encoder_2x2/experiment.md`, written before t8): t9's R arm is the trainable version of the same idea, and its premise
("no layer reaches 60 % of GaussianCaR in 3000 steps") is nearly overtaken by t8 on the maps, with a frozen encoder and no map-loss training (lane divider 0.35 =
51 % of GaussianCaR's 0.691, walkway 0.34 = 57 % of 0.594, t6 report table, same frames and tolerance), so the order of the lane should be re-read after t10.

## 1 Card

- **Question.** With the frozen DualBEV BEV and the frozen BEVFusion BEV both available to the trunk, does one chain keep the maps of t8_bevfeat AND the objects of t8_ctl?
- **Hypothesis.** The two camera BEVs carry complementary information (maps / ground semantics from BEVFusion's map-trained stride-8 features, object depth and
  object features from DualBEV's depth net); the merged BEV (same channel count into the same trunk) is used for both. Competing explanations the arm can tell apart:
  (a) the trunk uses both: maps AND objects recover; (b) the trunk takes the easier source and ignores the other (a shortcut): one family recovers only; (c) the
  loss of the objects with BEVFusion features is not only the weak depth but the missing object information in the map-trained context (then no mixing of depth helps;
  a hybrid of "BEVFusion context x DualBEV depth" would be the next test, t9's R arm is its trainable cousin).
- **Design.** One arm `t10_dual_s0` (seed 1 only if it passes a rule): everything identical to `t8_bevfeat_s0` / `t8_ctl_s0` (3 frames, window memory, radar, 3000 steps,
  Kendall, lr 3e-4, EMA, flip, GT signature 15efbde2d1, the code commit of t8 plus the default-off option below), cold VAL 1000 frames by `bev_suite eval --n 1000`.
  Controls are the existing same-code runs `t8_ctl_s0/s1` (DualBEV only) and `t8_bevfeat_s0/s1` (BEVFusion only): no new control run. Cost ~100 min per seed
  (two lifts, two reductions; the data loader reads both caches: 1.4 MB per frame instead of 0.3 / 1.1 MB).
- **The option (default off, bit-identical defaults).** `ModelConfig.camera_features = "dual"`:
  - dataset (`feature_cache.CachedFrontRadarDataset`): a second `FeatureStore` from `feature_cache2` (= `data/cache/cam_feats_bevfusion`; the first is
    `data/cache/cam_feats`); each frame gets `prob2` (118, 32, 88) / `ctx2` (80, 32, 88) next to `prob` / `ctx`; the item carries `ida2 = bevfusion_ida()`
    beside the DualBEV `ida`; `collate_cached` stacks `prob2`, `ctx2`, `ida2`.
  - model (`BEVFusionModel.__init__` and `CachedBEVFusionModel._Cached.camera_bevs`): branch 1 as today (`self.lift` with `depth_centres()` on the stride-16 cache);
    branch 2 = `self.lift2 = LiftPD(grid, depth_centres=bevfusion_camera.DEPTHS, frustum_cells=True, oversample=cfg.oversample)` with its own `self.cam_reduce2`
    (same type as `cam_reduce`); the two reduced BEVs (c_cam each) are concatenated and merged by `self.cam_merge = Conv2d(2 c_cam, c_cam, 1) + BN + ReLU`, so
    everything downstream (temporal window memory, aux head, fusion, trunk, heads) sees the same shapes as in the control. `camera_visibility` keeps using branch 1 (same
    camera geometry; the visibility is calibration only). `camera_off` still returns zeros. `lift == "s"` stays unsupported.
  - config `t10_dual_s0.json` = `t8_bevfeat_s0.json` + `"model": {"camera_features": "dual"}` + `"feature_cache2": "data/cache/cam_feats_bevfusion"` (and `feature_cache` back to
    `data/cache/cam_feats`).
- **Gate (layer 0, before the run; synthetic tensors, CPU, a few seconds; I write `temp/t10/gate.py` as soon as the option exists).**
  G1 shapes: the camera BEV is (B, F, c_cam, X, Y) as in the control; G2 branch select: with `cam_merge` set to [I, 0] the dual BEV equals the DualBEV-only camera BEV
  (same weights, same inputs), with [0, I] it equals the BEVFusion-only camera BEV (each branch uses its own depth centres, frustum convention and `ida`); G3 gradients: a loss on the
  merged BEV gives non-zero gradient to both reductions and the merge, and none to the cached inputs; G4 `camera_off` gives zeros without reading the features; G5 the
  default configs reproduce their own output bit for bit (existing selftests of model.py / feature_cache.py pass); G6 the first real step: peak GPU memory and seconds per step
  (the card is shared: peak below 7.5 GB, else micro-batch 1 with accum 3).
- **Prediction (before the run; cold VAL 1000, dt 0).**

  | layer | ctl (DualBEV) | bevfeat | predicted dual |
  |---|---|---|---|
  | lane divider@0.5 | 0.191 | 0.351 / 0.345 | 0.34 (0.30 - 0.37) |
  | drivable@0.5 | 0.716 | 0.804 / 0.809 | 0.80 |
  | walkway | 0.178 | 0.339 / 0.342 | 0.33 |
  | front mAP2 (0-50 m) | 0.367 | 0.183 / 0.180 | 0.33 (0.25 - 0.38) |
  | vehicle AP@2 | 0.670 | 0.455 / 0.447 | 0.62 |
  | pedestrian AP@2 | 0.210 | 0.001 / 0.002 | 0.15 |

  Reason for "most of both": the DualBEV BEV reaches the trunk unchanged, so the object path of the control stays available; the maps need the stride-8 BEV, and t8 shows the trunk can use it.
  The weak spot is the shortcut (b): the pedestrian AP is the sensitive number.
- **Decision rule (written now; bands from the t8 / t9 cards: maps max(0.02, 2 |s0 - s1|), walkway 0.054; front mAP2 the paired bootstrap of `temp/t4/compare_front.py`, 60 draws).**
  - E1 both: lane divider and drivable >= bevfeat - 0.02 and walkway >= bevfeat - 0.054, and front mAP2 >= ctl - 0.054 -> run the seed twin (s1); if it agrees, the chain with both feature
    sources is the new reference and the target for the single encoder (E2E, plan 2.8 step 1 / t9 U and UR, initialised from BEVFusion for the maps and distilled/trained
    against the DualBEV depth for objects); report ms per frame of both image sides (dual frozen is a diagnostic arm, not the product: Swin-T + FPN + ResNet-50 + DepthNet).
  - E2 maps kept, objects between (front mAP2 in [0.22, 0.31]) -> the shared trunk dilutes the object path: next arm = branch-specific heads (object head reads the trunk of branch 1 only, or
    a stop-gradient on branch 2 for the object loss), seed 1 first only if the gain over bevfeat on mAP2 exceeds the band.
  - E3 objects back (>= ctl - 0.054) but maps <= ctl + 0.05 -> shortcut: the trunk ignores branch 2. Check the gradient norm of `cam_reduce2` / `cam_merge`, add an auxiliary map head on branch 2's
    BEV, and rerun once; if it persists, the DualBEV BEV dominates by convergence speed, and the mixing has to be at the feature level (hybrid H or t9 R).
  - E4 both mid (maps and objects each lose more than the band against their best arm) -> capacity or optimisation, not information: longer run (6000 steps) before any conclusion.
  - INCONCLUSIVE (a metric inside its band): add s1, do not adopt.
- **Kill / validity criteria.** (1) the loss rule of t7 at step 1000 (group.lines mean over steps 900 - 1000 above 1.48 while the references are 1.22 - 1.25): stop and report; (2) a NaN or a
  CUDA unknown error: resume from the last checkpoint once (as for t8); (3) a missing frame in either cache (the token sets are identical by construction: BEVFusion cache = built from
  the tokens of `cam_feats`): the dataset raises, the run stops.
- **What it does not test.** Speed of a single product encoder, a better depth for the BEVFusion side, the hybrid (BEVFusion context x DualBEV depth on one grid), lens/roll behaviour, longer training.
- **Cost.** One seed ~100 min of lane time after the build (the lane is free except `t7_rgb_s1`); the build is about 60 lines in three files owned by the lane owner (`feature_cache.py`, `model.py`,
  the config) and the gate is mine.

## 2 Log

- 2026-10-31: card written after the t8 result; nothing built or queued. Needs the lane owner's go for the option and its place in the queue relative to t9 (see the first paragraph).
- 2026-10-31 (evening): the lane owner built the option (cc9fbde: `camera_features "dual"`, `feature_cache2`, `store2`, `prob2` / `ctx2` / `ida2`, `lift2`, `cam_reduce2`, `cam_merge`; 27.8 M parameters;
  start script 5bb32c6) and queued the run through `run.py` as `t10.dual_s0` (status waiting: starts when GPU memory in use falls below 3 GB; `t7_rgb_s1` is training). GT signature 15efbde2d1 as in
  t8_ctl / t8_bevfeat. Gate: G1 - G5 are covered by `tests/test_camera_features.py` (shapes, branch select [I,0] / [0,I], gradients to both reductions and the merge, refusals, defaults untouched; CPU, passes).
  G6 (peak memory, s/step) is read from the first real steps in `runs/t10_dual_s0/train.log`. Added by this chat, `tests/test_dual_camera_geometry.py` (first run as a scratch `temp/t10/gate.py`, moved into tests/ the same evening): the geometry of the pair with the real image transforms.
  Each branch gets a feature map storing the original-image pixel (x, y) of every cell (ramp; DualBEV cells at 16 k + 7.5 with `nuscenes_front_ida`, BEVFusion cells at j (W - 1) / (w - 1) with
  `bevfusion_ida`), the model's own lifts (`md.lift`, `md.lift2`, oversample 4) carry it into the BEV, and the result is compared with the analytic projection of every voxel: G7a DualBEV branch
  max error 0.0005 original pixel (55,379 voxels), G7b BEVFusion branch 0.0006 pixel (52,657 voxels), G7c the two branches agree to 0.0006 pixel on 52,649 common voxels, G8 the mix-up (branch 2 lifted
  with the DualBEV ida) gives 61 pixels, so the gate can see it: PASS. (The first version failed on the DualBEV branch by 5.9 pixels at an image row 0.5 pixel from the border: not a lift error, the
  sub-heights of a slice spread its row and one of them was clamped by the border padding; the gate now keeps all sub-heights inside, margin by fy dz / z_c.) What the gate does not cover: the content of the
  caches (that every BEVFusion shard was built with the crop of `bevfusion_ida`: the builder and `bevfusion_ida` take the same constants 0.48 / crop 32, 176, checked by hand, not by a test).
- 2026-10-31 (night): `t10.dual_s0` ran 3000 steps (run.py exit 0, 6900 s, GT signature 15efbde2d1) and was evaluated cold on VAL 1000 / 82 scenes (`evaluation/bev_suite.py runs/t10_dual_s0 eval --n 1000`; scripts `temp/t10/eval.sh`, `compare.sh`, `maps.py`).
  Result (maps = full area, boxes = front sector x < 50 m, `results_x50.md`; control = `t8_ctl_s0`, same code and GT signature, one seed; bevfeat = s0 / s1):

  | metric | t8_ctl_s0 | t10_dual_s0 | t8_bevfeat_s0 / s1 |
  |---|---|---|---|
  | lane divider IoU@0.5 | 0.191 | 0.335 | 0.351 / 0.345 |
  | drivable IoU@0.5 | 0.716 | 0.806 | 0.803 / 0.809 |
  | walkway IoU | 0.178 | 0.340 | 0.339 / 0.342 |
  | ped crossing IoU | 0.002 | 0.154 | 0.145 / 0.165 |
  | front mAP2 | 0.367 | 0.361 | 0.183 / 0.180 |
  | ped AP@2 (front) | 0.210 | 0.199 | 0.001 / 0.002 |
  | vehicle seg IoU | 0.415 | 0.413 | 0.309 / 0.305 |
  | pedestrian seg IoU (one seed; the control's second seed later scored 0.090: noise, see the 03:15 entry) | 0.000 | 0.091 | 0.000 / 0.000 |

  mAP2 difference to the control -0.006 [-0.016, -0.000] (paired scene bootstrap, 100 draws): inside the 0.054 band. Rule: **E1** (lane divider within 0.02 of bevfeat, drivable and walkway equal, mAP2 >= ctl - 0.054):
  the two frozen feature sets together keep the maps of BEVFusion AND the objects of DualBEV. Prediction hit on four of five numbers (front mAP2 0.33 (0.25-0.38) -> 0.361; lane divider 0.34 -> 0.335; drivable 0.80 -> 0.806;
  walkway 0.33 -> 0.340); the fifth, ped AP@2 0.15 -> 0.199, is better than predicted (outside the tolerance). Caveats: one seed, the control also one seed (s1 crashed), no latency measured (two image encoders: not a product design, an
  information test); the maps of dual vs bevfeat are within the band, so "no loss" not "gain". Decision: seed twin `t10.dual_s1` next (E1), then the cost question (can the map side be had with one encoder).
- 2026-11-01 03:15: the seed twin `t10_dual_s1` (3000 steps; config identical but the seed; killed once by the out-of-memory killer at step 760 and resumed from a checkpoint on a later code
  snapshot, commit 77aed27 against 40648bd for s0) was evaluated cold on VAL 1000 (`temp/t10/eval_s1.sh`, run `t10.dual_s1.eval`); front-sector comparison `results_x50_2seeds.md`; board
  `../../records/BOARD.md` (base `t8_ctl`, two seeds). Seed 1: lane divider 0.350, drivable@0.5 0.807, walkway 0.354, crossing 0.173, carpark 0.139, road divider 0.442; front mAP2 0.364 (control
  0.367 / 0.368, dual s0 0.361; difference to `t8_ctl_s0` -0.004 [-0.010, +0.001]), pedestrian AP@2 (front) 0.202. **Rule E1 holds on both seeds**: maps within 0.02 of s0 and of the BEVFusion-only
  runs, vehicle + pedestrian mAP2 within 0.006 of the control. Not claimed: the rare two-wheeler class and the 3-class mAP (two-wheeler AP@2 control 0.121 / 0.051, dual 0.016 / 0.061; 3-class mAP
  control 0.251 / 0.235, dual 0.221 / 0.234: inside the control's own seed spread) and the earlier one-seed gain of the pedestrian segmentation (0.000 -> 0.091), which the control's second seed
  (0.090) shows to be noise. The board reads "7 better, 0 worse" for `t10_dual`. The dual chain is the proposed new base; `records/champion.json` "promoted" stays null: Mirko's decision.

## 3 Lever 1: the dual chain at 6000 steps (card written 2026-11-01 before any run; proposed for the queue, not started)

- **Question.** What is the slope of the dual features with training length, and does it carry the maps toward the SOA at a cost this machine can pay?
- **Why now.** `../t6_soa_gap/report.md` section 5: for the OLD features the first doubling of the steps (3000 -> 6000) bought +0.053 on the lane divider, +0.034 walkway, +0.055
  crossing and +0.023 drivable, and the remaining gap to GaussianCaR then needed 4 - 10 more doublings (16 - 1000 x the compute). With the new features the slope is unknown; it decides whether
  "the SOA is reachable on this hardware" is a claim or a wish (the sceptical review of the case file, 2026-11-01, objection 7).
- **Hypothesis.** The dual chain is still under-trained at 3000 steps (0.55 epoch). At 6000 steps the maps gain at least 0.75 x the old features' first-doubling rate (the new features
  start closer to the SOA, so there is less to gain).
- **Prediction** (cold VAL 1000, `seg_fov`, before the run): lane divider@0.5 0.375 (0.335 - 0.415), walkway 0.370 (0.340 - 0.410), crossing 0.19 (0.15 - 0.25), drivable@0 0.78
  (0.76 - 0.80), front-sector mAP2 0.372 (0.35 - 0.40).
- **Decision rule** (written now): gain of the lane divider over `t10_dual_s0` (3000 steps; the mean of the two dual seeds when `t10_dual_s1` exists) >= +0.04 -> KEEP the lever (continue with the
  stride-4 branch and the longer budget); < +0.02 -> CUT: drop "the SOA is reachable on this hardware" and keep "SOA-level on camera detection, partial on maps"; in between -> INCONCLUSIVE
  (add a seed, do not adopt). Guards: front mAP2 not below 0.361 - 0.054 (objects kept); walkway and crossing change with the same sign.
- **Design.** One arm `t10_dual_long_s0` = `t10_dual_s0` with `steps` 6000 and `eval_every` 2000 (exactly the difference of `t3_ref_long` to `t2_ref_s0`, the matched reference of the old
  features: lane divider 0.234, walkway 0.225, crossing 0.057, drivable 0.689 at 6000 steps, same frames). Controls: `t10_dual_s0` / `s1` (3000 steps) and `t3_ref_long`. One seed; the seed
  twin of the 3000-step chain is the noise reference.
- **Budget.** About 3 - 4 GPU hours (115 min per 3000 steps on the shared GPU) plus 10 min for the evaluation.
- **Gate / kill.** The t7 loss rule at step 1000 (`group.lines` mean over steps 900 - 1000 above 1.48: stop); a NaN or a CUDA unknown error: resume from the last checkpoint once. It needs a free
  GPU slot and less than 3 GB in use; the velocity arms and the lane GPU arms come first (D28). Config `training/configs/t10_dual_long_s0.json`, start line `temp/t10/start_long.sh`.
- **Decision of Mirko (chat, 2026-11-01 02:4x, quoted):** "while you work on trainings and evaluations, let the architecture chat restructure the auto research in a branch. Once it is ready,
  merge its branch to main and then use the 6000 step run after a first smoke test as a test for the new system." So this run starts only after the restructured autoresearch is merged and a
  smoke test through it has passed, and it is the first real run of the new system (the start line above is written for today's `run.py` / `run_ar.sh` and is adapted to the new interface then).
- **Decision of Mirko (chat, 2026-11-01 about 02:50, button "Yes, keep it on"):** "cancel the 09:30 shutdown; the machine stays on until the 6000-step run and its evaluation are done." The
  planned shutdown of D28 is therefore not to be executed; the lead of the overnight round owns it and was told.

### 3b Amendment (2026-11-01 08:15, before any run): the lead's version of the 6000-step run

Source of the change: the lead of the overnight round (`../../records/morning_2026-11-01.md` section 4 and `../t11_vel_hurdle/experiment.md` section 3: the run "should carry the hurdle + accumulation 2"), relayed
by the queue owner as Mirko's wish; Mirko's own quoted words in D28 name only "the 6000 step run". The start is after the Architecture merge and smoke test, so Mirko can still veto the version below (the cost is the second bullet).

- **Design change.** `t10_dual_long_s0` = `t10_dual_s0` + `steps` 6000 + `eval_every` 2000 (§3 as written) + `accum` 2 (effective batch 6) + `model.object_moving` true (the velocity hurdle of `../t11_vel_hurdle`
  from step 0; `init_from` empty, `warmup_hold` / `warmup_ramp` at the t10 defaults). Samples seen: 6000 x 6 = 36,000 against 9,000 of the 3000-step chain (4x; the budget of the `t1_r4_soa` recipe, 7000 x 2).
  Config `training/configs/t10_dual_long_s0.json` (its `diff` to `t10_dual_s0.json`: name, steps, eval_every, accum, object_moving, a comment).
- **Cost.** Arm B2 needed 5.06 s per optimiser step (ledger `t11.vel_hurdle_acc2`: 5056 s for 1000 steps with 2 in-training evaluations; `t10_dual_s0` 2.30 s per step): 6000 steps are about 8.4 h, plus about 15 min
  for the evaluation, on the whole GPU (a t10-size job holds 11 of 12 GB). Started at 10:00 it ends about 18:30. The cheaper variant (accum 1, hurdle, 6000 steps) would take about 3.8 h, but arm B' showed that at batch 3 the
  static error misses the 0.30 clause (0.336).
- **What it tests now: three changes at once, so nobody reads it as one.** (i) the training budget x4 on the maps: the lane divider rule of §3 is unchanged and read as "the lever is 4x the samples (2x the optimiser steps)";
  attributing the gain to the budget assumes that the hurdle does not move the maps: B2 against `t10_dual_s0` was within 0.01 on the maps (drivable 0.757 / 0.756, walkway 0.345 / 0.340, lane divider 0.343 / 0.335; band 0.02), an argument and
  not a test (no arm without the hurdle at this budget exists); (ii) the hurdle from scratch (so far only as a 1000-step warm start of `t10_dual_s0` with a fresh velocity branch); (iii) the first real run of the
  restructured autoresearch (merge and smoke test come first).
- **Prediction** (cold VAL 1000, 82 scenes, before the run; the maps from the old-feature ladder: 3000 x 1 -> 6000 x 1 gave +0.053 lane divider, 3000 x 1 -> 7000 x 2 gave +0.092 on the same 1000 frames (0.273 against 0.181; corrected in 3c: version 1 of this line said "+0.093 on a different frame set", the 3109-frame number), times 0.75 because the dual chain
  starts higher; velocity and static error from B2 with a margin for the from-scratch start):

  | metric | t10_dual_s0 / s1 (3000 x 1) | predicted (range) |
  |---|---|---|
  | lane divider IoU@0.5 | 0.335 / 0.350 | 0.41 (0.36 - 0.46) |
  | walkway IoU@0 | 0.340 / 0.354 | 0.39 (0.35 - 0.44) |
  | crossing IoU@0 | 0.154 / 0.173 | 0.22 (0.17 - 0.30) |
  | drivable IoU@0 | 0.756 / 0.757 | 0.78 (0.765 - 0.80) |
  | front mAP2 (0 - 50 m) | 0.361 / 0.364 | 0.355 (0.33 - 0.385): about +0.01 from the longer training, -0.014 from the hurdle (B2) |
  | vehicle velocity skill on moving cars (`ave_skill_moving`) | 0.000 / 0.000 (B2 0.566) | 0.55 (0.35 - 0.70) |
  | vehicle static error `ave_static` (m/s) | 0.029 / 0.029 (B2 0.284) | 0.25 (0.15 - 0.35) |

- **Decision rule** (written now). Lever 1 as in §3: lane divider gain over the mean of the two dual seeds (0.3425) >= +0.04 KEEP, < +0.02 CUT (drop "the SOA is reachable on this hardware"), between INCONCLUSIVE (one seed; the seed
  spread of the dual chain on this layer is 0.015). Hurdle from scratch: vehicle `ave_skill_moving` >= 0.50 and `ave_static` <= 0.30 -> holds (criterion b of the case is then met by a from-scratch run, still one seed); skill
  0.25 - 0.50 or static error above 0.30 -> partly; skill < 0.25 -> the warm-start result is not a recipe. Guards: front mAP2 >= 0.307 (0.361 - 0.054), walkway and crossing move with the same sign as the lane divider.
  Reading when the guards or the hurdle fail but the maps gain: the budget lever stands, the hurdle goes back to a warm-start step.
- **Gate / kill.** As §3 (loss rule at step 1000: `group.lines` mean over steps 900 - 1000 above 1.48 -> stop); at step 2000 the in-training val `objhead.mov@+0.0` below 0.531 (20 % under the base-rate level) says the hurdle has started,
  above it is a note, not a stop; NaN or a CUDA unknown error: resume once from the last checkpoint (a resume with accumulation 2 has never been exercised: the code review checks it). Preconditions: (1) the Architecture merge and
  smoke test passed; (2) the `ar-reviewer` code pass on the t11 code (the hurdle, `init_from` / `init_reset`, logging) has no open blocker and the code is committed; (3) the GPU is free (the map scoring `pitch.maps_lfl` ended);
  (4) Mirko did not veto the version. If the run is not done by 19:00, the last checkpoint (step >= 4000) is evaluated and labelled as such. (Superseded in 3c: item d.)

### 3c Amendment 2 (2026-11-01 08:50, before any run): the critic's six fixes and the code review

Sources: the `ar-critic` read of this card (smoke step b of the D32 merge, 08:25, verdict "revise", G2 "no", six text fixes: `../../temp/smoke_d32/critic_long_run_card.md`) and the `ar-reviewer` code pass on the t11
code (08:30, mode code). Everything below supersedes the lines of 3b it names; the run starts only after these edits.

- **a. One run cannot cut a claim.** The noise band of this card's own rule is max(0.02, 2 |s0 - s1|) = 0.029 on the lane divider (s0 0.3352, s1 0.3498). Tool: `autoresearch/tools/decide.py --variant <long lane divider IoU@0.5>
  --control 0.3352,0.3498 --band 0.029 --id t10.dual_long_s0 --write` (one variant seed against the two dual seeds, unpaired); by that tool a gain inside the band is a TIE and a gain beyond it on one seed is INCONCLUSIVE (never KEEP).
  Actions on the gain g = long - 0.3425: **g >= +0.04**: lever 1 continues provisionally (the tool says INCONCLUSIVE until the seed twin of the long run exists, so no headline claim yet); **+0.029 < g < +0.04**: INCONCLUSIVE, the
  twin decides; **g <= +0.029**: INCONCLUSIVE for the claim, nothing is dropped on one run (the 3b rule "< +0.02 = CUT" is withdrawn; the tool's TIE only means that the cheaper 3000-step arm stays the base for other work); a second long run (the seed twin, or the arm without `object_moving`, about 8 h each) decides, and
  only a mean gain of the two long runs below +0.02 drops "the SOA is reachable on this hardware". Because the three changes of 3b come together and no equal-budget arm without the hurdle exists, a null result is read as "this bundle
  did not move the maps", not as "more training does not help".
- **b. Hypothesis and prediction on one ladder.** Two statements, so that a hit and a decision cannot disagree: the decision hypothesis is "at 4x the samples the lane divider gains at least +0.04" (0.75 x the +0.053 of the old
  features at 2x the samples, the KEEP threshold); the expectation is +0.07 (0.75 x their +0.092 at 4x the samples, the prediction 0.41). A gain of +0.04 to +0.07 is a KEEP-side result that falls short of the expectation (a prediction
  miss on the low side). The predicted range 0.36 - 0.46 stays as written (the honest uncertainty); its lower part (0.36 - 0.3825) is INCONCLUSIVE, no longer a CUT.
- **c. Matched references and tools.** The old-feature reference for 4x the samples is `t1_r4_soa` (7000 x 2: lane divider 0.273 on the same 1000 frames, walkway 0.246), next to `t3_ref_long` (6000 x 1: 0.234); the new frame-set remark of 3b
  is withdrawn (all rows are VAL-1000; the 3109-frame stored number of `t1_r4_soa` is 0.274). Maps: `evaluation/bev_suite.py compare runs/t10_dual_s0 runs/t10_dual_long_s0` and the same against `t10_dual_s1` (paired scene
  bootstrap, refused when the GT signatures differ); front mAP2: `temp/t4/compare_front.py` (100 paired scene draws), as for t11; the verdict through the `decide.py` line of item a, written into the ledger.
- **d. Intermediate checkpoints carry no verdict.** The 19:00 fallback of 3b is withdrawn (the cosine schedule is still at lr x 0.25 at step 4000, so that checkpoint cannot be compared with the annealed 3000-step chain; the machine
  stays on until the run is done, by Mirko's decision). If the run is not finished when the numbers are needed, the deck says "running"; an evaluation of a checkpoint at step 4000 may be added as a labelled note, no verdict.
- **e. Validity: the GT signature.** `runs/t10_dual_long_s0/gt.json` and the final card must carry `0.5_4_15efbde2d1`; `bev_suite.py compare` refuses otherwise. The five autogt signature files (grid, layers, flow, nuscenes_adapter,
  nuscenes_gt) and `bev_fusion/training/*.py` stay unchanged on disk during the run (a resume after an out-of-memory kill snapshots the working tree again; the reviewer's note).
- **f. Owner of precondition (2), and a referee line.** Owner: this chat (the case chat, BEV system), executor an `ar-reviewer`. Result 08:30: **no blocker for the run** (the hurdle code passes its gate G1 - G5 on CPU, bit-identical
  default paths, hurdle edge cases clean, main's code makes the gate fail, 20-step smoke run with exactly this config shape passed through run.py at 08:1x); **blockers for the commit to main only**: main's own `train.py` has a broken
  slim-checkpoint path (NameError on resume and evaluation load; the working tree repairs it) and four test gaps (the velocity terms in `log.jsonl`, `w_obj_mov`, the `mov` bias, the new test missing from `run_gates.sh`); the fixes follow after the launch
  and are recorded in this card (3d) when done: the test gaps are being closed by an `ar-coder` in tests/ only (training/ stays untouched), the commit of the reviewed code to main follows. The run trains from the snapshot of the working tree through run.py (tag `bf/t10.dual_long_s0`), never from the older tag `bf/t11.vel_hurdle_acc2` (its config file is the old one).
  Referee: `ar-critic` 08:25 revise (this amendment answers it); a second pass on the amended card is requested before the start; its verdict is written here: [pending].
