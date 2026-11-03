# t9 encoder 2 x 2 - does a trained, finer image side lift the maps? (card written 2026-10-31, before any code or run; Mirko: "combine both into one experiment")

Context: `../../results/experiments/t6_soa_gap/report.md` and `experiment.md`. Our maps reach 17 - 85 % of GaussianCaR's IoU; the gap is not range, threshold, overfitting,
BEV trunk or task competition (the maps-only run closes only 7 - 12 % of it). What is left is the image side and the budget: the camera
features are DETECTION-trained, FROZEN and at stride 16 (M6), and the runs are 9 - 27 x shorter (M8). The two levers offered to Mirko were
"map-supervised finer features" (plan 2.7 / 2.8) and "an unfreezing arm"; this card runs them in ONE factorial experiment so that each can be
read alone and together. Related and already running or done: t7 (a FROZEN stride-4 branch: lane divider +0.065, road divider +0.125, nothing on
walkway / carpark / crossing; seeds 0 / 1: +0.065 / +0.042 on the lane divider), t8 (frozen map-trained BEVFusion features, stride 8, running in
lane 4; same-code controls `t8_ctl_s*`).

## 1 Card

- **Question.** Starting from the reference chain (frozen DualBEV image side, stride 16), what does it add (a) to let the map loss train the
  existing DualBEV encoder, (b) to add a finer, trainable, ImageNet-initialised stride-8 image branch, and (c) to do both?
- **Hypothesis.** (a) lets the encoder learn the semantic ground classes that a detector never needed (walkway, carpark, crossing) and
  helps drivable; (b) helps the thin lines through resolution (as t7) and, being trainable and pretrained, a little more than the frozen
  stride-4 branch on the dividers; (a) + (b) is roughly additive on the dividers.
- **Design: 2 x 2 around the reference.** Everything else identical to `t2_ref_s0` (3 frames, clips of 3, flip augmentation, Kendall, 3000 steps,
  lr 3e-4 for the new parts, EMA, seed 0), one pinned code commit for all four arms (the GT generator changed after 0a3d73c: no comparison
  with ref_s*), scenes 428 / 82, cold VAL 1000 keyframes by `bev_suite eval --n 1000`.

  | arm | DualBEV image side | finer branch | what changes |
  |---|---|---|---|
  | **C** control | frozen, cached (stride 16) | none | nothing (same-code control; also gives the seed spread with C_s1) |
  | **R** | frozen, cached | trainable ImageNet ResNet-18 to stride 8, 1 x 1 / 3 x 3 head, lifted with the cached DualBEV depth, zero-initialised residual into the camera BEV (the t7 `LaneBranch` with a new kind `r18`) | resolution + a trained branch |
  | **U** | UNFROZEN for the current frame: ResNet-50 + neck + DepthNet run live on the key-frame image (`lane_img` cache of t7), map + object loss reach them; lr 0.1 x of the rest, BatchNorm in eval mode (the camera-vector BatchNorm1d keeps its statistics); the two older window frames stay cached | none | map supervision into the encoder |
  | **UR** | as U | as R | both |

  The depth stays the teacher's in R (the finer branch is read at stride 8 and lifted with the stride-16 depth, the sampling convention of t7)
  and is trainable in U (so the map loss may re-shape the depth distribution as well: this is the part of U that R does not have).
- **Controls and noise.** C_s0 and C_s1 (same code, two seeds) give the band per layer: max(0.02, 2 x |C_s0 - C_s1|) (from the earlier pair:
  walkway about 0.054, the others 0.02). Related arms read, not re-run: t7 res4 / rgb, t8_bevfeat (frozen map-trained stride 8), the maps-only
  run. One seed per arm first; any arm that passes a rule gets its seed twin before it is adopted.
- **Headline layers** (cold VAL, dt 0, camera cells, `seg_fov`): semantic - walkway IoU@0, carpark IoU@0, pedestrian crossing IoU@0;
  lines - lane divider IoU@0.5 m, road divider IoU@0.5 m; area - drivable IoU@0. Also reported: vehicle AP@1 / AP@2 and pedestrian AP@1 / AP@2
  (U changes the detector's own features, so detection may move either way) and ms per frame of every arm (idle GPU, batch 1).
- **Prediction (before the run; gain over C).**
  | layer | U | R | UR |
  |---|---|---|---|
  | walkway | +0.06 | +0.01 | +0.07 |
  | carpark | +0.05 | +0.02 | +0.06 |
  | crossing | +0.05 | 0.00 | +0.05 |
  | lane divider | +0.04 | +0.06 | +0.10 |
  | road divider | +0.05 | +0.10 | +0.14 |
  | drivable | +0.02 | +0.01 | +0.03 |
  No layer reaches 60 % of the GaussianCaR value in 3000 steps (walkway < 0.36, lane divider < 0.41); a surprise above that would mean the encoder was the whole story.
- **Decision rule (written now).** Gain = arm - C_s0 (the seed twin must agree in sign), bands as above.
  - D1 map supervision into the encoder is a lever: U gains more than the band on >= 2 of the 3 semantic layers. Next: longer (6000 steps) and
    all 700 scenes, then the distilled / BEVFusion-initialised light encoders of plan 2.8 step 1.
  - D2 resolution is the lever for the lines only: R gains more than the band on both dividers and U does not on the semantic layers. Next: the
    thin-line head and the horizon strip; for the semantic classes a map-trained teacher (BEVFusion initialisation).
  - D3 both are needed: UR - max(U, R) exceeds the band on >= 3 layers. Next: UR is the base of E8a.
  - D4 nothing exceeds the band in U, R, UR: encoder training at this budget does not help; the lever is length and data (M8): 6000-step arms and the 700-scene shards.
  - D5 mixed: add the seed twin of the arm before deciding; INCONCLUSIVE means add a seed, not adopt.
- **Gate (before any arm; known answers, `temp/t9/`).** G1 the live DualBEV on the JPEG key-frame image reproduces the cached features of 8 val
  frames (max |dprob| < 0.02, relative error of the context < 0.03); G2 R and UR at step 0 give exactly the output of C (zero start); G3 gradients:
  in U the ResNet-50, neck and DepthNet parameters receive non-zero gradients, in C and R none; BN of the DualBEV side stays in eval mode; G4 the
  lr multiplier reaches the right parameter group; G5 one UR step at the real micro-batch: peak GPU memory < 7.5 GB (the card is shared) and
  seconds per step measured - if it does not fit, micro-batch 1 clip with accum 3 and a gradient checkpoint on the ResNet-50 stages.
- **Kill / validity criteria.** (1) the t7 rule at step 1000: group.lines mean over steps 900 - 1000 above 1.48 (the references are 1.22 - 1.25):
  stop, lr x 0.3, rerun once; (2) U's detection collapses (vehicle AP@2 below half of C's): the encoder was damaged,
  report as such and rerun with lr 0.03 x; (3) a NaN: resume from the last checkpoint once.
- **Cost (estimate, to be replaced by the G5 measurement).** C 85 min incl. evaluation; R about 100 min; U and UR about 3 - 3.5 h each (live
  ResNet-50 forward and backward for 3 key-frame images per clip step); total about 9 h of lane time for the first seed of the four arms,
  plus 85 min for C_s1. Order: C_s0, R, U, UR, C_s1 (R and C first because they need no new encoder code beyond the branch kind).
- **What it does NOT test.** Distillation, the BEVFusion initialisation, 448 x 800 input, the horizon strip, more scenes, longer training,
  lidar depth: these are the later arms of plan 2.8 (they follow from D1 - D4).

## 2 Build list (default-off options, bit-identical defaults; owners as agreed in the loop)

1. `lane_branch.py`: kind `r18` = torchvision ResNet-18 from `models/pretrained/imagenet/` to stride 8, trainable, otherwise the t7 branch (zero start). (owner of the t7 branch)
2. `model.py`: `ModelConfig.camera_live = False`; if True the CURRENT frame of each keyframe runs `camera.image_features` live from `batch["lane_img"]` (normalised
   on the device) with gradient and BN in eval; `freeze_camera` is False for that part; older window frames keep the cached features. (owner of `model.py`)
3. `train.py`: parameter group for the live DualBEV side with `lr x camera_lr_mult` (default 1.0; 0.1 in U / UR); checkpoint keeps its state; `micro-batch` option if G5 fails.
4. Configs `t9_{ctl,r18,live,live_r18}_s0.json` (+ `t9_ctl_s1`), card log, `temp/t9/gate.py` (G1 - G5), queue lines in `temp/study/queue3.txt`.

## 3 Log

- 2026-10-31: card written; nothing built or queued. Needs Mirko's go and the owners of items 1 - 3.
