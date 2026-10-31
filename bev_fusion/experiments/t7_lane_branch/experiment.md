# t7_lane_branch - does a stride-4 lane branch into the BEV close part of the divider gap?

Started 2026-10-30 (Mirko: "Test a BEV lane branch", after the t5 probe v2: the stride-4 stage of the frozen ResNet-50 and an RGB CNN
draw the painted lines sharply in the image, the stride-16 context that feeds the lift gives blurred bands; the F1 of the probe could
not separate them (TIE, `../t5_lane_probe/experiment.md` section 5)). Related: `../t6_soa_gap/report.md` (divider gap 0.29 vs 0.69,
cause M6 = frozen, detection-trained, stride-16 image features, not isolated) and `../../plan.md` 2.8 (encoder fix plan, another
chat). This experiment tests the cheapest form of "more image resolution for the lines" inside the existing chain, without
touching the image encoder: a side branch.

## 1 Card (written before any code or run)

- **Question.** Does the BEV divider map improve when the chain also lifts stride-4 image features (frozen ResNet-50 layer 1, or an
  RGB CNN trained from scratch) into the BEV, next to the stride-16 context it uses now?
- **Hypothesis.** The stride-16 context loses the position of painted lines (one feature cell is 0.57 m x 7.9 m on the ground at
  20 m); lifting a stride-4 map (0.14 m x 2 m at 20 m) with the same depth distribution puts the line evidence into the right BEV
  cells, so the line head can find dividers where the reference only sees road.
- **Design (one change against the reference; everything else = `training/configs/t2_ref_s0.json`).**
  - Lane branch (`training/lane_branch.py`), current camera frame only (no temporal block):
    image (256 x 704, the frame the cached heads saw) -> feature map at stride 4 (64 x 176) -> 1 x 1 conv to 32 ch, BN, ReLU, 3 x 3
    conv 32 -> lift PD with the SAME cached depth probabilities of that frame (stride 16; the lift samples both maps in normalised
    image coordinates, so the feature map is read at stride 4 and the depth at stride 16) and the same 8 height slices x
    oversample 4 -> 1 x 1 conv over height x channels to 16 ch -> one 1 x 1 conv to c_cam (64), ZERO-initialised, ADDED to the camera
    BEV after the temporal block (the chain starts exactly as the reference; the pre-trained trunk sees the same input at step 0).
    Mirror augmentation applies to the lane BEV like to the camera BEV.
  - Image source: a key-frame image cache (`data/cache/img_front`, JPEG quality 95 of the 256 x 704 image after the dataset's own
    resize + crop, about 1.2 GB); the reference reads no image at all.
  - Arms (3000 steps, batch 3, clips of 3 keyframes, everything as `t2_ref`):
    - R: reference = the stored runs `t2_ref_s0`, `t2_ref_s1` (same config, same 3000-step protocol).
    - L4: lane branch on the frozen DualBEV ResNet-50 stem + layer 1 (stride 4, 256 ch; the weights of
      `models/pretrained/dualbev-r50-cbgs.pth`, BatchNorm in eval mode), seeds 0 and 1.
    - LR: lane branch on an RGB CNN trained from scratch (stride 4: 3 x 3 / 2 conv 32, 3 x 3 conv 32, 3 x 3 / 2 conv 64,
      3 x 3 conv 64, BN + ReLU each; the probe's control features), seeds 0 and 1 - is it the frozen backbone or only resolution +
      a trainable map?
  - Evaluation: `evaluation/bev_suite.py <run> eval --n 1000` (cold VAL, 1000 keyframes of the 82 val scenes, the queue's protocol).
    Layer-1 check first: the current suite re-evaluates `t2_ref_s0`; if its lane-divider IoU@0.5 differs from the stored 0.181 by more
    than 0.005, both reference seeds are re-evaluated and the new numbers are the reference.
- **Metrics.** Headline: `seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5` (front field of view, dt 0, 0.5 m tolerance). Secondary:
  lane divider at 0-20 m and IoU@1, road divider IoU@1, zero-crossing F1, drivable / walkway / crossing, detection mAP, ms per frame
  of the branch (idle GPU, batch 1).
- **Reference numbers (stored, VAL 1000).** lane divider IoU@0.5 0.181 / 0.179 (s0 / s1), IoU@1 0.274 / 0.267, road divider IoU@1
  0.279 / 0.286, drivable IoU@0.5 0.714 / 0.711, walkway 0.191 / 0.164, mAP 0.277 / 0.228.
- **Noise band.** Headline: 0.02 (2 x the reference seed spread on this metric is 0.004; 0.02 leaves room for arm seeds that may
  spread more). Other map classes 0.054 (t3), mAP 0.03 (2 x 0.015, t3).
- **Prediction (before the run).** L4: lane divider IoU@0.5 0.18 -> 0.22 (+0.04, mean of two seeds), the gain larger at 0-20 m;
  LR within 0.02 of L4; drivable, walkway, mAP unchanged inside their bands; branch cost +3 to +6 ms per frame.
- **Decision rule (headline, mean of the two seeds of an arm minus the mean of R; per seed a paired scene bootstrap of the arm
  against the reference seed with the same index, 1000 draws, 95 %).**
  1. gain >= 0.02 and both seed intervals exclude 0, and no other map class drops beyond 0.054 and mAP not beyond 0.03: KEEP the lane
     branch (if L4 and LR both pass and differ by less than 0.02: the cheaper one, by ms per frame).
  2. L4 passes and LR is worse than L4 by more than 0.02: the frozen backbone's features matter (keep L4); LR passes and is not worse:
     a small trainable image map is enough (keep LR).
  3. gain < 0.02, or an interval includes 0: TIE - resolution of the image features is not the limit of the dividers in this chain;
     the lever is how the features are trained (plan 2.8: map-trained encoder); nothing is built on the branch.
  4. An arm worse than R beyond the band on the headline: the branch disturbs the chain (check the zero start and the mirror).
- **Kill.** A non-finite loss, or the `group.lines` training loss at step 1000 more than 20 % above the reference's (a broken branch).
- **Gate before the runs.** Self-test of `lane_branch.py` (shapes; the zero start gives exactly the reference output; gradients reach
  the branch after one step; a ground line drawn in a synthetic image lands in the right BEV cells; mirror), the image cache equals
  the dataset's own image within JPEG error, and one smoke step of the training on the real config.
- **Budget.** CPU about 15 min (image cache); GPU 4 runs x about 80 min + 4 evaluations x about 10 min, in the GPU queue behind the
  jobs already waiting (`temp/study/queue3.txt`: lift_s, trunk_small, t6_maps_only). Order: L4 s0, LR s0, L4 s1, LR s1.

## 2 Log

- 2026-10-30 code: `training/lane_branch.py` (branch, image cache CLI, self-test), default-off hooks in `model.py`
  (`lane_branch`, `lane_pretrained`), `feature_cache.py` (`lane_images`, `lane_img` in the batch), `train.py`
  (`TrainConfig.lane_images`, check in `build`), `nuscenes_dataset.py` (`_image_uint8` split out of `_image`, bit-identical);
  configs `training/configs/t7_{res4,rgb}_s{0,1}.json` = `t2_ref_s<seed>` + the branch (diff: name, `lane_images`, `lane_branch`).
  Drawing `images/architecture.png` (`temp/t7/draw_branch.py`).
- Gates: `lane_branch.py` self-test passes (zero start adds nothing, gradient reaches the image CNN after one step, a stride-4 ground
  line lands in its BEV cells with the stride-16 depth: mean 0.0009 on the line, 0.000000 on the mirrored side, res4 frozen,
  res4 weights load); `tests/test_lane_branch_chain.py` passes for res4 (41 s) and rgb (36 s): JPEG error 1.44 of 255, outputs
  EXACTLY equal to the reference chain at the zero start (max difference 0.0), gradients, mirror; `model.py`, `nuscenes_dataset.py`,
  `feature_cache.py` self-tests pass unchanged.
- Kill threshold fixed now: reference `group.lines` (training) at steps 900-1000 = 1.253 / 1.217 (s0 / s1), so an arm is cut if
  its mean there exceeds 1.48 (1.2 x 1.235).
- Image cache: the first build with two forked workers ran the shared box out of memory at 8,133 images (the devkit copied into
  each worker, next to another chat's training); stopped through the task tool (a direct `kill` was refused), resumed in ONE
  process (`--workers 1`, the default now).
- Cache done: 19,398 keyframes (train + val of the t2 scene lists), 981 MB; check against the dataset's own image on 40 val
  samples: mean abs error 1.00 of 255 (max 1.30). CPU smoke of the real res4 config (2 steps, batch 1, one validation batch):
  runs, the checkpoint stores 84 lane tensors, the zero-initialised mix moved (|W| sum 0.43), the frozen layer 1 is unchanged.
  `train.py` self-test passes.
- Committed 4719d1d (code, configs, card); queued in `temp/study/queue3.txt` behind t6_maps_only as `cfg:t7_<arm>_s<seed>:4719d1d`
  (order res4 s0, rgb s0, res4 s1, rgb s1; run by the live queue `temp/study/lane3.sh`, log `temp/study/lane3b.log`).
  `temp/t7/finish.sh` (log `temp/t7/finish.log`) waits for the four, re-evaluates `t2_ref_s0` / `s1` with the current suite
  (legacy metrics copied to `metrics_legacy_2026-10-28.json` first: layer-1 check), writes the paired comparisons
  `temp/t7/compare_<arm>_s<seed>.txt` and the timing if the GPU is idle. `temp/t7/wait_step1000.sh <run>` = the kill check.
- 2026-10-31 kill check `t7_res4_s0`: `group.lines` mean over steps 900-1000 = 1.267 (reference 1.253 / 1.217, threshold 1.48):
  passes, the run continues. The same check is armed for `t7_rgb_s0`, `t7_res4_s1`, `t7_rgb_s1`.
