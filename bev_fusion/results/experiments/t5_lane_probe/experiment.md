# t5_lane_probe - do the frozen image features contain the lane lines?

Started 2026-10-30 (Mirko: "Run the probe", after the finding that lane dividers are far behind the state of the art:
mask IoU 0.27, zero-crossing F1 0.03 in `../t1_r4_soa/report.md`, against divider IoU 38.7 / 50.1 of GaussianLSS / GaussianCaR).
Code: `studies/recipes/t5/lane_probe.py` (self-test `python studies/recipes/t5/lane_probe.py selftest`).

## 1 Card

- **Question.** Is the lane failure already in the image features the lift receives (the frozen DualBEV context, 80 channels at
  stride 16, 16 x 44 cells), or does it arise later (lift depth, BEV trunk, line head)? And do the higher-resolution stages of the same
  frozen backbone keep the lines?
- **Hypothesis.** Stride-16 context features trained for detection lose painted lines (a 15 cm line is far below one feature cell:
  about 0.8 m wide at 30 m); the stride-4 / stride-8 stages of the same frozen ResNet-50 still carry them.
- **Design: image-plane line probe** (no lift, no BEV: the question is only what the features hold).
  - GT: the exact lane and road divider polylines of the GT cache (`meta["vectors"]`, ego frame at the front-camera time), placed on the
    ground (z = 0), sampled every 0.1 m from 2 to 60 m ahead, projected with K, T_cam_car and the image transform into the 704 x 256 input
    image, drawn as one class "divider" on the stride-4 grid (64 x 176, one cell = 4 x 4 input pixels).
  - Arms, all with the SAME small probe head (1 x 1 conv to 32, BN, ReLU, 3 x 3 conv 32, BN, ReLU, 1 x 1 to one logit, bilinear up-sampling
    of the logit to stride 4):
    - A `ctx16`: the 80 context channels at stride 16 (what lift PD receives) - the system as it is
    - B `fpn16`: the 512-channel neck output at stride 16 (before the DepthNet)
    - C `res8`: ResNet-50 layer 2, 512 channels at stride 8
    - D `res4`: ResNet-50 layer 1, 256 channels at stride 4
    - E `rgb` control: a small CNN trained from scratch on the image (3 strided 3 x 3 conv layers to stride 4, then the same head): is the
      task learnable from these frames at all with this budget
    - N `prior` null baseline: the per-pixel mean of the training masks
  - Data: the same frames for every arm; 600 TRAIN keyframes (4 per scene, 150 scenes of the t4 training list) and 300 VAL keyframes
    (up to 4 per scene of the 82 val scenes); TEST not used. Training: AdamW 1e-3, 15 epochs, batch 8, BCE with pos_weight 10 + soft Dice;
    one seed; decision threshold chosen per arm on the TRAIN frames, never on VAL.
- **Metric.** `evaluation/metrics.SegMeter` on the stride-4 image grid (res = 1 cell): divider IoU at tolerance 0 / 1 / 2 cells (1 cell =
  4 input pixels), all pixels and by ground distance (flat-road distance of the pixel: 2-20 / 20-40 / 40-60 m). Headline: IoU@1 (all).
  Paired scene bootstrap over the val scenes (1000 draws, 95 % interval) for every arm against A.
- **Prediction (before the run).** A IoU@1 <= 0.25; best of C / D >= A + 0.15; E >= 0.40; N <= 0.10.
- **Decision rule.**
  1. A >= 0.8 x E: the lift's features hold the lines -> the loss is downstream (depth of the lift, BEV trunk, line head) -> next test:
     the BEV line head on GT-depth lifted features.
  2. best(C, D) - A > 0.10 with the interval excluding 0: stride 16 loses the lines, the backbone keeps them -> next: a lane branch from the
     stride-4 / 8 frozen features into the BEV (or a higher-resolution lift for the line channels).
  3. A, C and D all < 0.5 x E: the frozen backbone does not encode paint -> next: an image lane head trained on the backbone (unfrozen
     stages) or a separate lane backbone.
  Differences inside the 95 % interval are ties.
- **Kill / validity criterion.** Before any number: overlays of the projected GT on 10 val images must lie on the painted markings in at
  least 9 of 10 (flat-road assumption, calibration, image transform); otherwise fix the GT first.
- **Budget.** CPU only, 2 threads, about 2 GB RAM (the GPU runs `t4_fpn_order` and its queue): features once (about 10 MB per frame,
  9 GB in `temp/t5/feats/`, deleted after the experiment), probe heads 10-20 min each.
- **Limits known before.** The divider GT on the image assumes a flat road at z = 0; slopes put far lines off by pixels (tolerance 1-2
  cells covers part of it); occluded markings (under cars) are in the GT but not visible, the same for every arm; one seed.

## 2 Log

- 2026-10-30, frames: 600 TRAIN keyframes of 150 scenes, 300 VAL keyframes of 75 of the 82 val scenes (the 300 cap is reached
  before the last 7 scenes); `temp/t5/frames.json`. Self-test passes.
- 2026-10-30, validity check (`temp/t5/gt_overlay.png`, 10 val frames): where the GT draws a divider it lies on the painted line
  (scene-0096, 0269, 0346, 0562, and the far pieces of 0105, 0554, 0630): placement passes. But visible paint is often NOT in the GT
  (double yellow lines in scene-0003, 0277, 0630; dashed lines in 0277). Hypothesis "our record query drops long dividers that cross
  the patch without a node inside" tested by `studies/recipes/t5/check_divider_records.py`: 40 val frames, 3887 m of divider in the grid
  windows kept, 0 m dropped - FALSE. The nuScenes map layers `lane_divider` + `road_divider` themselves lack these lines (scene-0003:
  9.5 m of divider in the whole 60 m x 52 m window). This is a property of the dataset shared by every published nuScenes map model
  (MapTR's "divider" class is the same two layers), so it cannot explain our gap to them; for this probe, unlabelled paint counts as a
  false positive for every arm alike.
- 2026-10-30, features (CPU, 2 threads, 900 frames, 9 GB in `temp/t5/feats/`) and the five heads (15 epochs each, 18 s to 10 min).
  Pre-registered metric, VAL 300 keyframes / 75 scenes, divider IoU@1 (all pixels): prior 0.135, ctx16 0.171, fpn16 0.190, res8 0.189,
  res4 0.148, rgb 0.112 (`temp/t5/results.md`). By the card's rule nothing applies: the RGB control is BELOW the prior, so the probe
  failed its own control and cannot rank the features with this metric. Train IoU is as low as val (0.16-0.27): the heads look
  underfit, but the picture (`images/probe_predictions.png`) shows why the metric is misleading: res8 / res4 / rgb draw the painted
  lines sharply, including the double yellow line of scene-0003 that has no label, while ctx16 gives wide blurred bands. With
  incomplete labels, correctly detected paint is a false positive and a sharp line one cell off is a miss.
- Post hoc (NOT pre-registered, added after seeing the picture): recall of the labelled line cells at 1 cell (all pixels) and
  precision at 1 cell only inside a 4-cell corridor around the labelled lines, F1 of the two; same heads retrained (same seed and
  frames, weights now saved), same threshold rule.
- Rerun (identical IoU numbers, so the training is reproducible). Post-hoc F1 (recall@1 / corridor precision@1): prior 0.402, ctx16
  0.409, fpn16 0.453 (+0.044 [+0.030, +0.060]), res8 0.391, res4 0.356, rgb 0.283. Where the labels are (`studies/recipes/t5/label_rows.py`):
  67 % of the labelled cells lie in image rows 16-31 of 64 (just below the horizon), 57 % at 20-60 m, where a line is compressed into a
  few rows, a flat-road projection error moves it by whole cells, and a per-pixel prior already hits many cells.

## 3 Result and verdict (2026-10-30; numbers from `results.md`, made by `studies/recipes/t5/lane_probe.py eval`)

VAL, 300 keyframes of 75 scenes, one seed; TRAIN 600 keyframes for the heads and the thresholds.

| arm | features | IoU@1 (pre-registered) | IoU@1 at 2-20 m | F1, post hoc | d F1 vs ctx16 [95 %] |
|---|---|---|---|---|---|
| prior | per-pixel mean of the TRAIN labels | 0.135 | 0.000 | 0.402 | -0.006 [-0.036, +0.028] |
| ctx16 | context, 80 ch, stride 16 (what the lift receives) | 0.171 | 0.089 | 0.409 | - |
| fpn16 | neck, 512 ch, stride 16 | 0.190 | 0.131 | 0.453 | +0.044 [+0.030, +0.060] |
| res8 | ResNet layer 2, stride 8 | 0.189 | 0.158 | 0.391 | -0.018 [-0.040, +0.006] |
| res4 | ResNet layer 1, stride 4 | 0.148 | 0.130 | 0.356 | -0.053 [-0.078, -0.026] |
| rgb | small CNN on the image, from scratch (control) | 0.112 | 0.104 | 0.283 | -0.126 [-0.157, -0.090] |

**Verdict: INCONCLUSIVE - the probe failed its own control.** The RGB control is below the prior on both measures, so this setup
cannot tell whether the features hold the lines; none of the three branches of the decision rule applies, and the prediction
(ctx16 <= 0.25: hit; best of res8 / res4 >= ctx16 + 0.15: miss; rgb >= 0.40: miss; prior <= 0.10: miss) is not evidence either way.

What can and cannot be said:
- **Seen, not measured:** on the three frames of `images/probe_predictions.png` the stride-4 / stride-8 features and the RGB CNN draw
  the painted lines sharply (also the unlabelled double yellow line of scene-0003); ctx16 gives wide blurred bands. This is the
  hypothesis in pictures, three frames chosen by index, not a measurement.
- **Exploratory only:** near the car (2-20 m), where the flat-road projection is exact and lines are wide, IoU@1 is 0.089 for ctx16 and
  0.130-0.158 for the frozen higher-resolution stages; this column was not the headline and has no interval.
- **Measured and solid:** the nuScenes divider labels miss much visible paint (validity check, 0 m lost by our query); this limits every
  line metric on nuScenes, ours and the published ones alike.
- Why the probe failed (readings): the labels sit mostly at 20-60 m just below the horizon, where the image-plane geometry is least exact
  and the prior is strong; unlabelled paint is penalised; the heads are small and short-trained (train IoU as low as val).

**Next (proposal, not run):** probe v2 with (1) the headline restricted to 2-30 m before the run, (2) the corridor precision and
recall as the pre-registered measures, (3) longer training (50 epochs) so that the RGB control can pass, (4) 300 fresh VAL frames (the
current ones shaped the post-hoc metric), (5) the control must beat the prior by a margin before any feature is ranked. The 9 GB of
features in `temp/t5/feats/` are kept for v2 (the TRAIN part is reusable).

## 4 Probe v2 - card (written 2026-10-30 before the v2 run; Mirko: "Run probe v2")

- **Question, hypothesis, arms, heads, TRAIN frames:** as in section 1 (the same 600 TRAIN keyframes and their features).
- **Changes against v1, all fixed before the run:**
  1. **Fresh VAL frames:** 300 keyframes of the val scenes that v1 did not use (v1's frames shaped the post-hoc metric).
  2. **Range:** every measure only on pixels whose flat-ground distance is 2-30 m (near enough that the flat-road projection is exact to
     about a cell and a painted line spans more than one cell). Training still uses all labels.
  3. **Measures (pre-registered now):** recall@1 = labelled line cells with a prediction within 1 cell (all 2-30 m pixels);
     corridor precision@1 = predicted cells within 1 cell of a label, counted only inside a 4-cell corridor around the labels (detected
     but unlabelled paint further away is not a false positive); headline **F1** of the two. All from `evaluation/metrics.SegMeter`
     counts, per val scene; paired scene bootstrap, 1000 draws, 95 % interval.
  4. **Training:** 50 epochs instead of 15 (the v1 heads were underfit); the threshold of each arm is the one with the best F1 on the
     TRAIN frames (2-30 m), never chosen on VAL.
  5. **Control gate:** the RGB control must beat the prior by at least 0.05 F1 with an interval excluding 0; otherwise the probe is
     invalid again and no feature is ranked.
- **Prediction (before the run).** prior F1 <= 0.30 at 2-30 m (the prior has no near-range advantage: v1 prior IoU@1 at 2-20 m was 0.000);
  rgb >= prior + 0.15; ctx16 the lowest of the four frozen arms; best of res8 / res4 >= ctx16 + 0.10.
- **Decision rule (F1, after the gate passes).**
  1. ctx16 >= 0.9 x rgb and no frozen arm beats ctx16 outside the interval: the lift's features hold the near lines -> the lane loss is
     downstream (lift depth, BEV, line head).
  2. best(res8, res4) - ctx16 > 0.05 with the interval excluding 0: stride 16 loses lines that the frozen backbone keeps -> a lane
     branch from the stride-4 / 8 features into the BEV.
  3. all frozen arms < 0.7 x rgb: the frozen backbone does not keep paint -> train an image lane head on unfrozen stages / a lane
     backbone.
  Otherwise (overlapping intervals): TIE between the features; report and do not build anything on it.
- **Budget.** CPU, 2 threads: 300 new frames of features (about 8 min), six heads x 50 epochs (about 1 h in total).

## 5 Probe v2 - result (2026-10-30)

Run: `bash studies/recipes/t5/run_v2.sh` (CPU, 2 threads), code `studies/recipes/t5/lane_probe_v2.py`; table `results_v2.md` (copied from
`temp/t5/results_v2.md`), picture `images/probe_v2_predictions.png` (not in git). VAL: 300 fresh keyframes of 82 val scenes (overlap with
v1: 0); 25937 labelled line cells at 2-30 m (mean 86 per frame, 30 frames without one).

| arm | recall@1 | corridor precision@1 | F1 | d F1 vs ctx16 [95 %] |
|---|---|---|---|---|
| prior (null) | 0.499 | 0.389 | 0.437 | -0.012 [-0.034, +0.010] |
| ctx16 (lift input) | 0.551 | 0.380 | 0.450 | - |
| fpn16 | 0.498 | 0.419 | 0.455 | +0.005 [-0.007, +0.017] |
| res8 | 0.430 | 0.464 | 0.446 | -0.003 [-0.025, +0.017] |
| res4 | 0.594 | 0.421 | 0.493 | +0.043 [+0.024, +0.061] |
| rgb (control) | 0.557 | 0.434 | 0.488 | +0.038 [+0.017, +0.058] |

- **Gate:** rgb - prior = +0.050 [+0.021, +0.077] -> passes, at the margin.
- **Predictions:** all four missed. Prior 0.437 (predicted <= 0.30); rgb only +0.050 over the prior (predicted +0.15); the lowest
  frozen arm is res8, not ctx16 (by 0.004, inside the noise); best(res8, res4) - ctx16 = +0.043 (predicted >= +0.10).
- **Decision (written rule): TIE.** Rule 1 fails (res4 beats ctx16 outside the interval); rule 2 fails (+0.043 < 0.05, although the
  interval excludes 0); rule 3 fails (every frozen arm >= 0.91 x rgb). Nothing is built on it.
- **What the picture shows (qualitative, not part of the rule):** ctx16 and fpn16 give blurry bands, mostly along the horizon; res4
  and rgb draw the painted lines sharply, including paint that has no label (the double yellow line in row 4, the curb line in row 1).
  Painted lines are clearly visible at stride 4 and clearly blurred at stride 16. The F1 hardly separates them, for two reasons: the
  labels miss much visible paint (section 3), and a per-pixel prior already reaches F1 0.44 because lines sit at similar image places.
  In this design the metric, not the features, limits what the probe can tell apart.
- **Consequence:** the probe question is closed as a TIE under nuScenes divider labels; a third probe on the same labels would not
  help. A sharper test needs either complete line labels (hand-checked frames, or the image-side lane labels of another dataset) or
  a direct BEV test (a stride-4/8 lane branch into the BEV against the reference, judged by the BEV divider metric).
