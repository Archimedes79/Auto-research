# t12 motion block on a frozen base - does a two-frame + Doppler velocity field beat the base's own velocity? (card written 2026-11-01, before the runs; Mirko: "use an existing training and just train this block if it improves anything")

Context: plan 7.4 (fused flow pyramid, `tests/test_flow_fused.py`): on GROUND-TRUTH maps the pyramid without correlation reached
0.93 / 0.99 m/s on moving vehicles (2 seeds, 48 held-out frames), the real radar Doppler halved the radial error. In the model today
every velocity is a per-cell 1 x 1 guess from one frame: the window head of the camera temporal fusion, the time mover's velocity
head (`TimeMover.motion`, "B") and the object head's box velocity ("C"); t11 found B / C nearly dead in `t10_dual_s0`. The open
question of the design review (2026-11-01): does the block still work on the model's own BEV features?

## 1 Card

- **Question.** With `t10_dual_s0` frozen (EMA weights, step 3000), does the motion block (`training/motion_block.py`: camera BEV of
  k + the oldest window frame (about 0.33 s) ego-aligned + the fixed radar raster with the linear Doppler, 3 levels 2 / 1 / 0.5 m,
  294 k parameters), trained alone, give a better velocity field than the base's velocity head on the same VAL frames?
- **Hypothesis.** Yes, clearly: the frozen camera BEV carries enough of the object's position at two times for the backward warp, and
  the radar Doppler gives the radial part; the field is worse than on GT maps (blurred features, 0.33 s instead of 0.5 s).
- **Design.** `python training/train_motion_block.py --base runs/t10_dual_s0 --ckpt ckpt_00003000.pt --out runs/t12_mb_s<seed>
  --seed <seed> --steps 3000 --val 500`. Only the block trains (AdamW 1e-3, warm-up 100, cosine, clip 5, batch 3); its velocity
  replaces `TimeMover.motion` (it moves the trunk state to -0.5 / +0.5 s for the timed heads; the outputs at dt 0 do not change).
  Loss: the base's loss with its frozen Kendall weights (motion term + the maps / boxes at +-0.5 s; the other terms carry no
  gradient) + the motion term on the two coarser levels with the motion group's weight. Cold keyframes (no world memory carried),
  the same for both arms. Arms: **base** (the frozen model as it is) and **block**, evaluated on the same 500 VAL samples
  (`val_subset`, spread over all val scenes). Seeds 0 and 1 of the block on the same base.
- **Controls.** v = 0 (null; the GT speed at the box centres); the base's own field B and box velocity C (matched reference, same
  frames); strong simple baseline: the box velocity of `t11_vel_hurdle_acc2` (arm B2, the run that revived the velocity branch: vehicle AVE moving 2.86 m/s on cold VAL 1000; the raw `vel` map at the GT centres, without its decode gate), read by the same
  script with `--steps 0` on its checkpoint; seed twin (s1). Layer 0 on the real features (in the base evaluation): the older frame
  backward-warped with the GT velocity must match the current frame better than with v = 0 on moving object cells (`epe_warp_gt` <
  `epe_warp_zero`), else the inputs or the sign are wrong and nothing else is read.
- **Metrics** (VAL, dt 0, `metrics.json`): EPE (m/s) on supervised object cells moving > 1 m/s (headline), moving vehicles, static,
  all; at the GT box centres of moving objects: the field vs the object head's box velocity vs v = 0; the base loss groups and
  `total_main` (the +-0.5 s maps / boxes moved with the new field); block ms per frame (batch 1, with the GPU memory in use at that
  moment) and peak GPU memory.
- **Prediction.** Moving-object EPE: base about the v = 0 level (dry run on 6 samples: 5.5 m/s), block <= 2.0 m/s on both seeds;
  at the box centres the block's field <= the object head of `t11_vel_hurdle_acc2` + 0.1 m/s; `total_main` not worse than the base by more
  than 0.5 %.
- **Decision rule** (band = max(0.1 m/s, 2 x |s0 - s1|) on the headline EPE):
  - KEEP (write the joint-training card: block inside the model, trained with it, under run.py, 2 seeds) if the block's moving EPE is
    below the base's by more than the band on BOTH seeds AND `total_main` is not worse by more than 0.5 %;
  - the centre comparison with `t11_vel_hurdle_acc2` decides whether the box velocity should be read from the field (field <= hurdle + band)
    or stay the object head's own (a second, separate decision);
  - inside the band: TIE (the base is cheaper: no block); worse: CUT, and the cause is looked for in the inputs (layer 0) before any
    other change.
- **Kill.** Layer 0 fails (warp with GT v not better than v = 0), or the loss is non-finite in more than 1 % of the steps.
- **Budget.** About 0.13 s per step on a free GPU (dry run): 3000 steps < 10 min + two evaluations of 500 samples; one GPU slot.
- **What it can and cannot show.** A win is a LOWER bound (the frozen features were never trained to carry motion; joint training can
  only add). A loss is not final (it can be the features). Not tested: the memory gate and the window velocity of the camera temporal
  fusion (the frozen fusion was trained with its own gate). Cold keyframes only.

## 2 Summary (2026-11-01, runs t12.mb_s0, t12.mb_s1, t12.hurdle_ctl; tags bf/t12.*)

VAL 500 samples (pinned GT 0.5_4_15efbde2d1, cold keyframes, dt 0), the same frames for every arm; numbers from `runs/t12_*/metrics.json`.

| EPE m/s (VAL 500) | v = 0 | base t10_dual_s0 (B / C) | hurdle_acc2 object head | block s0 | block s1 |
|---|---|---|---|---|---|
| moving object cells (headline, n 37901) | - | 6.674 | - | 3.319 | 3.318 |
| moving vehicle cells (n 33564) | - | 6.907 | - | 3.405 | 3.402 |
| static cells (n 111981) | 0 | 0.055 | - | 0.077 | 0.080 |
| moving box centres (n 1594) | 4.042 | 4.042 / 4.042 | 2.426 | 2.255 | 2.286 |
| total_main (+-0.5 s maps / boxes) | - | 11.507 | - | 10.785 | 10.792 |

- **Layer 0** passed: warp with GT v 0.410 < v = 0 0.566 on moving cells.
- **Verdict by the rule: KEEP.** Band = max(0.1, 2 x 0.0015) = 0.10; both seeds are 3.35 m/s below the base; total_main 6 % better (the moved +-0.5 s objects fit better), not worse.
- **Second decision (box velocity):** the field at the box centres beats the revived object head of t11_vel_hurdle_acc2 by 0.14 / 0.17 m/s (> band): read the box velocity from the field in the joint design.
- **Prediction:** moving EPE <= 2.0 missed (3.32; on GT maps the pyramid reached 0.93: the frozen features blur the motion); centre and total_main predictions hit.
- **Honest negatives:** static cells slightly worse (0.055 -> 0.08 m/s); the base's own velocity is dead (= v = 0), so the base is a weak reference; the strong one is the hurdle object head.
- **Cost:** 294 k parameters, 17 ms per frame at batch 1 (s1; s0 measured 58 ms while the GPU was shared: not an idle-GPU number), peak 2.2-2.6 GB.
- **Next:** the joint-training card (block inside the model, trained with it under run.py, 2 seeds), with the box velocity read from the field and the block's runtime cut (17 ms is too slow for 0.3 M parameters).
