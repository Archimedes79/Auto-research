# t17 more data - does adding the 272 train scenes of nuScenes parts 7 - 10 to the full network's recipe improve it? (card written 2026-11-02, before the run; D46 plan: a good working system by the afternoon of 2026-11-03)

## 0 Statement, referee, status
- **Statement.** The full network `t16_full` (card `../t16_full_network/experiment.md`) trains 6000 steps of 18 keyframes = about 108 k samples on 428 train scenes (about 17 k keyframes): about 6 passes over the data. If the train-to-val gap of its evaluation is large, more distinct scenes are the cheapest lever; this arm changes ONE thing: 700 train scenes (the 428 + the 272 train scenes of parts 7 - 10 that are in the trainval train split and whose DualBEV shards exist), everything else is t16's. The 82 val scenes stay, so the arm is read against `t16.full_s0` on the same 1000 frames.
- **Why now.** The BEVFusion image-side features exist for parts 1 - 6 only; building them for the 272 scenes costs 27 minutes of GPU (6 scenes took 36 s on 2026-11-02), the DualBEV shards, the radar cache and the v8 GT (all of trainval) exist.
- **Status.** Config `training/configs/t17_data.json` written (700 / 82 scenes, caches in `data/cache/all/`: DualBEV shards of all 850 scenes linked, BEVFusion shards built by `temp/t16/prep_more.sh extract` as GPU job `t17.feat`); static check passes; the extraction, the run `t17.data` and its evaluation are queued behind the evaluation of the baseline. Whether it stays in the queue is decided on the `generalisation` table of `t16.full_s0` (small train-to-val gap = no gain expected: the arm is cancelled and the GPU goes to the control first).
- **TEST is untouched** (VAL only).

## 1 Card
- **Mechanism.** About 6 epochs over 17 k keyframes overfit the scenes (the 400-step run on 8 scenes overfits after 100 steps); 64 % more scenes means fewer repeats of the same frames and more road geometry, lighting and traffic. The maps, the lines and the object heads gain; the velocity field (supervised on moving objects only) gains least.
- **Prediction (before the run; `t17.data` against `t16.full_s0`, one seed each, VAL 1000 frames, GT v8; guesses, no stored number exists).** The six map IoUs +0.01 ... +0.04, front mAP2 +0.02, vehicle AP@2 +0.01; the train-minus-val gap of `bev_suite.py generalisation` smaller by at least 20 %; velocity EPE (`vel/dt+0.00/cells/vehicle_moving/epe`) within +-0.3 m/s; ms per frame and parameters identical (data only); step time about equal (the GT cache of the new scenes is cold in the first pass: about 40 minutes more).
- **Decision rule (written before the run; single seed each, so by hand: composed from the table rows, never KEEP).** ADOPT as the delivered system if at least four of the six map IoUs and front mAP2 are better than `t16.full_s0` by more than their band (0.010; 0.020 for walkway / lane divider as in the t11 card) AND the paired scene-bootstrap interval of the mean over the maps excludes 0 (`bev_suite.py compare ... --curves --fusion`) AND no guard is worse than its band: vehicle moving EPE +0.3 m/s, static EPE +0.1 m/s, front mAP2 -0.010. TIE (inside the bands): keep `t16.full_s0` (cheaper to repeat, fewer files). WORSE on two or more guards: keep `t16.full_s0`, record the negative.
- **Design.** One arm, seed 0, same recipe and budget as `t16.full_s0` (6000 steps, batch 3 x accumulation 2, `grad_clip_split` ["motion_block"], probe on); control = `t16.full_s0` (same code: the snapshot of both is compared by `diff` of `runs/<name>/code`); no seed twin before tomorrow afternoon (named as the first open item).
- **Cost.** 27 min extraction + about 7 h training + 50 min evaluation; disk 41 GB on D: for the new BEVFusion shards (D: has 396 GB free).
- **Kill criterion.** Non-finite loss in 3 rows in a row, stalled > 15 min, `t_data` above 30 % of a step after step 1500 (the cold GT of 272 scenes: 10.9 k samples at 0.57 s per sample and worker, about 10 samples per second with 6 workers, 2.3 needed).

## 2 Summary (written after the run)
**Verdict by the written rule (by hand: one seed per arm): no ADOPT, TIE / negative, `t16.full_s0` stays the base.** Run `t17.data` (ledger row t17.data, evaluation t17.data.eval, 1000 val frames, GT v8, paired scene bootstrap `bev_suite.py compare runs/t16_full_s0 runs/t17_data --curves --fusion --velocity --n 1000`, output `temp/t16/compare_t17.txt`, rule applied by `temp/t16/rule_t17.py`):

| VAL 1000, higher is better unless noted | t16.full_s0 | t17.data | b - a | 95 % interval | reading |
|---|---|---|---|---|---|
| drivable area IoU | 0.7791 | 0.7670 | -0.0121 | [-0.0202, -0.0047] | worse |
| walkway IoU | 0.4135 | 0.4024 | -0.0111 | [-0.0247, +0.0022] | inside band |
| car park IoU | 0.2736 | 0.2392 | -0.0344 | | worse |
| pedestrian crossing IoU | 0.2561 | 0.2569 | +0.0008 | | inside band |
| lane divider IoU@0.5 | 0.4290 | 0.4299 | +0.0009 | [-0.0148, +0.0153] | inside band |
| road divider IoU@1 | 0.5722 | 0.5598 | -0.0124 | | worse |
| mAP (fusion area) | 0.4158 | 0.4147 | -0.0011 | [-0.0112, +0.0085] | tie |
| vehicle AP@2 (fusion area) | 0.7994 | 0.7879 | -0.0115 | [-0.0207, -0.0016] | worse |
| pedestrian AP@2 (fusion area) | 0.3757 | 0.3992 | +0.0235 | [+0.0078, +0.0410] | better |
| vehicle moving centre EPE, m/s (lower is better) | 1.112 | 1.069 | -0.043 | [-0.066, -0.020] | better |
| static cells EPE, m/s (lower is better) | 0.189 | 0.216 | +0.027 | [+0.012, +0.040] | worse, inside the guard (+0.1) |
| train-to-val gap, mean of the six map IoUs | +0.0906 | +0.0926 | +2 % | | prediction (-20 % or less) missed |

- **Rule.** At least four of six map IoUs and the mAP better than their bands: 0 of 6, mAP no. No guard (vehicle moving EPE +0.3, static EPE +0.1, mAP -0.010) is worse than its band. So the cell "inside the bands: keep `t16.full_s0`" applies; three maps are worse than their band, which is recorded as the negative.
- **Prediction.** The six map IoUs +0.01 ... +0.04: missed on all six (three at -0.012 ... -0.034). Front mAP2 +0.02: missed (-0.001). Vehicle AP@2 +0.01: missed (-0.012). Gap smaller by 20 %: missed (+2 %). Vehicle moving EPE within +-0.3 m/s: held. Parameters and ms per frame identical (data only).
- **Reading.** The mechanism of the card (the epochs over 17 k keyframes overfit the scenes) does not show: the train-to-val gap stays +0.09 (the car park +0.19 and the lane divider +0.10 are the largest), although every frame was seen about 4 times instead of about 6 (same 6000 steps, 64 % more frames). The gap is therefore not a data-quantity problem at this budget; it looks like a place problem (the same streets in train, other streets in val) or a limit of the cached camera features. Pedestrian AP and the moving-object velocity improved (both intervals exclude 0), the drivable area and vehicle AP@2 worsened (also excluding 0): a trade, not a gain. One seed per arm and an interrupted run (below): a seed twin would be needed to call the small differences anything but a tie.
- **Cost.** 27 min extraction, 6 h 52 min training (incl. the interruption), 29 min evaluation; 41 GB of BEVFusion shards on D:.

Run log (2026-11-02): the training started 12:48; at step 2600 (15:42) the trainer died with a CUDA `unknown error` (it hung ~2 min in D state, then the GPU was free again); `run.py` repeated the run by its retry rule (attempt 2 at 15:45) and the trainer resumed from `ckpt_00002500.pt` at step 2500. The run is therefore one interrupted-and-resumed run: the batches after step 2500 are not the ones an uninterrupted run would have seen (same recipe, different order), a small extra source of variation next to the single seed. The same code snapshot (pin 40e7ff5b3c) ran both attempts.
