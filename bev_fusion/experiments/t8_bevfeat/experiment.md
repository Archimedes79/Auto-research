# t8_bevfeat - feature swap: cached BEVFusion camera features (map-trained, stride 8) instead of the DualBEV cache (detection-trained, stride 16)

Step 0 of plan 2.8 (image-encoder chat); runs by the lane owner (`temp/study/lane4.sh`, code pinned at 96a02be, GT signature 15efbde2d1).
Arms: `t8_bevfeat_s0/s1` (model.camera_features = "bevfusion", feature_cache data/cache/cam_feats_bevfusion), controls `t8_ctl_s0` / `t8_ctl_s1` (= t2_ref_s0 / s1 at the same
commit and GT). `t8_ctl_s1` crashed at step 2500 with "CUDA unknown error"; the overnight chain resumed it from checkpoint 2500 on a newer code snapshot
(run `t8.ctl_s1`, commit 9b3febf265, not 96a02be) and it finished at step 3000 on 2026-11-01 00:37. The code diff of that snapshot is not only default-off options
(e.g. `sample_targets` keyed by side, a check in `dt_weight`, edits in `nuscenes_dataset.py` / `feature_cache.py`) and was not shown to be neutral; the agreement with
`t8_ctl_s0` below is the only evidence. Its evaluation is run `t8.ctl_s1.eval` (`python evaluation/bev_suite.py runs/t8_ctl_s1 --workers 4 eval --n 1000` in the gcar env from the bev_fusion folder, started through a one-line wrapper script that was not kept; GT signature 15efbde2d1 like the other arms).
3 camera frames, window memory, radar, 3000 steps, cold VAL 1000 frames.
Numbers: `runs/<name>/eval/table.txt`; front-area objects `results_t8_x50_2ctl.md` (`python evaluation/bev_suite.py front runs/t8_ctl_s0 runs/t8_ctl_s1 runs/t8_bevfeat_s0 runs/t8_bevfeat_s1 --n 1000 --out experiments/t4_soa_front/results_t8_x50_2ctl.md`, 1000 bootstrap draws, the suite's `front` command (`evaluation/front_compare.py`);
the earlier `results_t8_x50.md` has the three arms without the second control, made by the former script with 100 draws and its own random stream: same point numbers, slightly different intervals).

## Result (2026-10-31)

Maps (seg_fov, dt 0, IoU, 1000 VAL frames):

| layer | t8_ctl_s0 | t8_ctl_s1 | t8_bevfeat_s0 | t8_bevfeat_s1 |
|---|---|---|---|---|
| drivable area @0.5 | 0.716 | 0.716 | 0.804 | 0.809 |
| walkway | 0.178 | 0.176 | 0.339 | 0.342 |
| carpark area | 0.115 | 0.106 | 0.173 | 0.168 |
| ped crossing | 0.002 | 0.000 | 0.145 | 0.165 |
| lane divider @0.5 | 0.191 | 0.180 | 0.351 | 0.345 |
| road divider @1 | 0.284 | 0.284 | 0.501 | 0.501 |

Objects, front sector 0-50 m (mAP2 = vehicle + pedestrian, 0.5 / 1 / 2 / 4 m): control s0 0.367, control s1 0.368 (+0.001 [-0.006, +0.007]), bevfeat_s0 0.183 (-0.184 [-0.208, -0.161]), bevfeat_s1 0.180.
Vehicle AP@2 0.670 (s1 0.668) -> 0.455 / 0.447; pedestrian AP@2 0.210 (s1 0.213) -> 0.001 / 0.002 (pedestrians gone). By distance 0-20 / 20-40 / 40-50 m: 0.542 / 0.334 / 0.159 (s1 0.531 / 0.341 / 0.165) -> 0.253 / 0.184 / 0.096.
Full-area detection mAP (all classes, `det/dt+0.00/mean/all/mAP`): control 0.252 / 0.235, bevfeat 0.110 / 0.108. The control's second seed is 0.016 lower there, inside the band 0.054.

Control seed spread (2026-11-01): the six map layers differ by at most 0.010 (lane divider @0.5), front mAP2 by 0.001; the prediction of the evaluation card (drivable 0.716, lane divider 0.191, mAP 0.251, each +-0.03) hit on all three.
The smallest map effect of the swap (carpark area, +0.06) is 6 times the largest control spread of a map layer; the others are +0.09 (drivable) to +0.22 (road divider). The object effect (-0.18) is far outside the 0.001 spread of the controls.

**Reading.** The map-trained stride-8 features nearly double the map layers (two seeds agree; far outside the seed spread) and halve the object quality. This
supports "the frozen detection-trained features limit the maps" (M6) and shows the reverse for objects: BEVFusion's depth is a weak latent assignment
(corr 0.11 with flat-road depth vs 0.82 for DualBEV, plan 2.8), and objects, small ones above all, need depth. Neither feature set alone gives both.
Not measured: speed (the cache excludes the image encoder), per-layer bootstrap for the maps. The second control seed is measured now (above); it trained its last 500 steps on a different code snapshot (see the arms paragraph).

Consequence (proposal, not run): feed both (DualBEV heads for the object path, BEVFusion features for the map path, or concat the two camera BEVs), or end-to-end encoder training with both losses (plan 2.8 step 1).
