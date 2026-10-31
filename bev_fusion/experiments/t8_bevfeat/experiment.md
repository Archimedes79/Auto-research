# t8_bevfeat - feature swap: cached BEVFusion camera features (map-trained, stride 8) instead of the DualBEV cache (detection-trained, stride 16)

Step 0 of plan 2.8 (image-encoder chat); runs by the lane owner (`temp/study/lane4.sh`, code pinned at 96a02be, GT signature 15efbde2d1).
Arms: `t8_bevfeat_s0/s1` (model.camera_features = "bevfusion", feature_cache data/cache/cam_feats_bevfusion), control `t8_ctl_s0` (= t2_ref_s0 at the same
commit and GT; `t8_ctl_s1` crashed at step ~2500 with "CUDA unknown error", not yet resumed). 3 camera frames, window memory, radar, 3000 steps, cold VAL 1000 frames.
Numbers: `runs/<name>/eval/table.txt`; front-area objects `results_t8_x50.md` (made by `temp/t4/compare_front.py`, 60 bootstrap draws).

## Result (2026-10-31)

Maps (seg_fov, dt 0, IoU, 1000 VAL frames):

| layer | t8_ctl_s0 | t8_bevfeat_s0 | t8_bevfeat_s1 |
|---|---|---|---|
| drivable area @0.5 | 0.716 | 0.804 | 0.809 |
| walkway | 0.178 | 0.339 | 0.342 |
| carpark area | 0.115 | 0.173 | 0.168 |
| ped crossing | 0.002 | 0.145 | 0.165 |
| lane divider @0.5 | 0.191 | 0.351 | 0.345 |
| road divider @1 | 0.284 | 0.501 | 0.501 |

Objects, front sector 0-50 m (mAP2 = vehicle + pedestrian, 0.5 / 1 / 2 / 4 m): control 0.367, bevfeat_s0 0.183 (-0.184 [-0.208, -0.162]), bevfeat_s1 0.180.
Vehicle AP@2 0.670 -> 0.455 / 0.447; pedestrian AP@2 0.210 -> 0.001 / 0.002 (pedestrians gone). By distance 0-20 / 20-40 / 40-50 m: 0.542 / 0.334 / 0.159 -> 0.253 / 0.184 / 0.096.

**Reading.** The map-trained stride-8 features nearly double the map layers (two seeds agree; far outside the seed spread) and halve the object quality. This
supports "the frozen detection-trained features limit the maps" (M6) and shows the reverse for objects: BEVFusion's depth is a weak latent assignment
(corr 0.11 with flat-road depth vs 0.82 for DualBEV, plan 2.8), and objects, small ones above all, need depth. Neither feature set alone gives both.
Not measured: speed (the cache excludes the image encoder), per-layer bootstrap for the maps, the second control seed.

Consequence (proposal, not run): feed both (DualBEV heads for the object path, BEVFusion features for the map path, or concat the two camera BEVs), or end-to-end encoder training with both losses (plan 2.8 step 1).
