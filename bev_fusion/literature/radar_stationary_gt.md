# Radar stationary-structure target: how others supervise it, which GT can be reused, decision table

Scout: 2026-10-26, scan mode, for plan.md 8.4 (row "radar aux head", item (b) STATIONARY STRUCTURE map; decision (3) of that section). Notes of the new papers are in `literature/papers/`, rows in `literature/scout_tables/papers_new_stationary_gt.csv`.

Fact marks: `[c Sx]` checked (page read through a page summariser on 2026-10-26, source in the key at the end), `[d Sx]` digest (search snippet or a second-hand summary), `[m]` memory, `[calc]` computed from checked numbers, `[est]` estimate. Quotes are at most one sentence.

## 0. Answer in eight lines
1. No camera-radar fusion network supervises the radar's stationary part with its own loss. CRN, HyDRa, RCBEVDet++, BEVCar, GaussianCaR, REOcc, MetaOcc, Doracamom, OccFusion, CaR1: the radar enters through the shared detection / map / occupancy heads only (section 1). Radar-specific occupancy losses exist only in stand-alone radar papers (Sless et al. on nuScenes, deep radar ISM, RadarOcc on K-Radar, 4DRadarRBD), and there the labels are built from lidar (section 2).
2. The static world of the map-segmentation fusion networks comes from the nuScenes map expansion (drivable area, ped crossing, walkway, stop line, carpark area, road / lane divider [c S4, S11]); the occupancy fusion networks use Occ3D or SurroundOcc voxel labels (lidar-derived [c S2, S6, S9]). Neither contains a radar-visible structure layer.
3. Best reusable released GT without the raw lidar: Occ3D-nuScenes (MIT, 0.4 m, +-40 m, 2 Hz keyframes of all 850 trainval scenes, lidar visibility mask; trainval folder about 32 GB incl. images). It stops at x = 40 m, so our grid (x 0..60 m) is labelled only on 66.7 % of its area. SurroundOcc (0.5 m, +-50 m, same cell size as ours, 83.3 %) is the better geometric fit but sits on Baidu Pan.
4. Static classes are the ones radar helps most in the only controlled test found: on SurroundOcc labels adding radar to a camera model raises IoU of manmade 11.49 -> 16.16, vegetation 18.47 -> 21.81, barrier 18.39 -> 20.46 [c S9]. The same paper finds radar HURTS on Occ3D and blames the 40 m label extent and the camera visibility mask [c S9].
5. Nothing found validates a drivable-area-edge band as a radar target. The nuScenes map has no kerb / guard-rail / barrier / fence / wall layer [c S12]; the road-boundary radar papers use lidar-built labels [c S25].
6. Recommendation: D as the control arm, C (the proposal in 8.4) as the free first arm, but first measure it (section 4, test T0, no download); then ask for the Occ3D label download (A) as the second arm; B (80.2 GB lidar) only if A pays and the 40-60 m ring is the point.
7. All sets are derived from nuScenes lidar and carry the nuScenes non-commercial terms [c S28 for the nuScenes rule; c S14 for the Occ3D statement].
8. Not verified: gts-only download size of Occ3D, frame / timestamp convention of the labels, SurroundOcc size, the share of static radar returns that each proxy explains (section 4).

## 1. How the radar's stationary / occupancy part is supervised

| network (id) | radar-specific stationary / occupancy loss? | GT of the STATIC world | evidence |
|---|---|---|---|
| CRN 2304.00670 | no: radar occupancy is a sigmoid head inside the view transform; no occupancy loss; total = detection + depth (lidar depth) | none (detection only) | paper Sec. 3.4 [c S1]; code: `training_step` adds only `loss_detection + loss_depth` [c local `literature/code/crn_fusion.md`] |
| HyDRa 2403.07746 | no: radar guidance attention is an unsupervised 3x3 conv + sigmoid; consistency weight is an attention weight | Occ3D-nuScenes labels for the shared occupancy head (FB-OCC style); no loss formulas in the text | [c S2] |
| RCBEVDet++ 2409.04979 | no: focal loss on the segmentation head | classes vehicle, drivable area, lane; label source not named on the page | [c S3] |
| BEVCar 2403.11761 | no: BCE for vehicles, alpha-balanced focal (0.25, gamma 3) for map classes | nuScenes map annotations: drivable, carpark, ped crossing, walkway, stop line, road divider, lane divider; no visibility filter mentioned | [c S4] |
| GaussianCaR 2602.08784 | no: BCE + Dice + centerness + offset, plus an auxiliary copy at the first fusion stage | drivable area, lane divider; source not named | [c S5] |
| CaR1 2509.10139 | no: vehicle masks only | none | [c S10] |
| REOcc 2511.06666 | no: densifier is a fixed formula, amplifier trained end-to-end | Occ3D-nuScenes | [c S6] |
| MetaOcc 2501.15384 | no: "radar branch receives no direct supervision"; CE + Lovasz + two affinity terms | OmniHD-Scenes (manual annotation of lidar points), SurroundOcc-nuScenes | [c S7] |
| Doracamom 2501.15394 | no loss on the radar encoder; two auxiliary heads on SHARED features: BEV foreground mask from projected boxes (BCE + Dice) and a binary occupied map from the semantic occupancy (BCE + Dice) | OmniHD-Scenes occupancy labels | [c S8] |
| OccFusion 2403.01644 | no: focal + Lovasz + scene-class affinity | SurroundOcc and Occ3D | [c S9] |
| RaCFormer 2412.12725, RC-BEVFusion 2305.15883 | no radar loss (detection) | none | local notes |
| RadarDistill 2403.05061 | YES but feature-level: lidar-teacher activation / proposal distillation onto the radar BEV (the one radar-specific supervision on nuScenes among the nets here) | none | local note [c], needs a lidar network and the lidar data |
| stand-alone radar occupancy: Sless 1904.00415 (nuScenes), deep radar ISM 2305.12409 (own SRR520), RadarOcc 2405.14014 (K-Radar), 2112.04282 (Oxford scanning radar), 4DRadarRBD 2503.01930 (own 4D radar) | YES: own loss, Lovasz / Dice / CE, free-occupied-unknown classes | lidar-built (section 2a) | local notes [c], S22, S23, S25 |

Reading: Option D is the field's practice for fusion networks. The two things that come closest to our idea are Doracamom's auxiliary "occupied" map (BCE + Dice from the dense labels, on shared features) and the stand-alone radar occupancy papers (own loss, lidar labels, explicit "unknown").

## 2. GT recipes and released sets

### 2a. Recipes (how the lidar-built labels are made)

| recipe | accumulation | dynamic objects | free / unknown | meshing / filling | radar-specific step |
|---|---|---|---|---|---|
| Sless 1904.00415 | lidar over the whole scene in global coordinates, projected into each radar frame | radar dynamic clusters removed with the velocity; lidar handling not stated | ray tracing from the radar: free before the first return, occupied run, unobserved behind; concave hull / FOV set to ignore | dilation, hole filling, erosion | one network for all five radars |
| deep radar ISM 2305.12409 | lidar measurement grids fused over time into a dynamic grid map | oriented boxes, every cell inside = occupied | belief masses free / occupied / unknown | none | occupancy removed where NO radar detection exists in the partly observed zone, plus a dilation near detections [c S22] |
| RadarOcc 2405.14014 | consecutive sweeps superimposed (count not stated), background in a world frame | boxes + track ids, object points in box frames, majority vote | not described | none | only the front part of the sweep, 24 clear-weather sequences [c S23] |
| Occ3D [c S13] | static points in the world frame, dynamic points in box frames; KNN labels for non-keyframes | box based | ray casting from lidar origins (mask_lidar) and camera rays (mask_camera) | VDBFusion TSDF mesh, ground fitted locally, image-guided refinement | none |
| SurroundOcc [c S16] | same split, lidar at 20 Hz (about 13 M points per scene) | box based | no visibility mask released | Poisson mesh, voxelised, nearest-neighbour semantics | none |
| OpenOccupancy / AAP [c S17] | same superposition | box based | none | pseudo-label fill of empty voxels, then about 4000 human hours of purification | none |
| OpenOcc [c S18] | background global, objects in object frames, majority vote | box velocity gives flow (moving above 0.2 m/s) | camera invisibility by ray tracing | hole filling | none |

Take-away: every recipe splits static from dynamic BEFORE accumulating, so a stationary layer is a by-product; the only recipes that adapt the label to the radar's visibility are deep radar ISM (remove occupancy the radar cannot see) and Sless (an "unobserved" class, ignore masks).

### 2b. Released nuScenes occupancy sets (reuse without the raw lidar)

| set | voxel / range / classes | frames, scenes | format, size, hosting | licence | frame |
|---|---|---|---|---|---|
| Occ3D-nuScenes (2304.14365) | 0.4 m; x, y [-40, 40] m, z [-1, 5.4] m; 200 x 200 x 16; ids 0-16 as lidarseg incl. barrier (1), driveable_surface (11), other_flat (12), sidewalk (13), terrain (14), manmade (15), vegetation (16), 17 = free [c S14; id-to-name order m] | keyframes 2 Hz [c S13]: 28,130 train + 6,019 val + 6,008 test frames [c S14] = the nuScenes train / val keyframes of the 850 trainval scenes [calc, with S17] | per frame `gts/<scene>/<sample_token>/labels.npz` (`semantics`, `mask_lidar`, `mask_camera`) + `annotations.json`; trainval folder about 32 GB incl. imgs, mini 440 MB, test 6 GB; gts alone NOT stated; Google Drive, Baidu, OpenDataLab [c S14] | MIT (data + generation code) plus nuScenes terms [c S15] | ego frame [d, search snippet of a README mirror]; timestamp not stated |
| SurroundOcc (2303.09551) | 0.5 m; x, y [-50, 50] m, z [-5, 3] m; 200 x 200 x 16; 16 classes + empty [c S16] | train / val of nuScenes (counts not stated) | Baidu Pan for data (README of 2026-10-26; an older README mirror listed Tsinghua Cloud [d]); size not stated; mesh vertices released | code Apache-2.0; data terms not stated [c S16] | not stated (OpenOcc sibling: lidar coordinates [c S18]) |
| OpenOccupancy / nuScenes-Occupancy (2303.03991) | 0.2 m; x, y [-51.2, 51.2] m, z [-5, 3] m; 512 x 512 x 40; 16 classes + empty [c S17] | 850 scenes, 34 K keyframes (28,130 / 6,019) [c S17] | `nuScenes-Occupancy-v0.1.7z` about 5 GB, about 130 GB unpacked; Google Drive, Baidu [c S17] | code Apache-2.0; data terms not stated | lidar coordinates [m: the z range suggests it] |
| OpenOcc v2.1 (2306.02851, OccNet) | repo: 0.4 m, same range as Occ3D, 17 classes, `semantics` + `flow` [c S18]; the paper describes 0.5 m, +-50 m [c S18] | 28,130 / 6,019 / 6,008 frames | `openocc_v2.1.zip` + `infos.zip`, OpenDataLab / Google Drive; size not stated [c S18] | CC BY-NC-SA 4.0 [c S18] | lidar coordinates in the paper version |
| UniOcc mirror (2503.24381) | the three variants above, 2 Hz, with voxel flow | as the originals | Hugging Face `tasl-lab/uniocc`; 776 GB total, per-folder sizes not shown [c S20] | card: MIT (toolkit); nuScenes terms apply | not stated |
| Cam4DOcc (2311.17663) | not a new set: movable = boxes, static (GSO) = nuScenes-Occupancy [c S19] | | | MIT (code) | |

Not reusable for this: nuScenes-lidarseg / Panoptic nuScenes (per-point labels, need the lidar files); RadarOcc's labels (K-Radar, not nuScenes).

Coverage of OUR grid (x 0..60 m forward, y -26..26 m, 0.5 m cells = 120 x 104 = 12,480 cells) [calc]:

| set | defined x | cells with a label | share |
|---|---|---|---|
| Occ3D | 0..40 m | 80 x 104 = 8,320 | 66.7 % |
| SurroundOcc | 0..50 m | 100 x 104 = 10,400 | 83.3 % |
| OpenOccupancy | 0..51.2 m | about 10,650 | 85.3 % |
| none | 50..60 m | 0 | the far ring stays unlabelled in every released set |

How a released set collapses to our layer (proposal, to be verified on the data): (1) choose structure classes {barrier, manmade, vegetation} (+ "others" optional); radar helps these most [c S9]; leave out ground classes and moving classes; (2) parked vehicles from OUR boxes (speed < 0.5 m/s over the track, already in `training/gt_bev.py` [c S29]) instead of the vehicle voxels, because the set does not separate parked from moving cars (OpenOcc flow could, but only for box classes); (3) height rule per column, not fixed slice numbers: structure voxels from about 0.3 m above the highest ground voxel (classes 11-14) up to about 3 m (radar elevation is unreliable, other nets collapse z; RaCFormer sets radar z = 1 m [c local note]); (4) mask = `mask_lidar` AND inside the set's range AND our radar coverage mask (NOT `mask_camera`); (5) re-pose by the ego pose difference from the keyframe time to the front-camera time (below one cell at 15 m/s for tens of ms [m]); (6) resample 0.4 m -> 0.5 m by area coverage (soft target in [0, 1], matching the BCE on soft targets of 8.4); a static layer is exact at every 12 Hz frame and every dt by re-posing, which no moving-object layer is.

### 2c. Annotation / map proxies (the proposal in 8.4): what the papers say

- The map has no kerb, guard-rail, barrier, fence or wall layer [c S12]; the border of `drivable_area` is a road-boundary prior, and the border to `walkway` / `ped_crossing` is a kerb by inference only.
- Radar road-boundary papers build their labels from lidar [c S25] or from surveyed points [c S26], not from a map; I found no paper that checks a map edge against radar returns, and no per-class share of static radar returns (qualitative only: guard rails and roadside structures give many strong returns [d, search snippets]). A TU Delft thesis (CURBE) could not be read; an earlier snippet claiming it used nuScenes was NOT confirmed (the table of contents mentions Astyx), so it is not used.
- Parked objects: radar sees them (strong reflectors, Doppler zero after ego compensation), and nuScenes marks stationary candidates in `dyn_prop` / `ambig_state` [c local NUSC_RADAR note]; the box footprint is the better-founded half of the proxy.
- Our own registration offset (0.5-1 m in places, plan 9.5) is as large as the proposed 1 m band.
- Scale of the unknown: none of the fusion papers tested a radar-specific static target against a no-target control, so the benefit of ANY radar aux target is unproven; only the shared-head gain on static classes is (OccFusion, section 0 item 4).

## 3. Decision table

| | A: reuse released occupancy GT (Occ3D first) | B: build from lidar | C: proxy from boxes + map (as in 8.4) | D: no radar-specific target |
|---|---|---|---|---|
| gives | lidar-derived structure classes (barrier, manmade, vegetation, terrain), dense, with `mask_lidar`, keyframes of all trainval scenes; up to x = 40 m (SurroundOcc 50 m) | exactly the target we define, up to the lidar's real range (x to 60 m, sparse far), a known / unknown mask by ray casting, own height band, time-exact re-posing | parked / standing object footprints (reliable) + a drivable-area border band (a prior) | shared heads only: areas, lines, objects (as CRN, HyDRa, BEVCar ...); radar learns what helps those |
| download | Occ3D trainval folder about 32 GB incl. images (gts alone unknown, probably a few GB [est]); SurroundOcc on Baidu, size unknown | LIDAR_TOP parts 1-6: 80.2 GB [c S28] (about 3.7 h at the 6 MB/s of six parallel connections [calc from S28]); needs Mirko's OK and space on D: | none | none |
| effort | 0.5-1 day: loader, join by sample token, column collapse, ego re-pose, resample, tests [est] | 3-5 days: accumulate static sweeps with ego poses (nuScenes z drift: [d S18]), remove moving boxes, ground height, 2D histogram, ray cast for unknown, tests; CPU about 20 K keyframes, hours to a day [est] | already specified; 0.5 day for the adequacy test T0 | 0 |
| fit to range (x 0..60) | 66.7 % (Occ3D) / 83.3 % (SurroundOcc); far ring unlabelled | full range (labels sparse beyond about 40 m [m]) | full | full |
| fit to grid | 0.4 m -> 0.5 m resample (Occ3D); 0.5 m aligned (SurroundOcc) | native 0.5 m | native | n/a |
| fit to time | keyframes 2 Hz, re-posed (static); ego-frame vs front-camera-time shift to correct | any time (12 Hz sweeps) | any time | n/a |
| licence | Occ3D MIT + nuScenes non-commercial; OpenOcc CC BY-NC-SA | nuScenes non-commercial (already our case) | same | same |
| risk | frame / timestamp convention unverified; label stops at 40 m (OccFusion: radar looks worse under that label); parked cars not separated; Poisson / TSDF filling in the 25-40 m ring | most effort for a benefit that is unproven; pose z noise; 80 GB on a crowded disk | band may not be what radar sees (guard rails, walls, bushes missing; map offset 0.5-1 m) so the aux head may learn a map prior | no extra supervision for the radar; but no paper shows that an extra target helps either |

## 4. Recommendation and checks

Order of work (nothing needs a download until step 2):
1. **D is always the control arm** of the radar aux head (8.4): same network without the stationary term. Without it no gain can be attributed.
2. **T0, adequacy of C on our own data (no download, 0.5 day)**: for the radar sweeps of the keyframes, keep returns that are Doppler-static (ego-compensated speed below a threshold; the closed formula of 2511.20003 [c local note] or `vx_comp, vy_comp` with `dyn_prop` flags) and report (i) recall: share of static returns within one cell of the proxy mask (stationary boxes + 1 m drivable-area band); (ii) precision: share of proxy cells with at least one static return in 5 accumulated sweeps; (iii) the same two numbers for a null (proxy shifted 5 m sideways) and per range bin (0-20, 20-40, 40-60 m). If recall of C is near the null, the 8.4 target mostly misses what radar sees: go to A. Known answer for the harness: a mask built from the static returns themselves gives recall 1 by construction; the null must give the chance rate.
3. **A as the second arm, after Mirko's OK for the Occ3D download** (ask first whether only the `gts` part and `annotations.json` can be fetched; the folder is about 32 GB with images that we have). Build the collapse with the rules in 2b, join to our 510 scenes by `sample_token` (check: share of our 510 scenes found in `annotations.json`; expect all, inference), mask beyond x = 40 m with weight 0. Spec tests: (a) alignment: the overlap of our stationary-box footprints with the Occ3D vehicle voxels peaks at zero shift when the sweep is moved +-1 m (verifies frame and timestamp, and gives the re-pose); (b) a one-voxel toy map resamples to the analytic coverage (0.16 m^2 of a 0.25 m^2 cell = 0.64 when the voxel lies fully inside); (c) a label of another frame scores chance in the T0 numbers; (d) reproduce the class counts of the data card for mini (404 frames). Then run T0 for A against C.
4. **SurroundOcc** only if the 40 m cut is the problem in the first A run (OccFusion's finding [c S9] says it can be): Baidu Pan access and the size are unknown today.
5. **B** only if A pays and the 40-60 m ring is where the radar must teach (it is also the only way to get a lidar-derived `unknown` mask of our own). The generation recipe is Occ3D / SurroundOcc without meshing: static sweeps in the world frame, boxes removed, height band, 2D histogram, ray casting for unknown. Do not start before steps 2-3 show that a radar aux target helps at all.
6. Idea, not from a paper, untested: a self-supervised static target = static radar returns of OTHER keyframes (not in the input window), re-posed; no lidar, no map, but it only teaches densification.

Decisions for Mirko (to go into DECISIONS.md): (1) OK to download the Occ3D-nuScenes labels (size to be confirmed, at most 32 GB); (2) OK for the 80.2 GB lidar only after step 3; (3) accept C as the first radar-aux arm only together with T0 and the D control.

## Source key
- S1 https://arxiv.org/html/2304.00670 (CRN, Sec. 3.4) ; S2 https://arxiv.org/html/2403.07746 (HyDRa) ; S3 https://arxiv.org/html/2409.04979 (RCBEVDet++) ; S4 https://arxiv.org/html/2403.11761v2 (BEVCar)
- S5 https://arxiv.org/html/2602.08784v2 (GaussianCaR) ; S6 https://arxiv.org/html/2511.06666v1 (REOcc) ; S7 https://arxiv.org/html/2501.15384v3 (MetaOcc) ; S8 https://arxiv.org/html/2501.15394 (Doracamom)
- S9 https://arxiv.org/html/2403.01644v4 (OccFusion, Tables I, II) ; S10 https://arxiv.org/html/2509.10139 (CaR1)
- S11 https://raw.githubusercontent.com/mit-han-lab/bevfusion/main/mmdet3d/datasets/pipelines/loading.py ; S12 https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/map_expansion/map_api.py
- S13 https://arxiv.org/html/2304.14365 (Occ3D) ; S14 https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction (data card) ; S15 https://github.com/Tsinghua-MARS-Lab/Occ3D and https://tsinghua-mars-lab.github.io/Occ3D/
- S16 https://arxiv.org/html/2303.09551 and https://github.com/weiyithu/SurroundOcc/blob/main/README.md ; S17 https://arxiv.org/html/2303.03991 and https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md
- S18 https://arxiv.org/html/2306.02851 and https://github.com/OpenDriveLab/OccNet ; S19 https://arxiv.org/html/2311.17663 and https://github.com/haomo-ai/Cam4DOcc ; S20 https://huggingface.co/datasets/tasl-lab/uniocc and https://uniocc.github.io
- S22 https://arxiv.org/html/2305.12409v3 ; S23 https://arxiv.org/html/2405.14014v4 ; S25 https://arxiv.org/html/2503.01930v1 ; S26 https://arxiv.org/html/2309.08341
- S28 local `datasets/NuScenes/README.md` (lidar parts 1-6: 80.2 GB not downloaded; download speed; non-commercial rule) ; S29 local `bev_fusion/plan.md` section 8.4 and `training/gt_bev.py`
- local notes (read in full): `literature/papers/1904.00415_*`, `2305.12409_*`, `2405.14014_*`, `2112.04282_*`, `2403.05061_*`, `2511.20003_*`, `NUSC_RADAR_*`, `2412.12725_*`, `literature/code/crn_fusion.md`
