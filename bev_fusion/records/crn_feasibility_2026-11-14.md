# CRN on our frames: feasibility (card t26, track B3; read-only investigation 2026-11-14, nothing built, run or installed beyond `--help`)

**Verdict: yes, on a GPU, with no download and no install, on 1620 of our VAL frames (43 of the 82 scenes), not on the 1000 frames of `--n 1000`.** Evidence: `~/work/CRN` (git 5e9d2fa, env `crn`: python 3.8, torch 1.9.1+cu111, mmcv 1.6.0, mmdet 2.25.2, pytorch-lightning 1.6.0; `--help` of the R50 exp runs), both checkpoints on disk (`ckpt/CRN_r50_256x704_128x128_4key.pth` 236 MB, `..._r18_...` 144 MB), CRN's CUDA ops built (`ops/*/*.so`).

## What it needs, and what is already there
| need | state |
|---|---|
| info files (CRN format: 6 cameras, key-frame history, lidar calibration only) | `~/data/nuscenes_full/nuscenes_infos_val_crn.pkl` (made by the lane project 2026-10-12 with CRN's `gen_info.py`): **1706 samples = exactly the 1706 GaussianCaR frames, 43 scenes, all inside our 82 val scenes**. `nuscenes_infos_val.pkl` next to it is another (older) format: the CRN exp reads that name, so it needs a data dir whose `nuscenes_infos_val.pkl` points at the `_crn` file |
| radar | `radar_pv_filter/<cam image>.bin` (CRN's `gen_radar_pv.py` output) exists for all six cameras of all 1706 samples (60,720 files = 10,120 samples: the lane project's train 8414 + val 1706); format = pixel x / y, depth, rcs, vx, vy, sweep index per point, 8 sweeps accumulated. We have the radar sweeps and the lidar calibration records, so more samples could be made (`gen_radar_bev.py` + `gen_radar_pv.py`, CPU), but no lidar files |
| images | `samples/CAM_*` of all six cameras exist for our 82 val scenes (parts 1 - 6); the 4 key frames (now, -1 s, -2 s, -3 s: `key_idxes [-2, -4, -6]`) are key frames, no camera sweeps (infos hold none) |
| depth GT | not on disk (no lidar), and not needed: the exp sets `return_depth = True`, the dataset then reads `depth_gt/*.bin`; set it `False` in a launcher subclass (the batch keeps its slot, `eval_step` ignores it) |
| data root | `~/work/CRN` has no `data/`: a folder `data/nuScenes` of symlinks (images, maps, v1.0-trainval, the two radar folders, the info file under the name CRN reads). A change in the third-party folder, no install |
| GPU, not CPU | the voxel-pooling ops are CUDA-only and the geometry is cast `.half()`: no CPU path. RTX 3060 12 GB, 2.5 GB held by the desktop. Memory at batch 1 not measured (24 images per sample, fp16): expected 3 - 5 GB, measured in the smoke |
| output | `DetNuscEvaluator._format_bbox` writes the nuScenes json (global boxes, 10 classes, top 500 per sample). Our `front_compare.py` takes a pickle `{"det": {scene: {0.0: [(dets, gt boxes) per frame]}}, "meta": {frames, gt_signature}}` as `bev_suite.py front ref ... LABEL=FILE.pkl` (the format of `soa_front_v8.py`, DualBEV) |

## Steps (time)
1. Launcher outside the CRN repo: sets `return_depth = False`, the info name, `torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False`, `-b 1 --gpus 1`; dumps raw predictions per sample (not only the json). 0.5 h coder.
2. Smoke: 20 frames on the GPU, memory, ms per frame, boxes drawn over our GT by eye. 0.2 h GPU.
3. Converter (global json -> our dicts): global -> ego frame of the CAM_FRONT time through `meta["T_e2g"]` of our GT (the GT boxes are at camera time, CRN's at the lidar key-frame time: move each box by its own velocity x the gap, as the GT does), yaw and velocity rotated; classes car / truck / bus / trailer / construction_vehicle -> `vehicle`, pedestrian -> `pedestrian` (two-wheelers kept, barrier and cone dropped); GT boxes by `gt_boxes_from_meta` of the stored GT; pickle meta with the frame list and the GT signature. 1.5 h coder, a test with a planted box.
4. Full run on the 1706 frames (the 1620 samples with a previous and a next key frame score): about 5 - 8 min (GPU 2 - 4 min: the paper's 22.7 FPS is an RTX 3090; the JPEG decoding of 24 images per sample is the other half).
5. Our network on the SAME 1620 frames (cold, like `soa_maps.py`): the frame list is the one thing the rows must share; `bev_suite.py eval --n 1000` stores only its 1000 frames, of which **521 are in the 43 scenes**.
6. `bev_suite.py front`: vehicle and pedestrian AP at 0.5 / 1 / 2 / 4 m, mAP2, per-range, with the paired scene bootstrap (43 scenes), sector x < 50 m, +-32.3 degrees. The rows CRN can be scored on: exactly those (vehicle + pedestrian x 4 distances); also its velocity at matched boxes (AVE), not the map or velocity-field rows.

## Reported numbers and the reproduction gate
README model zoo for this checkpoint (val, 150 scenes, 10 classes, 360 degrees, 6 cameras, 4 key frames, no CBGS): **R50 NDS 56.2, mAP 47.3**, 22.7 FPS, 61.4 M parameters; R18 54.2 / 44.9. (The paper's 56.0 / 49.0 in `results/build/inputs/soa_digest.md` is another training; the gate target is the README.) **The card's gate "reproduce the repository's mAP to 0.01" cannot be met as written**: the official devkit evaluation asserts the full 150-scene val set, we hold images and radar for 43 of its scenes (1706 samples). Honest gate: the same devkit code on the 1706 samples (sample list filtered), mAP within +-0.05 of 47.3 (the spread of a 43-scene subset), and the boxes on our frames look right. A different, stronger check that needs no published number: the same sample run twice, TF32 on and off, centres within 0.1 m.

## Risks
- **TF32**: torch 1.9.1 has matmul and cuDNN TF32 ON by default (read in the env), as `bev_fusion_lane_center/CLAUDE.md` warns. CRN's geometry is cast to double and its convs run in fp16 (`precision=16`), so the effect should be small; switched off in the launcher and measured on 50 frames both ways.
- Frame set: the references see 1620 frames of 43 scenes, our `--n 1000` set has 521 of them; every row of the SOA table must be re-scored on the common frames (the maps scorer takes `--frames-from`). 43 scenes make the bootstrap intervals wide (the card's 0.030 margin should be read against them).
- CRN's labels differ from ours (nuScenes boxes against our care boxes in the fusion area, GT at camera time); the card already says so.
- Old runs (`t20_base_s0`, `t16_full_s0`) import `autogt`, which no longer exists (D78 / D81): scoring the base on GT v13 needs a re-pin or an alias in the process that runs it (a scratch launcher exists: `temp/t26_b/with_autogt_alias.py`); not a CRN question but it blocks `t26.base_eval`.

## What needs the owner's OK
Nothing in the CRN run itself: no download, no install, no GPU job of the card's training. For the lead to decide: (a) the symlink folder inside `~/work/CRN`, (b) the GPU slot (about 10 min, 4 GB), (c) only if the 39 scenes without CRN inputs should be covered: generating infos and radar PV for them (derived data, CPU, an hour or two; no new download).

Fair alternatives if this is not wanted: CRN-R18 (checkpoint on disk) as a lighter second row; DualBEV front-only is the camera reference already stored (GT v8; its detections do not depend on the GT, only the GT boxes it is paired with).
