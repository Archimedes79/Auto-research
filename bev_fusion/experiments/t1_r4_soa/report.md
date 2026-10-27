# t1_r4_soa - audit, fixes and a second full training of the camera + radar BEV chain (report of 2026-10-27)

All numbers: nuScenes VAL (82 scenes, 3109 keyframes of the parts downloaded so far; the only exception is the train-subset check, labelled TRAIN: 800 samples of 428 scenes), EMA weights, front camera + five radars,
0.5 m grid, 60 m range, one training seed per run, TEST split not used. Sources: `runs/<run>/eval/*.json|txt`, `results/tables_vs_r0.md`,
`results/tables_vs_r3.md` (built by `temp/train/build_tables.py`). Cold = every frame from an empty memory (the table default);
warm = memory carried over the whole scene.

## 1 Summary

- **Does it work?** Yes, as a research system: camera + radar are both used (modality audit), every output is produced at
  t - 0.5 / t / t + 0.5 s, the chain runs in 50.7 ms (+17.2 ms frozen camera heads, RTX 3060, batch 1, fp32 eager).
- **Is it competitive with published systems?** No. Detection mAP 0.309 (3 classes, centre-distance AP, front view <= 60 m),
  drivable-area IoU 0.70, walkway 0.24, lane-divider mask 0.27. Published six-camera systems are far ahead on detection
  (CRN with ResNet-50 camera + radar: 49.0 mAP / 56.0 NDS on val; DualBEV, whose image heads we use frozen: about 38 mAP / 50 NDS
  in a detection ablation, multi-frame) and on map segmentation (BEVFusion camera-only baseline 57.09 mIoU, the mean over six map
  classes, from the table of the official repository https://github.com/mit-han-lab/bevfusion, not in our literature folder; the paper
  itself should be checked before this number is quoted elsewhere). Not like for like (front camera
  only, 428 of 700 training scenes, 3 classes, our own box/IoU definitions), so these are orders of magnitude, not a ranking.
- **The fixes of this round did not improve the system over the first baseline `t1_r0`**: mAP 0.307 -> 0.309 (tie by size; no
  bootstrap was run for mAP), maps lower (mask IoU up to 0.054, zero-crossing F1 up to 0.08, carpark 0.038), vehicle AP@2 m
  -0.016 (paired-scene interval excludes 0). Heading error 1.78 -> 0.64 rad is a gain of the direction head from t1 (r3 had 0.59,
  so r4 is 0.05 rad worse than r3); the velocity error against r3 fell 2.32 -> 1.82 m/s (the r4 effect of the velocity weight) and is
  still poor.
- **Why (hypothesis; train-subset check = TRAIN, 800 samples of 428 scenes, vs VAL 3109 samples of 82 scenes):** detection
  overfits strongly (mAP 0.448 vs 0.309; baseline 0.468 vs 0.307), drivable area and walkway barely (0.711 vs 0.698, 0.282 vs 0.244),
  but lane dividers and carpark clearly (0.346 vs 0.274 and 0.296 vs 0.162 for r4; 0.446 vs 0.327 and 0.370 vs 0.201 for r0). So a
  data-amount limit is plausible for detection and for the thin / rare map classes, and the blurry maps point at the frozen,
  detection-trained image features; no data-scaling or unfreezing experiment exists yet, so this is a reading, not a result.
  The component swaps tried here fixed neither.

## 2 Flaws found by the audit (`temp/audit/`) and what was done

| flaw | evidence | action | effect |
|---|---|---|---|
| detection overfits | train mAP 0.47 vs val 0.31 (r0), 0.28 (r3) | left-right mirror augmentation of whole clips (`training/augment.py`, self-tested) | r0 -> r4 (all changes together, no separate ablation): train mAP 0.468 -> 0.448, val unchanged |
| velocity not learned | probe at GT centres (`temp/audit/diag3.log`): zero prediction 1.74 m/s, trained head 1.70 m/s (vehicles); loss weight 0.2 | `w_obj_vel` 1.0 | AVE 2.32 (r3) -> 1.82; r0 1.92; still poor |
| non-standard BEV encoder | plain small trunk | DualBEV ResNet pyramid + FPN-LSS, initialised from the DualBEV checkpoint, + stride-1 detail path, 96-ch light heads | no gain on val, +7 ms |
| box refiner harmful and slow | AP lower (r3: 0.632 -> 0.596), +3.6 ms (`runs/t1_r3_combined/eval/latency.json`: 47.5 vs 43.9) | `refine: false` | removed |
| heading | AOE 1.78 rad (sign-free axis) | direction class (SECOND-style) | AOE 0.64 rad (kept from t1) |
| BN statistics | checked, fine | none | - |
| modality use | audit with one modality zeroed | none needed | both sensors contribute |

Training: 7000 optimiser steps x 18 samples (3 x 2 clips of 3 keyframes, BPTT), lr 3e-4 cosine, Kendall task weights, about 6.3 h (3.24 s per step at the end).
Validation total loss during training (`log.jsonl`, 50 val batches, all-keyframe total): 13.8 (step 1000) -> 11.5 (2000) -> 8.20
(7000; last-keyframe warm 8.05, cold 8.14). Full-VAL EMA evaluation of the same checkpoint: 8.275 (cold), 7.978 (warm).

## 3 Numbers (VAL, dt = 0, cold unless stated)

| metric | prior map | `t1_r0` | `t1_r3_combined` | `t1_r4_soa` | r4 - r0 |
|---|---|---|---|---|---|
| mAP (3 classes) | - | 0.307 | 0.282 | 0.309 (warm 0.311) | +0.001 (tie) |
| vehicle AP@2 m | - | 0.643 | 0.632 | 0.627 (warm 0.635) | -0.016 |
| pedestrian AP@2 m | - | 0.214 | 0.199 | 0.228 | +0.014 (interval includes 0) |
| vehicle AOE (rad) | - | 1.78 | 0.59 | 0.64 | -1.14 |
| vehicle AVE (m/s) | - | 1.92 | 2.32 | 1.82 | -0.10 |
| drivable area IoU | 0.429 | 0.699 | 0.696 | 0.698 (warm 0.698) | -0.001 (tie) |
| walkway IoU | 0.000 | 0.278 | 0.261 | 0.244 | -0.034 |
| lane-divider mask IoU@0.5 m | 0.000 | 0.327 | 0.306 | 0.274 (warm 0.283) | -0.054 |
| lane-divider zero-crossing F1 | - | 0.068 | - | 0.028 | -0.041 (interval excludes 0) |
| chain latency, ms | - | 43.4 | 43.9 | 50.7 | +7.3 |
| three time offsets, ms | - | 62.6 | 62.8 | 68.2 | +5.6 |

Time offsets (r4): mAP 0.243 (t - 0.5 s), 0.309 (t), 0.193 (t + 0.5 s); drivable IoU 0.709 / 0.698 / 0.670.
Memory (600 val clips, last keyframe of clips of 3, warm vs cold, dt 0; no interval computed): outside the camera view drivable +0.028,
walkway +0.009, lane-divider mask IoU@1 m +0.015; in view no change. mAP: no bootstrap was run (tie judged by size, +0.001). Paired scene bootstrap (r4 vs r0): drivable area tie; walkway, lane dividers, vehicle AP@2 m worse with intervals excluding 0.
Baselines and controls: prior map (constant per-cell mean), `t1_r0` (matched reference on the same frames), warm/cold twin. Not done:
null/oracle sanity runs for this round, a second training seed, an ablation of the individual fixes.

Train vs val gap (TRAIN subset, 800 samples of 428 scenes, same code path, vs VAL 3109 samples of 82 scenes; per-class AP on 800 samples is
noisy, e.g. two-wheeler n_gt 123 vs 524; also carpark 0.296 / 0.162 (r4), 0.370 / 0.201 (r0), pedestrian crossing 0.140 / 0.093 (r4),
0.176 / 0.106 (r0)): mAP 0.448 / 0.309, drivable 0.711 / 0.698, walkway 0.282 / 0.244,
lane dividers 0.346 / 0.274 (r4); mAP 0.468 / 0.307, drivable 0.725 / 0.699, walkway 0.328 / 0.278, lane dividers 0.446 / 0.327 (r0).

## 4 What the images show (`images/compare_*.png`: `t1_r0` on top, `t1_r4_soa` below, same val frame)

Both models place the road, crossings, walkways and the nearby vehicles roughly right, but the predicted maps are blurry blobs next
to a sharp ground truth; thin structures (lane dividers, stop lines) are barely resolved. r4 draws the near vehicles a little
cleaner and shows less speckle in the splat mass, but its drivable-area errors (red false positives at the road edges) are the same as
the baseline's. The bottleneck is the resolution and content of the 16 x 44 image features that are cached and frozen.

## 5 Architecture

`images/architecture.png` (blue = trained, grey = frozen, green = outputs, sand = training only; no red blocks): frozen DualBEV image
heads -> lift PD -> camera temporal fusion (window + carried memory) -> radar BEV block -> concat fusion -> DualBEV BEV encoder +
detail path -> time mover (velocity splat) -> semantic and CenterPoint-style object heads at t - 0.5 / t / t + 0.5 s.

## 6 Honest negatives and open points

- Three rounds (`t1_r0`, `t1_r3_combined`, `t1_r4_soa`) never beat the first baseline on maps or on vehicle AP@2 m (heading and, for r4, velocity
  improved); component swaps are not the lever, as far as one seed per run can say.
- The line decoder (zero-crossing of a distance field) is weak (F1 0.03); the motion head starts slowly (zero init).
- Single seed; differences of about 0.02 are inside what a second seed could move; no ablation separating mirror, encoder and velocity weight.
- Radar smear and the 428-scene training set are unsolved.

## 7 What would move the numbers (needs Mirko's decision where marked)

1. More data: nuScenes parts 7-10 (the remaining 272 scenes) and the other cameras - **download, needs permission**. Directly attacks
   the detection overfit (train 0.45 vs val 0.31).
2. Un-freeze or replace the image features so they carry map content (needs raw images, not the cache; heavy on 12 GB).
3. Line head redesign (mask + centreline in standard focal/dice form instead of the distance field).
4. A second seed and single-fix ablations of this round (about 6 h per run, 2-3 h at 2500 steps).

## 8 Files

`runs/t1_r4_soa` (config, log, 3 checkpoints, `eval/`), `training/configs/t1_r4_soa.json`, `training/augment.py`, `evaluation/evaluate_run.py`
(`--warm`, `--on-train`), `temp/train/` (launch, pipeline, tables, drawing; scratch), `temp/audit/` (diagnostics),
`experiments/t1_r4_soa/{images,results}`.
