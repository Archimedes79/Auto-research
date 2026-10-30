# t4_soa_front - state-of-the-art comparison with only the front area visible

Started 2026-10-30 (request of Mirko: "Do a state of the art comparison considering only the front area as visible").
Code: `training/dualbev_front.py` (self-test), `temp/t4/dualbev_front_eval.py`, `temp/t4/compare_front.py`.

## 1 Card

- **Question.** How far is our camera + radar system from a state-of-the-art camera detector when both are judged in the same area
  (the front-camera sector, 0-50 m, +-32.3 deg) on the same frames with the same metric?
- **Why a new run.** Published numbers (nuScenes mAP / NDS of CRN, RCBEVDet, ...) are 360 degrees, ten classes, six cameras: they
  cannot be cut to the front sector. The only honest front-area SOA number is a SOA detector run by us on our frames.
- **Reference.** The pre-trained DualBEV single-frame R50 detector (official nuScenes val 36.8 mAP / 43.4 NDS with six cameras, README of the
  authors), the model whose image heads and BEV encoder our system already uses. Run with ONLY CAM_FRONT (the other five cameras are
  absent: "front area visible"), unchanged weights (no fine-tuning), same 1000 frames of the 82 official val scenes (the DualBEV training
  set is the official train split: no overlap, checked in `data/README.md` / `datasets/NuScenes/README.md`: our split is the official one).
- **Metric.** The study protocol: `evaluation.metrics.DetMeter`, AP over centre distances 0.5 / 1 / 2 / 4 m, groups vehicle (car, truck, bus,
  trailer, construction vehicle) and pedestrian, mAP2 = their mean; GT and detections restricted to x in [0, 50) m and +-32.3 deg
  (DualBEV's BEV ends at 51.2 m); heading, size and velocity are not compared.
- **Prediction (before the run).** DualBEV front-only mAP2 0.40-0.55 on this area: front objects are the best seen, the detector has had
  20 epochs on 700 scenes and the full image-to-BEV transformer; our reference 0.339 (60 m). Pedestrians are hard for every camera model.
- **Decision rule.** Report both systems per class, per range, with paired scene bootstrap. A gap larger than the noise band 0.054 and an
  interval excluding 0 is a real gap; the size of the gap tells where the next work goes (training length vs. blocks).
- **Kill / validity criterion.** The port must reproduce plausible detections: scores of the best car detection > 0.4 on frames with
  nearby cars, nearest-detection distance to GT small (checked on frames, `temp/t4/debug_records.py`). The official 36.8 mAP cannot be
  reproduced here (six cameras, 10 classes, 360 degrees, lidar-free), which is a limit of this check.

## 2 Log

- Port written from the checkpoint tensor names and the DualBEV code reading (`literature/code/dualbev.md`, extra reads of the view
  transformer fuser / ProbNet / geometry attention / CenterHead config on 2026-10-30). Every tensor of the file loads (only the loss
  buffer is left over). Geometry self-test: a depth plane 20 m ahead lands at 23 m / 0 m in both BEV pools.
- First run: scores 0.08-0.15 and detections on the BEV border = no signal. Cause found: our `bev_trunk.FPNLSS` concatenates
  [up(stage 2), stage 0]; BEVDet / DualBEV concatenate [stage 0, up(stage 2)]. The convolution weights after the concatenation are
  order-specific, so the pre-trained FPN-LSS was scrambled. With the original order the port detects (best car score 0.8, 7.7 m / 0.1 m
  against GT 7.6 m / 0.1 m). **This also affects our own system**: every run since t1 loaded the pre-trained trunk with the scrambled order
  (fine-tuning repairs part of it). New flag `trunk_fpn_original` (default off, earlier runs unchanged) and arm `t4_fpn_order` (try 10).

## 3 Result (2026-10-30; numbers from `results_x50.md`, made by `temp/t4/compare_front.py` from the `scene_records.pkl` of each run)

Front sector (0-50 m, +-32.3 deg), the same 1000 cold-VAL frames of 82 official val scenes (3061 vehicle and 1435 pedestrian GT boxes),
centre-distance AP at 0.5 / 1 / 2 / 4 m, `mAP2` = vehicle + pedestrian. Difference to DualBEV with the paired scene bootstrap (100 draws).

| system | mAP2 | difference to DualBEV [95% interval] |
|---|---|---|
| DualBEV pre-trained, front camera only (SOA reference) | 0.250 | - |
| ours, camera only, 3000 steps | 0.242 | -0.009 [-0.021, -0.002] (tie inside the noise band 0.054) |
| ours, camera + radar, 3000 steps (two seeds) | 0.375 / 0.359 | +0.124 / +0.109 |
| ours, camera + radar, 6000 steps | 0.386 | +0.135 |
| ours, camera + radar, 7000 x 2 steps | 0.400 | +0.150 [+0.133, +0.169] |

- Vehicles: equal at the loose threshold (AP@4 0.733 both, 3000 steps), far apart at tight ones (AP@1 0.277 vs 0.524, AP@0.5 0.076 vs 0.271).
  Pedestrians: DualBEV AP@2 0.105 vs 0.226 (ours), AP@0.5 0.000 vs 0.066.
- By distance: DualBEV 0.447 / 0.211 / 0.063 (0-20 / 20-40 / 40-50 m); ours 0.557 / 0.341 / 0.161. The camera-only arm of our system equals DualBEV
  in all three bins (0.439 / 0.204 / 0.054).
- Localisation: matched confident detections (score >= 0.3, nearest same-group GT within 2 m) miss the GT centre by 0.76 m (DualBEV) and 0.50 m
  (ours); no systematic offset (median dx +0.14 m, dy 0.00 m for DualBEV): the gap is precision, not a coordinate error (`temp/t4/bias_check.py`).

**Reading.** In the front area the pre-trained SOA camera detector and our camera-only system are the same (0.250 vs 0.242). Everything our
system has beyond it comes from the radar branch (+0.13 at the same schedule) and, to a smaller degree, from longer training (+0.025 to 0.4);
the SOA camera path itself is not exceeded. The prediction 0.40-0.55 for DualBEV was wrong: its camera-only front-sector number is 0.25.

**Limits (do not over-read).**
- DualBEV was run with its other five cameras absent, which is the requested condition but not its published one (36.8 mAP with six cameras
  cannot be reproduced here: 360 degrees, ten classes, not the same metric). A camera + radar SOA system (CRN, RCBEVDet) was NOT run; the
  comparison with it remains the secondary-source table in `../t3_improve_loop/soa_reference.md` (not area-comparable).
- Our port is read from the checkpoint tensor names and a code summary, not run against the original: no NMS (3 x 3 peaks, threshold 0.05,
  top 100 as for our heads), RGB input chosen on a 100-frame check (mAP2 0.206 RGB vs 0.175 BGR, other frames, 60 m), no test-time tricks.
  That our own camera-only arm lands on the same number is a consistency check of the port, not proof that it is exact.
- Groups are merged (vehicle = car, truck, bus, trailer, construction vehicle), as in the study; the metric rewards our heads, which were trained
  on exactly this grouping and metric, DualBEV on ten classes.
- Our systems use three camera frames (and radar); DualBEV is single frame. One seed except where two are shown.

## 4 Consequence: trunk bug (try 10)

`t4_fpn_order` (same as the reference, FPN-LSS concatenation in the order the pre-trained weights expect) is running on the GPU lane
(commit 1ac3757); result to be appended here and in the ledger of tries.
