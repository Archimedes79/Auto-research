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
