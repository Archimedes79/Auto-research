#!/bin/bash
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/ledger.py note t16.full_s0 \
  --result "6000 steps done (04:47 - 11:50, 3.9 s per step at accumulation 2, cold GT cache in the first epoch); VAL 1000 frames, GT v8 (c643a7539d), EMA weights: drivable IoU@0 0.779, walkway 0.413, carpark 0.274, crossing 0.256, lane divider IoU@0.5 0.429, road divider IoU@1 0.572; mAP (fusion area) 0.416, vehicle AP@2 0.799, pedestrian 0.376, two-wheeler 0.341; velocity EPE at moving box centres 1.112 m/s (zero predictor 4.268), vehicle_moving cells 1.441, static cells 0.189; val loss total 7.442; train-vs-val on 1000 frames each: drivable 0.819 / 0.779, lane divider 0.554 / 0.429, carpark 0.445 / 0.274; radar_kept recovered to 0.90 (0.04 at step 400). Inference 96.4 ms per frame (batch 1, 3 dts, idle GPU; measured on the identical architecture t16_small_final_s0)." \
  --decision "by hand: single seed, GT v8, bring-up baseline, no verdict; the new baseline row of the system; read against t16.ctl_s0 (control, queued) and t17.data (700 scenes, queued)"
