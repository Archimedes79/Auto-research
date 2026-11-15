#!/bin/bash
# scratch: result + decision of the stage-B2 rows (ledger.py note writes csv and md)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
L=../autoresearch/tools/ledger.py

python3 $L note t22.B2_s0 \
  --result 'VAL 1000 frames, one seed, vs t21.pair4_s0 (paired scene bootstrap, b-a [95%]): lane divider IoU@1 0.5495->0.4895 -0.0600 [-0.0770,-0.0443] (m 0.0164); mAP fusion area 0.4154->0.3385 -0.0769 [-0.0972,-0.0529] (m 0.0222); drivable IoU@0 -0.0286 [-0.0419,-0.0142] (m 0.0139); walkway -0.0706 [-0.1014,-0.0463]; lane divider IoU@0.5 -0.0605; vehicle IoU@0 dt0 -0.0348 [-0.0526,-0.0204]; vehicle AP@2 40-60 m -0.0071 fusion / -0.0271 plain (point); velocity: static-cell EPE 0.2172->0.2150 -0.0022 [-0.0290,+0.0178], vehicle-cell +0.0191 [-0.0030,+0.0396], moving-centre EPE 1.0911->1.1292 +0.0381 [+0.0125,+0.0643] (limit +0.05). Vs t22.B_s0: lane IoU@1 -0.0554 [-0.0713,-0.0409], mAP -0.0642 [-0.0812,-0.0406], moving EPE -0.0138 [-0.0409,+0.0090]. Fine-tune recovered 30% (lane IoU@1), 51% (drivable), 25% (mAP) of the loss of the frozen stand-in with student_frames all (stage B: 87/100/60%). Val loss 8.483 (B 8.389, control 8.284); 1500 steps, 6.5 s/step, no non-finite loss' \
  --decision 'CUT by the written rule (by hand: composed from the table rows): two named guards (lane divider IoU@1, mAP fusion area) below the control by more than 2 m with the whole interval below -m. Prediction hit on the velocity rows (static-cell, moving-centre point estimate), missed on maps and boxes (-0.077 against -0.012; my 10 pct for STAND-IN did not come true). The student route stays INCONCLUSIVE at stage B (one seed); the variant student also on the older frame is closed for this recipe; nothing adopted, frozen encoders stay the baseline. Open for Mirko: R arm, second seeds of stage B, B3 with gradient through the older frame (new code), or stop. Card: experiments/t22_student_encoder, Stage B2 result'

python3 $L note t22.B2_s0.eval \
  --result 'diagnostic evaluation of t22.B2_s0 (VAL 1000 frames, --quant --curves --fusion --velocity); numbers and verdict in the row t22.B2_s0' \
  --decision 'no verdict (diagnostic, --no-card)' \
  --notes 'evaluator hash a38635126e2f = the one of the gate C0 evaluations (difference to the control e084cbc548a7 is perturb_camera only)'

python3 $L note smoke.t22_B2_s0 \
  --decision 'no verdict (20-step end-to-end smoke of B2, passed before the run)' \
  --notes 'attempt 1 of this id was snapshotted 2026-11-08 22:03 by the first waiter and cut by the reboot of 22:09; B2 itself had not started'
python3 $L tail -n 5 2>&1 | cut -c1-230
