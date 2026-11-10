#!/bin/bash
# scratch: gate C0 -> ledger notes of the two diagnostic rows and the schedule state (numbers from temp/t22/compare_C0_*.log, read by hand from the files, 2026-11-07 night)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py note t22.standin_all_pair4 \
  --result 'diagnostic gate C0 (a), VAL 1000 frames, paired scene bootstrap, stand-in with student_frames all (A2 student frozen makes BOTH frames) against the stand-in with current: static-cell EPE 0.3168 -> 0.1833 (-0.1335 [-0.2197, -0.0652]; control 0.2172, so now below it: -0.0339 [-0.0630, -0.0119]); moving-centre EPE 1.1912 -> 1.2207 (+0.0295 [-0.0089, +0.0611]; against the control +0.1296 [+0.0944, +0.1669]); vehicle-cell EPE 0.7617 -> 0.6646 (-0.0971 [-0.1605, -0.0398]); maps and boxes FALL, every interval excludes 0: drivable IoU@0 -0.0376, lane divider IoU@1 -0.0512, walkway -0.0522, mAP fusion area (detf) 0.3838 -> 0.3126 (-0.0712 [-0.0923, -0.0457]), vehicle AP@2 -0.0406; the card predicted a move under 0.01' \
  --decision 'no verdict row (diagnostic, --no-card); rule C0 of the card: INCONCLUSIVE (static-cell EPE meets the rule, moving-centre EPE does not, the map / box prediction is missed); stage B2 only on Mirko word' \
  --notes 'the two evaluations carry different evaluator hashes (e084cbc548a7 for the earlier runs, a38635126e2f here): the only number-shaping file that differs between the commits 91184f8612 and e796f3aef1 is evaluation/bev_adapter.py, in perturb_camera (LANE_IMAGE_KEYS), which a plain evaluation never calls (only _perturb with a --control does), so the warning of bev_suite compare changes no number here'
python3 ../autoresearch/tools/ledger.py note t22.B_all \
  --result 'diagnostic gate C0 (b), VAL 1000 frames: the stage-B checkpoint (adapted to the mixed pair) evaluated with student_frames all against B with current: static-cell EPE 0.2999 -> 0.1970 (-0.1029 [-0.1864, -0.0368]); moving-centre EPE 1.1431 -> 1.1757 (+0.0326 [+0.0057, +0.0598]); vehicle-cell EPE 0.7192 -> 0.6541; maps and boxes fall (the model never saw a student older frame): drivable IoU@0 -0.0543, lane divider IoU@1 -0.0830, mAP fusion area (detf) 0.4027 -> 0.3286 (-0.0741 [-0.0949, -0.0481]); against the control mAP -0.0868' \
  --decision 'no verdict (diagnostic, --no-card); read together with t22.standin_all_pair4: gate C0 INCONCLUSIVE' \
  --notes 'same evaluator-hash remark as t22.standin_all_pair4 (perturb_camera only)'
python3 ../autoresearch/tools/schedule.py set t22.refC_gate done
python3 ../autoresearch/tools/schedule.py show 2>&1 | grep -E "t22|refC" | cut -c1-200
git -C /mnt/c/Auto-research status --short bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/schedule.csv
