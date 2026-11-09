#!/bin/bash
# scratch: the judgement columns of the rows t17.data (the more-data arm) and t17.data.eval in the BEV ledger (AR_PROJECT is set HERE, inside WSL)
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
L="python3 autoresearch/tools/ledger.py"
RES="VAL 1000 frames, GT v8, one seed each, against t16.full_s0 (paired scene bootstrap, 1000 resamples): none of the six map IoUs beats its band (drivable area -0.012 [-0.020,-0.005], car park -0.034, road divider -0.012 worse; walkway, ped. crossing, lane divider inside the band), front mAP -0.001, vehicle AP@2 -0.012 [-0.021,-0.002], pedestrian AP@2 +0.024 [+0.008,+0.041], vehicle moving EPE 1.112 -> 1.069 m/s [-0.066,-0.020], static EPE 0.189 -> 0.216 [+0.012,+0.040]; train-to-val map-IoU gap +0.091 -> +0.093 (+2 %, predicted <= -20 %); predicted +0.01...+0.04 on the maps: missed"
DEC="by hand: one seed, written rule of the t17 card: no ADOPT (0 of 6 maps and the mAP not better by their band), no guard worse than its band, so TIE/negative: keep t16.full_s0 as the base; 64 % more scenes at the same 6000-step budget (about 4 instead of about 6 passes per frame) did not shrink the train-to-val gap, the gap is not a data-quantity problem at this budget"
NOTE="attempt 1 died at step 2600 (CUDA unknown error, trainer hung ~2 min), run.py retried at 15:45 and the trainer resumed from ckpt_00002500.pt at step 2500: not bit-identical to an uninterrupted run; the dependents of this run were blocked at 17:34 (their --wait-after of 720 min ran out) and requeued with 1800 min at 19:44"
$L note t17.data --result "$RES" --decision "$DEC"
tail -c 600 bev_fusion/records/LEDGER.csv | head -c 300
echo
git status --short bev_fusion_lane_center/experiment_execution/ | head -3
echo "lane ledger untouched: $(git diff bev_fusion_lane_center/experiment_execution/ | wc -l) diff lines"
