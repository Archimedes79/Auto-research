#!/bin/bash
# scratch: ledger notes for the t23 second-seed pair (numbers from temp/lane_tests/t23_s1_readout.txt and t23_s1_compare.txt, i.e. from the cards of the runs)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
L=../autoresearch/tools/ledger.py

python3 $L note t23.C_s1 \
  --result 'VAL 1000, dt 0, seed 1, against the matched control t21.ctl_s1: lane divider IoU@0.5 0.4336 -> 0.4329 (d1 -0.0007, paired 95% [-0.0079, +0.0065], m1 = max(0.010, half-width 0.0072) = 0.010); IoU@1 0.5489 -> 0.5623 (+0.0134); AP 0.2141 -> 0.2542 (+0.0401); roc_auc 0.8362 -> 0.9287 (+0.0925); zero-crossing F1@0.5 0.1795 -> 0.1472 (-0.0323, interval excludes 0); guards: road divider IoU@1 -0.0046, drivable +0.0018, walkway IoU@0 -0.0065 (interval [-0.0108, -0.0021], inside m), ped crossing -0.0051, mAP fusion area -0.0041, velocity EPE moving +0.0061, lane-centre rows within +0.0027, vehicle IoU@0 -0.0018; step-1 check equal; arm spread |C_s1 - C_s0| 0.0042, control spread |ctl_s1 - ctl_s0| 0.0005; the three predictions of card section 4 hit' \
  --decision 'TIE on two seeds by hand (card section 4 rule: |d1| <= m1 and d0 -0.0054 also inside): the BCE as set here (weight 10, positives x3, ignore 1.0 m) does not move lane divider IoU@0.5; the ranking gain reproduces (AP +0.038 / +0.040, roc_auc +0.090 / +0.093) and is reported separately; walkway IoU@0 is lower in both seeds (-0.0102 / -0.0065); goes into closed_questions.md for these settings; no third seed, no weight step' || exit 1

python3 $L note t21.ctl_s1 \
  --result 'VAL 1000, dt 0: lane divider IoU@0.5 0.4336 (ctl_s0 0.4341, spread 0.0005), IoU@1 0.5489 (0.5516), AP 0.2141 (0.2149), roc_auc 0.8362; matched partner of t23.C_s1 (same code state); read in card experiments/t23_lane_line_loss section 5' \
  --decision 'control of the second seed pair, no verdict of its own; with ctl_s0 it gives the fine-tune spread of this recipe: 0.0005 in IoU@0.5 (a pair of seeds, code state differs: ctl_s0 ran on its own snapshot)' || exit 1
echo done
