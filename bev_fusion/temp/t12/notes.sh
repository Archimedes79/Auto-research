#!/bin/bash
# t12 ledger decisions (rule of the card, written before the runs)
cd "$(dirname "$0")/../.."
export AR_PROJECT=../bev_fusion
L=../autoresearch/tools/ledger.py
python3 $L note t12.mb_s0 --decision "KEEP (by the card rule): moving EPE 6.674 -> 3.319 m/s (band 0.10), total_main 11.507 -> 10.785 (-6.3 %); write the joint-training card" \
  --notes "VAL 500 (pinned GT 0.5_4_15efbde2d1), same frames: moving vehicles 6.907 -> 3.405, static 0.055 -> 0.077 (worse), box centres field 2.255 vs base objhead 4.042 = v0 4.042 vs t11_vel_hurdle_acc2 objhead 2.426 (t12.hurdle_ctl): field better by 0.17 > band -> read the box velocity from the field (2nd decision); prediction <= 2.0 m/s missed (3.32); warp check passed (0.410 < 0.566); block 17 ms/frame batch 1 (s1; s0 timing 58 ms on a shared GPU), peak 2.2-2.6 GB"
python3 $L note t12.mb_s1 --decision "KEEP (seed twin, same rule): moving EPE 6.674 -> 3.318 m/s, |s0-s1| 0.0015, total_main 11.507 -> 10.792 (-6.2 %)" \
  --notes "moving vehicles 6.907 -> 3.402, static 0.055 -> 0.080, box centres field 2.286 vs hurdle objhead 2.426; block 17.1 ms/frame (not an idle-GPU number)"
python3 $L note t12.hurdle_ctl --decision "control only (no verdict): objhead at moving box centres 2.426 m/s vs v0 4.042; its time-mover field 6.618 (also near v0)"
