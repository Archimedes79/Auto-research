#!/bin/bash
# scratch: the judgement columns of the row t18.fix (the bundle arm: hygiene flags + motion block lr 1e-4) in the BEV ledger (AR_PROJECT is set HERE, inside WSL)
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
L="python3 autoresearch/tools/ledger.py"
RES="VAL 1000 frames, GT v8, one seed each, against t17.data (paired scene bootstrap, 1000 resamples; temp/t16/rule_t18.py, compare_t18.txt): static EPE 0.216 -> 0.241 m/s (+0.025, band 0.020: predicted at most 0.130, missed), moving vehicle EPE 1.364 -> 1.575 (+0.211), moving box-centre EPE 1.069 -> 1.204 (+0.135 [+0.107,+0.163], band 0.10, worse); guards hold: mean of the six map IoUs +0.009 (drivable area +0.007 [+0.003,+0.010], walkway +0.021 [+0.009,+0.035]), front mAP +0.005, vehicle IoU +0.007; lane divider zero-crossing F1 -0.013, vehicle moving AVE skill 0.860 -> 0.826 (worse); kill criteria not hit (motion term at step 2000 +3 percent, no non-finite loss)"
DEC="by hand: one seed per arm, written rule of the t18 card, label 'bundle: lr + hygiene flags' (the flags change the velocity losses that M learns from, so the lr alone is not isolated): CUT, moving and centre EPE worse than the band, static not better; keep t16.full_s0 as the base; the M lr 1e-4 is not adopted, the card names the next arm (a lower output gain of M, not a smaller lr); no training is started (Mirko: stop all BEV system trainings after t18.fix)"
$L note t18.fix --result "$RES" --decision "$DEC"
tail -c 700 bev_fusion/records/LEDGER.csv | head -c 500
echo
echo "lane ledger untouched: $(git diff bev_fusion_lane_center/experiment_execution/ | wc -l) diff lines"
