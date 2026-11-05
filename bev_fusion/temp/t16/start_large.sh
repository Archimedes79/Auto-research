#!/bin/bash
# scratch: the large run t16.full_s0 (baseline of the full network) and its matched control t16.ctl_s0, through the launcher; usage: start_large.sh [--go]
cd "$(dirname "$0")/../../../autoresearch" || exit 1
export AR_PROJECT=../bev_fusion
GO="$1"
HYP="The complete chain (dual camera features, delta-slot radar R, motion block M in its joint form with the box velocity read from its field, velocity hurdle; GT v8, 6000 steps, batch 3 x accumulation 2) trains without a defect on the 428 / 82 scenes and gives the baseline of the system: M and R do not cost the maps, and the velocity field leaves the zero predictor (t16 card, bring-up)."
PRED="Against the matched control t16_ctl (single seed each, VAL 1000 frames, GT v8, guesses): the six map IoUs and front mAP2 within +-0.02; vehicle moving EPE (vel/dt+0.00/cells/vehicle_moving/epe) below 4.0 m/s; static EPE at most 0.03 m/s above the control; ms per frame at most 25 ms above; step time at most 1.3 x the control."
RULE="No KEEP / CUT (one seed never gives a verdict): the table is the new baseline, labelled by hand: single seed, GT v8, bring-up baseline. Kill: non-finite loss in 3 rows in a row, stalled > 15 min. Findings, not kills: group.motion flat at 1.0945, dead_module, data_bound (card section 2)."
python3 -m trainsuite start ../bev_fusion/training/configs/t16_full.json --id t16.full_s0 --hyp "$HYP" --pred "$PRED" --rule "$RULE" $GO "${@:2}"
