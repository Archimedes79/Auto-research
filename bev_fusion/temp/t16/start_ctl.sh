#!/bin/bash
# scratch: the large run t16.full_s0 (baseline of the full network) and its matched control t16.ctl_s0, through the launcher; usage: start_large.sh [--go]
cd "$(dirname "$0")/../../../autoresearch" || exit 1
export AR_PROJECT=../bev_fusion
GO="$1"
HYP="The matched control of t16.full_s0: the recipe of t10_dual_long_s0 (dual camera features, velocity hurdle, 6000 steps, batch 3 x accumulation 2) without delta slots, motion block and split clip, on GT v8 and the 428 / 82 scenes, same seed: the reference the baseline of the full network is read against (t16 card, section 2, S7)."
PRED="Reference row, no prediction of its own; the prediction of the card is about its difference to t16.full_s0 (guesses: maps within +-0.02, vehicle moving EPE of the control above 4.0 m/s, ms per frame 25 ms lower)."
RULE="No KEEP / CUT: a reference row, read with t16.full_s0 by hand (single seed each, GT v8). Kill: non-finite loss in 3 rows in a row, stalled > 15 min."
python3 -m trainsuite start ../bev_fusion/training/configs/t16_ctl.json --id t16.ctl_s0 --hyp "$HYP" --pred "$PRED" --rule "$RULE" $GO "${@:2}"
