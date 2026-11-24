#!/bin/bash
# t25 (D145): queue ONE arm, no chain (the lead's rule of 2026-11-24 for the shared GPU: a waiting lane job gets the gate between two arms).
#   bash experiments/t25_one_frame_one_scan/scripts/queue_arm.sh R|A1|A2
# Called by hand when the evaluation of the arm before has finished and no lane job waits for the gate; the evaluation job of the arm
# (t25.<arm>_s0.eval) is started here with it (launch.sh day started those of R and A1 too early: a launcher gives up after 12 h; they were stopped on 2026-11-24 09:21).
# The hypothesis, rule and predictions are the ones of launch.sh (the card, section 1).
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$PWD
export AR_PROJECT=../bev_fusion
HYP="t25 (card experiments/t25_one_frame_one_scan, D91, D145): one camera frame and ONE radar sweep per step with the past carried recursively at feature level keeps what the 0.5 s radar window and the pair buffer give"
RULE="by hand: composed from the table rows (card section 1, added 2026-11-24): HOLDS (provisional, one seed) when d >= -0.010 on detf mean mAP against ctl with the interval's lower end >= -0.015 and no guard CUT; FAILS when d < -0.020 with the interval's upper end below -0.010; else INCONCLUSIVE; the memory PAYS when A2 - A1 >= +0.010 with the interval above 0, HURTS at <= -0.010 with the interval below 0, else TIE (A1 taken); ctl outside its sanity band voids the comparison; the card's guards and kill criteria as written"
case "$1" in
  R)  CFG=t25_rec_r;        PRED="R (recurrent camera, 0.5 s radar window): detf mean mAP -0.005 (-0.010 .. 0.000) against ctl, velocity EPE +0.05 (0.00 .. +0.10)" ;;
  A1) CFG=t25_rec_scan;     PRED="A1 (recurrent camera + one radar sweep): detf mean mAP -0.025 (-0.040 .. -0.015) against ctl, pedestrian AP@1 -0.04 or worse, velocity EPE +0.15 (+0.08 .. +0.25)" ;;
  A2) CFG=t25_rec_scan_mem; PRED="A2 (A1 + recursive radar memory): detf mean mAP -0.015 (-0.030 .. -0.005) against ctl, A2 - A1 about +0.010, velocity EPE +0.12 (+0.05 .. +0.22)" ;;
  *) echo "usage: queue_arm.sh R|A1|A2"; exit 2 ;;
esac
cd "$ROOT/autoresearch" || exit 1
python3 -m Training queue add "training/configs/$CFG.json" --id "t25.${1}_s0" --hyp "$HYP; arm $1" --pred "$PRED" --rule "$RULE" --no-check --note "D145; card t25 section 3; queued by hand after the evaluation of the arm before; static check by hand"
# the evaluation job of the arm waits for the arm (run.py --after; its launcher gives up after 12 h of waiting, so it is started here, with the arm)
cd "$ROOT/bev_fusion" && python3 ../autoresearch/tools/run.py --id "t25.${1}_s0.eval" --no-card --gpu --gpu-max-mb 3000 --split val --after "t25.${1}_s0" -- bash experiments/t25_one_frame_one_scan/scripts/eval_arm.sh "runs/t25_${1}_s0" > "runs/t25_${1}_s0.eval.launcher.log" 2>&1 &
wait
