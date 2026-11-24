#!/bin/bash
# t25 launch (D145), the exact commands of card section 3.   bash experiments/t25_one_frame_one_scan/scripts/launch.sh day|night   (in the WSL, from anywhere)
#   day   : queue ctl, R, A1 (one training at a time, each next arm after the evaluation of the one before) and start the three evaluation jobs;
#   night : queue A2 after the evaluation of A1 (only from 22:00 on: no job may straddle the window end 17:30) and start its evaluation job.
# The script stays in the foreground until its evaluation launchers have ended (they wait for their arm): run it as a background job.
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$PWD
export AR_PROJECT=../bev_fusion
HYP="t25 (card experiments/t25_one_frame_one_scan, D91, D145): one camera frame and ONE radar sweep per step with the past carried recursively at feature level keeps what the 0.5 s radar window and the pair buffer give"
RULE="by hand: composed from the table rows (card section 1, added 2026-11-24): HOLDS (provisional, one seed) when d >= -0.010 on detf mean mAP against ctl with the interval's lower end >= -0.015 and no guard CUT; FAILS when d < -0.020 with the interval's upper end below -0.010; else INCONCLUSIVE; the memory PAYS when A2 - A1 >= +0.010 with the interval above 0, HURTS at <= -0.010 with the interval below 0, else TIE (A1 taken); ctl outside its sanity band voids the comparison; the card's guards and kill criteria as written"

queue_add() {   # arm config prediction [after]
  local arm=$1 cfg=$2 pred=$3 after=$4
  local extra=()
  [ -n "$after" ] && extra=(--after "$after")
  (cd "$ROOT/autoresearch" && python3 -m Training queue add "training/configs/$cfg.json" --id "t25.${arm}_s0" --hyp "$HYP; arm $arm" --pred "$pred" --rule "$RULE" --no-check --note "D145; card t25 section 3; static check by hand" "${extra[@]}")
}

eval_job() {    # arm
  local arm=$1
  (cd "$ROOT/bev_fusion" && python3 ../autoresearch/tools/run.py --id "t25.${arm}_s0.eval" --no-card --gpu --gpu-max-mb 3000 --split val --after "t25.${arm}_s0" -- bash experiments/t25_one_frame_one_scan/scripts/eval_arm.sh "runs/t25_${arm}_s0" > "runs/t25_${arm}_s0.eval.launcher.log" 2>&1) &
}

PRED_CTL="ctl (matched control, pair P4, retrained on GT v13): the sanity band: detf mean mAP within 0.03 of the base 0.4092, lane divider IoU@0.5 within 0.03 of 0.4257, velocity EPE of moving centres <= 1.2537; it is the reference of the other arms"
PRED_R="R (recurrent camera, 0.5 s radar window): detf mean mAP -0.005 (-0.010 .. 0.000) against ctl, velocity EPE +0.05 (0.00 .. +0.10)"
PRED_A1="A1 (recurrent camera + one radar sweep): detf mean mAP -0.025 (-0.040 .. -0.015) against ctl, pedestrian AP@1 -0.04 or worse, velocity EPE +0.15 (+0.08 .. +0.25)"
PRED_A2="A2 (A1 + recursive radar memory): detf mean mAP -0.015 (-0.030 .. -0.005) against ctl, A2 - A1 about +0.010, velocity EPE +0.12 (+0.05 .. +0.22)"

case "$1" in
  day)
    eval_job ctl; eval_job R; eval_job A1
    queue_add ctl t25_ctl "$PRED_CTL" ""
    queue_add R t25_rec_r "$PRED_R" t25.ctl_s0.eval
    queue_add A1 t25_rec_scan "$PRED_A1" t25.R_s0.eval
    wait
    ;;
  night)
    eval_job A2
    queue_add A2 t25_rec_scan_mem "$PRED_A2" t25.A1_s0.eval
    wait
    ;;
  *) echo "usage: launch.sh day|night"; exit 2 ;;
esac
