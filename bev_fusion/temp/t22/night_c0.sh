#!/bin/bash
# scratch: gate C0 of the t22 card (student_frames 'all'): two diagnostic evaluations through the launcher (--no-card, never a verdict).
# Conditions (the lane chat's answer of 2026-11-07, which lifts its READY / CHAIN_DONE hold for this one short job): inside my window 22:00 - 06:45 (this script
# sleeps until 22:00 and gives run.py only the minutes left until 06:30 to wait, so nothing can start later); fewer than 2 GPU jobs of run.py are running (run.py's own slot
# limit); GPU memory in use below 7000 MB at the start (run.py --gpu-max-mb 7000). The BEV t23 chain4 and my stage B2 wait for CHAIN_DONE as agreed (D61).
#   (a) t22_standin_all_pair4: the A2 student frozen makes BOTH frames, in the control's checkpoint (against t22_standin_pair4 = current frame only, and the control)
#   (b) t22_B_all: the stage-B checkpoint with student_frames all
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
[ -d runs/t22_standin_all_pair4 ] || $PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 runs/t22_standin_all_pair4 --student runs/t22_A2_s0 --frames all || exit 1
[ -d runs/t22_B_all ] || $PY temp/t22/chain/make_b_all.py runs/t22_B_s0 runs/t22_B_all || exit 1

minutes_left() {   # minutes from now until 06:30 (0 or less: the window is over); only called inside 22:00 - 06:30
  now=$(( 10#$(date +%H) * 60 + 10#$(date +%M) ))
  if [ $now -ge 1320 ]; then echo $(( 1440 - now + 390 )); else echo $(( 390 - now )); fi
}
wait_window() {
  while true; do
    h=$(date +%H%M)
    if [ "$h" -ge 0630 ] && [ "$h" -lt 2200 ]; then
      echo "$(date +%H:%M:%S) outside my window (22:00 - 06:45): sleeping"; sleep 300; continue
    fi
    return 0
  done
}
run_one() {   # $1 id, $2 run folder, $3 checkpoint folder
  wait_window
  W=$(minutes_left)
  [ "$W" -le 5 ] && { echo "$(date +%H:%M:%S) window over, not starting $1"; exit 3; }
  echo "$(date +%H:%M:%S) $1: asking run.py for a GPU slot (at most $W min, memory in use < 7000 MB)"
  python3 ../autoresearch/tools/run.py --id $1 --gpu --gpu-max-mb 7000 --wait-gpu $W --split val --no-card -- \
    bash -c "$PY -u evaluation/evaluate.py $2 --quant --curves --fusion --velocity --n 1000 --ckpt-dir $3 >> $2.log 2>&1"
  echo "$(date +%H:%M:%S) $1 exit $?"
}
run_one t22.standin_all_pair4 runs/t22_standin_all_pair4 runs/t21_pair4_s0
run_one t22.B_all runs/t22_B_all runs/t22_B_s0
echo "$(date +%H:%M:%S) gate C0 finished"
