#!/bin/bash
# scratch: the evaluation of the control t16.ctl_s0 behind it (wait 1800 min), after the 17:34 block of the chain
cd "$(dirname "$0")/../.." || exit 1               # bev_fusion/
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id t16.ctl_s0.eval --gpu --split val --no-card --wait-after 1800 --after t16.ctl_s0 -- bash temp/t16/eval_chain.sh runs/t16_ctl_s0 > runs/t16_ctl_s0.eval.launcher.log 2>&1 &
sleep 3
echo "queued t16.ctl_s0.eval pid $!"
ps -eo pid,etimes,cmd | grep 'run.py --id t16.ctl_s0.eval' | grep -v grep | cut -c1-120
