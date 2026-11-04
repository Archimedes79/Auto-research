#!/bin/bash
# scratch: the chain behind t17.data was blocked at 17:34 (the dependents' --wait-after of 720 minutes ran out while t17.data trained): queue it again with 1800 minutes.
# t17.data is done, so its evaluation starts at once; then t18.smoke -> t18.fix -> t18.fix.eval (queue_t18.sh with the longer wait), then the control
cd "$(dirname "$0")/../.." || exit 1               # bev_fusion/
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
RUN=/mnt/c/Auto-research/autoresearch/tools/run.py
nohup python3 $RUN --id t17.data.eval --gpu --split val --no-card --wait-after 1800 --after t17.data -- bash temp/t16/eval_chain.sh runs/t17_data > runs/t17_data.eval.launcher.log 2>&1 &
sleep 4
bash temp/t16/queue_t18_long.sh
sleep 3
bash temp/t16/start_ctl.sh --go --run-arg=--wait-after --run-arg=1800 --run-arg=--after --run-arg=t18.fix.eval 2>&1 | grep -E 'started|refus|error|ended' | cut -c1-200
sleep 3
ps -eo pid,etimes,cmd | grep 'run.py --id t1[678]' | grep -v grep | cut -c1-120
