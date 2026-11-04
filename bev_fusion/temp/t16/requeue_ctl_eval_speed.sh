#!/bin/bash
# scratch: after the control's launcher was replaced (22:00) its evaluation and the Training chat's three speed runs went to "blocked" (they saw the stale record):
# queue them again with the same parameters the Training chat used (--ram-mb 20000, --wait-after 3000, --wait-gpu 1500, pin a16e696..., no card, no git) and a longer wait for the evaluation
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
RUN=/mnt/c/Auto-research/autoresearch/tools/run.py
PIN=a16e6969c165d54d93187fb183e6a2b0471213a3
nohup python3 $RUN --id t16.ctl_s0.eval --gpu --split val --no-card --wait-after 1800 --after t16.ctl_s0 -- bash temp/t16/eval_chain.sh runs/t16_ctl_s0 > runs/t16_ctl_s0.eval.launcher.log 2>&1 &
sleep 3
prev=t16.ctl_s0.eval
for id in speed.d speed.e speed.f; do
  f=${id/./_}
  nohup python3 $RUN --id $id --no-card --no-git --gpu --ram-mb 20000 --split train --after $prev --wait-after 3000 --wait-gpu 1500 -- bash temp/train/run_ar.sh temp/compose/$f.json $PIN > runs/$f.launcher.log 2>&1 &
  sleep 3
  prev=$id
done
ps -eo pid,etimes,args | grep "[r]un.py --id" | cut -c1-170
for r in t16.ctl_s0.eval speed.d speed.e speed.f; do echo "$r $(grep -o '"status": "[a-z_]*"' records/runs/$r/run.json | head -1) $(cut -c1-70 records/runs/$r/heartbeat)"; done
