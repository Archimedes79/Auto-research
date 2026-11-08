#!/bin/bash
# Lead 2026-11-02 (after the freeze a16e696): queue the three 70-step timing runs of the NEW code (the frozen frame, commit a16e696 = tag bf/frame-2026-11-02; t10 config) behind the
# control's EVALUATION job t16.ctl_s0.eval, so that they run last (20 minutes exclusive) and nothing of mine is left running afterwards. One after the other through run.py's queue
# (RAM gate 20 GB, FIFO line). Final checkpoints are slim (weights only).
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PIN=$(git -C .. rev-parse --verify a16e696^{commit}) || exit 1
for x in d e f; do
  ~/miniconda3/envs/gcar/bin/python - <<EOF
import json
p = "temp/compose/speed_$x.json"
c = json.load(open(p))
c["slim_at_end"] = True
json.dump(c, open(p, "w"), indent=1)
EOF
done
prev=t16.ctl_s0.eval
for x in d e f; do
  setsid nohup python3 ../autoresearch/tools/run.py --id speed.$x --no-card --no-git --gpu --ram-mb 20000 --split train --after $prev --wait-after 3000 --wait-gpu 1500 \
    -- bash temp/train/run_ar.sh temp/compose/speed_$x.json $PIN > runs/speed_$x.launcher.log 2>&1 < /dev/null &
  prev=speed.$x
  sleep 2
done
pgrep -af "run.py --id speed" | cut -c1-150
# afterwards (from autoresearch/, AR_PROJECT=../bev_fusion): python3 -m trainsuite --root ../bev_fusion/runs speed speed_a3 speed_d speed_e speed_f --skip 3   (a3 = the old code, 2.66 s/step)
