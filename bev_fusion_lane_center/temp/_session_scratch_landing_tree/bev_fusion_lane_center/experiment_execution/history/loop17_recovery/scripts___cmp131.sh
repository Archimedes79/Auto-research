#!/bin/bash
# E131 comparison tables: champion single / spatial hold / EMA + monitor hold against the control with the same input fusion
S=/mnt/c/ML-Experiment/scripts
for p in dark blur cam_front3_off; do
  bash $S/_py14.sh p17_compare.py --pert $p --md tables/e131.md \
    --entries champ=builder:e130_champion/${p}_single champ_md=builder:e131_hold/${p}_maxdecay_0.95 champ_em=builder:e131_hold/${p}_ema_hold_mon_0.5 ctrl=control:single ctrl_md=control:maxdecay_0.95 ctrl_em=control:ema_hold_mon_0.5 \
    --boot "champ_md,champ;champ_em,champ;champ_md,ctrl_md;champ_em,ctrl_em" 2>&1 | tail -16
done
