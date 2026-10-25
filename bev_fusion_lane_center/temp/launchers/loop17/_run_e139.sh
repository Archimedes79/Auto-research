#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E139: map-conditioned builder trained on REAL failure frames (TRAIN rows, seven types, out-of-fold dense maps) + clean frames; arm B with the map, arm C (control) with the map always zero; then evaluation on the VAL failure series
S=$REPO/scripts
L=$REPO/out/loop2/e17/logs
until [ -f $L/e139_dense.done ]; do sleep 20; done
bash $S/_py14.sh p17_mapft.py --name e17_map_b --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 > $L/e139_train_b.log 2>&1 &
bash $S/_py14.sh p17_mapft.py --name e17_map_c --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 --p_none 1.0 --p_wrong 0.0 > $L/e139_train_c.log 2>&1 &
wait
V=single,maxdecay:0.95
V4=single_e37,maxdecay:0.95_e37
P4=fog,radar_off,extr3,cam_front_off
for mode in "exact 0 _m0" "exact 2 _m2" "none 0 _mnone" "wrong 0 _mwrong"; do
  set -- $mode
  bash $S/_py14.sh p17_eval.py --out e139_b --mapnet 1 --ckpt e17_map_b --map_mode $1 --map_sigma $2 --suffix $3 --variants $V > $L/e139_eval_b$3.log 2>&1
done
bash $S/_py14.sh p17_eval.py --out e139_c --mapnet 1 --ckpt e17_map_c --map_mode none --suffix _mnone --variants $V > $L/e139_eval_c.log 2>&1
bash $S/_py14.sh p17_eval.py --out e139_b --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --suffix _m0 --perts $P4 --variants $V4 > $L/e139_eval_b4.log 2>&1
bash $S/_py14.sh p17_eval.py --out e139_c --mapnet 1 --ckpt e17_map_c --map_mode none --suffix _mnone --perts $P4 --variants $V4 > $L/e139_eval_c4.log 2>&1
echo finished > $L/e139.done
