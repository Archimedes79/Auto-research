#!/bin/bash
# D113 / E365: the predicted-node caches, one GPU slot at a time (each run.py in the foreground, one after the other), K = 512 (the K rule of the card, VAL seed 0: 0.9878 < 0.99).
# `--split val` is the only data split run.py accepts besides test / confirm; these jobs read TRAIN and VAL frames and make derived data, no number of a verdict.
# usage: wsl bash temp/_session_scratch_d113/start_cache.sh [dry]       (E365.tc_a_s0_val is already done)
cd "$(dirname "$0")/../.."
MODE=$1
for J in "a e292_a 0 train" "a e292_a 1 val" "a e292_a 1 train" "p e273_p 0 val" "p e273_p 0 train" "p e273_p 1 val" "p e273_p 1 train"; do
  set -- $J
  K=$1; ID=$2; S=$3; SPLIT=$4
  NAME="E365.tc_${K}_s${S}_${SPLIT}"
  LINE=(python3 ../autoresearch/tools/run.py --id $NAME --seed $S --gpu --ram-mb 6000 --retries 0 --timeout 3600 --no-card --split val
    -- bash tools/env/_pyr.sh lane_tokcache.py build --run runs/${ID}_s$S --split $SPLIT --K 512 --workers 2)
  if [ "$MODE" = "dry" ]; then printf '%q ' "${LINE[@]}"; echo; else "${LINE[@]}" > temp/_session_scratch_d113/launch_${K}_s${S}_${SPLIT}.log 2>&1 || { echo "FAILED $NAME"; exit 1; }; fi
done
echo "all caches done $(date +%H:%M:%S)"
