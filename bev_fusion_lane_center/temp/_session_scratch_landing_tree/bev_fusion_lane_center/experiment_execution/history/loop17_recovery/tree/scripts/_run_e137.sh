#!/bin/bash
# E137: time per frame of the whole chain on an IDLE GPU: waits for the E139 chain (its evaluation uses the GPU), then four configurations
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
until [ -f $L/e139.done ]; do sleep 30; done
sleep 20
for cfg in "288 144 1" "288 144 0" "192 96 0" "192 96 1"; do
  set -- $cfg
  bash $S/_py14.sh p17_latency.py --M $1 --K0 $2 --rec $3 --n 60 > $L/e137_M$1_rec$3.log 2>&1
done
echo finished > $L/e137.done
