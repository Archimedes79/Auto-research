#!/bin/bash
# the mini measurement Mirko allowed (2026-11-01): speed.a = HEAD code, speed.b = + speed changes + profiler, speed.c = b + bf16 + cudnn.benchmark; one after the other
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
nohup nvidia-smi dmon -s um -d 2 -o T > temp/speed_dmon.txt 2>&1 &
prev=""
for x in a b c; do
  tree=speed
  [ "$x" = a ] && tree=speed_base
  after=""
  [ -n "$prev" ] && after="--after speed.$prev"
  nohup python3 ../autoresearch/tools/run.py --id speed.$x --no-card --no-git --gpu --ram-mb 20000 --split train $after -- bash temp/speed_run.sh $tree temp/compose/speed_$x.json > runs/speed_$x.launcher.log 2>&1 &
  prev=$x
  sleep 2
done
sleep 5
tail -n 5 runs/speed_a.launcher.log
