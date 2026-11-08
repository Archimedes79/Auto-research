#!/bin/bash
# CPU only: the three configs of the timing runs on the NEW code (commit f13e593): speed_d = amp off, speed_e = amp fp16 (GradScaler), speed_f = amp bf16; t10 config, 70 steps, probe on, no profiler
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=../autoresearch AR_PROJECT=../bev_fusion
common="--set steps=70 --set warmup=10 --set eval_every=0 --set save_every=70 --set keep=1 --set log_every=5 --set grad_diag_every=0 --set probe=true"
for p in "d off" "e fp16" "f bf16"; do
  set -- $p
  python3 -m trainsuite compose training/configs/t10_dual_s0.json $common --set amp=$2 --out temp/compose/speed_$1.json | tail -1
  sed -i "2s/\"name\": \"t10_dual_s0\"/\"name\": \"speed_$1\"/" temp/compose/speed_$1.json
  echo "speed_$1: $(sed -n 2p temp/compose/speed_$1.json) amp=$(grep -o '"amp": "[a-z0-9]*"' temp/compose/speed_$1.json)"
done
