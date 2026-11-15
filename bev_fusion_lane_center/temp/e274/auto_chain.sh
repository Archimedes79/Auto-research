#!/bin/bash
# one-off (E274): start chain_after_p.sh when both p runs have their done flags (or both p trainers are gone for two checks in a row)
cd "$(dirname "$0")/../.."
gone=0
for i in $(seq 1 2000); do
  if [ -f experiment_execution/runs/E274.p_s0/done ] && [ -f experiment_execution/runs/E274.p_s1/done ]; then break; fi
  if pgrep -f "qgraph_train.py --name e274_p_s" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  [ $gone -ge 2 ] && break
  sleep 30
done
sleep 20
bash temp/e274/chain_after_p.sh
