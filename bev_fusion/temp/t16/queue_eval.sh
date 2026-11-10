#!/bin/bash
# scratch: queue the evaluation job of a run behind it, then (re)queue a successor behind the evaluation. usage: queue_eval.sh <run id> <run folder name>
cd "$(dirname "$0")/.." || exit 1
cd .. || exit 1                                   # bev_fusion/
ID=$1; NAME=$2
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id "$ID.eval" --gpu --split val --no-card --after "$ID" -- bash temp/t16/eval_chain.sh runs/$NAME > runs/$NAME.eval.launcher.log 2>&1 &
sleep 2
echo "queued $ID.eval pid $!"
