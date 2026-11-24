#!/bin/bash
# t25 (D145): the numbers of one finished arm, as the card's decision protocol asks: the cold score on VAL 1000 through the front door
# (evaluation/evaluate.py --quant), then the warm evaluation on 12 whole val scenes (reported next to it, not decisive).
#   bash experiments/t25_one_frame_one_scan/scripts/eval_arm.sh runs/<name>      (from bev_fusion/; run.py starts it as t25.<arm>.eval with --after <arm>)
# Not in this script on purpose: the timing (needs an idle GPU), the memory command and the stream drift (diagnostics of t21).
cd "$(dirname "$0")/../../.." || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
run=$1
$PY evaluation/evaluate.py "$run" --quant --curves --fusion --velocity --n 1000
rc1=$?
scenes=$($PY -c "import json; v = json.load(open('$run/config.json'))['val_scenes']; print(' '.join(v[::7][:12]))")
$PY evaluation/bev_suite.py "$run" --fusion --velocity eval --warm --scenes $scenes > "$run/eval/warm_cmd.txt" 2>&1
echo "warm eval exit $? ($scenes)"; tail -n 3 "$run/eval/warm_cmd.txt" | cut -c1-200
exit $rc1
