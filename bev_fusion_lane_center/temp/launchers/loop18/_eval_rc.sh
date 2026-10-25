#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# wait for the end of a recall-weighted fine-tune, then evaluate it on all VAL frames with fork pruning (cached raw outputs in ~/exp_gcar_b123/p18, json for the report in runs/e18/rc_eval)
# usage: _eval_rc.sh <run name> <weight> <tag>
R=$BFLC
until [ -f $R/runs/e123/$1/model.pt ] && [ -f $R/runs/e123/$1/status.json ]; do sleep 10; done
sleep 5
bash $R/tools/env/_py14.sh p18_variant_prune.py --name $1 --weight $2 --tag $3 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_$3.log 2>&1
cat $R/runs/e18/rc_eval_$3.log
