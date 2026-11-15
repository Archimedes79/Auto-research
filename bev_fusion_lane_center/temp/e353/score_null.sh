#!/bin/bash
cd "$(dirname "$0")/../.."
R=$PWD
S=$1; P=$2; K=$3
bash tools/env/_py14.sh jprune_b.py commands --seed $S --point $P 2>/dev/null | grep -A1 "^# $P $K\$" | tail -1 > /tmp/score_cmd_$P_$K.txt
CMD=$(cat /tmp/score_cmd_$P_$K.txt)
echo "$(eval echo "$CMD")"
eval "nice $CMD"
