#!/bin/bash
# stop E273.p_s0 (phase 3 running on the code before the critic's fixes), file its folders away, start again on the fixed code
cd "$(dirname "$0")/../.."
pid=$(pgrep -f "python qgraph_train.py --name e273_p_s0" | head -1)
echo "training pid: $pid"
[ -n "$pid" ] && kill -TERM $pid
for i in $(seq 1 30); do sleep 2; pgrep -f "python qgraph_train.py --name e273_p_s0" >/dev/null || break; done
pgrep -f "run.py --id E273.p_s0" >/dev/null && echo "launcher still running" || echo "launcher gone"
mkdir -p runs/_superseded
for k in 2 3; do [ -d runs/e273_p_s0_p$k ] && mv runs/e273_p_s0_p$k runs/_superseded/e273_p_s0_v1_p$k; done
ls runs/_superseded
