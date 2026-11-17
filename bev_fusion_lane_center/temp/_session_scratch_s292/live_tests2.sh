#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P="bash $R/tools/env/_py14.sh"
for t in "grid/test_mnms_layer.py --real" "grid/qgraph.py" "startable.py" "run_config.py --selftest" "qends_eval.py selftest" "jends_eval.py selftest" "grid/test_soft_nms.py" "grid/qmnms.py"; do
  out=$(nice $P $t 2>&1); rc=$?; echo "rc=$rc  $t  | $(echo "$out" | tail -1 | cut -c1-150)"
  [ $rc -ne 0 ] && echo "$out" | tail -8 | cut -c1-220
done
