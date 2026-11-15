#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P="bash $R/tools/env/_py14.sh"
for t in "grid/qmnms.py" "grid/qgraph.py" "grid/test_mnms_layer.py" "grid/test_soft_nms.py" "startable.py" "run_config.py --selftest" "grid/test_tier_loss.py" "test_qgraph_readout.py"; do
  out=$($P $t 2>&1); rc=$?; echo "rc=$rc  $t  | $(echo "$out" | tail -1 | cut -c1-140)"
done
