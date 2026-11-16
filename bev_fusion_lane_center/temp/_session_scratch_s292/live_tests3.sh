#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P="bash $R/tools/env/_py14.sh"
for t in "grid/jwire.py" "qends_eval.py selftest" "jends_eval.py selftest" "jwiring.py selftest" "jdup.py selftest" "jtime.py selftest" "grid/test_mnms_layer.py"; do
  out=$(nice $P $t 2>&1); rc=$?; echo "rc=$rc  $t  | $(echo "$out" | tail -1 | cut -c1-230)"
  [ $rc -ne 0 ] && echo "$out" | tail -6 | cut -c1-220
done
