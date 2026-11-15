#!/bin/bash
# the two tests of the last code commit (E319), one line each, then DONE
cd "$(dirname "$0")/../.."
for t in grid/jmaps.py grid/test_jmaps_dirx.py; do
  s=$(date +%s)
  out=$(bash tools/env/_py14.sh $t 2>&1); rc=$?
  echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL) $t ($(( $(date +%s) - s ))s) | $(echo "$out" | tail -1 | cut -c1-150)"
done
echo DONE
