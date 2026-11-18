#!/bin/bash
# throw-away: run the full test_jtoken suite at low priority, summarise the log
cd "$(dirname "$0")/../.." || exit 1
LOG=temp/e374hook/test_jtoken_run.log
OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh training/grid/test_jtoken.py > "$LOG" 2>&1
echo "exit $?"
grep -E '^[a-z_]+( \(.*\))?: |test_jtoken ok|Error|Traceback' "$LOG" | cut -c1-240
