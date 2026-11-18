#!/bin/bash
kill $(pgrep -f "[r]un_side_tests.sh") 2>/dev/null
kill $(pgrep -f "_py14.sh training/grid/test_jpast_train.py") 2>/dev/null
kill $(pgrep -f "^python test_jpast_train.py") 2>/dev/null
sleep 1
kill $(pgrep -f "training/grid/test_jpast_train.py one") 2>/dev/null
sleep 2
pgrep -af "test_jpast_train|test_jmap_train|run_side_tests" | cut -c1-120 || echo none left
