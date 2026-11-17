#!/bin/bash
# wait (at most ~14 min) until a log ends with a line rc=..., then print the summary lines of the test run
LOG=${1:?log}
for i in $(seq 1 165); do grep -q '^rc=' "$LOG" && break; sleep 5; done
grep -E '^(rc=|[a-z_0-9]+: .*(ok|SKIPPED)|test_jtoken_eval ok|Traceback|AssertionError|RuntimeError|  File )' "$LOG" | cut -c1-240
