#!/bin/bash
# one-off: stream the key lines of an E273 run log (phase starts, gates, failures, every 1000th step)
cd "$(dirname "$0")/../.."
f=experiment_execution/runs/$1/run.log
for i in $(seq 1 600); do [ -f $f ] && break; sleep 3; done
tail -n 0 -F $f 2>/dev/null | grep --line-buffered -E "phase [0-9]:|GATE|FAILED|Traceback|ROLLBACK|all phases done|step [0-9]*000/|gate of phase|exit"
