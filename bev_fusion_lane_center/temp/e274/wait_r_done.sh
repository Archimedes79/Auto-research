#!/bin/bash
# one-off: return when both r runs have ended (done flags) or no r trainer is left; the harness wakes me on exit
cd "$(dirname "$0")/../.."
for i in $(seq 1 2000); do
  [ -f experiment_execution/runs/E274.r_s0/done ] && [ -f experiment_execution/runs/E274.r_s1/done ] && { echo "both r runs done"; break; }
  if ! pgrep -f "qgraph_train.py --name e274_r_s" > /dev/null; then echo "no r trainer process left"; break; fi
  sleep 30
done
date
for s in 0 1; do echo "== s$s"; ls experiment_execution/runs/E274.r_s$s | tr '\n' ' '; echo; tail -n 3 runs/e274_r_s${s}_p7/log.txt 2>/dev/null | cut -c1-300; done
