#!/bin/bash
# one-off: return when the phase-3 node gate of BOTH r runs is in the logs, or when an r run has ended (done flag or no trainer process) -- the harness wakes me on exit
cd "$(dirname "$0")/../.."
for i in $(seq 1 720); do
  g=0
  for s in 0 1; do grep -q " GATE " runs/e274_r_s${s}_p3/log.txt 2>/dev/null && g=$((g + 1)); done
  [ $g -ge 2 ] && { echo "both node gates are in"; break; }
  if ! pgrep -f "qgraph_train.py --name e274_r_s" > /dev/null; then echo "no r trainer process left"; break; fi
  sleep 30
done
date
for s in 0 1; do echo "== s$s"; grep -h " GATE " runs/e274_r_s${s}_p3/log.txt 2>/dev/null | tail -1 | cut -c1-700; tail -n 2 runs/e274_r_s${s}_p*/log.txt 2>/dev/null | tail -4 | cut -c1-200; done
ls experiment_execution/runs/E274.r_s0 experiment_execution/runs/E274.r_s1 2>/dev/null | head -12
