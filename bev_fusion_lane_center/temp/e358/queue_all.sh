#!/bin/bash
# E358: queue the GPU live checks, the two training runs (after both live checks) and the two scoring jobs (after their training runs); all wait in run.py's FIFO queues behind the E361 runs of the Fable session
cd "$(dirname "$0")/../.."
for S in 0 1; do bash temp/e358/start_live.sh $S > temp/e358/launch_live_s$S.log 2>&1 & sleep 4; done
for S in 0 1; do bash temp/e358/start_w.sh $S > temp/e358/launch_w_s$S.log 2>&1 & sleep 4; done
for S in 0 1; do bash temp/e358/start_sc.sh $S > temp/e358/launch_sc_s$S.log 2>&1 & sleep 4; done
wait
