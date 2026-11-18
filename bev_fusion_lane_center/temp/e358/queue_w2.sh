#!/bin/bash
# E358: re-queue the two training runs (after both faithful live checks) and the two scoring jobs; only to be started when BOTH faithful live checks are ACCEPTED
cd "$(dirname "$0")/../.."
for S in 0 1; do bash temp/e358/start_w2.sh $S > temp/e358/launch_w2_s$S.log 2>&1 & sleep 4; done
for S in 0 1; do bash temp/e358/start_sc.sh $S > temp/e358/launch_sc2_s$S.log 2>&1 & sleep 4; done
wait
