#!/bin/bash
# GPU smokes (20 steps, --no-card, never feed a verdict) of the new options before the day chains: E302, E305 (e2e), E306 (exact raster)
cd "$(dirname "$0")/../.."
L=temp/gt2/smokes.log
echo "start $(date)" > $L
bash temp/e302/start_cap.sh 0 j1 smoke >> $L 2>&1; echo "E302 smoke exit $? $(date)" >> $L
bash temp/e305/start_e2e.sh e 0 j1 smoke >> $L 2>&1; echo "E305 smoke exit $? $(date)" >> $L
bash temp/e306/start_exact.sh 0 smoke >> $L 2>&1; echo "E306 smoke exit $? $(date)" >> $L
echo SMOKES_DONE >> $L
