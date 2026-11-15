#!/bin/bash
cd "$(dirname "$0")/../.."
L=temp/gt2/smokes2.log
echo "start $(date)" > $L
bash temp/e302/start_cap.sh 0 j1 smoke >> $L 2>&1; echo "E302 smoke exit $? $(date)" >> $L
bash temp/e305/start_e2e.sh e 0 j1 smoke >> $L 2>&1; echo "E305 smoke exit $? $(date)" >> $L
echo SMOKES_DONE >> $L
