#!/bin/bash
# 00:20: the selftest and the l2 kill row passed, the s20 kill row differed from the PRINTED E276 P1 numbers by their print resolution only (tolerance widened, named in e346_arm.py): now the smokes, then the 9 E355 arms (serial --after chain, s20 first)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
S=temp/_session_scratch_s292
say() { echo "$(date +%H:%M:%S) $*" >> $S/night_go.log; }
say "smokes start"
nice -n 19 bash temp/e346/check_e355.sh smokes >> temp/e346/out/check_stdout.log 2>&1
rc=$?
say "smokes exit $rc"
if [ $rc -eq 0 ]; then
  grep -E '^env ' temp/e346/start_arms_e355.txt | while IFS= read -r line; do
    eval "$line" >> temp/e346/out/arms_launch.log 2>&1 &
    sleep 3
  done
  say "E355 arms launched (serial --after chain)"
  wait
else
  say "smokes FAILED: the arms are NOT launched; read temp/e346/out/check_e355.log"
fi
