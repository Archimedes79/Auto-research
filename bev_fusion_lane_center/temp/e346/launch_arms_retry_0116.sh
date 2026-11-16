#!/bin/bash
# E355 step 0, restart of the arm chain at 2026-11-16 01:25 (lane lead, supervision): E355.s25 failed at 01:11 with OSError 12 (WSL page cache, gzip read), so s31 ... c25 were blocked.
# The page cache was dropped (root, drop_caches) at 01:22. This script resubmits the 8 remaining arms with the SAME command lines as start_arms_e355.txt (serial --after chain, s25 after the finished s20),
# only --retries 0 -> 1 (one more attempt on an infrastructure error). Nothing of the card (arms, rule, data) changes. usage: wsl bash temp/e346/launch_arms_retry_0116.sh ; log temp/e346/out/arms_launch_retry.log
cd "$(dirname "$0")/../.."
echo "retry launch $(date)" >> temp/e346/out/arms_launch_retry.log
while IFS= read -r line; do
  bash -c "$line" >> temp/e346/out/arms_launch_retry.log 2>&1 &
  sleep 3
done < <(grep -E '^env ' temp/e346/start_arms_e355.txt | grep -vE -- '--id E355\.s20 ' | sed 's/--retries 0/--retries 1/')
wait
echo "retry launchers ended $(date)" >> temp/e346/out/arms_launch_retry.log
