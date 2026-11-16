#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
while pgrep -f 'mut_chain3' > /dev/null; do sleep 3; done
python3 temp/e291/mutate.py p17_ok_is_valid_only p11b_no_detach_no_nograd > temp/_session_scratch_e291/mut_p2.log 2>&1
echo "rc $?" >> temp/_session_scratch_e291/mut_p2.log
python3 temp/_session_scratch_e291/mut_jdiag.py > temp/_session_scratch_e291/mut_jd.log 2>&1
echo "rc $?" >> temp/_session_scratch_e291/mut_jd.log
