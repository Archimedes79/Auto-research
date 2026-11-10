#!/bin/bash
# scratch: self-test of the release condition of rearm_B2_after_chain7.sh (dry mode; never calls the gate)
cd /mnt/c/Auto-research/bev_fusion || exit 1
bash -n temp/t22/rearm_B2_after_chain7.sh && echo "syntax ok"
touch temp/t22/_fake_done7
B2_DRY=1 B2_FLAG=temp/t22/_fake_done7 timeout 20 bash temp/t22/rearm_B2_after_chain7.sh
echo "exit $? (expected 0 with 'exists' + 'dry run' lines)"
rm -f temp/t22/_fake_done7
# real flag must not exist now, chain7 must be alive
[ -f temp/lane_tests/DONE7 ] && echo "DONE7 exists (unexpected)" || echo "DONE7 absent (expected)"
pgrep -f "lane_tests/chain7\.sh" > /dev/null && echo "chain7 alive (expected)" || echo "chain7 NOT alive"
pgrep -f "t22/night_B2\.sh" > /dev/null && echo "night_B2 pattern matches something (unexpected)" || echo "no process matches t22/night_B2.sh (expected)"
