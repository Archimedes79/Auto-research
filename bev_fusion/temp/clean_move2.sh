#!/bin/bash
# scratch: the last scripts of the round into runner_round_b, tagged and moved by clean_temp.py
cd /mnt/c/Auto-research/bev_fusion/temp || exit 1
mkdir -p runner_round_b
for f in final_real.sh final_real.log move_mine_copy.sh restore_others.sh clean_dry.sh clean_move.sh; do [ -e "$f" ] && cp "$f" runner_round_b/; done
ls runner_round_b
cd /mnt/c/Auto-research || exit 1
AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py runner_round_b --move --force 2>&1 | tail -6 | cut -c1-230
echo "exit ${PIPESTATUS[0]}"
cd bev_fusion/temp && rm -f final_real.sh final_real.log move_mine_copy.sh restore_others.sh clean_dry.sh clean_move.sh
ls /mnt/c/Auto-research/bev_fusion/_to_delete
