#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 ../autoresearch/tools/experiment.py check E279 2>&1 | tail -2
bash -n temp/e276/launch_chain.sh && echo syntax ok || exit 2
[ -f temp/e276/CHAIN_DONE ] && mv temp/e276/CHAIN_DONE temp/e276/CHAIN_DONE.e277
bash temp/e276/restart_chain.sh
sleep 5
tail -3 temp/e276/chain.log
