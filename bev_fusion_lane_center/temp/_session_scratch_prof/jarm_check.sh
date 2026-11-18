#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 tools/jarm.py selftest 2>&1 | tail -12
python3 tools/jarm.py table e373_cb_s0 e378_x_s0 e376_mx_s0 2>&1 | tail -7
python3 tools/jarm.py run E383 f_s0 --seed 0 --head-seed 2 --ref e373_cb_s0 --dry -- --none-weight 0.25 2>&1 | cut -c1-120
