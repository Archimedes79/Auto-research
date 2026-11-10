#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
python3 ../autoresearch/tools/schedule.py set holds.close done || exit 1
python3 ../autoresearch/tools/hold.py show | cut -c1-110
