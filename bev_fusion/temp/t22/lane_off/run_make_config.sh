#!/bin/bash
# scratch: write training/configs/t20_base_nol4.json and preview it through trainsuite (nothing starts: no --go)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY temp/t22/lane_off/make_config.py || exit 1
cd ../autoresearch && AR_PROJECT=../bev_fusion python3 -m trainsuite start ../bev_fusion/training/configs/t20_base_nol4.json --id t20.base_nol4_s0 --no-card 2>&1 | tail -15
