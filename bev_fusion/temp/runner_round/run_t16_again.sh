#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
log=temp/cleanup/base_t16_again.log
~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py runs/t16_full_s0 > "$log" 2>&1
echo "exit $?" >> "$log"
md5sum evaluation/bev_adapter.py evaluation/bev_free.py evaluation/bev_eval.py >> "$log"
