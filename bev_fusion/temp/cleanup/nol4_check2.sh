#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
AR_PROJECT=../bev_fusion python3 -m Training check ../bev_fusion/temp/cleanup/nol4_job.json 2>&1 | tail -n 14 | cut -c1-230
