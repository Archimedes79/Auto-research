#!/bin/bash
# scratch: schedule state of the frozen update (done) and the gate C0 (armed)
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
python3 tools/schedule.py set frozen.update done 2>&1 | cut -c1-200
python3 tools/schedule.py show 2>&1 | head -6 | cut -c1-200
