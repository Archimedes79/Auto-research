#!/bin/bash
cd ~/mutk2/bev_fusion || exit 1
grep -n -E 'FAIL|Traceback|CheckpointError|RuntimeError' temp/t22/mut_K4.out | head -6 | cut -c1-250
tail -3 temp/t22/mut_K4.out | cut -c1-250
