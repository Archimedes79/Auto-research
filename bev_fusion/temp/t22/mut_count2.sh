#!/bin/bash
# scratch: counts the planted-bug results of the critic-round fixes
cp ~/mutk2/mut_K2.log /mnt/c/Auto-research/bev_fusion/temp/t22/mut_K2.log
echo "killed:   $(grep -c '^KILLED' ~/mutk2/mut_K2.log)"
echo "survived: $(grep -c '^SURVIVED' ~/mutk2/mut_K2.log)"
echo "skipped:  $(grep -c '^SKIP' ~/mutk2/mut_K2.log)"
grep -E '^SURVIVED|^SKIP' ~/mutk2/mut_K2.log | cut -c1-200
tail -1 ~/mutk2/mut_K2.log
