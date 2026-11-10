#!/bin/bash
# scratch: counts the planted-bug results of section K and keeps a copy of the log in temp/t22
cp ~/mutk/mut_K.log /mnt/c/Auto-research/bev_fusion/temp/t22/mut_K.log
echo "killed:   $(grep -c '^KILLED' ~/mutk/mut_K.log)"
echo "survived: $(grep -c '^SURVIVED' ~/mutk/mut_K.log)"
echo "skipped:  $(grep -c '^SKIP' ~/mutk/mut_K.log)"
tail -1 ~/mutk/mut_K.log
