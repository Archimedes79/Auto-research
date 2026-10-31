#!/usr/bin/env bash
# The settings of the thin-line experiment (steps seeds sigma noise); results to trunk_thin_lines_result2.txt.
# Run from the bev_fusion folder: wsl bash temp/checks/run_thin_lines.sh
py=~/miniconda3/envs/gcar/bin/python
out=temp/checks/trunk_thin_lines_result2.txt
: > "$out"
for cfg in "100 3 1.0 0.05" "300 3 1.0 0.05" "1500 2 2.0 0.15"; do
  $py temp/checks/trunk_thin_lines.py $cfg 2>&1 | tail -4 >> "$out"
done
cat "$out"
