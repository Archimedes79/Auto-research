#!/bin/bash
f=$($HOME/miniconda3/envs/gcar/bin/python -c "import nuscenes.nuscenes as n; print(n.__file__)")
echo $f
grep -n "__load_table__" -B1 -A6 $f | head -60
