#!/bin/bash
PY=~/miniconda3/envs/gcar/bin/python
$PY temp/audit/bench.py temp/audit/t1_r4_b1.json 4 2>&1 | grep -v -i warn | tail -3
