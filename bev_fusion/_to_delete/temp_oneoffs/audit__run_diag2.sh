#!/bin/bash
# one-off audit: BatchNorm statistics of the EMA weights, and the velocity probe (run names keep "t1_r" in the command line)
PY=~/miniconda3/envs/gcar/bin/python
$PY temp/audit/diag_bn.py runs/t1_r3_combined --n 300
$PY temp/audit/probe_velocity.py runs/t1_r3_combined --steps 400
$PY temp/audit/probe_velocity.py runs/t1_r0 --steps 400
