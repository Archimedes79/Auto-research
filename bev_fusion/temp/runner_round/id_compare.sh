#!/bin/bash
# scratch: is the id of the stored predictions the same from every working folder / entry point?
PY=~/miniconda3/envs/gcar/bin/python
B=/mnt/c/Auto-research/bev_fusion; A=/mnt/c/Auto-research/autoresearch
echo "1 runner from bev_fusion:            $(cd $B && $PY -m runner Runner.plan runs/t23_C_s1 small.val 2>&1 | grep '^results')"
echo "2 runner from autoresearch:          $(cd $A && PYTHONPATH=$B $PY -m runner Runner.plan runs/t23_C_s1 small.val --project $B 2>&1 | grep '^results')"
echo "3 evaluation from autoresearch:      $(cd $A && AR_PROJECT=../bev_fusion $PY -m evalsuite Evaluation.check runs/t23_C_s1 small.val 2>&1 | grep '^results')"
echo "4 evaluation from bev_fusion:        $(cd $B && PYTHONPATH=$A AR_PROJECT=$B $PY -m evalsuite Evaluation.check runs/t23_C_s1 small.val 2>&1 | grep '^results')"
echo "5 runner again from bev_fusion:      $(cd $B && $PY -m runner Runner.plan runs/t23_C_s1 small.val 2>&1 | grep '^results')"
