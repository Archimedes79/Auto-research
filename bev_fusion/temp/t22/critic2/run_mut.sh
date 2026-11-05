#!/bin/bash
# usage: run_mut.sh <label> <file relative to mut/bev_fusion/training> <python-snippet that edits text variable s>
set -u
M=/mnt/c/Auto-research/bev_fusion/temp/t22/critic2
label=$1; rel=$2; edit=$3
f=$M/mut/bev_fusion/training/$rel
cp $f $f.orig
~/miniconda3/envs/gcar/bin/python - "$f" <<PY
import sys
f = sys.argv[1]
s = open(f, encoding="utf-8").read()
t = s
$edit
assert t != s, "mutation did not change the file"
open(f, "w", encoding="utf-8").write(t)
PY
cd $M/mut/bev_fusion && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python tests/test_student_chain.py > $M/mut_$label.log 2>&1
echo "$label exit=$?"
grep -c "  FAIL" $M/mut_$label.log
grep "  FAIL" $M/mut_$label.log | cut -c1-220 | head -8
tail -3 $M/mut_$label.log | cut -c1-200
mv $f.orig $f
