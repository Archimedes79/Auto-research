#!/bin/bash
# scratch: the autocast-order bug on the private copy ~/mutk2: the older-frame no_grad call BEFORE the current-frame call with a graph; the new autocast case of test_student_chain must fail
cd ~/mutk2/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
cp /mnt/c/Auto-research/bev_fusion/tests/test_student_chain.py tests/test_student_chain.py
cp /mnt/c/Auto-research/bev_fusion/training/feature_cache.py training/feature_cache.py
cp training/feature_cache.py temp/t22/feature_cache.keep
python3 - <<'EOF'
p = "training/feature_cache.py"
s = open(p, encoding="utf-8").read()
a = "            old = self._run_student_older(batch) if stud is not None and self.cfg.student_frames == \"all\" and Fn > 1 else None\n"
b = "            stud, side = (self._run_student(batch) if self.student is not None else None), {}\n"
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, "")
s = s.replace(b, "            old = self._run_student_older(batch) if self.student is not None and self.cfg.student_frames == \"all\" and Fn > 1 else None\n" + b)
open(p, "w", encoding="utf-8").write(s)
EOF
$PY tests/test_student_chain.py > temp/t22/mut_K4.out 2>&1
rc=$?
cp temp/t22/feature_cache.keep training/feature_cache.py
if [ $rc -ne 0 ]; then echo "KILLED   older-frame call before the current-frame call (exit $rc): $(grep -m1 -E 'FAIL|Error' temp/t22/mut_K4.out | cut -c1-200)"; else echo "SURVIVED the swapped order"; fi
