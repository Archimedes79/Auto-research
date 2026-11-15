#!/bin/bash
# scratch: the three test files edited after the review chat's findings (CPU, low priority); the student-encoder test is also run as on a clean checkout (weight file 'missing' by a wrapper, nothing on disk touched)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
one() {
  name="$1"; shift
  t0=$(date +%s)
  out=$("$@" 2>&1); rc=$?
  echo "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )  $name ($(( $(date +%s) - t0 )) s, exit $rc) | $(echo "$out" | tail -1 | cut -c1-200)"
  [ $rc -ne 0 ] && echo "$out" | grep -E 'FAIL|Error' | head -6 | cut -c1-240
}
ls -d temp/student_chain_* 2>/dev/null | wc -l | sed 's/^/student_chain folders before: /'
one "student chain"   env CUDA_VISIBLE_DEVICES= nice -n 10 $PY tests/test_student_chain.py
ls -d temp/student_chain_* 2>/dev/null | wc -l | sed 's/^/student_chain folders after: /'
one "student encoder (weights present)" nice -n 10 $PY tests/test_student_encoder.py
one "student encoder (weights 'missing')" nice -n 10 $PY temp/t22/hide_weights_wrapper.py
one "stream tools"    nice -n 10 $PY tests/test_stream_tools.py
