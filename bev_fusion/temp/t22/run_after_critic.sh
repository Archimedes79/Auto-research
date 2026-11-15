#!/bin/bash
# scratch: the tests touched by the critic's fixes (CPU, 2 threads, low priority), one line each; the GPU stays hidden except for none of them needing it
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
one() {
  name="$1"; shift
  t0=$(date +%s)
  out=$("$@" 2>&1); rc=$?
  echo "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )  $name ($(( $(date +%s) - t0 )) s, exit $rc) | $(echo "$out" | tail -1 | cut -c1-220)"
  [ $rc -ne 0 ] && echo "$out" | grep -E 'FAIL|Error|assert' | head -8 | cut -c1-260
  return 0
}
one "student chain"        env CUDA_VISIBLE_DEVICES= nice -n 10 $PY tests/test_student_chain.py
one "startable selftest"   env CUDA_VISIBLE_DEVICES= nice -n 10 $PY training/startable.py
one "interface selftest"   env CUDA_VISIBLE_DEVICES= nice -n 10 $PY training/interface.py
one "bev_modules selftest" env CUDA_VISIBLE_DEVICES= nice -n 10 $PY evaluation/bev_modules.py --selftest
one "lane_branch selftest" env CUDA_VISIBLE_DEVICES= nice -n 10 $PY training/lane_branch.py
ls -d temp/student_chain_* 2>/dev/null | wc -l | sed 's/^/student_chain folders left: /'
