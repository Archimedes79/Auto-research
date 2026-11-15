#!/bin/bash
# scratch: the gates after "stream temporal" of tests/run_gates.sh, which did not run (another chat edited run_gates.sh while it was running); CPU, low priority; one line per gate
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
one() {
  name="$1"; shift
  t0=$(date +%s)
  out=$("$@" 2>&1); rc=$?
  echo "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )  $name ($(( $(date +%s) - t0 )) s, exit $rc) | $(echo "$out" | tail -1 | cut -c1-160)"
  [ $rc -ne 0 ] && echo "$out" | grep -E 'FAIL|Error|error' | head -5 | cut -c1-220
}
one "stream tools"          nice -n 10 $PY tests/test_stream_tools.py
one "student encoder"       nice -n 10 $PY tests/test_student_encoder.py
one "evaluate door"         nice -n 10 $PY tests/test_evaluate_door.py
one "report build"          nice -n 10 $PY tests/test_report_build.py
one "trainkit selftest"     nice -n 10 bash -c "cd ../autoresearch && $PY -m trainkit.selftest"
one "trainkit equivalence"  nice -n 10 $PY tests/test_trainkit_equivalence.py
echo done
