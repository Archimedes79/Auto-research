#!/bin/bash
# E320 scratch: install qends_eval / jmaneuver / jdiag atomically (cp to a temp name in the same folder, then mv), then py_compile + the self-tests of the installed files
cd "$(dirname "$0")/../.."
for n in qends_eval jmaneuver jdiag; do
  cmp -s evaluation/$n.py temp/e320/orig/$n.pysrc || { echo "REFUSED: evaluation/$n.py changed since the orig copy was made"; exit 1; }
done
for n in qends_eval jmaneuver jdiag; do
  cp temp/e320/dev/$n.pysrc evaluation/.$n.py.e320tmp && mv -f evaluation/.$n.py.e320tmp evaluation/$n.py && echo "installed evaluation/$n.py"
done
python3 -m py_compile evaluation/qends_eval.py evaluation/jmaneuver.py evaluation/jdiag.py && echo "py_compile ok"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh evaluation/qends_eval.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jmaneuver.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jdiag.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jends_eval.py selftest 2>&1 | tail -n 1
