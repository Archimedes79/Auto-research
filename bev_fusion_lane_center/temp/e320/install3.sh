#!/bin/bash
# E320 scratch: install the critic fixes + the lead's read-outs atomically (cp to a temp name in the same folder, then mv), then py_compile + the self-tests of the installed files
cd "$(dirname "$0")/../.."
L="jends training/grid|jends_train training/train|qends_eval evaluation|jmaneuver evaluation|jends_eval evaluation|jdiag evaluation"
IFS='|'
for pair in $L; do
  set -- $pair; n=${pair%% *}; dir=${pair##* }
  cmp -s $dir/$n.py temp/e320/orig/$n.pysrc || { echo "REFUSED: $dir/$n.py changed since the orig copy was made"; exit 1; }
done
for pair in $L; do
  n=${pair%% *}; dir=${pair##* }
  cp temp/e320/dev/$n.pysrc $dir/.$n.py.e320tmp && mv -f $dir/.$n.py.e320tmp $dir/$n.py && echo "installed $dir/$n.py"
done
unset IFS
python3 -m py_compile training/train/jends_train.py training/grid/jends.py evaluation/qends_eval.py evaluation/jmaneuver.py evaluation/jends_eval.py evaluation/jdiag.py && echo "py_compile ok"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh training/grid/jends.py 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/qends_eval.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jmaneuver.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jends_eval.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jdiag.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh jends_train.py --help 2>&1 | grep -c "attach-r\|--attach\|--geom-w"
