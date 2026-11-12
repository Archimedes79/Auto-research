cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash tools/env/_py14.sh temp/gt2/pair_diag.py e284_h2_s0 > temp/gt2/e284_diag.log 2>&1
echo DIAG284_DONE >> temp/gt2/e284_diag.log
