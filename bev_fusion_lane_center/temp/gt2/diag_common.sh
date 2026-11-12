cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash tools/env/_py14.sh temp/gt2/pair_diag_common.py e284_h2_s0 e287_p2_s0 e308_t_s0 > temp/gt2/common_s0.log 2>&1
bash tools/env/_py14.sh temp/gt2/pair_diag_common.py e284_h2_s1 e287_p2_s1 e308_t_s1 > temp/gt2/common_s1.log 2>&1
echo DIAGC_DONE >> temp/gt2/common_s1.log
