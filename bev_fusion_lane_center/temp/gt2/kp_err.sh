cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash tools/env/_py14.sh temp/gt2/kp_err.py e284_h2_s0 e287_p2_s0 e308_t_s0 > temp/gt2/kp_err_s0.log 2>&1
echo KPERR_DONE >> temp/gt2/kp_err_s0.log
